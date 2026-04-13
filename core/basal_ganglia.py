"""
Basal Ganglia — D1/D2 dual-pathway action selection with Active Inference.

Reference:
  Frank (2005)  "Dynamic dopamine modulation in the basal ganglia"
  Gurney, Prescott & Redgrave (2001)  "A computational model of action
      selection in the basal ganglia"
  Wilson & Kawaguchi (1996)  "In vivo intracellular recording ... MSNs"
  Friston (2010)  "The free-energy principle: a unified brain theory"
  Schultz (1998)  "Predictive reward signal of DA neurons"

Architecture:
  D1-MSN (direct/Go):   DA excites → disinhibits thalamus → facilitates action
  D2-MSN (indirect/NoGo): DA inhibits → maintains STN inhibition → suppresses
  Critic (ventral striatum): V(s) estimation via LIF population
  Active Inference:      G(a) = -pragmatic(a) + ambiguity - β × epistemic(a)
                         BG D1 ≈ pragmatic, D2 ≈ cost/risk, world model ≈ epistemic

MSN bistable dynamics (Wilson & Kawaguchi 1996):
  Down state: τ_m ≈ 80ms, high threshold (quiescent)
  Up state:   τ_m ≈ 25ms, low threshold (ready to fire)
  Transition gated by cortical input level

Changes from legacy:
  1. D1/D2 pathway split replaces single actor
  2. Bistable MSN membrane dynamics (Up/Down state)
  3. Active Inference merged: BG directly computes G(a)
  4. Input gain derived from biophysics (no arbitrary 3.0× multiplier)
  5. Uses BasalGangliaConfig + ActiveInferenceConfig from config.py
  6. Weight init via init_weights() (PSP-target scaling)
  7. InhibitoryPool from new interneuron.py with inhibitory STDP
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .config import (
    BasalGangliaConfig,
    ActiveInferenceConfig,
    InhibitoryPoolConfig,
    NeuronConfig,
    init_weights,
)
from .free_energy import expected_free_energy
from .interneuron import InhibitoryPool
from .receptor import hill_response

if TYPE_CHECKING:
    from .astrocyte import AstrocyteField
    from .world_model import SNNWorldModel


# =====================================================================
# Bistable MSN helpers
# =====================================================================

# Bi & Poo (2001): A_minus / A_plus ≈ 1.05 — slight LTD bias ensures
# long-term weight stability.  Normalized: LTP amplitude = 1.0,
# LTD amplitude = 1.05 (preserves ratio without changing lr scale).
_STDP_LTD_LTP_RATIO: float = 1.05


def _msn_decay(
    cortical_drive: NDArray[np.float32],
    cfg: BasalGangliaConfig,
    threshold: float = 0.3,
) -> NDArray[np.float32]:
    """Per-neuron membrane decay switching between Up and Down state.

    Wilson & Kawaguchi (1996): MSN bistability.
      cortical_drive > threshold → Up state (fast τ, ready to fire)
      cortical_drive ≤ threshold → Down state (slow τ, quiescent)

    Returns per-neuron decay factor exp(-dt / τ_effective).
    """
    up_mask = cortical_drive > threshold
    tau = np.where(
        up_mask,
        cfg.tau_m_msn_up,
        cfg.tau_m_msn_down,
    )
    return np.exp(-cfg.ctx.dt / tau).astype(np.float32)


def _derive_input_gain(
    fan_in: int,
    ncfg: NeuronConfig,
    w_clip: float,
    target_rate: float = 0.05,
    mean_input_rate: float | None = None,
    headroom: float = 1.0,
) -> float:
    """Derive synaptic gain from AdEx rheobase — proper pA-scale current.

    The AdEx membrane equation (Brette & Gerstner 2005) is:
      C_m dV/dt = -g_L(V - E_L) + g_L Δ_T exp((V - V_T)/Δ_T) + I_syn - w

    At the saddle-node bifurcation, the minimum constant current for
    spiking (rheobase) is:
      I_rheo = g_L × (V_T - E_L - Δ_T) = g_L × (gap - delta_t)

    Gain scales I_syn so that expected_active inputs produce ~I_rheo,
    accounting for homeostatic column normalization (Turrigiano 2008)
    that clips weight column L2 norms to ``w_clip``.  After clipping,
    the effective per-synapse weight is approximately
    ``w_clip / sqrt(fan_in)``.

    Parameters
    ----------
    mean_input_rate : float, optional
        Actual mean input firing rate from the upstream population encoder.
        If provided, overrides target_rate for expected_active computation.
    headroom : float
        Multiplicative factor ensuring neurons reach threshold within the
        finite integration window despite DA modulation and Poisson variance.
        Accounts for: (a) finite n_substeps × dt vs τ_m, and
        (b) DA-pathway suppression at baseline for D1D2Actor.

    Returns gain in pA per unit input, matching the AdEx current scale.
    """
    gap = abs(ncfg.v_thresh - ncfg.v_rest)
    i_rheo = ncfg.g_L * (gap - ncfg.delta_t)  # pA (Brette & Gerstner 2005)
    rate = mean_input_rate if mean_input_rate is not None else target_rate
    expected_active = max(1.0, fan_in * rate)
    # Effective per-synapse weight after homeostatic column normalization
    effective_w = w_clip / np.sqrt(max(1.0, float(fan_in)))
    return float(headroom * i_rheo / (expected_active * effective_w))


def _adex_step_bg(
    v: NDArray[np.float32],
    w_adapt: NDArray[np.float32],
    I_syn: NDArray[np.float32],
    in_refrac: NDArray[np.bool_],
    ncfg: NeuronConfig,
    eff_v_thresh: float | NDArray[np.float32] | None = None,
    eff_g_L: float | NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    """Single AdEx integration step for BG populations.

    Returns: (v_new, w_adapt_new, spiked)
    """
    ctx = ncfg.ctx
    vt = eff_v_thresh if eff_v_thresh is not None else ncfg.v_thresh
    gL = eff_g_L if eff_g_L is not None else ncfg.g_L
    exp_term = np.exp(
        np.clip((v - vt) / ncfg.delta_t, -20.0, 10.0),
    )
    inv_Cm = 1.0 / ncfg.C_m
    F_v = inv_Cm * (
        -gL * (v - ncfg.v_rest)
        + gL * ncfg.delta_t * exp_term
        + I_syn - w_adapt
    )
    J_v = inv_Cm * (-gL + gL * exp_term)
    integrated = ctx.exp_euler_step(v, F_v, J_v)
    v_new = np.where(in_refrac, ncfg.v_reset, integrated)

    spiked = (v_new >= ncfg.v_spike_cutoff) & ~in_refrac
    v_new[spiked] = ncfg.v_reset
    w_adapt[spiked] += ncfg.b

    # Subthreshold adaptation
    w_adapt_new = (
        w_adapt * ncfg.w_decay
        + ncfg.a * (v_new - ncfg.v_rest) * ncfg.w_gain
    )
    return v_new, w_adapt_new, spiked


# =====================================================================
# LIF Critic (ventral striatum)
# =====================================================================

class SNNDeepCritic:
    """LIF-based Critic for value estimation (ventral striatum).

    Ventral striatal neurons encode V(s) as population firing rate
    (Schultz 1998; Samejima et al. 2005). Value read out as linear
    projection from hidden-layer rate-coded activity.

    Single LIF step per forward() call — no inner integration loops.
    V_trace EMA replaces peek/snapshot for TD error computation.

    Learning: three-factor STDP modulated by dopaminergic TD error
    with causal ±20ms timing window (Bi & Poo 2001).
    """

    # ── NetworkGraph layer interface ─────────────────────────────

    @property
    def num_inputs(self) -> int:
        return self._state_size

    @property
    def num_neurons(self) -> int:
        return self.config.hidden_size

    def __init__(self, state_size: int, config: BasalGangliaConfig,
                 mean_input_rate: float | None = None) -> None:
        self.config = config
        self._state_size = state_size
        h = config.hidden_size

        # ── AdEx NeuronConfig for critic population ───────────────────
        # Biophysical consistency: τ_m = C_m / g_L ⇒ g_L = C_m / τ_m
        # Default g_L=30nS assumes cortical τ≈9.4ms; ventral striatal
        # neurons with τ=15ms require g_L=18.7nS (C_m=281pF).
        #
        # Adaptation scaling: NeuronConfig defaults (a=4nS, b=80.5pA)
        # are calibrated for g_L=30nS.  When g_L changes, the voltage
        # effect of adaptation (b/g_L, a/g_L) must be preserved.
        # Without scaling, b/g_L grows from 2.68mV to 4.30mV, making
        # adaptation over-suppress spiking.
        _C_m = 281.0  # NeuronConfig default (Brette & Gerstner 2005)
        _g_L_eff = _C_m / config.tau_m_critic
        _g_L_ref = 30.0  # NeuronConfig default (Destexhe & Paré 1999)
        _adapt_scale = _g_L_eff / _g_L_ref
        self._ncfg = NeuronConfig(
            ctx=config.ctx,
            v_rest=config.v_rest,
            v_thresh=config.v_thresh,
            v_reset=config.v_reset,
            tau_m=config.tau_m_critic,
            g_L=_g_L_eff,
            a=4.0 * _adapt_scale,
            b=80.5 * _adapt_scale,
        )

        # ── Precomputed membrane decay ────────────────────────────────
        self._mem_decay: float = config.ctx.decay(config.tau_m_critic)

        # ── Input gain from biophysics (AdEx rheobase) ─────────────────
        # Integration headroom: with n_substeps = τ/dt, neurons reach
        # ~63% of steady-state.  Factor 1/(1-e^{-1}) ≈ 1.58 ensures
        # reliable spiking despite Poisson input variance.
        self._input_gain: float = _derive_input_gain(
            state_size, self._ncfg, config.w_clip_critic,
            mean_input_rate=mean_input_rate,
            headroom=1.58,
        )

        # ── Weights via principled init ───────────────────────────────
        self.w_h: NDArray[np.float32] = init_weights(state_size, h, excitatory=True)
        # Pre-normalize to homeostatic clip so the first update() does
        # not destructively rescale the weight distribution.  Column
        # normalization (Turrigiano 2008) then maintains this scale.
        for j in range(h):
            col_norm = float(np.linalg.norm(self.w_h[:, j]))
            if col_norm > config.w_clip_critic:
                self.w_h[:, j] *= config.w_clip_critic / col_norm

        # ── LIF state (Down state at rest) ────────────────────────────
        self.v_hidden: NDArray[np.float32] = np.full(
            h, config.v_rest, dtype=np.float32,
        )
        self.spikes_hidden: NDArray[np.bool_] = np.zeros(h, dtype=bool)
        self.refrac_hidden: NDArray[np.int32] = np.zeros(h, dtype=np.int32)

        # ── AdEx adaptation current ───────────────────────────────────
        self.w_adapt_hidden: NDArray[np.float32] = np.zeros(h, dtype=np.float32)

        # ── Astrocyte ATP coupling (Krok 1.3) ───────────────────────
        self._astrocyte: AstrocyteField | None = None
        self._zone_idx: NDArray[np.int32] | None = None

        # ── Rate-coded activation (EMA of spikes) ─────────────────────
        self.activation: NDArray[np.float32] = np.zeros(h, dtype=np.float32)
        self._rate_decay: float = config.ctx.decay(config.tau_m_critic)

        # ── NMDA slow synaptic current trace (Wang 2002) ─────────────
        self._i_nmda: NDArray[np.float32] = np.zeros(h, dtype=np.float32)
        self._nmda_decay: float = config.ctx.decay(100.0)  # τ_NMDA=100ms

        # ── NOTE: Value readout (w_v, b_v) moved to VTA circuit ──────
        # The VTA's VTACircuit.w_value replaces the old w_v linear readout.
        # VP pathway reads V(s) and PPTg reads V(s') from critic.activation
        # via shared w_value weight.  See core/vta.py.

        # ── InhibitoryPool for sparsity ───────────────────────────────
        self.inh_pool = InhibitoryPool(
            n_excitatory=h,
            config=InhibitoryPoolConfig(
                ctx=config.ctx,
                n_interneurons=max(4, h // 4),
                target_sparsity=0.15,
            ),
        )

        # ── Eligibility traces (three-factor STDP) ────────────────────
        self.e_h: NDArray[np.float32] = np.zeros((state_size, h), dtype=np.float32)
        self._trace_decay: float = config.ctx.decay(config.tau_e_critic)

        # ── Pre/post STDP traces ──────────────────────────────────────
        self._x_pre: NDArray[np.float32] = np.zeros(state_size, dtype=np.float32)
        self._x_post: NDArray[np.float32] = np.zeros(h, dtype=np.float32)
        self._pre_decay: float = config.ctx.decay(20.0)
        self._post_decay: float = config.ctx.decay(20.0)

        # ── Spike timing for causal STDP window (±20ms) ──────────────
        self._t_since_pre: NDArray[np.int32] = np.full(state_size, 1000, dtype=np.int32)
        self._t_since_post: NDArray[np.int32] = np.full(h, 1000, dtype=np.int32)
        self._stdp_window: int = 20

        # ── Receptor dose-response modulation (D2) ───────────────────
        self._receptor_gain: float = 1.0
        self._receptor_lr: float = 1.0

        # ── Homeostatic synaptic scaling state (Turrigiano 2004) ──────
        # Slow EMA of per-neuron firing rate.  NOT reset between episodes
        # — homeostatic regulation operates on a timescale of hours/days.
        self._homeo_rate: NDArray[np.float32] = np.zeros(h, dtype=np.float32)
        self._homeo_decay: float = config.ctx.decay(config.homeo_tau)
        self._homeo_counter: int = 0

        # ── Inference mode ─────────────────────────────────────────────
        # When True, forward() skips eligibility trace updates (pre/post
        # synaptic traces, STDP eligibility).  Used during V(s')
        # computation in observe() so that next-state integration does
        # not corrupt the act()-phase eligibility needed for credit
        # assignment.  Membrane dynamics and activation EMA still run
        # normally — only plasticity-related state is frozen.
        self._inference_mode: bool = False

    def set_astrocyte(
        self,
        astrocyte: AstrocyteField,
        zone_idx: NDArray[np.int32] | None = None,
    ) -> None:
        """Attach astrocyte for ATP-based threshold/leak modulation."""
        self._astrocyte = astrocyte
        if zone_idx is not None:
            self._zone_idx = zone_idx
        else:
            n = self.config.hidden_size
            self._zone_idx = np.linspace(
                0, astrocyte.config.n_zones - 1, n,
            ).astype(np.int32)

    def forward(self, state_spikes: NDArray[np.float32]) -> NDArray[np.float32]:
        """Single LIF step: compute hidden-layer spike activity and update eligibility.

        Returns:
            (hidden_size,) spike activity (float32) for downstream layers.
            Population activation (EMA of rates) readable via self.activation
            for VTA value readout.
        """
        state_f32 = state_spikes.astype(np.float32)
        cfg = self.config
        h = cfg.hidden_size

        # ── Event-based pre-synaptic trace ────────────────────────────
        if not self._inference_mode:
            self._x_pre *= self._pre_decay
            pre_binary = (state_f32 > 0.5).astype(np.float32)
            self._x_pre += pre_binary
            self._t_since_pre += 1
            self._t_since_pre[pre_binary > 0.5] = 0

        # ── Synaptic current ──────────────────────────────────────────
        current = (state_f32 @ self.w_h) * self._input_gain * self._receptor_gain

        # ── NMDA temporal integration (Wang 2002; Jahr & Stevens 1990) ─
        # Slow synaptic current: τ_NMDA=100ms extends effective integration
        # window by ~4-5× beyond membrane τ=15ms.
        # AMPA:NMDA ratio from config (Myme et al. 2003).
        _ampa_frac = cfg.ampa_nmda_ratio / (1.0 + cfg.ampa_nmda_ratio)
        _nmda_frac = 1.0 - _ampa_frac
        _nmda_compl = 1.0 - self._nmda_decay
        self._i_nmda = self._i_nmda * self._nmda_decay + _nmda_compl * current
        current = _ampa_frac * current + _nmda_frac * self._i_nmda

        # ── Single AdEx step ──────────────────────────────────────────
        in_refrac = self.refrac_hidden > 0
        self.refrac_hidden[in_refrac] -= 1

        # Noise in pA: g_L × noise_std_mV gives physiological current noise
        noise_std_pA = self._ncfg.g_L * cfg.membrane_noise_std
        noise = np.random.normal(
            0, noise_std_pA, h,
        ).astype(np.float32)
        I_total = current + noise

        # ATP modulation (Krok 1.3)
        if self._astrocyte is not None:
            zc = self._zone_idx
            vt_c = self._ncfg.v_thresh + self._astrocyte.threshold_shift[zc]
            gL_c = self._ncfg.g_L * self._astrocyte.leak_gain[zc]
        else:
            vt_c = gL_c = None

        self.v_hidden, self.w_adapt_hidden, self.spikes_hidden = _adex_step_bg(
            self.v_hidden, self.w_adapt_hidden, I_total, in_refrac, self._ncfg,
            eff_v_thresh=vt_c, eff_g_L=gL_c,
        )

        inh_current = self.inh_pool.step(
            self.spikes_hidden.astype(np.float32), v_exc=self.v_hidden,
        )
        self.v_hidden -= inh_current

        self.refrac_hidden[self.spikes_hidden] = cfg.refrac_period

        rate_compl = 1.0 - self._rate_decay
        self.activation = (
            self.activation * self._rate_decay
            + self.spikes_hidden.astype(np.float32) * rate_compl
        )

        # ── Homeostatic rate tracker (slow EMA, Turrigiano 2004) ──────
        hc = 1.0 - self._homeo_decay
        self._homeo_rate = (
            self._homeo_rate * self._homeo_decay
            + self.spikes_hidden.astype(np.float32) * hc
        )
        self._homeo_counter += 1

        # ── Event-based post-synaptic trace ───────────────────────────
        if not self._inference_mode:
            self._x_post *= self._post_decay
            self._x_post[self.spikes_hidden] += 1.0
            self._t_since_post += 1
            self._t_since_post[self.spikes_hidden] = 0

        # ── Membrane potential normalisation ──────────────────────────
        # Continuous [0,1] signal: 0 = at rest, 1 = at threshold.
        # Used for eligibility, readout, and evidence accumulation.
        ncfg = self._ncfg
        v_normalized = np.clip(
            (self.v_hidden - ncfg.v_rest) / (ncfg.v_thresh - ncfg.v_rest),
            0.0, 1.0,
        ).astype(np.float32)

        # ── Voltage-based eligibility (Clopath et al. 2010) ──────────
        # EMA form: e = e * decay + (1 - decay) * outer(v_pre, v_post)
        # (1-decay) converts sum to bounded EMA — eligibility stays in
        # [-1, 1] independent of n_substeps.  Captures ALL neurons'
        # membrane state, not just spiking neurons (~25% of population).
        if not self._inference_mode:
            _e_compl = 1.0 - self._trace_decay
            _v_post_centered = v_normalized - float(np.mean(v_normalized))
            self.e_h = self.e_h * self._trace_decay + _e_compl * np.outer(state_f32, _v_post_centered)

        # NOTE: Evidence accumulation and V(s) readout moved to VTA circuit.
        # VTA reads critic.activation via w_value (VP/PPTg pathways).
        # See core/vta.py — VTACircuit.store_prediction() and compute_rpe().

        return self.spikes_hidden.astype(np.float32)

    def update(self, td_error: float) -> None:
        """Three-factor STDP: Δw = lr × DA(td_error) × eligibility.

        Value readout (w_v, b_v) moved to VTA circuit — this method
        now only updates the hidden-layer weights (w_h).

        No arbitrary clips — learning bounded by:
        - VTA D2-autoreceptor-normalized RPE (Tobler 2005)
        - EMA eligibility traces (Clopath 2010)
        - Homeostatic column norm bounds (Turrigiano 2008)
        """
        td = float(td_error)
        cfg = self.config
        effective_lr = cfg.critic_lr * self._receptor_lr

        dw_h = effective_lr * td * self.e_h
        self.w_h += dw_h

        # Continuous homeostatic synaptic scaling (Turrigiano 2004, 2008)
        # Per-step multiplicative correction: dw/dt = -(1/τ_homeo) × (rate - target)/target × w
        # Discretized: scale = 1 + (dt/τ_homeo) × (target - rate)/target
        # Removes the periodic counter/modulo pattern — homeostasis is
        # a continuous biophysical process operating on every timestep.
        #
        # Silent neurons (homeo_rate < 1% of target) get a fixed
        # upscaling factor — metaplasticity (Abraham & Bear 1996).
        # Biologically, silent AMPA synapses retain NMDA receptors and
        # undergo spontaneous potentiation via stochastic vesicle release
        # (Bhatt et al. 2009; Kerchner & Nicoll 2008).
        # The correction is clipped to ±_HOMEO_CLIP to prevent single-step
        # over-correction (maximum ±50% of target per τ_homeo).  Silent
        # neurons receive the maximum positive correction = _HOMEO_CLIP.
        _HOMEO_CLIP = 0.5  # max fractional correction per τ_homeo window
        _alpha_h = cfg.ctx.dt / cfg.homeo_tau
        target = cfg.homeo_target_rate
        rate_err = np.where(
            self._homeo_rate > target * 0.01,  # active neuron
            (target - self._homeo_rate) / max(target, 1e-8),
            _HOMEO_CLIP,  # silent neuron: max positive drift (metaplasticity)
        ).astype(np.float32)
        np.clip(rate_err, -_HOMEO_CLIP, _HOMEO_CLIP, out=rate_err)
        scale = 1.0 + _alpha_h * rate_err
        self.w_h *= scale[np.newaxis, :]

    def reset_state(self) -> None:
        """Reset transient state between episodes. Weights preserved."""
        cfg = self.config
        # Neurons start in Down state at v_rest (Wilson & Kawaguchi 1996).
        # Transition to Up state requires cortical input.  Random uniform
        # initialization [-70, -55] adds 15mV noise that drowns the learned
        # weight signal — biologically unrealistic.
        self.v_hidden[:] = cfg.v_rest
        self.spikes_hidden.fill(False)
        self.refrac_hidden.fill(0)
        self.w_adapt_hidden.fill(0.0)
        self.activation.fill(0.0)
        self.e_h.fill(0.0)
        self._i_nmda.fill(0.0)
        self._x_pre.fill(0.0)
        self._x_post.fill(0.0)
        self._t_since_pre.fill(1000)
        self._t_since_post.fill(1000)
        self.inh_pool.reset_state()

    def set_plasticity_timescales(self, ne: float) -> None:
        """NE modulates eligibility trace decay (Aston-Jones & Cohen 2005)."""
        ne = float(np.clip(ne, 0.0, 1.0))
        ne_factor = 1.0 + ne * (self.config.tau_ne_compression - 1.0)
        eff_tau = self.config.tau_e_critic / ne_factor
        self._trace_decay = float(np.exp(-self.config.ctx.dt / eff_tau))

    def set_receptor_modulation(self, gain_mod: float, lr_mod: float) -> None:
        """Apply receptor dose-response modulation (Hill equation effects)."""
        self._receptor_gain = float(gain_mod)
        self._receptor_lr = float(lr_mod)


# =====================================================================
# D1/D2 MSN Actor (dorsal striatum)
# =====================================================================

class D1D2Actor:
    """Dual-pathway MSN actor with bistable dynamics and DA-modulated STDP.

    D1 (direct/Go):   DA excites → facilitates selected action
    D2 (indirect/NoGo): DA inhibits → suppresses competing actions

    High DA → D1 dominance → exploitation
    Low DA  → D2 dominance → caution / exploration

    Learning via DA-modulated Hebbian STDP (Frank 2005):
      D1: standard Hebbian STDP, LTP gated by positive DA (reward)
      D2: anti-Hebbian STDP, LTP gated by negative DA (punishment)
    No REINFORCE policy gradient — purely biological three-factor rule.

    Single LIF step per forward() call — no integration loops.
    """

    # ── NetworkGraph layer interface ─────────────────────────────────

    @property
    def num_inputs(self) -> int:
        return self._state_size

    @property
    def num_neurons(self) -> int:
        return self.action_dim

    def __init__(
        self,
        state_size: int,
        motor_dim: int,
        internal_dim: int,
        config: BasalGangliaConfig,
        mean_input_rate: float | None = None,
    ) -> None:
        self.config = config
        self._state_size = state_size
        # ── Population coding (Georgopoulos 1986) ─────────────────────
        # Each motor action is represented by a population of MSNs, not
        # a single neuron.  This gives robust rate estimates for spike-
        # based action selection and meaningful inhibitory competition.
        self.n_per_action: int = max(1, config.neurons_per_action)
        self._total_motor: int = motor_dim * self.n_per_action
        self.action_dim = self._total_motor + internal_dim
        self.motor_dim = motor_dim

        # ── D1 pathway weights (cortex → D1-MSN) ─────────────────────
        self.w_d1: NDArray[np.float32] = init_weights(
            state_size, self.action_dim, excitatory=True,
        )
        # ── D2 pathway weights (cortex → D2-MSN) ─────────────────────
        # D1 and D2 MSNs are separate neuronal populations.  While they
        # share cortical afferents, individual synaptic strengths are
        # independently established during synaptogenesis (Gerfen &
        # Surmeier 2011).  Independent random initialization provides
        # symmetry breaking so net evidence (D1-D2) is non-zero from
        # the start, enabling Go/NoGo learning immediately.
        self.w_d2: NDArray[np.float32] = init_weights(
            state_size, self.action_dim, excitatory=True,
        )
        # Pre-normalize to homeostatic clip (same rationale as critic)
        for w_mat in (self.w_d1, self.w_d2):
            for j in range(self.action_dim):
                col_norm = float(np.linalg.norm(w_mat[:, j]))
                if col_norm > config.w_clip:
                    w_mat[:, j] *= config.w_clip / col_norm

        # ── AdEx NeuronConfig for MSN (Up-state defaults) ────────────
        # Biophysical consistency: τ_m = C_m / g_L ⇒ g_L = C_m / τ_m
        # MSN Up-state τ=25ms with C_m=281pF → g_L=11.24nS
        # (Wilson & Kawaguchi 1996).  Forward pass overrides g_L per
        # neuron via bistable C_m/τ_eff, but ncfg.g_L must match
        # Up-state for consistent gain derivation.
        #
        # Adaptation scaling: same rationale as critic.  MSN g_L=11.24
        # vs reference 30 → b/g_L would be 7.16mV (47.7% of gap!)
        # without scaling.  With scaling, b/g_L = 2.68mV (17.9%).
        _C_m = 281.0  # NeuronConfig default (Brette & Gerstner 2005)
        _g_L_eff = _C_m / config.tau_m_msn_up
        _g_L_ref = 30.0  # NeuronConfig default (Destexhe & Paré 1999)
        _adapt_scale = _g_L_eff / _g_L_ref
        self._ncfg = NeuronConfig(
            ctx=config.ctx,
            v_rest=config.v_rest,
            v_thresh=config.v_thresh,
            v_reset=config.v_reset,
            tau_m=config.tau_m_msn_up,
            g_L=_g_L_eff,
            a=4.0 * _adapt_scale,
            b=80.5 * _adapt_scale,
        )

        # ── Input gain from biophysics (AdEx rheobase) ─────────────────
        # Integration headroom: with n_substeps × dt ≈ τ_m, the neuron
        # only reaches 63 % of steady-state voltage → multiply by
        # 1/(1−e^{-1}) ≈ 1.58 so mean input ≈ I_rheo within the window.
        self._input_gain: float = _derive_input_gain(
            state_size, self._ncfg, config.w_clip,
            mean_input_rate=mean_input_rate,
            headroom=1.58,
        )
        # D2-MSN gain compensation: at baseline DA the D2-receptor
        # Hill suppression reduces current to d2_net_mod ≈ 0.52 × I.
        # Without compensation D2 sits below rheobase and never fires,
        # killing NoGo learning.  Exact compensation — no safety margin.
        _baseline_da = config.baseline_da
        _d2_resp = hill_response(_baseline_da, config.d2_ec50, config.d2_hill_n)
        _d2_mod = 1.0 - config.d2_receptor_density * _d2_resp
        _d2_tonic = 1.0 + config.d2_tonic_boost_max * (1.0 - _baseline_da)
        _d2_net_mod = max(_d2_mod * _d2_tonic, 0.1)
        self._d2_gain_comp: float = 1.0 / _d2_net_mod

        # ── D1-MSN membrane state ────────────────────────────────────
        self.v_d1: NDArray[np.float32] = np.full(
            self.action_dim, config.v_rest, dtype=np.float32,
        )
        self.spikes_d1: NDArray[np.bool_] = np.zeros(self.action_dim, dtype=bool)
        self.refrac_d1: NDArray[np.int32] = np.zeros(self.action_dim, dtype=np.int32)
        self.rate_d1: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)
        self.w_adapt_d1: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)

        # ── NMDA slow synaptic current traces (Wang 2002) ─────────────
        self._i_nmda_d1: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)
        self._i_nmda_d2: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)
        self._nmda_decay: float = config.ctx.decay(100.0)  # τ_NMDA=100ms

        # ── D2-MSN membrane state ────────────────────────────────────
        self.v_d2: NDArray[np.float32] = np.full(
            self.action_dim, config.v_rest, dtype=np.float32,
        )
        self.spikes_d2: NDArray[np.bool_] = np.zeros(self.action_dim, dtype=bool)
        self.refrac_d2: NDArray[np.int32] = np.zeros(self.action_dim, dtype=np.int32)
        self.rate_d2: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)
        self.w_adapt_d2: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)

        # ── Rate EMA decay (matched to MSN membrane τ for consistent
        #    spike-rate estimation timescale) ──────────────────────────
        self._rate_decay: float = config.ctx.decay(config.tau_m_msn_up)

        # ── InhibitoryPool for D1 competition ─────────────────────────
        # Planert et al. (2010): MSN in vivo rates ~1-5 Hz.  The pool
        # target is a per-timestep firing rate, NOT a per-decision
        # action-level sparsity.  0.05 matches cortical target_rate.
        self.inh_pool_d1 = InhibitoryPool(
            n_excitatory=self.action_dim,
            config=InhibitoryPoolConfig(
                ctx=config.ctx,
                n_interneurons=max(2, self.action_dim // 2),
                target_sparsity=0.05,
            ),
        )
        # ── InhibitoryPool for D2 competition ─────────────────────────
        self.inh_pool_d2 = InhibitoryPool(
            n_excitatory=self.action_dim,
            config=InhibitoryPoolConfig(
                ctx=config.ctx,
                n_interneurons=max(2, self.action_dim // 2),
                target_sparsity=0.05,
            ),
        )

        # ── DA modulation state ───────────────────────────────────────
        self._da_level: float = 0.5

        # ── NE level for temperature modulation (Usher & Damasio 2000) ─
        self._ne_level: float = 0.3  # Baseline from NeuromodulatorConfig

        # ── Eligibility traces (DA-modulated Hebbian STDP, Frank 2005) ─
        self.e_d1: NDArray[np.float32] = np.zeros(
            (state_size, self.action_dim), dtype=np.float32,
        )
        self.e_d2: NDArray[np.float32] = np.zeros(
            (state_size, self.action_dim), dtype=np.float32,
        )
        self._trace_decay: float = config.ctx.decay(config.tau_e_actor)

        # ── Pre-synaptic traces ───────────────────────────────────────
        self._x_pre: NDArray[np.float32] = np.zeros(state_size, dtype=np.float32)
        self._pre_decay: float = config.ctx.decay(20.0)

        # ── Post-synaptic traces (Bi & Poo 2001: asymmetric window) ──
        # Required for the LTD arm of STDP: when post fires before pre,
        # synaptic weight should decrease.  Without these, eligibility
        # is always ≥ 0 and STDP degenerates to rate-based Hebbian.
        self._x_post_d1: NDArray[np.float32] = np.zeros(
            self.action_dim, dtype=np.float32,
        )
        self._x_post_d2: NDArray[np.float32] = np.zeros(
            self.action_dim, dtype=np.float32,
        )
        self._post_decay: float = config.ctx.decay(20.0)

        # ── Spike timing for causal STDP window (±20ms) ──────────────
        self._t_since_pre: NDArray[np.int32] = np.full(state_size, 1000, dtype=np.int32)
        self._t_since_d1_spike: NDArray[np.int32] = np.full(
            self.action_dim, 1000, dtype=np.int32,
        )
        self._t_since_d2_spike: NDArray[np.int32] = np.full(
            self.action_dim, 1000, dtype=np.int32,
        )
        self._stdp_window: int = 20

        # ── Action tracking ───────────────────────────────────────────
        self._last_net_evidence: NDArray[np.float32] | None = None
        self._last_action: int = -1
        self.last_internal_action: NDArray[np.float32] = np.array(
            [], dtype=np.float32,
        )
        # D1/D2 net evidence (exposed for Active Inference)
        # Per-action aggregated rates (population sum).
        self._d1_rates: NDArray[np.float32] = np.zeros(motor_dim, dtype=np.float32)
        self._d2_rates: NDArray[np.float32] = np.zeros(motor_dim, dtype=np.float32)

        # ── Evidence accumulators (Gold & Shadlen 2007) ───────────────
        # GPi/SNr integrates D1/D2 spikes over the decision window as
        # cumulative counts, not instantaneous rates.  This gives integer-
        # scale evidence with meaningful softmax discrimination.
        # Separate from rate EMA (used by Active Inference readout).
        self._spike_count_d1: NDArray[np.float32] = np.zeros(
            self.action_dim, dtype=np.float32,
        )
        self._spike_count_d2: NDArray[np.float32] = np.zeros(
            self.action_dim, dtype=np.float32,
        )
        self._n_forward: int = 0

        # ── Membrane potential accumulators ────────────────────────────
        # Graded voltage carries more information than binary spikes
        # in small MSN populations.  Accumulated normalised membrane
        # potential = continuous-valued analogue of spike count
        # (Cisek 2007; Priebe & Ferster 2008).
        self._v_accum_d1: NDArray[np.float32] = np.zeros(
            self.action_dim, dtype=np.float32,
        )
        self._v_accum_d2: NDArray[np.float32] = np.zeros(
            self.action_dim, dtype=np.float32,
        )

        # ── Fast epistemic drive (error neuron → D1 excitability) ─────
        self._epistemic_drive: float = 0.0

        # ── Receptor dose-response modulation (D2) ───────────────────
        self._receptor_gain: float = 1.0
        self._receptor_lr: float = 1.0

        # ── Homeostatic synaptic scaling state (Turrigiano 2004) ──────
        self._homeo_rate_d1: NDArray[np.float32] = np.zeros(
            self.action_dim, dtype=np.float32,
        )
        self._homeo_rate_d2: NDArray[np.float32] = np.zeros(
            self.action_dim, dtype=np.float32,
        )
        self._homeo_decay: float = config.ctx.decay(config.homeo_tau)
        self._homeo_counter: int = 0

        # ── Astrocyte ATP coupling (Krok 1.3) ─────────────────────────
        self._astrocyte: AstrocyteField | None = None
        self._zone_idx: NDArray[np.int32] | None = None

        # ── Inference mode ─────────────────────────────────────────────
        # See SNNDeepCritic._inference_mode for rationale.
        self._inference_mode: bool = False

    def set_astrocyte(
        self,
        astrocyte: AstrocyteField,
        zone_idx: NDArray[np.int32] | None = None,
    ) -> None:
        """Attach astrocyte for ATP-based threshold/leak modulation."""
        self._astrocyte = astrocyte
        if zone_idx is not None:
            self._zone_idx = zone_idx
        else:
            n = self.action_dim
            self._zone_idx = np.linspace(
                0, astrocyte.config.n_zones - 1, n,
            ).astype(np.int32)

    def set_da_level(self, da: float) -> None:
        """Set DA modulation: high DA → D1 excitation, D2 suppression."""
        self._da_level = float(np.clip(da, 0.0, 1.0))

    def set_ne_level(self, ne: float) -> None:
        """Set NE level for temperature-modulated action selection.

        Reference: Humphries, Stewart & Gurney (2006); Usher & Damasio (2000).
        NE at 0.5 → focused exploitation; extremes → exploration.
        """
        self._ne_level = float(np.clip(ne, 0.0, 1.0))

    def set_epistemic_drive(self, error_rate: NDArray[np.float32]) -> None:
        """Fast epistemic path: error neuron error_rate → D1 excitability boost.

        High prediction error → explore novel states → boost Go pathway.
        """
        self._epistemic_drive = float(np.clip(np.mean(error_rate), 0.0, 1.0))

    def forward(
        self,
        state_spikes: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Single-step MSN dynamics with DA-modulated Hebbian STDP.

        No integration loops, no forced_action, no softmax fallback.

        Returns:
            (motor_dim,) motor action probabilities rescaled to [-1, 1].
            Internal action stored in self.last_internal_action.
        """
        state_f32 = state_spikes.astype(np.float32)
        cfg = self.config
        ad = self.action_dim
        self._n_forward += 1

        # ── Event-based pre-synaptic trace ────────────────────────────
        if not self._inference_mode:
            self._x_pre *= self._pre_decay
            pre_binary = (state_f32 > 0.5).astype(np.float32)
            self._x_pre += pre_binary
            self._t_since_pre += 1
            self._t_since_pre[pre_binary > 0.5] = 0

        # ── Synaptic currents ─────────────────────────────────────────
        current_d1 = (state_f32 @ self.w_d1) * self._input_gain * self._receptor_gain
        current_d2 = (state_f32 @ self.w_d2) * self._input_gain * self._receptor_gain * self._d2_gain_comp

        # ── NMDA temporal integration (Wang 2002; Jahr & Stevens 1990) ─
        # Slow synaptic current: τ_NMDA=100ms extends effective
        # integration window by ~4× beyond membrane τ=25ms.
        # AMPA:NMDA ratio from config (Myme et al. 2003).
        _ampa_frac = cfg.ampa_nmda_ratio / (1.0 + cfg.ampa_nmda_ratio)
        _nmda_frac = 1.0 - _ampa_frac
        _nmda_compl = 1.0 - self._nmda_decay
        self._i_nmda_d1 = self._i_nmda_d1 * self._nmda_decay + _nmda_compl * current_d1
        self._i_nmda_d2 = self._i_nmda_d2 * self._nmda_decay + _nmda_compl * current_d2
        current_d1 = _ampa_frac * current_d1 + _nmda_frac * self._i_nmda_d1
        current_d2 = _ampa_frac * current_d2 + _nmda_frac * self._i_nmda_d2

        # ── DA pathway-specific modulation (Frank 2005; Surmeier 2007) ─
        # D1 (Go): DA activates D1 receptors → cAMP↑/PKA → NMDA
        #   potentiation → INCREASED excitability.
        #   Phasic DA burst (high DA from TD>0) → D1 excited → Go.
        # D2 (NoGo): DA activates D2 receptors → cAMP↓ → REDUCED
        #   excitability.  High tonic DA suppresses D2; DA dip
        #   (negative TD) releases D2 from suppression → NoGo.
        # This replaces the uniform receptor_gain that applied identical
        # modulation to both pathways.
        da = float(np.clip(self._da_level, 0.0, 1.0))
        # D1: excitatory Hill response — high DA → more D1 current
        d1_resp = hill_response(da, cfg.d1_ec50, cfg.d1_hill_n)
        d1_mod = 1.0 + cfg.d1_receptor_density * d1_resp
        # D2: inhibitory Hill response — high DA → less D2 current
        d2_resp = hill_response(da, cfg.d2_ec50, cfg.d2_hill_n)
        d2_mod = 1.0 - cfg.d2_receptor_density * d2_resp
        # D2 tonic boost: low tonic DA → high-affinity D2 tonically
        # activated → D2 more excitable (Dreyer et al. 2010).
        # When DA is high, D2 is suppressed (d2_mod < 1) AND no tonic
        # boost.  When DA is low, D2 is released AND gets tonic boost.
        d2_tonic = 1.0 + cfg.d2_tonic_boost_max * (1.0 - da)
        current_d1 *= d1_mod
        current_d2 *= d2_mod * d2_tonic

        # ── Fast epistemic drive: error neurons → D1 excitability ─────
        # High prediction error boosts Go pathway (explore novel states)
        if self._epistemic_drive > 0.01:
            current_d1 *= 1.0 + self._epistemic_drive

        # ── Bistable MSN decay (Wilson & Kawaguchi 1996) ──────────────
        # Competitive threshold: median cortical drive gives natural
        # 50% up/down split (Plenz & Kitai 1998).  Floor at 0:
        # when cortical input is absent, all MSN stay in down state.
        cortical_drive = np.abs(current_d1) + np.abs(current_d2)
        drive_threshold = float(np.median(cortical_drive))
        up_mask = cortical_drive > drive_threshold
        tau_eff = np.where(up_mask, cfg.tau_m_msn_up, cfg.tau_m_msn_down).astype(np.float32)
        g_L_eff = self._ncfg.C_m / tau_eff  # per-neuron effective g_L

        # ── Shared AdEx constants ─────────────────────────────────────
        ncfg = self._ncfg
        ctx = ncfg.ctx
        inv_Cm = 1.0 / ncfg.C_m
        h = ctx.dt

        # ── ATP modulation (Krok 1.3) ─────────────────────────────────
        if self._astrocyte is not None:
            za = self._zone_idx
            eff_v_thresh = ncfg.v_thresh + self._astrocyte.threshold_shift[za]
            g_L_eff = g_L_eff * self._astrocyte.leak_gain[za]
        else:
            eff_v_thresh = ncfg.v_thresh

        # ── AdEx step: D1 ─────────────────────────────────────────────
        in_refrac_d1 = self.refrac_d1 > 0
        self.refrac_d1[in_refrac_d1] -= 1
        # Noise in pA: g_L_eff × noise_std_mV (biophysical current noise)
        noise_d1 = np.random.normal(0, g_L_eff * cfg.membrane_noise_std, ad).astype(np.float32)

        exp_term_d1 = np.exp(np.clip((self.v_d1 - eff_v_thresh) / ncfg.delta_t, -20.0, 10.0))
        F_d1 = inv_Cm * (
            -g_L_eff * (self.v_d1 - ncfg.v_rest)
            + g_L_eff * ncfg.delta_t * exp_term_d1
            + current_d1 + noise_d1 - self.w_adapt_d1
        )
        J_d1 = inv_Cm * (-g_L_eff + g_L_eff * exp_term_d1)
        integrated_d1 = ctx.exp_euler_step(self.v_d1, F_d1, J_d1)
        self.v_d1 = np.where(in_refrac_d1, ncfg.v_reset, integrated_d1)

        inh_d1 = self.inh_pool_d1.step(
            self.spikes_d1.astype(np.float32), v_exc=self.v_d1,
        )
        self.v_d1 -= inh_d1

        self.spikes_d1 = (self.v_d1 >= ncfg.v_spike_cutoff) & ~in_refrac_d1
        self.v_d1[self.spikes_d1] = ncfg.v_reset
        self.w_adapt_d1[self.spikes_d1] += ncfg.b
        self.w_adapt_d1 = (
            self.w_adapt_d1 * ncfg.w_decay
            + ncfg.a * (self.v_d1 - ncfg.v_rest) * ncfg.w_gain
        )
        self.refrac_d1[self.spikes_d1] = cfg.refrac_period

        # ── AdEx step: D2 ─────────────────────────────────────────────
        in_refrac_d2 = self.refrac_d2 > 0
        self.refrac_d2[in_refrac_d2] -= 1
        # Noise in pA: g_L_eff × noise_std_mV (biophysical current noise)
        noise_d2 = np.random.normal(0, g_L_eff * cfg.membrane_noise_std, ad).astype(np.float32)

        exp_term_d2 = np.exp(np.clip((self.v_d2 - eff_v_thresh) / ncfg.delta_t, -20.0, 10.0))
        F_d2 = inv_Cm * (
            -g_L_eff * (self.v_d2 - ncfg.v_rest)
            + g_L_eff * ncfg.delta_t * exp_term_d2
            + current_d2 + noise_d2 - self.w_adapt_d2
        )
        J_d2 = inv_Cm * (-g_L_eff + g_L_eff * exp_term_d2)
        integrated_d2 = ctx.exp_euler_step(self.v_d2, F_d2, J_d2)
        self.v_d2 = np.where(in_refrac_d2, ncfg.v_reset, integrated_d2)

        inh_d2 = self.inh_pool_d2.step(
            self.spikes_d2.astype(np.float32), v_exc=self.v_d2,
        )
        self.v_d2 -= inh_d2

        self.spikes_d2 = (self.v_d2 >= ncfg.v_spike_cutoff) & ~in_refrac_d2
        self.v_d2[self.spikes_d2] = ncfg.v_reset
        self.w_adapt_d2[self.spikes_d2] += ncfg.b
        self.w_adapt_d2 = (
            self.w_adapt_d2 * ncfg.w_decay
            + ncfg.a * (self.v_d2 - ncfg.v_rest) * ncfg.w_gain
        )
        self.refrac_d2[self.spikes_d2] = cfg.refrac_period

        # ── Rate EMA (for Active Inference readout) ─────────────────────
        rc = 1.0 - self._rate_decay
        self.rate_d1 = self.rate_d1 * self._rate_decay + self.spikes_d1.astype(np.float32) * rc
        self.rate_d2 = self.rate_d2 * self._rate_decay + self.spikes_d2.astype(np.float32) * rc

        # ── Homeostatic rate tracker (slow EMA, Turrigiano 2004) ──────
        hc = 1.0 - self._homeo_decay
        self._homeo_rate_d1 = (
            self._homeo_rate_d1 * self._homeo_decay
            + self.spikes_d1.astype(np.float32) * hc
        )
        self._homeo_rate_d2 = (
            self._homeo_rate_d2 * self._homeo_decay
            + self.spikes_d2.astype(np.float32) * hc
        )
        self._homeo_counter += 1

        # ── Event-based post-synaptic traces (Bi & Poo 2001) ─────────
        # Required for the LTD arm of STDP in eligibility below.
        if not self._inference_mode:
            self._x_post_d1 *= self._post_decay
            self._x_post_d1[self.spikes_d1] += 1.0
            self._t_since_d1_spike += 1
            self._t_since_d1_spike[self.spikes_d1] = 0

            self._x_post_d2 *= self._post_decay
            self._x_post_d2[self.spikes_d2] += 1.0
            self._t_since_d2_spike += 1
            self._t_since_d2_spike[self.spikes_d2] = 0

        # ── Evidence accumulation (Gold & Shadlen 2007) ───────────────
        # Spike counts over the decision window — integer-scale evidence
        # for robust action discrimination.
        self._spike_count_d1 += self.spikes_d1.astype(np.float32)
        self._spike_count_d2 += self.spikes_d2.astype(np.float32)

        # ── Membrane potential accumulation (Cisek 2007) ──────────────
        # Graded D1/D2 voltage captures continuous action competition.
        # Normalise to [0, 1]: 0 = rest, 1 = threshold.
        v_d1_norm = np.clip(
            (self.v_d1 - ncfg.v_rest) / (ncfg.v_thresh - ncfg.v_rest),
            0.0, 1.0,
        ).astype(np.float32)
        v_d2_norm = np.clip(
            (self.v_d2 - ncfg.v_rest) / (ncfg.v_thresh - ncfg.v_rest),
            0.0, 1.0,
        ).astype(np.float32)
        self._v_accum_d1 += v_d1_norm
        self._v_accum_d2 += v_d2_norm

        # ── Store per-action aggregated rates (population sum) ────────
        self._d1_rates = self.rate_d1[:self._total_motor].reshape(
            self.motor_dim, self.n_per_action,
        ).sum(axis=1)
        self._d2_rates = self.rate_d2[:self._total_motor].reshape(
            self.motor_dim, self.n_per_action,
        ).sum(axis=1)

        # ── GPi/SNr action competition via spike-count evidence ────────
        # GPi/SNr disinhibition is gated by MSN spike rate, not graded
        # membrane potential (Mink 1996; Gurney, Prescott & Redgrave 2001).
        # D1 (direct) spikes disinhibit thalamus → facilitate action (Go).
        # D2 (indirect) spikes maintain STN→GPi inhibition → suppress (NoGo).
        # Net evidence = D1 spike count − D2 spike count per action population.
        #
        # Exploration arises from intrinsic Poisson variability of spike
        # counts (Gold & Shadlen 2007).  With N spikes from a population
        # of n_per_action neurons × n_substeps timesteps, the coefficient
        # of variation CV ≈ 1/√N provides natural stochastic exploration
        # that decreases as evidence accumulates — no artificial softmax
        # temperature parameter needed.
        d1_counts = self._spike_count_d1[:self._total_motor].reshape(
            self.motor_dim, self.n_per_action,
        ).sum(axis=1)
        d2_counts = self._spike_count_d2[:self._total_motor].reshape(
            self.motor_dim, self.n_per_action,
        ).sum(axis=1)

        net_evidence = d1_counts - d2_counts

        # WTA: highest net evidence wins; ties broken randomly
        # (equal D1−D2 balance = no preference → uniform selection).
        max_ev = net_evidence.max()
        winners = np.flatnonzero(net_evidence == max_ev)
        action = int(np.random.choice(winners))
        self._last_action = action
        self._last_net_evidence = net_evidence.copy()

        # ── Voltage-based eligibility (Clopath et al. 2010) ──────────
        # EMA form captures ALL neurons' membrane state for credit
        # assignment matching the membrane-voltage readout in action
        # selection.  (1-decay) converts sum to bounded EMA.
        # Non-centered: v_d1_norm ∈ [0,1] gives positive eligibility
        # for active neurons.  Ratchet prevented by WTA-mediated
        # inhibition (suppresses losing actions' membrane → lower
        # eligibility) + column norm clipping + OpAL sign separation.
        if not self._inference_mode:
            _e_compl = 1.0 - self._trace_decay
            self.e_d1 = self.e_d1 * self._trace_decay + _e_compl * np.outer(state_f32, v_d1_norm)
            self.e_d2 = self.e_d2 * self._trace_decay + _e_compl * np.outer(state_f32, v_d2_norm)

        # ── Internal actions (WM gate) via MSN dynamics ─────────────
        # WM gate uses the same bistable MSN membrane state as motor
        # actions — no sigmoid.  The normalised D1 membrane potential
        # of the internal neuron(s) IS the gate signal [0, 1].
        if self.action_dim > self._total_motor:
            self.last_internal_action = v_d1_norm[self._total_motor:].copy()
        else:
            self.last_internal_action = np.array([], dtype=np.float32)

        return net_evidence.astype(np.float32)

    def update(self, td_error: float) -> None:
        """DA-modulated three-factor STDP (Frank 2005 Go/NoGo model).

        D1 (Go): positive TD → DA burst → strengthen active Go synapses (LTP)
        D2 (NoGo): negative TD → DA dip → strengthen active NoGo synapses (LTP)

        No REINFORCE grad_log_pi — purely Hebbian with dopaminergic gating.
        No arbitrary clips — bounded by Welford normalization + column norms.
        """
        td = float(td_error)
        cfg = self.config
        effective_lr = cfg.actor_lr * self._receptor_lr

        # ── Bidirectional DA-modulated STDP (Shen et al. 2008) ─────────
        # D1 (Go):  LTP on DA burst (positive TD);
        #           LTD on DA dip  (negative TD).
        # D2 (NoGo): LTP on DA dip  (negative TD).
        #
        # Shen et al. 2008 (Fig. 3, Table 1) demonstrated bidirectional
        # plasticity in BOTH D1- and D2-MSNs: spike-timing with DA
        # pause → D1 LTD (25% depression vs 35% LTP, ratio ≈ 0.7).
        # The earlier pure-OpAL model blocked D1 LTD citing "prolonged
        # depletion", but Shen et al. used brief 10-min antagonist
        # application — matching phasic DA dips, not chronic depletion.
        #
        # D1 LTD is critical for reversal learning: without it, D1
        # weights for a previously-rewarded action never decrease,
        # preventing exploration of alternatives after contingency
        # shift (Kravitz et al. 2012; Tai et al. 2012).
        #
        # D2 LTD on DA burst is NOT included: Shen et al. showed it
        # requires eCB signaling (Kreitzer & Malenka 2007) that is
        # minimal at resting rates.  Omitting D2 LTD is conservative
        # and preserves NoGo learning stability.

        # D1 (Go): LTP on positive TD (DA burst)
        if td > 0:
            dw_d1 = effective_lr * td * self.e_d1
            self.w_d1 += dw_d1

        # D2 (NoGo): LTP on negative TD (DA dip)
        # D1 (Go):  LTD on negative TD (DA dip) — Shen et al. 2008
        if td < 0:
            dw_d2 = effective_lr * (-td) * self.e_d2
            self.w_d2 += dw_d2

            dw_d1_ltd = effective_lr * cfg.ltd_ratio * td * self.e_d1
            self.w_d1 += dw_d1_ltd

        # Synaptic protein turnover removed: voltage-based EMA
        # eligibility inherently bounds updates via (1-decay) factor,
        # making explicit weight decay unnecessary.

        # Dale's law: excitatory synapses cannot go negative.
        # Floor = 0 (pure Dale's law).  Silent synapses exist
        # biologically (Kerchner & Nicoll 2008) and STDP reactivates
        # them when needed — no artificial minimum required.
        np.maximum(self.w_d1, 0.0, out=self.w_d1)
        np.maximum(self.w_d2, 0.0, out=self.w_d2)

        # Continuous homeostatic synaptic scaling (Turrigiano 2004, 2008)
        # Per-step multiplicative correction derived from homeo_tau.
        # D2 naturally gets more upscaling because DA suppresses D2
        # firing (Frank 2005) — the rate error is larger for D2.
        _HOMEO_CLIP = 0.5  # max fractional correction per τ_homeo window
        _alpha_h = cfg.ctx.dt / cfg.homeo_tau
        target = cfg.homeo_target_rate
        for w, hr in ((self.w_d1, self._homeo_rate_d1),
                      (self.w_d2, self._homeo_rate_d2)):
            rate_err = np.where(
                hr > target * 0.01,  # active neuron
                (target - hr) / max(target, 1e-8),
                _HOMEO_CLIP,  # silent neuron: max positive drift (metaplasticity)
            ).astype(np.float32)
            np.clip(rate_err, -_HOMEO_CLIP, _HOMEO_CLIP, out=rate_err)
            scale = 1.0 + _alpha_h * rate_err
            w *= scale[np.newaxis, :]
            np.maximum(w, 0.0, out=w)  # Maintain Dale's law

    def get_action(self) -> int:
        return self._last_action

    @property
    def action_entropy(self) -> float:
        """Decision uncertainty from evidence margin (Kiani & Shadlen 2009).

        Returns [0, 1]: 0 = one action clearly dominant, 1 = all equal.
        Based on winner–runner-up margin relative to evidence range.
        """
        if self._last_net_evidence is None:
            return 1.0
        ev = self._last_net_evidence
        if len(ev) < 2:
            return 0.0
        sorted_ev = np.sort(ev)[::-1]
        margin = sorted_ev[0] - sorted_ev[1]
        ev_range = sorted_ev[0] - sorted_ev[-1]
        if ev_range < 1e-8:
            return 1.0
        confidence = margin / ev_range
        return float(np.clip(1.0 - confidence, 0.0, 1.0))

    @property
    def pragmatic_values(self) -> NDArray[np.float32]:
        """Per-action D1 rate (population sum, Georgopoulos 1986)."""
        return self._d1_rates.copy()

    @property
    def cost_values(self) -> NDArray[np.float32]:
        """Per-action D2 rate (population sum)."""
        return self._d2_rates.copy()

    def reset_state(self) -> None:
        """Reset transient state. Weights preserved."""
        cfg = self.config
        # Start in Down state at v_rest (Wilson & Kawaguchi 1996).
        self.v_d1[:] = cfg.v_rest
        self.v_d2[:] = cfg.v_rest
        self.spikes_d1.fill(False)
        self.spikes_d2.fill(False)
        self.refrac_d1.fill(0)
        self.refrac_d2.fill(0)
        self.w_adapt_d1.fill(0.0)
        self.w_adapt_d2.fill(0.0)
        self.rate_d1.fill(0.0)
        self.rate_d2.fill(0.0)
        self.e_d1.fill(0.0)
        self.e_d2.fill(0.0)
        self._i_nmda_d1.fill(0.0)
        self._i_nmda_d2.fill(0.0)
        self._x_pre.fill(0.0)
        self._x_post_d1.fill(0.0)
        self._x_post_d2.fill(0.0)
        self._t_since_pre.fill(1000)
        self._t_since_d1_spike.fill(1000)
        self._t_since_d2_spike.fill(1000)
        self._last_net_evidence = None
        self._last_action = -1
        self._d1_rates.fill(0.0)
        self._d2_rates.fill(0.0)
        self._spike_count_d1.fill(0.0)
        self._spike_count_d2.fill(0.0)
        self._v_accum_d1.fill(0.0)
        self._v_accum_d2.fill(0.0)
        self._n_forward = 0
        self.inh_pool_d1.reset_state()
        self.inh_pool_d2.reset_state()

    def set_plasticity_timescales(self, ne: float) -> None:
        """NE modulates policy trace decay and exploration temperature."""
        ne = float(np.clip(ne, 0.0, 1.0))
        self._ne_level = ne  # used by softmax temperature in forward()
        ne_factor = 1.0 + ne * (self.config.tau_ne_compression - 1.0)
        eff_tau = self.config.tau_e_actor / ne_factor
        self._trace_decay = float(np.exp(-self.config.ctx.dt / eff_tau))

    def reset_spike_counts(self) -> None:
        """Reset evidence accumulators for a new decision cycle.

        Called at the start of each act() to clear accumulated spike
        counts from the previous decision window.  Models the GPi/SNr
        reset between decisions (Lo & Wang 2006).
        """
        self._spike_count_d1.fill(0.0)
        self._spike_count_d2.fill(0.0)
        self._v_accum_d1.fill(0.0)
        self._v_accum_d2.fill(0.0)
        self._n_forward: int = 0

    def gate_eligibility(self, selected_action: int) -> None:
        """Gate eligibility to the selected action channel only.

        DA reinforcement targets the synaptic pathways of the winning
        action channel (Redgrave, Gurney & Reynolds 2010).  Non-selected
        motor populations’ eligibility is zeroed so that update() only
        modifies the selected action’s weights.  Internal (non-motor)
        neurons are left intact.
        """
        for a in range(self.motor_dim):
            if a != selected_action:
                start = a * self.n_per_action
                end = start + self.n_per_action
                self.e_d1[:, start:end] = 0.0
                self.e_d2[:, start:end] = 0.0

    def set_receptor_modulation(self, gain_mod: float, lr_mod: float) -> None:
        """Apply receptor dose-response modulation (Hill equation effects)."""
        self._receptor_gain = float(gain_mod)
        self._receptor_lr = float(lr_mod)

    def efference_copy(self) -> NDArray[np.float32]:
        """Generate sub-threshold efference copy of D1/D2 activity.

        Returns the net D1−D2 sub-threshold activity for each motor action.
        Used during theta-sweep planning: efference copy feeds into the
        world model encoder to predict outcome of contemplated actions.
        No spike threshold — captures graded intent (Cisek 2007).
        Per-action population mean for consistent readout.
        """
        # Sub-threshold voltages relative to rest, normalized
        d1_sub = np.clip(
            (self.v_d1[:self._total_motor] - self.config.v_rest)
            / (self.config.v_thresh - self.config.v_rest),
            0.0, 1.0,
        ).reshape(self.motor_dim, self.n_per_action).mean(axis=1)
        d2_sub = np.clip(
            (self.v_d2[:self._total_motor] - self.config.v_rest)
            / (self.config.v_thresh - self.config.v_rest),
            0.0, 1.0,
        ).reshape(self.motor_dim, self.n_per_action).mean(axis=1)
        return (d1_sub - d2_sub).astype(np.float32)


# =====================================================================
# Active Inference integration
# =====================================================================

class ActiveInferenceModule:
    """Expected Free Energy action selection wrapping world model + BG.

    G(a) = -pragmatic(a) + ambiguity - β×epistemic(a)
      pragmatic ≈ D1-MSN rate (expected reward)
      cost      ≈ D2-MSN rate (expected risk)
      epistemic ≈ world model prediction uncertainty
      β modulated by NE (curiosity drive → exploration)
    """

    def __init__(
        self,
        world_model: "SNNWorldModel",
        config: ActiveInferenceConfig | None = None,
    ) -> None:
        self.world_model = world_model
        self.config = config or ActiveInferenceConfig()
        self.action_size = world_model.action_size

        # Diagnostic outputs
        self.last_epistemic_values: dict[int, float] = {}
        self.last_pragmatic_values: dict[int, float] = {}
        self.last_ambiguity_values: dict[int, float] = {}
        self.last_total_values: dict[int, float] = {}
        self.last_selected_action: int = 0

    def compute_epistemic_values(
        self,
        state_spikes: NDArray[np.float32],
        candidate_actions: list[int],
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Epistemic value and ambiguity per action.

        Returns:
            (epistemic_dict, ambiguity_dict) per candidate action.
            Ambiguity = ensemble variance = expected sensory entropy.
        """
        results = self.world_model.mental_rehearsal(
            state_spikes, candidate_actions,
        )
        epistemic: dict[int, float] = {}
        ambiguity: dict[int, float] = {}
        for action in candidate_actions:
            info = results[action]
            if self.config.uncertainty_method == "novelty":
                epistemic[action] = info.novelty
            else:
                epistemic[action] = self._variance_uncertainty(
                    state_spikes, action,
                )
            # Ambiguity = ensemble variance (expected observation entropy)
            ambiguity[action] = info.ensemble_variance
        return epistemic, ambiguity

    def _variance_uncertainty(
        self,
        state_spikes: NDArray[np.float32],
        action: int,
        n_samples: int = 3,
    ) -> float:
        """Prediction uncertainty via variance across perturbed inputs."""
        predictions: list[NDArray[np.float32]] = []
        for _ in range(n_samples):
            noise = np.random.normal(
                0, 0.05, state_spikes.shape,
            ).astype(np.float32)
            perturbed = np.clip(state_spikes + noise, 0.0, 1.0)
            result = self.world_model.mental_rehearsal(perturbed, [action])
            predictions.append(result[action].predicted_state)
        if len(predictions) < 2:
            return 0.0
        stacked = np.stack(predictions)
        return float(np.mean(np.var(stacked, axis=0)))

    def select_action(
        self,
        state_spikes: NDArray[np.float32],
        candidate_actions: list[int],
        actor: D1D2Actor | None = None,
        ne_level: float = 0.3,
    ) -> int:
        """Select action minimizing expected free energy G(a).

        If actor is provided, uses D1/D2 rates as pragmatic/cost values.
        """
        epistemic, ambiguity = self.compute_epistemic_values(
            state_spikes, candidate_actions,
        )
        self.last_epistemic_values = epistemic
        self.last_ambiguity_values = ambiguity

        # NE-modulated epistemic weight
        beta = (
            self.config.epistemic_weight
            + ne_level * self.config.ne_epistemic_boost
        )

        # Build G(a) per action
        total: dict[int, float] = {}
        prag_dict: dict[int, float] = {}
        for action in candidate_actions:
            if actor is not None and action < actor.motor_dim:
                pragmatic = float(actor.pragmatic_values[action])
                cost = float(actor.cost_values[action])
            else:
                pragmatic = 0.0
                cost = 0.0
            epist = epistemic.get(action, 0.0)
            amb = ambiguity.get(action, 0.0)
            g = expected_free_energy(
                pragmatic_value=pragmatic - cost,
                epistemic_value=epist,
                ambiguity=amb,
                epistemic_weight=beta,
            )
            total[action] = -g  # Higher = better (negate for selection)
            prag_dict[action] = pragmatic - cost

        self.last_pragmatic_values = prag_dict
        self.last_total_values = total

        # Softmax selection
        actions = list(total.keys())
        values = np.array([total[a] for a in actions], dtype=np.float32)
        shifted = values - np.max(values)
        temp = max(self.config.pragmatic_temperature, 1e-6)
        exp_vals = np.exp(shifted / temp)
        probs = exp_vals / (np.sum(exp_vals) + 1e-8)

        selected = int(np.random.choice(actions, p=probs))
        self.last_selected_action = selected
        return selected

    def select_action_greedy(
        self,
        state_spikes: NDArray[np.float32],
        candidate_actions: list[int],
        actor: D1D2Actor | None = None,
        ne_level: float = 0.3,
    ) -> int:
        """Greedy (argmax) variant for evaluation."""
        epistemic, ambiguity = self.compute_epistemic_values(
            state_spikes, candidate_actions,
        )
        beta = (
            self.config.epistemic_weight
            + ne_level * self.config.ne_epistemic_boost
        )
        total: dict[int, float] = {}
        for action in candidate_actions:
            if actor is not None and action < actor.motor_dim:
                pragmatic = float(actor.pragmatic_values[action])
                cost = float(actor.cost_values[action])
            else:
                pragmatic = 0.0
                cost = 0.0
            g = expected_free_energy(
                pragmatic_value=pragmatic - cost,
                epistemic_value=epistemic.get(action, 0.0),
                ambiguity=ambiguity.get(action, 0.0),
                epistemic_weight=beta,
            )
            total[action] = -g
        self.last_total_values = total
        self.last_selected_action = max(total, key=lambda k: total[k])
        return self.last_selected_action
