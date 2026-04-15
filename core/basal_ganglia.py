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
    SynapseConfig,
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


def _derive_conductance_scale(
    fan_in: int,
    ncfg: NeuronConfig,
    w_clip: float,
    target_rate: float = 0.05,
    mean_input_rate: float | None = None,
    headroom: float = 1.0,
    actual_mean_w: float | None = None,
) -> float:
    """Derive weight scaling factor so that expected input reaches rheobase.

    Weights are in nS (conductance).  At runtime: I = g × (E_exc − V).

    The AdEx rheobase (Brette & Gerstner 2005):
      I_rheo = g_L × (gap − Δ_T)  [pA]

    Ohm's law correction: the driving force (E_exc − V) decreases as V
    rises toward V_thresh, reducing current at the very moment the neuron
    needs it most.  We calibrate at V_thresh (worst-case driving force)
    so the expected synaptic current still exceeds rheobase there.

    Required (at V = V_thresh):
      expected_active × mean_g × effective_df × ampa_eff × headroom ≈ I_rheo

    Parameters
    ----------
    headroom : float
        Factor ≥ 1 compensating for finite integration window and
        Poisson input variance.  1/(1−e^{−1}) ≈ 1.58 for one-τ window.
    actual_mean_w : float | None
        Mean weight from init_weights (before scaling).  If provided,
        used as the per-synapse conductance estimate instead of the
        heuristic w_clip / √fan_in.

    Returns:
        Multiplicative scaling factor (dimensionless) for weight matrices.
    """
    gap = abs(ncfg.v_thresh - ncfg.v_rest)
    # Use driving force at V_thresh (the bottleneck): as V rises toward
    # threshold, both driving force and NMDA Mg²⁺ block change, and the
    # product ampa_eff × df is LOWEST at V_thresh (self-limiting).
    # Calibrating at this worst-case point ensures reliable spiking.
    effective_df = ncfg.e_exc - ncfg.v_thresh  # E_exc − V_thresh ≈ 55 mV
    i_rheo = ncfg.g_L * (gap - ncfg.delta_t)  # pA

    # AMPA/NMDA effective fraction at threshold (Mg²⁺ partly unblocked)
    ampa_ratio = 3.0  # AMPA:NMDA peak conductance ratio (Myme et al. 2003)
    ampa_frac = ampa_ratio / (1.0 + ampa_ratio)  # 0.75
    nmda_frac = 1.0 - ampa_frac                  # 0.25
    mg_block = SynapseConfig.nmda_mg_block(ncfg.v_thresh)  # B(V_thresh)
    ampa_eff = ampa_frac + nmda_frac * mg_block   # ~0.78 at V_thresh

    rate = mean_input_rate if mean_input_rate is not None else target_rate
    expected_active = max(1.0, fan_in * rate)
    if actual_mean_w is not None and actual_mean_w > 0:
        effective_g = actual_mean_w
    else:
        effective_g = w_clip / np.sqrt(max(1.0, float(fan_in)))  # nS
    return float(
        headroom * i_rheo
        / (expected_active * effective_g * effective_df * ampa_eff)
    )


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

        # ── Weights in nS via principled init ─────────────────────────
        # init_weights outputs nS; _cond_scale adjusts for rheobase.
        self.w_h: NDArray[np.float32] = init_weights(
            state_size, h, excitatory=True,
            g_L=self._ncfg.g_L,
            driving_force=self._ncfg.e_exc - self._ncfg.v_rest,
        )
        self._init_mean_w: float = float(self.w_h.mean())

        # ── Conductance scale from biophysics (AdEx rheobase) ──────────
        # Applied at forward-time (like the old _input_gain), NOT baked
        # into weights: STDP operates on the init-scale weights so that
        # critic_lr doesn't need rescaling after the nS conversion.
        _cond_scale: float = _derive_conductance_scale(
            state_size, self._ncfg, config.w_clip_critic,
            mean_input_rate=mean_input_rate,
            headroom=1.58,
            actual_mean_w=float(self.w_h.mean()),
        )
        self._cond_scale: float = _cond_scale

        # Column-norm clip (homeostatic target, Turrigiano 2008).
        _w_clip: float = config.w_clip_critic
        _init_col_norms = np.linalg.norm(self.w_h, axis=0)
        self._w_clip_nS: float = float(1.5 * _init_col_norms.mean())
        for j in range(h):
            if _init_col_norms[j] > self._w_clip_nS:
                self.w_h[:, j] *= self._w_clip_nS / _init_col_norms[j]

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

        # ── NMDA slow synaptic conductance trace (Wang 2002) ──────────
        self._g_nmda: NDArray[np.float32] = np.zeros(h, dtype=np.float32)
        self._nmda_decay: float = config.ctx.decay(100.0)  # τ_NMDA=100ms

        # ── Excitatory reversal potential (for I = g × (E_exc − V)) ──
        self._e_exc: float = self._ncfg.e_exc  # 0.0 mV

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
                0, astrocyte.n_zones - 1, n,
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

        # ── Synaptic conductance (nS) ─────────────────────────────────
        # _cond_scale converts init-scale weights → nS at forward time
        # (preserves STDP operating on the smaller init scale).
        g_total = (state_f32 @ self.w_h) * np.float32(self._cond_scale) * self._receptor_gain

        # ── NMDA temporal integration + Mg²⁺ block (Wang 2002; Jahr & Stevens 1990)
        # AMPA: fast conductance, no voltage gating.
        # NMDA: slow τ=100ms EMA of conductance, with voltage-dependent
        #   Mg²⁺ block B(V) applied at current conversion (not on g).
        # AMPA:NMDA ratio (3:1) is PEAK conductance (Myme et al. 2003).
        _ampa_frac = cfg.ampa_nmda_ratio / (1.0 + cfg.ampa_nmda_ratio)
        _nmda_frac = 1.0 - _ampa_frac
        _nmda_compl = 1.0 - self._nmda_decay
        self._g_nmda = self._g_nmda * self._nmda_decay + _nmda_compl * g_total
        # Ohm's law: I = g × (E_exc − V), with Mg²⁺ block on NMDA
        driving_force = self._e_exc - self.v_hidden
        mg_block = SynapseConfig.nmda_mg_block(self.v_hidden)
        current = (_ampa_frac * g_total + _nmda_frac * self._g_nmda * mg_block) * driving_force

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
        np.clip(self.v_hidden, -90.0, None, out=self.v_hidden)  # K+ reversal floor

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
        # over-correction (maximum ±50% of target per τ_homeo; Turrigiano
        # 2008 — gradual scaling prevents runaway oscillation).
        # Silent neurons receive the maximum positive correction = _HOMEO_CLIP.
        _HOMEO_CLIP = 0.5
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
        self._g_nmda.fill(0.0)
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

        # ── AdEx NeuronConfig for MSN (Up-state defaults) ────────────
        # Must be created BEFORE weight init so g_L and driving_force
        # are available for conductance scaling.
        #
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

        # ── Excitatory reversal potential (for I = g × (E_exc − V)) ──
        self._e_exc: float = self._ncfg.e_exc  # 0.0 mV

        # ── D1 pathway weights (cortex → D1-MSN, in nS) ──────────────
        self.w_d1: NDArray[np.float32] = init_weights(
            state_size, self.action_dim, excitatory=True,
            g_L=self._ncfg.g_L,
            driving_force=self._ncfg.driving_force_exc,
        )
        # ── D2 pathway weights (cortex → D2-MSN, in nS) ──────────────
        # D1 and D2 MSNs are separate neuronal populations.  While they
        # share cortical afferents, individual synaptic strengths are
        # independently established during synaptogenesis (Gerfen &
        # Surmeier 2011).  Independent random initialization provides
        # symmetry breaking so net evidence (D1-D2) is non-zero from
        # the start, enabling Go/NoGo learning immediately.
        self.w_d2: NDArray[np.float32] = init_weights(
            state_size, self.action_dim, excitatory=True,
            g_L=self._ncfg.g_L,
            driving_force=self._ncfg.driving_force_exc,
        )
        # ── Conductance scale from biophysics (AdEx rheobase) ──────────
        # Integration headroom: 1/(1-e^{-1}) ≈ 1.58 for one-τ window.
        # Driving force reduction and NMDA Mg²⁺ block are handled
        # inside _derive_conductance_scale.
        self._init_mean_w: float = float(
            (self.w_d1.mean() + self.w_d2.mean()) / 2.0
        )
        _cond_scale: float = _derive_conductance_scale(
            state_size, self._ncfg, config.w_clip,
            mean_input_rate=mean_input_rate,
            headroom=1.58,
            actual_mean_w=self._init_mean_w,
        )
        self._cond_scale: float = _cond_scale
        # Scale applied at forward-time (preserves STDP operating on
        # the smaller init scale, same rationale as critic).
        # Column-norm clip on init-scale weights
        _all_col_norms = np.concatenate([
            np.linalg.norm(self.w_d1, axis=0),
            np.linalg.norm(self.w_d2, axis=0),
        ])
        self._w_clip_nS: float = float(1.5 * _all_col_norms.mean())
        for w_mat in (self.w_d1, self.w_d2):
            for j in range(self.action_dim):
                col_norm = float(np.linalg.norm(w_mat[:, j]))
                if col_norm > self._w_clip_nS:
                    w_mat[:, j] *= self._w_clip_nS / col_norm

        # D2-MSN gain compensation: at baseline DA the D2-receptor
        # Hill suppression reduces current to d2_net_mod ≈ 0.52 × I,
        # while D1 receptors BOOST D1 current (d1_mod ≈ 1.53).
        # Without compensation D2 sits far below D1 and NoGo learning
        # fails because D2 eligibility traces are too weak.
        #
        # Planert et al. (2010) showed D1 and D2 MSN firing rates are
        # similar in vivo at tonic DA.  Day et al. (2008) confirmed
        # balanced baseline excitability.  To match this, compensate
        # D2 to equal D1's boosted drive at baseline DA — not just to
        # reach the unmodulated level.
        _baseline_da = config.baseline_da
        _d1_resp = hill_response(_baseline_da, config.d1_ec50, config.d1_hill_n)
        _d1_mod_base = 1.0 + config.d1_receptor_density * _d1_resp
        _d2_resp = hill_response(_baseline_da, config.d2_ec50, config.d2_hill_n)
        _d2_mod = 1.0 - config.d2_receptor_density * _d2_resp
        _d2_tonic = 1.0 + config.d2_tonic_boost_max * (1.0 - _baseline_da)
        _d2_net_mod = max(_d2_mod * _d2_tonic, 0.1)
        self._d2_gain_comp: float = _d1_mod_base / _d2_net_mod

        # ── D1-MSN membrane state ────────────────────────────────────
        self.v_d1: NDArray[np.float32] = np.full(
            self.action_dim, config.v_rest, dtype=np.float32,
        )
        self.spikes_d1: NDArray[np.bool_] = np.zeros(self.action_dim, dtype=bool)
        self.refrac_d1: NDArray[np.int32] = np.zeros(self.action_dim, dtype=np.int32)
        self.rate_d1: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)
        self.w_adapt_d1: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)

        # ── NMDA slow synaptic conductance traces (Wang 2002) ─────────
        self._g_nmda_d1: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)
        self._g_nmda_d2: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)
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
        # Tonic DA level for high-affinity D2 receptor modulation
        # (Dreyer et al. 2010).  Initialised at phasic baseline;
        # overwritten by set_tonic_da_level() before first forward().
        self._tonic_da_level: float = config.baseline_da

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

        # ── Per-action EFE conductance (Pezzulo et al. 2018) ─────────
        # Prefrontal → striatal projection carrying expected free energy
        # scores.  Per-action conductance (nS) injected as excitatory
        # current I = g_efe × (E_exc − V_d1) in forward().
        self._efe_g: NDArray[np.float32] = np.zeros(
            self._total_motor, dtype=np.float32,
        )

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
                0, astrocyte.n_zones - 1, n,
            ).astype(np.int32)

    def set_da_level(self, da: float) -> None:
        """Set phasic DA: high DA → D1 excitation, D2 suppression."""
        self._da_level = float(np.clip(da, 0.0, 1.0))

    def set_tonic_da_level(self, tonic_da: float) -> None:
        """Set tonic DA for high-affinity D2 receptor modulation.

        Tonic DA tracks average reward rate (Niv et al. 2007).
        High tonic DA (rich environment) → D2 suppressed → exploitation.
        Low tonic DA (poor/novel environment) → D2 boosted → caution.
        Operates on minute-scale (τ ≈ 60 s, Grace 1991), independent
        of phasic DA transients (τ ≈ 200 ms).
        """
        self._tonic_da_level = float(np.clip(tonic_da, 0.0, 1.0))

    def set_ne_level(self, ne: float) -> None:
        """NE interface stub — NE effect handled in set_plasticity_timescales()."""
        pass

    def set_epistemic_drive(self, error_rate: NDArray[np.float32]) -> None:
        """Fast epistemic path: error neuron error_rate → D1 excitability boost.

        High prediction error → explore novel states → boost Go pathway.
        """
        self._epistemic_drive = float(np.clip(np.mean(error_rate), 0.0, 1.0))

    def set_efe_conductance(self, per_action_g: NDArray[np.float32]) -> None:
        """Set per-action EFE-derived conductance (nS) for D1 bias.

        Prefrontal → striatal projection carrying active inference
        expected free energy scores (Pezzulo et al. 2018).  Each
        action's EFE is mapped to excitatory conductance applied
        uniformly across that action's MSN subpopulation.  The
        resulting current I = g × (E_exc - V) follows Ohm's law,
        identical to regular synaptic input.

        Args:
            per_action_g: (motor_dim,) conductance in nS per action.
        """
        g = np.clip(per_action_g[:self.motor_dim], 0.0, None).astype(np.float32)
        expanded = np.repeat(g, self.n_per_action)
        self._efe_g[:len(expanded)] = expanded

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

        # ── Synaptic conductances (nS) ─────────────────────────────────
        # _cond_scale applied at forward time (same as critic)
        _cs = np.float32(self._cond_scale)
        g_d1 = (state_f32 @ self.w_d1) * _cs * self._receptor_gain
        g_d2 = (state_f32 @ self.w_d2) * _cs * self._receptor_gain * self._d2_gain_comp

        # ── NMDA temporal integration + Mg²⁺ block (Wang 2002; Jahr & Stevens 1990)
        # AMPA:NMDA ratio (3:1) is peak conductance (Myme et al. 2003).
        # NMDA slow EMA on conductance; Mg²⁺ block applied at current conversion.
        _ampa_frac = cfg.ampa_nmda_ratio / (1.0 + cfg.ampa_nmda_ratio)
        _nmda_frac = 1.0 - _ampa_frac
        _nmda_compl = 1.0 - self._nmda_decay
        self._g_nmda_d1 = self._g_nmda_d1 * self._nmda_decay + _nmda_compl * g_d1
        self._g_nmda_d2 = self._g_nmda_d2 * self._nmda_decay + _nmda_compl * g_d2
        # Ohm's law: I = g × (E_exc − V), with Mg²⁺ block on NMDA
        df_d1 = self._e_exc - self.v_d1  # driving force D1 (mV)
        df_d2 = self._e_exc - self.v_d2  # driving force D2 (mV)
        mg_d1 = SynapseConfig.nmda_mg_block(self.v_d1)
        mg_d2 = SynapseConfig.nmda_mg_block(self.v_d2)
        current_d1 = (_ampa_frac * g_d1 + _nmda_frac * self._g_nmda_d1 * mg_d1) * df_d1
        current_d2 = (_ampa_frac * g_d2 + _nmda_frac * self._g_nmda_d2 * mg_d2) * df_d2

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
        # D2 tonic boost: HIGH-AFFINITY D2 receptors respond to the
        # slow background DA level, not phasic transients (Dreyer et al.
        # 2010).  _tonic_da_level tracks average reward rate via
        # minute-scale integration (Niv et al. 2007; Grace 1991).
        #
        # Low tonic DA (unknown/poor environment) → high-affinity D2
        # tonically active → D2 excitable → NoGo/caution bias.
        # High tonic DA (rich environment) → D2 saturated → reduced
        # tonic boost → Go/exploitation bias.
        #
        # The _d2_gain_comp (computed at init using phasic baseline_da)
        # calibrates D1≈D2 at the equilibrium point (Planert et al.
        # 2010).  Since baseline_tonic_da (0.0) < baseline_da (0.5),
        # the initial D2 boost exceeds the calibration → natural
        # "caution prior" without any explicit bias parameter.
        da_tonic = float(np.clip(self._tonic_da_level, 0.0, 1.0))
        d2_tonic = 1.0 + cfg.d2_tonic_boost_max * (1.0 - da_tonic)
        current_d1 *= d1_mod
        current_d2 *= d2_mod * d2_tonic

        # ── STN-GPe hyperdirect pathway (Frank 2006; Bogacz & Gurney 2007)
        # Low DA → STN disinhibited → global GPi excitation → raises
        # effective threshold for ALL action channels.  Multiplicative
        # suppression relative to tonic DA baseline: only active when DA
        # drops below baseline (phasic dip).  At DA=baseline → stn_factor=1
        # (no suppression).  At DA=0 → stn_factor = 1 - stn_strength × baseline_da.
        # This ensures exploration is driven by DA dips (reward omission),
        # not by the normal tonic state.
        da_deficit = max(cfg.baseline_da - da, 0.0)
        stn_factor = max(1.0 - cfg.stn_strength * da_deficit, 0.0)
        current_d1 *= stn_factor
        current_d2 *= stn_factor

        # ── Fast epistemic drive: error neurons → D1 excitability ─────
        # High prediction error boosts Go pathway (explore novel states)
        if self._epistemic_drive > 0.01:
            current_d1 *= 1.0 + self._epistemic_drive

        # ── Per-action EFE conductance (Pezzulo et al. 2018) ─────────
        # Prefrontal → striatal top-down bias from active inference.
        # Ohm's law: I = g_efe × (E_exc − V_d1).  Only affects D1 (Go):
        # actions with higher EFE get more excitation → WTA selects.
        tm = self._total_motor
        if np.any(self._efe_g > 1e-6):
            efe_I = self._efe_g * (self._e_exc - self.v_d1[:tm])
            current_d1[:tm] += efe_I

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
        np.clip(self.v_d1, -90.0, None, out=self.v_d1)  # K+ reversal floor

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
        np.clip(self.v_d2, -90.0, None, out=self.v_d2)  # K+ reversal floor

        self.spikes_d2 = (self.v_d2 >= ncfg.v_spike_cutoff) & ~in_refrac_d2
        self.v_d2[self.spikes_d2] = ncfg.v_reset
        self.w_adapt_d2[self.spikes_d2] += ncfg.b
        self.w_adapt_d2 = (
            self.w_adapt_d2 * ncfg.w_decay
            + ncfg.a * (self.v_d2 - ncfg.v_rest) * ncfg.w_gain
        )
        self.refrac_d2[self.spikes_d2] = cfg.refrac_period

        # ── Cross-action lateral inhibition (Taverna et al. 2008) ─────
        # D1 and D2 MSNs inhibit each other through lateral GABAergic
        # collaterals.  Connection probabilities and unitary iPSPs
        # from Taverna et al. (2008, J Neurosci, Fig 5):
        #
        #   D1→D1: p=14%, iPSP=0.14 mV
        #   D2→D2: p=11%, iPSP=0.12 mV
        #   D1→D2: p= 6%, iPSP=0.10 mV (weakest cross-pathway)
        #
        # The model sums spikes from the competing population and
        # applies uniform lateral inhibition to the target channel.
        # Since actual connectivity is sparse (~14%), g_lat must
        # account for connection probability (mean-field scaling):
        #   g_lat = p_conn × iPSP / (V_thresh − E_inh)
        #
        # Self-limiting: as V → E_inh, driving force → 0 (Brunel & Wang 2003).
        _E_INH = -75.0   # GABA-A reversal (Buhl et al. 1995)
        _DF_INH = cfg.v_thresh - _E_INH  # 20 mV driving force at threshold
        # Lateral iPSP conductances from striatal patch-clamp
        # (Taverna, Ilijic & Bhardwaj 2008, Table 1):
        #   g_lat = p_connect × mean_iPSP_mV / driving_force
        _G_LAT_D1 = 0.14 * 0.14 / _DF_INH  # p=14%, iPSP=0.14mV — D1→D1
        _G_LAT_D2 = 0.11 * 0.12 / _DF_INH  # p=11%, iPSP=0.12mV — D2→D2
        _G_LAT_X  = 0.06 * 0.10 / _DF_INH  # p=6%,  iPSP=0.10mV — D1→D2 cross
        if self.motor_dim > 1:
            npa = self.n_per_action
            tm = self._total_motor
            _d1_act_spikes = self.spikes_d1[:tm].reshape(
                self.motor_dim, npa,
            ).sum(axis=1).astype(np.float32)
            _d2_act_spikes = self.spikes_d2[:tm].reshape(
                self.motor_dim, npa,
            ).sum(axis=1).astype(np.float32)
            total_d1 = _d1_act_spikes.sum()
            total_d2 = _d2_act_spikes.sum()
            if total_d1 > 0 or total_d2 > 0:
                for a in range(self.motor_dim):
                    s = a * npa
                    e = s + npa
                    d1_others = total_d1 - _d1_act_spikes[a]
                    d2_others = total_d2 - _d2_act_spikes[a]
                    # D1→D1 within-pathway lateral inhibition
                    self.v_d1[s:e] += (_G_LAT_D1 * d1_others) * (_E_INH - self.v_d1[s:e])
                    # D2→D2 within-pathway lateral inhibition
                    self.v_d2[s:e] += (_G_LAT_D2 * d2_others) * (_E_INH - self.v_d2[s:e])
                    # D1→D2 cross-pathway (weak, Taverna 2008 Fig. 5)
                    self.v_d2[s:e] += (_G_LAT_X * d1_others) * (_E_INH - self.v_d2[s:e])

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
        # Preserve the act()-phase action during inference_mode forward
        # passes (e.g., V(s') critic integration in observe()).  The
        # action-gated eligibility in update() needs the act()-phase
        # action, not a spurious action from zero-input decay dynamics.
        if not self._inference_mode:
            self._last_action = action
        self._last_net_evidence = net_evidence.copy()

        # ── Spike-based eligibility (Bi & Poo 2001; Clopath et al. 2010) ─
        # Post-synaptic spike traces (_x_post_d1/d2) provide credit to
        # active neurons.  Combined with cross-action lateral inhibition,
        # the winner fires more → higher eligibility.  Voltage floor
        # ensures both D1 and D2 get baseline credit even with sparse
        # spiking, preventing initialization-dependent lock-in.

        if not self._inference_mode:
            _e_compl = 1.0 - self._trace_decay
            # Hybrid: spike trace + voltage floor (ensures both pathways learn)
            _floor = 0.07  # minimum eligibility from membrane proximity to threshold
            post_d1 = np.maximum(self._x_post_d1, _floor * v_d1_norm)
            post_d2 = np.maximum(self._x_post_d2, _floor * v_d2_norm)
            self.e_d1 = self.e_d1 * self._trace_decay + _e_compl * np.outer(state_f32, post_d1)
            self.e_d2 = self.e_d2 * self._trace_decay + _e_compl * np.outer(state_f32, post_d2)

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
        base_lr = cfg.actor_lr * self._receptor_lr

        # ── DA-gated plasticity asymmetry (Collins & Frank 2014) ──────
        # D1/D2 learning rates modulated by receptor occupancy via the
        # SAME Hill parameters used for excitability (no new constants).
        # D1 LTP requires D1 receptor activation: cAMP↑ → PKA →
        #   DARPP-32 phosphorylation → CaMKII → LTP (Surmeier 2007).
        # D2 LTP requires D2 receptor DE-activation: DA dip →
        #   cAMP↓ relief → permissive for LTP (Shen et al. 2008).
        # D1 LTD requires D1 receptor vacancy: eCB-mediated depression
        #   only when D1/PKA signalling is low (Shen et al. 2008).
        da = float(np.clip(self._da_level, 0.0, 1.0))
        _d1_r = hill_response(da, cfg.d1_ec50, cfg.d1_hill_n)
        _d2_r = hill_response(da, cfg.d2_ec50, cfg.d2_hill_n)

        # ── Action-gated eligibility (Frank 2005; Wickens et al. 2003) ─
        # Three-factor STDP requires the post-synaptic MSN to have been
        # in UP state during the action — only responding MSNs accumulate
        # full synaptic tags (Reynolds & Wickens 2002; Gurney et al. 2015).
        # With weak lateral inhibition (Taverna et al. 2008: p_conn ≤ 14%,
        # iPSP ≤ 0.14 mV), spiking WTA alone does not produce sufficient
        # activity contrast between chosen and non-chosen action channels
        # to gate eligibility.  Explicit scaling implements the GPi
        # selection gate: the winning action keeps full eligibility, while
        # non-chosen actions retain only DOWN-state residual plasticity.
        #
        # DOWN-state MSNs fire at 0.5–2 Hz vs UP-state 10–40 Hz
        # (Wilson & Kawaguchi 1996; Mahon et al. 2006), giving a
        # spike-rate ratio of ~5–15%.  Subthreshold Ca²⁺ transients
        # (Yuste & Denk 1995) support weak STDP even in DOWN state.
        # Factor 0.1 ≈ geometric mean of spike-rate and voltage ratios.
        _DOWN_STATE_FACTOR = 0.1
        _act = self._last_action
        if 0 <= _act < self.motor_dim:
            npa = self.n_per_action
            for a in range(self.motor_dim):
                if a != _act:
                    s = a * npa
                    e = s + npa
                    self.e_d1[:, s:e] *= _DOWN_STATE_FACTOR
                    self.e_d2[:, s:e] *= _DOWN_STATE_FACTOR

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
        # lr ∝ D1 occupancy: high DA → strong LTP (Surmeier et al. 2007)
        if td > 0:
            lr_d1_ltp = base_lr * (1.0 + cfg.d1_receptor_density * _d1_r)
            dw_d1 = lr_d1_ltp * td * self.e_d1
            self.w_d1 += dw_d1

        # D2 (NoGo): LTP on negative TD (DA dip)
        # lr ∝ D2 receptor VACANCY (1 − occupancy): low DA → strong NoGo
        # learning (Shen et al. 2008; Collins & Frank 2014).
        # D1 (Go): LTD on negative TD — requires D1 receptor vacancy;
        # lr ∝ (1 − D1 occupancy) × ltd_ratio (Shen et al. 2008).
        if td < 0:
            lr_d2_ltp = base_lr * (1.0 + cfg.d2_receptor_density * (1.0 - _d2_r))
            dw_d2 = lr_d2_ltp * (-td) * self.e_d2
            self.w_d2 += dw_d2

            lr_d1_ltd = base_lr * cfg.ltd_ratio * (1.0 - cfg.d1_receptor_density * _d1_r)
            dw_d1_ltd = lr_d1_ltd * td * self.e_d1
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
        # Clip ±50%: prevents oscillatory instability (Turrigiano 2008).
        _HOMEO_CLIP = 0.5
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

        # ── Per-column norm clipping (Sabatini et al. 2002) ───────────
        # PSD surface area limits total receptor count per dendritic
        # spine.  The init-time clip (1.5× mean_col_norm) is tight for
        # symmetry breaking.  During learning, allow 10× growth before
        # clipping — enough for reliable D1/D2 discrimination while
        # preventing runaway drift (D2 has no LTD pathway).
        _learn_clip = self._w_clip_nS * 10.0
        for w_mat in (self.w_d1, self.w_d2):
            for j in range(self.action_dim):
                col_norm = float(np.linalg.norm(w_mat[:, j]))
                if col_norm > _learn_clip:
                    w_mat[:, j] *= _learn_clip / col_norm

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
        self._g_nmda_d1.fill(0.0)
        self._g_nmda_d2.fill(0.0)
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
        self._efe_g.fill(0.0)
        self.inh_pool_d1.reset_state()
        self.inh_pool_d2.reset_state()

    def set_plasticity_timescales(self, ne: float) -> None:
        """NE modulates policy trace decay via tau compression."""
        ne = float(np.clip(ne, 0.0, 1.0))
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

    # gate_eligibility() removed - Phase 2 HACK B.
    # Voltage-based eligibility (Clopath et al. 2010) naturally decays
    # for suppressed action channels: InhibitoryPool drives losing
    # MSNs near V_rest -> v_norm ~ 0 -> eligibility ~ 0.  No explicit
    # zeroing needed (Wang 2002; Wickens et al. 2003).

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
        """Select action by injecting EFE bias into D1 MSNs.

        Computes expected free energy G(a) per candidate action, then
        converts to conductance bias on actor D1 MSN subpopulations.
        Actual selection occurs via WTA in actor.forward() during
        subsequent integration substeps (Pezzulo et al. 2018).

        If no actor, falls back to argmax on -G(a).
        """
        total, prag_dict = self._compute_efe_values(
            state_spikes, candidate_actions, actor, ne_level,
        )
        self.last_pragmatic_values = prag_dict
        self.last_total_values = total

        # Inject EFE as D1 conductance bias (replaces softmax)
        if actor is not None:
            self._inject_efe_to_actor(total, actor)
            selected = actor.get_action()
        else:
            selected = max(total, key=lambda k: total[k])

        self.last_selected_action = selected
        return selected

    def select_action_greedy(
        self,
        state_spikes: NDArray[np.float32],
        candidate_actions: list[int],
        actor: D1D2Actor | None = None,
        ne_level: float = 0.3,
    ) -> int:
        """Greedy variant: injects EFE bias, returns argmax of -G(a)."""
        total, _ = self._compute_efe_values(
            state_spikes, candidate_actions, actor, ne_level,
        )
        self.last_total_values = total

        if actor is not None:
            self._inject_efe_to_actor(total, actor)

        self.last_selected_action = max(total, key=lambda k: total[k])
        return self.last_selected_action

    def _compute_efe_values(
        self,
        state_spikes: NDArray[np.float32],
        candidate_actions: list[int],
        actor: D1D2Actor | None,
        ne_level: float,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Compute -G(a) and pragmatic values for all candidates."""
        epistemic, ambiguity = self.compute_epistemic_values(
            state_spikes, candidate_actions,
        )
        self.last_epistemic_values = epistemic
        self.last_ambiguity_values = ambiguity

        beta = (
            self.config.epistemic_weight
            + ne_level * self.config.ne_epistemic_boost
        )

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
            total[action] = -g
            prag_dict[action] = pragmatic - cost

        return total, prag_dict

    def _inject_efe_to_actor(
        self,
        total: dict[int, float],
        actor: D1D2Actor,
    ) -> None:
        """Convert -G(a) scores to D1 conductance bias.

        Normalized to [0, 1] then scaled by g_L × efe_drive_fraction.
        Sherman & Guillery (2002): top-down cortical projections provide
        ~10-30% of total excitatory drive to striatum.
        """
        actions = sorted(total.keys())
        values = np.array([total[a] for a in actions], dtype=np.float32)

        v_min = values.min()
        v_range = values.max() - v_min
        if v_range > 1e-8:
            normalized = (values - v_min) / v_range
        else:
            normalized = np.zeros_like(values)

        g_ref = actor._ncfg.g_L * self.config.efe_drive_fraction
        efe_g = np.zeros(actor.motor_dim, dtype=np.float32)
        for i, a in enumerate(actions):
            if a < actor.motor_dim:
                efe_g[a] = normalized[i] * g_ref

        actor.set_efe_conductance(efe_g)
