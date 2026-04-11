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

if TYPE_CHECKING:
    from .astrocyte import AstrocyteField
    from .world_model import SNNWorldModel


# =====================================================================
# Bistable MSN helpers
# =====================================================================

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
    target_rate: float = 0.05,
) -> float:
    """Derive synaptic gain from AdEx rheobase — proper pA-scale current.

    The AdEx membrane equation (Brette & Gerstner 2005) is:
      C_m dV/dt = -g_L(V - E_L) + g_L Δ_T exp((V - V_T)/Δ_T) + I_syn - w

    At the saddle-node bifurcation, the minimum constant current for
    spiking (rheobase) is:
      I_rheo = g_L × (V_T - E_L - Δ_T) = g_L × (gap - delta_t)

    Gain scales I_syn so that expected_active inputs produce ~I_rheo,
    putting neurons in the responsive regime where input variations
    translate to firing rate differences.

    Returns gain in pA per unit input, matching the AdEx current scale.
    """
    gap = abs(ncfg.v_thresh - ncfg.v_rest)
    i_rheo = ncfg.g_L * (gap - ncfg.delta_t)  # pA (Brette & Gerstner 2005)
    expected_active = max(1.0, fan_in * target_rate)
    return float(i_rheo / expected_active)


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

    def __init__(self, state_size: int, config: BasalGangliaConfig) -> None:
        self.config = config
        self._state_size = state_size
        h = config.hidden_size

        # ── AdEx NeuronConfig for critic population ───────────────────
        # Biophysical consistency: τ_m = C_m / g_L ⇒ g_L = C_m / τ_m
        # Default g_L=30nS assumes cortical τ≈9.4ms; ventral striatal
        # neurons with τ=15ms require g_L=18.7nS (C_m=281pF).
        _C_m = 281.0  # NeuronConfig default (Brette & Gerstner 2005)
        self._ncfg = NeuronConfig(
            ctx=config.ctx,
            v_rest=config.v_rest,
            v_thresh=config.v_thresh,
            v_reset=config.v_reset,
            tau_m=config.tau_m_critic,
            g_L=_C_m / config.tau_m_critic,
        )

        # ── Precomputed membrane decay ────────────────────────────────
        self._mem_decay: float = config.ctx.decay(config.tau_m_critic)

        # ── Input gain from biophysics (AdEx rheobase) ─────────────────
        self._input_gain: float = _derive_input_gain(
            state_size, self._ncfg,
        )

        # ── Weights via principled init ───────────────────────────────
        self.w_h: NDArray[np.float32] = init_weights(state_size, h, excitatory=True)

        # ── LIF state (warm start near threshold) ─────────────────────
        self.v_hidden: NDArray[np.float32] = np.random.uniform(
            config.v_rest, config.v_thresh, h,
        ).astype(np.float32)
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

        # ── Readout weights: hidden rates → V(s) ─────────────────────
        v_std = 1.0 / np.sqrt(h)
        self.w_v: NDArray[np.float32] = np.random.uniform(
            -v_std, v_std, h,
        ).astype(np.float32)
        self.b_v: float = 0.0

        # ── V_trace EMA (replaces peek/snapshot, τ ≈ 200ms) ──────────
        self._v_trace_decay: float = np.exp(-config.ctx.dt / 200.0)
        self.v_trace: float = 0.0
        self.last_value: float = 0.0

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
        self.e_v: NDArray[np.float32] = np.zeros(h, dtype=np.float32)
        self.e_bv: float = 0.0
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
        """Single LIF step: compute V(s), update V_trace and eligibility.

        Returns:
            (hidden_size,) spike activity (float32) for downstream layers.
            Value estimate stored in self.last_value and tracked via self.v_trace.
        """
        state_f32 = state_spikes.astype(np.float32)
        cfg = self.config
        h = cfg.hidden_size

        # ── Event-based pre-synaptic trace ────────────────────────────
        self._x_pre *= self._pre_decay
        pre_binary = (state_f32 > 0.5).astype(np.float32)
        self._x_pre += pre_binary
        self._t_since_pre += 1
        self._t_since_pre[pre_binary > 0.5] = 0

        # ── Synaptic current ──────────────────────────────────────────
        current = (state_f32 @ self.w_h) * self._input_gain * self._receptor_gain

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

        inh_current = self.inh_pool.step(self.spikes_hidden.astype(np.float32))
        self.v_hidden -= inh_current

        self.refrac_hidden[self.spikes_hidden] = cfg.refrac_period

        rate_compl = 1.0 - self._rate_decay
        self.activation = (
            self.activation * self._rate_decay
            + self.spikes_hidden.astype(np.float32) * rate_compl
        )

        # ── Event-based post-synaptic trace ───────────────────────────
        self._x_post *= self._post_decay
        self._x_post[self.spikes_hidden] += 1.0
        self._t_since_post += 1
        self._t_since_post[self.spikes_hidden] = 0

        # ── Eligibility: causal STDP window (±20ms) ──────────────────
        self.e_h *= self._trace_decay
        if np.any(self.spikes_hidden):
            post_idx = np.where(self.spikes_hidden)[0]
            ltp_mask = (self._t_since_pre <= self._stdp_window).astype(np.float32)
            self.e_h[:, post_idx] += (self._x_pre * ltp_mask)[:, np.newaxis]
        if np.any(pre_binary > 0.5):
            pre_idx = pre_binary > 0.5
            ltd_mask = (self._t_since_post <= self._stdp_window).astype(np.float32)
            self.e_h[pre_idx, :] -= (self._x_post * ltd_mask)[np.newaxis, :]

        self.e_v = self.e_v * self._trace_decay + self.activation
        self.e_bv = self.e_bv * self._trace_decay + float(np.mean(self.activation))

        # ── V(s) readout + V_trace EMA ────────────────────────────────
        current_v = float(np.dot(self.w_v, self.activation) + self.b_v)
        self.last_value = current_v
        self.v_trace = (
            self.v_trace * self._v_trace_decay
            + current_v * (1.0 - self._v_trace_decay)
        )

        return self.spikes_hidden.astype(np.float32)

    def update(self, td_error: float) -> None:
        """Three-factor STDP: Δw = lr × DA(td_error) × eligibility."""
        td = float(np.clip(td_error, -50.0, 50.0))
        cfg = self.config
        effective_lr = cfg.critic_lr * self._receptor_lr

        dw_v = effective_lr * td * self.e_v
        np.clip(dw_v, -0.1, 0.1, out=dw_v)
        self.w_v += dw_v

        db_v = effective_lr * td * self.e_bv
        self.b_v += float(np.clip(db_v, -0.1, 0.1))

        dw_h = effective_lr * td * self.e_h
        np.clip(dw_h, -0.1, 0.1, out=dw_h)
        self.w_h += dw_h

        # Homeostatic column normalisation
        wc = cfg.w_clip_critic
        np.clip(self.w_v, -wc, wc, out=self.w_v)
        for j in range(cfg.hidden_size):
            col_norm = float(np.linalg.norm(self.w_h[:, j]))
            if col_norm > wc:
                self.w_h[:, j] *= wc / col_norm

    def reset_state(self) -> None:
        """Reset transient state between episodes. Weights preserved."""
        cfg = self.config
        self.v_hidden[:] = np.random.uniform(
            cfg.v_rest, cfg.v_thresh, cfg.hidden_size,
        ).astype(np.float32)
        self.spikes_hidden.fill(False)
        self.refrac_hidden.fill(0)
        self.w_adapt_hidden.fill(0.0)
        self.activation.fill(0.0)
        self.e_h.fill(0.0)
        self.e_v.fill(0.0)
        self.e_bv = 0.0
        self.v_trace = 0.0
        self.last_value = 0.0
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
    ) -> None:
        self.config = config
        self._state_size = state_size
        self.action_dim = motor_dim + internal_dim
        self.motor_dim = motor_dim

        # ── D1 pathway weights (cortex → D1-MSN) ─────────────────────
        self.w_d1: NDArray[np.float32] = init_weights(
            state_size, self.action_dim, excitatory=True,
        )
        # ── D2 pathway weights (cortex → D2-MSN) ─────────────────────
        self.w_d2: NDArray[np.float32] = init_weights(
            state_size, self.action_dim, excitatory=True,
        )

        # ── AdEx NeuronConfig for MSN (Up-state defaults) ────────────
        # Biophysical consistency: τ_m = C_m / g_L ⇒ g_L = C_m / τ_m
        # MSN Up-state τ=25ms with C_m=281pF → g_L=11.24nS
        # (Wilson & Kawaguchi 1996).  Forward pass overrides g_L per
        # neuron via bistable C_m/τ_eff, but ncfg.g_L must match
        # Up-state for consistent gain derivation.
        _C_m = 281.0  # NeuronConfig default (Brette & Gerstner 2005)
        self._ncfg = NeuronConfig(
            ctx=config.ctx,
            v_rest=config.v_rest,
            v_thresh=config.v_thresh,
            v_reset=config.v_reset,
            tau_m=config.tau_m_msn_up,
            g_L=_C_m / config.tau_m_msn_up,
        )

        # ── Input gain from biophysics (AdEx rheobase) ─────────────────
        self._input_gain: float = _derive_input_gain(
            state_size, self._ncfg,
        )

        # ── D1-MSN membrane state ────────────────────────────────────
        self.v_d1: NDArray[np.float32] = np.random.uniform(
            config.v_rest, config.v_thresh, self.action_dim,
        ).astype(np.float32)
        self.spikes_d1: NDArray[np.bool_] = np.zeros(self.action_dim, dtype=bool)
        self.refrac_d1: NDArray[np.int32] = np.zeros(self.action_dim, dtype=np.int32)
        self.rate_d1: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)
        self.w_adapt_d1: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)

        # ── D2-MSN membrane state ────────────────────────────────────
        self.v_d2: NDArray[np.float32] = np.random.uniform(
            config.v_rest, config.v_thresh, self.action_dim,
        ).astype(np.float32)
        self.spikes_d2: NDArray[np.bool_] = np.zeros(self.action_dim, dtype=bool)
        self.refrac_d2: NDArray[np.int32] = np.zeros(self.action_dim, dtype=np.int32)
        self.rate_d2: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)
        self.w_adapt_d2: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)

        # ── Rate EMA decay ────────────────────────────────────────────
        self._rate_decay: float = 0.9  # Fast averaging for action selection

        # ── InhibitoryPool for D1 competition ─────────────────────────
        self.inh_pool_d1 = InhibitoryPool(
            n_excitatory=self.action_dim,
            config=InhibitoryPoolConfig(
                ctx=config.ctx,
                n_interneurons=max(2, self.action_dim // 2),
                target_sparsity=1.0 / max(motor_dim, 1),
            ),
        )
        # ── InhibitoryPool for D2 competition ─────────────────────────
        self.inh_pool_d2 = InhibitoryPool(
            n_excitatory=self.action_dim,
            config=InhibitoryPoolConfig(
                ctx=config.ctx,
                n_interneurons=max(2, self.action_dim // 2),
                target_sparsity=1.0 / max(motor_dim, 1),
            ),
        )

        # ── DA modulation state ───────────────────────────────────────
        self._da_level: float = 0.5

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
        self._last_probs: NDArray[np.float32] | None = None
        self._last_action: int = -1
        self.last_internal_action: NDArray[np.float32] = np.array(
            [], dtype=np.float32,
        )
        # D1/D2 net evidence (exposed for Active Inference)
        self._d1_rates: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)
        self._d2_rates: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)

        # ── Fast epistemic drive (error neuron → D1 excitability) ─────
        self._epistemic_drive: float = 0.0

        # ── Receptor dose-response modulation (D2) ───────────────────
        self._receptor_gain: float = 1.0
        self._receptor_lr: float = 1.0

        # ── Astrocyte ATP coupling (Krok 1.3) ─────────────────────────
        self._astrocyte: AstrocyteField | None = None
        self._zone_idx: NDArray[np.int32] | None = None

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

        # ── Event-based pre-synaptic trace ────────────────────────────
        self._x_pre *= self._pre_decay
        pre_binary = (state_f32 > 0.5).astype(np.float32)
        self._x_pre += pre_binary
        self._t_since_pre += 1
        self._t_since_pre[pre_binary > 0.5] = 0

        # ── Synaptic currents ─────────────────────────────────────────
        current_d1 = (state_f32 @ self.w_d1) * self._input_gain * self._receptor_gain
        current_d2 = (state_f32 @ self.w_d2) * self._input_gain * self._receptor_gain

        # ── DA modulation (Frank 2005) ────────────────────────────────
        da = self._da_level
        d1_mod = cfg.d1_bias + da * (1.0 - cfg.d1_bias)
        d2_mod = cfg.d2_bias * (1.0 - da) + (1.0 - cfg.d2_bias)
        current_d1 *= d1_mod
        current_d2 *= d2_mod

        # ── Fast epistemic drive: error neurons → D1 excitability ─────
        # High prediction error boosts Go pathway (explore novel states)
        if self._epistemic_drive > 0.01:
            current_d1 *= 1.0 + self._epistemic_drive

        # ── Bistable MSN decay (Wilson & Kawaguchi 1996) ──────────────
        cortical_drive = np.abs(current_d1) + np.abs(current_d2)
        drive_max = np.max(cortical_drive) + 1e-8
        cortical_norm = cortical_drive / drive_max
        up_mask = cortical_norm > 0.3
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

        inh_d1 = self.inh_pool_d1.step(self.spikes_d1.astype(np.float32))
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

        inh_d2 = self.inh_pool_d2.step(self.spikes_d2.astype(np.float32))
        self.v_d2 -= inh_d2

        self.spikes_d2 = (self.v_d2 >= ncfg.v_spike_cutoff) & ~in_refrac_d2
        self.v_d2[self.spikes_d2] = ncfg.v_reset
        self.w_adapt_d2[self.spikes_d2] += ncfg.b
        self.w_adapt_d2 = (
            self.w_adapt_d2 * ncfg.w_decay
            + ncfg.a * (self.v_d2 - ncfg.v_rest) * ncfg.w_gain
        )
        self.refrac_d2[self.spikes_d2] = cfg.refrac_period

        # ── Rate EMA ──────────────────────────────────────────────────
        rc = 1.0 - self._rate_decay
        self.rate_d1 = self.rate_d1 * self._rate_decay + self.spikes_d1.astype(np.float32) * rc
        self.rate_d2 = self.rate_d2 * self._rate_decay + self.spikes_d2.astype(np.float32) * rc

        # ── Store pathway rates for Active Inference readout ──────────
        self._d1_rates = self.rate_d1.copy()
        self._d2_rates = self.rate_d2.copy()

        # ── Net evidence: D1 - D2 per action (Go - NoGo) ─────────────
        motor_d1 = self.rate_d1[:self.motor_dim]
        motor_d2 = self.rate_d2[:self.motor_dim]
        net_evidence = motor_d1 - motor_d2

        # ── Action probabilities: softmax over spike-based net evidence ─
        total_rate = np.sum(np.abs(motor_d1)) + np.sum(np.abs(motor_d2))
        if total_rate < 1e-6:
            # Network silent → uniform random exploration (no fallback hack)
            probs = np.ones(self.motor_dim, dtype=np.float32) / self.motor_dim
        else:
            shifted = net_evidence - np.max(net_evidence)
            exp_val = np.exp(shifted)
            probs = exp_val / (np.sum(exp_val) + 1e-10)
        probs = probs.astype(np.float32)
        self._last_probs = probs

        # ── Action selection ──────────────────────────────────────────
        action = int(np.random.choice(self.motor_dim, p=probs))
        self._last_action = action

        # ── Spike timing updates ─────────────────────────────────────
        self._t_since_d1_spike += 1
        self._t_since_d1_spike[self.spikes_d1] = 0
        self._t_since_d2_spike += 1
        self._t_since_d2_spike[self.spikes_d2] = 0

        # ── DA-modulated Hebbian STDP eligibility (Frank 2005) ────────
        # D1 (Go): standard Hebbian — pre × post_spike
        # D2 (NoGo): standard Hebbian — pre × post_spike
        # Weight update modulated by TD error sign in update()
        self.e_d1 *= self._trace_decay
        if np.any(self.spikes_d1):
            d1_idx = np.where(self.spikes_d1)[0]
            ltp_mask = (self._t_since_pre <= self._stdp_window).astype(np.float32)
            self.e_d1[:, d1_idx] += (self._x_pre * ltp_mask)[:, np.newaxis]

        self.e_d2 *= self._trace_decay
        if np.any(self.spikes_d2):
            d2_idx = np.where(self.spikes_d2)[0]
            ltp_mask = (self._t_since_pre <= self._stdp_window).astype(np.float32)
            self.e_d2[:, d2_idx] += (self._x_pre * ltp_mask)[:, np.newaxis]

        # ── Motor output ──────────────────────────────────────────────
        motor_action = probs * 2.0 - 1.0  # [0,1] → [-1,1]

        # ── Internal actions (WM gate, etc.) ──────────────────────────
        if self.action_dim > self.motor_dim:
            internal_logits = state_f32 @ self.w_d1[:, self.motor_dim:]
            self.last_internal_action = 1.0 / (
                1.0 + np.exp(-np.clip(internal_logits, -10, 10))
            )
        else:
            self.last_internal_action = np.array([], dtype=np.float32)

        return motor_action.astype(np.float32)

    def update(self, td_error: float) -> None:
        """DA-modulated three-factor STDP (Frank 2005 Go/NoGo model).

        D1 (Go): positive TD → DA burst → strengthen active Go synapses (LTP)
        D2 (NoGo): negative TD → DA dip → strengthen active NoGo synapses (LTP)

        No REINFORCE grad_log_pi — purely Hebbian with dopaminergic gating.
        Soft weight clipping: dw = clip(dw, -0.1, 0.1) per update.
        """
        td = float(np.clip(td_error, -50.0, 50.0))
        cfg = self.config
        effective_lr = cfg.actor_lr * self._receptor_lr

        # D1: positive DA (reward) drives Go pathway LTP
        dw_d1 = effective_lr * max(td, 0.0) * self.e_d1
        np.clip(dw_d1, -0.1, 0.1, out=dw_d1)
        self.w_d1 += dw_d1

        # D2: negative DA (punishment) drives NoGo pathway LTP
        dw_d2 = effective_lr * max(-td, 0.0) * self.e_d2
        np.clip(dw_d2, -0.1, 0.1, out=dw_d2)
        self.w_d2 += dw_d2

        # Homeostatic column normalisation + Dale's law
        for w in (self.w_d1, self.w_d2):
            np.maximum(w, 0.0, out=w)  # Dale's law: excitatory weights
            for j in range(self.action_dim):
                col_norm = float(np.linalg.norm(w[:, j]))
                if col_norm > cfg.w_clip:
                    w[:, j] *= cfg.w_clip / col_norm

    def get_action(self) -> int:
        return self._last_action

    @property
    def action_entropy(self) -> float:
        """Normalized entropy of action distribution [0, 1]."""
        if self._last_probs is None:
            return 1.0
        p = self._last_probs
        entropy = -float(np.sum(p * np.log(p + 1e-10)))
        max_entropy = float(np.log(max(self.motor_dim, 2)))
        if max_entropy < 1e-8:
            return 0.0
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    @property
    def pragmatic_values(self) -> NDArray[np.float32]:
        """D1 rates as pragmatic value proxy per action."""
        return self._d1_rates[:self.motor_dim].copy()

    @property
    def cost_values(self) -> NDArray[np.float32]:
        """D2 rates as cost/risk proxy per action."""
        return self._d2_rates[:self.motor_dim].copy()

    def reset_state(self) -> None:
        """Reset transient state. Weights preserved."""
        cfg = self.config
        self.v_d1[:] = np.random.uniform(
            cfg.v_rest, cfg.v_thresh, self.action_dim,
        ).astype(np.float32)
        self.v_d2[:] = np.random.uniform(
            cfg.v_rest, cfg.v_thresh, self.action_dim,
        ).astype(np.float32)
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
        self._x_pre.fill(0.0)
        self._t_since_pre.fill(1000)
        self._t_since_d1_spike.fill(1000)
        self._t_since_d2_spike.fill(1000)
        self._last_probs = None
        self._last_action = -1
        self._d1_rates.fill(0.0)
        self._d2_rates.fill(0.0)
        self.inh_pool_d1.reset_state()
        self.inh_pool_d2.reset_state()

    def set_plasticity_timescales(self, ne: float) -> None:
        """NE modulates policy trace decay."""
        ne = float(np.clip(ne, 0.0, 1.0))
        ne_factor = 1.0 + ne * (self.config.tau_ne_compression - 1.0)
        eff_tau = self.config.tau_e_actor / ne_factor
        self._trace_decay = float(np.exp(-self.config.ctx.dt / eff_tau))

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
        """
        # Sub-threshold voltages relative to rest, normalized
        d1_sub = np.clip(
            (self.v_d1[:self.motor_dim] - self.config.v_rest)
            / (self.config.v_thresh - self.config.v_rest),
            0.0, 1.0,
        )
        d2_sub = np.clip(
            (self.v_d2[:self.motor_dim] - self.config.v_rest)
            / (self.config.v_thresh - self.config.v_rest),
            0.0, 1.0,
        )
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


# =====================================================================
# Integrated BG System
# =====================================================================

class BasalGangliaAGISystem:
    """Integrated BG: D1/D2 Actor + LIF Critic + Active Inference.

    TD error uses V_trace EMA instead of peek/snapshot:
      δ = r + γ × V_now - V_trace
    """

    def __init__(
        self,
        state_size: int,
        motor_dim: int,
        internal_dim: int = 1,
        config: BasalGangliaConfig | None = None,
    ) -> None:
        self.config = config or BasalGangliaConfig()
        self.critic = SNNDeepCritic(state_size, self.config)
        self.actor = D1D2Actor(state_size, motor_dim, internal_dim, self.config)

    def step(
        self,
        state_spikes: NDArray[np.float32],
        reward: float,
        is_terminal: bool = False,
        da_level: float = 0.5,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        """One BG step: critic → TD (via V_trace) → update → actor."""
        self.critic.forward(state_spikes)
        current_v = self.critic.last_value

        # TD error: δ = r + γ × V_now - V_trace (Schultz 1998)
        if is_terminal:
            td_error = reward - self.critic.v_trace
        else:
            td_error = reward + self.config.gamma * current_v - self.critic.v_trace

        td_error = float(np.clip(td_error, -50.0, 50.0))

        self.actor.set_da_level(da_level)
        self.critic.update(td_error)
        self.actor.update(td_error)

        motor_action = self.actor.forward(state_spikes)

        return motor_action, self.actor.last_internal_action, td_error

    def reset_state(self) -> None:
        """Reset transient state between episodes."""
        self.critic.reset_state()
        self.actor.reset_state()

    def set_plasticity_timescales(self, ne: float) -> None:
        self.critic.set_plasticity_timescales(ne)
        self.actor.set_plasticity_timescales(ne)

    def compute_exploration_noise(
        self,
        serotonin: float,
        tonic_da: float,
    ) -> float:
        """Exploration noise from DA × 5-HT (Doya 2002)."""
        min_exploration = 0.15
        da_noise = max(0.3, 1.0 - tonic_da)
        sero_noise = max(0.3, 1.0 - serotonin)
        return max(min_exploration, da_noise * sero_noise)
