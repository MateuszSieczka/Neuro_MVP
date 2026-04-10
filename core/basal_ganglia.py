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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .config import (
    BasalGangliaConfig,
    ActiveInferenceConfig,
    InhibitoryPoolConfig,
    init_weights,
)
from .free_energy import expected_free_energy
from .interneuron import InhibitoryPool

if TYPE_CHECKING:
    from .world_model import SNNWorldModel


# =====================================================================
# Snapshot for replay
# =====================================================================

@dataclass
class BGSnapshot:
    """BG eligibility traces stored per experience for replay."""
    critic_e_h: NDArray[np.float32]
    critic_e_v: NDArray[np.float32]
    critic_e_bv: float
    d1_e: NDArray[np.float32]
    d2_e: NDArray[np.float32]

    def __post_init__(self) -> None:
        for name in ('critic_e_h', 'critic_e_v', 'd1_e', 'd2_e'):
            val = getattr(self, name)
            if isinstance(val, np.ndarray):
                object.__setattr__(self, name, val.copy())


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
    tau_m: float,
    cfg: BasalGangliaConfig,
    target_rate: float = 0.05,
) -> float:
    """Derive synaptic gain from biophysics — no arbitrary multiplier.

    I_thresh = gap × (1 - decay): current needed to reach threshold in one dt.
    Expected active inputs = fan_in × target_rate.
    Gain = I_thresh / (expected_active × w_mean), where w_mean ≈ PSP_target init.

    This replaces the old `3.0 × gap × sqrt(fan_in)` formula.
    """
    decay = np.exp(-cfg.ctx.dt / tau_m)
    gap = abs(cfg.v_thresh - cfg.v_rest)
    i_thresh = gap * (1.0 - decay)
    expected_active = max(1.0, fan_in * target_rate)
    return float(i_thresh / expected_active)


# =====================================================================
# LIF Critic (ventral striatum)
# =====================================================================

class SNNDeepCritic:
    """LIF-based Critic for value estimation (ventral striatum).

    Ventral striatal neurons encode V(s) as population firing rate
    (Schultz 1998; Samejima et al. 2005). Value is read out as linear
    projection from hidden-layer rate-coded activity. Learning uses
    three-factor STDP modulated by dopaminergic TD error.
    """

    def __init__(self, state_size: int, config: BasalGangliaConfig) -> None:
        self.config = config
        self._state_size = state_size
        h = config.hidden_size

        # ── Precomputed membrane decay ────────────────────────────────
        self._mem_decay: float = config.ctx.decay(config.tau_m_critic)

        # ── Input gain from biophysics ────────────────────────────────
        self._input_gain: float = _derive_input_gain(
            state_size, config.tau_m_critic, config,
        )

        # ── Weights via principled init ───────────────────────────────
        self.w_h: NDArray[np.float32] = init_weights(state_size, h, excitatory=True)

        # ── LIF state (warm start near threshold) ─────────────────────
        self.v_hidden: NDArray[np.float32] = np.random.uniform(
            config.v_rest, config.v_thresh, h,
        ).astype(np.float32)
        self.spikes_hidden: NDArray[np.bool_] = np.zeros(h, dtype=bool)
        self.refrac_hidden: NDArray[np.int32] = np.zeros(h, dtype=np.int32)

        # ── Rate-coded activation (EMA of spikes) ─────────────────────
        self.activation: NDArray[np.float32] = np.zeros(h, dtype=np.float32)
        self._rate_decay: float = config.ctx.decay(config.tau_m_critic)

        # ── Readout weights: hidden rates → V(s) ─────────────────────
        v_std = 1.0 / np.sqrt(h)
        self.w_v: NDArray[np.float32] = np.random.uniform(
            -v_std, v_std, h,
        ).astype(np.float32)
        self.b_v: float = 0.0

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
        self._pre_decay: float = config.ctx.decay(20.0)  # 20ms STDP window
        self._post_decay: float = config.ctx.decay(20.0)

    def forward(self, state_spikes: NDArray[np.float32]) -> float:
        """Compute V(s) with LIF dynamics; update eligibility traces."""
        state_f32 = state_spikes.astype(np.float32)
        cfg = self.config

        # ── Pre-synaptic trace ────────────────────────────────────────
        self._x_pre *= self._pre_decay
        self._x_pre += np.clip(state_f32, 0.0, 1.0)

        # ── Synaptic current ──────────────────────────────────────────
        current = (state_f32 @ self.w_h) * self._input_gain

        h = cfg.hidden_size
        for _ in range(cfg.integration_steps):
            in_refrac = self.refrac_hidden > 0
            self.refrac_hidden[in_refrac] -= 1

            leaked = (
                self.v_hidden * self._mem_decay
                + cfg.v_rest * (1.0 - self._mem_decay)
            )
            noise = np.random.normal(
                0, cfg.membrane_noise_std, h,
            ).astype(np.float32)
            integrated = leaked + current + noise
            self.v_hidden = np.where(in_refrac, cfg.v_reset, integrated)

            inh_current = self.inh_pool.step(self.spikes_hidden.astype(np.float32))
            self.v_hidden -= inh_current

            self.spikes_hidden = (self.v_hidden >= cfg.v_thresh) & ~in_refrac
            self.v_hidden[self.spikes_hidden] = cfg.v_reset
            self.refrac_hidden[self.spikes_hidden] = cfg.refrac_period

            rate_compl = 1.0 - self._rate_decay
            self.activation = (
                self.activation * self._rate_decay
                + self.spikes_hidden.astype(np.float32) * rate_compl
            )

        # ── Post-synaptic trace ───────────────────────────────────────
        self._x_post *= self._post_decay
        self._x_post[self.spikes_hidden] += 1.0

        # ── Eligibility: STDP correlation ─────────────────────────────
        self.e_h *= self._trace_decay
        if np.any(self.spikes_hidden):
            self.e_h[:, self.spikes_hidden] += self._x_pre[:, np.newaxis]
        pre_active = state_f32 > 0.1
        if np.any(pre_active):
            self.e_h[pre_active, :] += self._x_post[np.newaxis, :]

        self.e_v = self.e_v * self._trace_decay + self.activation
        self.e_bv = self.e_bv * self._trace_decay + float(np.mean(self.activation))

        return float(np.dot(self.w_v, self.activation) + self.b_v)

    def peek(self, state_spikes: NDArray[np.float32]) -> float:
        """Estimate V(s') without modifying internal state."""
        return float(np.dot(self.w_v, self.activation) + self.b_v)

    def update(self, td_error: float) -> None:
        """Three-factor STDP: Δw = lr × DA(td_error) × eligibility."""
        td = float(np.clip(td_error, -50.0, 50.0))
        cfg = self.config

        self.w_v += cfg.critic_lr * td * self.e_v
        self.b_v += cfg.critic_lr * td * self.e_bv

        self.w_h += cfg.critic_lr * td * self.e_h

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
        self.activation.fill(0.0)
        self.e_h.fill(0.0)
        self.e_v.fill(0.0)
        self.e_bv = 0.0
        self._x_pre.fill(0.0)
        self._x_post.fill(0.0)
        self.inh_pool.reset_state()

    def set_plasticity_timescales(self, ne: float) -> None:
        """NE modulates eligibility trace decay (Aston-Jones & Cohen 2005)."""
        ne = float(np.clip(ne, 0.0, 1.0))
        ne_factor = 1.0 + ne * (self.config.tau_ne_compression - 1.0)
        eff_tau = self.config.tau_e_critic / ne_factor
        self._trace_decay = float(np.exp(-self.config.ctx.dt / eff_tau))


# =====================================================================
# D1/D2 MSN Actor (dorsal striatum)
# =====================================================================

class D1D2Actor:
    """Dual-pathway MSN actor with bistable dynamics.

    D1 (direct/Go):   DA excites → facilitates selected action
    D2 (indirect/NoGo): DA inhibits → suppresses competing actions

    High DA → D1 dominance → exploitation
    Low DA  → D2 dominance → caution / exploration

    Each action has D1 + D2 populations; net evidence = D1_rate - D2_rate.
    Action probabilities via softmax over net evidence.
    """

    def __init__(
        self,
        state_size: int,
        motor_dim: int,
        internal_dim: int,
        config: BasalGangliaConfig,
    ) -> None:
        self.config = config
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

        # ── Input gain from biophysics ────────────────────────────────
        # Use Up-state τ for gain calculation (active selection mode)
        self._input_gain: float = _derive_input_gain(
            state_size, config.tau_m_msn_up, config,
        )

        # ── D1-MSN membrane state ────────────────────────────────────
        self.v_d1: NDArray[np.float32] = np.random.uniform(
            config.v_rest, config.v_thresh, self.action_dim,
        ).astype(np.float32)
        self.spikes_d1: NDArray[np.bool_] = np.zeros(self.action_dim, dtype=bool)
        self.refrac_d1: NDArray[np.int32] = np.zeros(self.action_dim, dtype=np.int32)
        self.rate_d1: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)

        # ── D2-MSN membrane state ────────────────────────────────────
        self.v_d2: NDArray[np.float32] = np.random.uniform(
            config.v_rest, config.v_thresh, self.action_dim,
        ).astype(np.float32)
        self.spikes_d2: NDArray[np.bool_] = np.zeros(self.action_dim, dtype=bool)
        self.refrac_d2: NDArray[np.int32] = np.zeros(self.action_dim, dtype=np.int32)
        self.rate_d2: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)

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

        # ── Eligibility traces (policy gradient via STDP) ─────────────
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

        # ── Action tracking ───────────────────────────────────────────
        self._last_probs: NDArray[np.float32] | None = None
        self._last_action: int = -1
        self._last_state: NDArray[np.float32] | None = None
        # D1/D2 net evidence (exposed for Active Inference)
        self._d1_rates: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)
        self._d2_rates: NDArray[np.float32] = np.zeros(self.action_dim, dtype=np.float32)

    def set_da_level(self, da: float) -> None:
        """Set DA modulation: high DA → D1 excitation, D2 suppression."""
        self._da_level = float(np.clip(da, 0.0, 1.0))

    def forward(
        self,
        state_spikes: NDArray[np.float32],
        forced_action: int | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """MSN competition through D1/D2 pathways."""
        state_f32 = state_spikes.astype(np.float32)
        self._last_state = state_f32
        cfg = self.config

        # ── Pre-synaptic trace ────────────────────────────────────────
        self._x_pre *= self._pre_decay
        self._x_pre += np.clip(state_f32, 0.0, 1.0)

        # ── Synaptic currents ─────────────────────────────────────────
        current_d1 = (state_f32 @ self.w_d1) * self._input_gain
        current_d2 = (state_f32 @ self.w_d2) * self._input_gain

        # ── DA modulation (Frank 2005) ────────────────────────────────
        # D1: DA excites (scales up drive)
        # D2: DA inhibits (scales down drive)
        da = self._da_level
        d1_mod = cfg.d1_bias + da * (1.0 - cfg.d1_bias)
        d2_mod = cfg.d2_bias * (1.0 - da) + (1.0 - cfg.d2_bias)
        current_d1 *= d1_mod
        current_d2 *= d2_mod

        # ── Bistable MSN decay (Wilson & Kawaguchi 1996) ──────────────
        cortical_drive = np.abs(current_d1) + np.abs(current_d2)
        # Normalize to [0, 1] range for threshold comparison
        drive_max = np.max(cortical_drive) + 1e-8
        cortical_norm = cortical_drive / drive_max
        decay_d1 = _msn_decay(cortical_norm, cfg)
        decay_d2 = _msn_decay(cortical_norm, cfg)

        # ── Multi-tick D1 integration ─────────────────────────────────
        ad = self.action_dim
        for _ in range(cfg.integration_steps):
            # D1
            in_refrac_d1 = self.refrac_d1 > 0
            self.refrac_d1[in_refrac_d1] -= 1
            noise_d1 = np.random.normal(
                0, cfg.membrane_noise_std, ad,
            ).astype(np.float32)
            leaked_d1 = (
                self.v_d1 * decay_d1
                + cfg.v_rest * (1.0 - decay_d1)
            )
            self.v_d1 = np.where(in_refrac_d1, cfg.v_reset, leaked_d1 + current_d1 + noise_d1)
            inh_d1 = self.inh_pool_d1.step(self.spikes_d1.astype(np.float32))
            self.v_d1 -= inh_d1
            self.spikes_d1 = (self.v_d1 >= cfg.v_thresh) & ~in_refrac_d1
            self.v_d1[self.spikes_d1] = cfg.v_reset
            self.refrac_d1[self.spikes_d1] = cfg.refrac_period

            # D2
            in_refrac_d2 = self.refrac_d2 > 0
            self.refrac_d2[in_refrac_d2] -= 1
            noise_d2 = np.random.normal(
                0, cfg.membrane_noise_std, ad,
            ).astype(np.float32)
            leaked_d2 = (
                self.v_d2 * decay_d2
                + cfg.v_rest * (1.0 - decay_d2)
            )
            self.v_d2 = np.where(in_refrac_d2, cfg.v_reset, leaked_d2 + current_d2 + noise_d2)
            inh_d2 = self.inh_pool_d2.step(self.spikes_d2.astype(np.float32))
            self.v_d2 -= inh_d2
            self.spikes_d2 = (self.v_d2 >= cfg.v_thresh) & ~in_refrac_d2
            self.v_d2[self.spikes_d2] = cfg.v_reset
            self.refrac_d2[self.spikes_d2] = cfg.refrac_period

            # Rate EMA
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

        # ── Sub-threshold fallback (drift-diffusion readout) ──────────
        total_rate = np.sum(np.abs(motor_d1)) + np.sum(np.abs(motor_d2))
        if total_rate < 1e-6:
            v_diff = (self.v_d1[:self.motor_dim] - self.v_d2[:self.motor_dim])
            v_diff = v_diff - np.mean(v_diff)
            probs = np.exp(v_diff) / (np.sum(np.exp(v_diff)) + 1e-10)
        else:
            shifted = net_evidence - np.max(net_evidence)
            exp_val = np.exp(shifted)
            probs = exp_val / (np.sum(exp_val) + 1e-10)
        probs = probs.astype(np.float32)
        self._last_probs = probs

        # ── Action selection ──────────────────────────────────────────
        if forced_action is not None:
            action = forced_action
        else:
            action = int(np.random.choice(self.motor_dim, p=probs))
        self._last_action = action

        # ── Eligibility traces (STDP-based REINFORCE) ─────────────────
        one_hot = np.zeros(self.motor_dim, dtype=np.float32)
        one_hot[action] = 1.0
        grad_log_pi = np.zeros(self.action_dim, dtype=np.float32)
        grad_log_pi[:self.motor_dim] = one_hot - probs

        self.e_d1 = self.e_d1 * self._trace_decay + np.outer(state_f32, grad_log_pi)
        # D2 trace: anti-correlation (NoGo pathway opposes selected action)
        self.e_d2 = self.e_d2 * self._trace_decay - np.outer(state_f32, grad_log_pi)

        # ── Motor output ──────────────────────────────────────────────
        motor_action = probs * 2.0 - 1.0  # [0,1] → [-1,1]

        # ── Internal actions (WM gate, etc.) ──────────────────────────
        if self.action_dim > self.motor_dim:
            internal_logits = state_f32 @ self.w_d1[:, self.motor_dim:]
            internal_action = 1.0 / (1.0 + np.exp(-np.clip(internal_logits, -10, 10)))
        else:
            internal_action = np.array([], dtype=np.float32)

        return motor_action.astype(np.float32), internal_action.astype(np.float32)

    def update(self, td_error: float) -> None:
        """Three-factor STDP: D1 learns from +TD, D2 from -TD (Frank 2005)."""
        td = float(np.clip(td_error, -50.0, 50.0))
        cfg = self.config

        # D1 (Go): positive TD → strengthen selected action
        self.w_d1 += cfg.actor_lr * td * self.e_d1
        # D2 (NoGo): negative TD → strengthen suppression of bad actions
        self.w_d2 += cfg.actor_lr * (-td) * self.e_d2

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
        self.rate_d1.fill(0.0)
        self.rate_d2.fill(0.0)
        self.e_d1.fill(0.0)
        self.e_d2.fill(0.0)
        self._x_pre.fill(0.0)
        self._last_probs = None
        self._last_action = -1
        self._last_state = None
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
        self.last_total_values: dict[int, float] = {}
        self.last_selected_action: int = 0

    def compute_epistemic_values(
        self,
        state_spikes: NDArray[np.float32],
        candidate_actions: list[int],
    ) -> dict[int, float]:
        """Epistemic value = prediction uncertainty per action."""
        results = self.world_model.mental_rehearsal(
            state_spikes, candidate_actions,
        )
        epistemic: dict[int, float] = {}
        for action in candidate_actions:
            info = results[action]
            if self.config.uncertainty_method == "novelty":
                epistemic[action] = info.novelty
            else:
                epistemic[action] = self._variance_uncertainty(
                    state_spikes, action,
                )
        return epistemic

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
        epistemic = self.compute_epistemic_values(state_spikes, candidate_actions)
        self.last_epistemic_values = epistemic

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
            g = expected_free_energy(
                pragmatic_value=pragmatic - cost,
                epistemic_value=epist,
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
        epistemic = self.compute_epistemic_values(state_spikes, candidate_actions)
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

    Exploration via LIF membrane noise + DA modulation of D1/D2 balance.
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
        self.last_v: float = 0.0

    def step(
        self,
        state_spikes: NDArray[np.float32],
        reward: float,
        is_terminal: bool = False,
        da_level: float = 0.5,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        """One BG step: critic → TD → update → actor forward."""
        current_v = self.critic.forward(state_spikes)

        if is_terminal:
            td_error = reward - self.last_v
        else:
            td_error = reward + self.config.gamma * current_v - self.last_v

        td_error = float(np.clip(td_error, -50.0, 50.0))

        self.actor.set_da_level(da_level)
        self.critic.update(td_error)
        self.actor.update(td_error)

        motor_action, internal_action = self.actor.forward(state_spikes)
        self.last_v = 0.0 if is_terminal else current_v

        return motor_action, internal_action, td_error

    def reset_state(self) -> None:
        """Reset transient state between episodes."""
        self.last_v = 0.0
        self.critic.reset_state()
        self.actor.reset_state()

    def set_plasticity_timescales(self, ne: float) -> None:
        self.critic.set_plasticity_timescales(ne)
        self.actor.set_plasticity_timescales(ne)

    def snapshot_traces(self) -> BGSnapshot:
        """Capture current eligibility traces for replay."""
        return BGSnapshot(
            critic_e_h=self.critic.e_h,
            critic_e_v=self.critic.e_v,
            critic_e_bv=self.critic.e_bv,
            d1_e=self.actor.e_d1,
            d2_e=self.actor.e_d2,
        )

    def restore_traces(self, snap: BGSnapshot) -> None:
        """Restore eligibility traces from replay snapshot."""
        self.critic.e_h[:] = snap.critic_e_h
        self.critic.e_v[:] = snap.critic_e_v
        self.critic.e_bv = snap.critic_e_bv
        self.actor.e_d1[:] = snap.d1_e
        self.actor.e_d2[:] = snap.d2_e

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
