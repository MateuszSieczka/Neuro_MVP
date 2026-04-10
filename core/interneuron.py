"""
InhibitoryPool — GABAergic E→I→E feedback with inhibitory STDP.

Reference: Brunel & Wang (2003), Woodin et al. (2003), Isaacson & Scanziani (2011)

Changes from legacy:
  1. Uses InhibitoryPoolConfig from config.py (derived tau decays).
  2. Inhibitory STDP (Woodin et al. 2003): E→I Hebbian, I→E anti-Hebbian.
  3. E/I balance homeostatic rate — slow correction factor.
  4. DA modulation via D2 receptors on PV+ interneurons.
  5. Proper current naming (i_gaba_a, not g_gaba_a for current-based model).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import InhibitoryPoolConfig, SynapseConfig


class InhibitoryPool:
    """Fast-spiking PV+ interneuron pool for E→I→E lateral inhibition.

    Replaces algorithmic k-WTA with biophysical competition.
    GABA-A (fast, ~70-80%) + GABA-B (slow, ~20-30%) dual channels.
    """

    def __init__(
        self,
        n_excitatory: int,
        config: InhibitoryPoolConfig | None = None,
    ) -> None:
        self.config = config or InhibitoryPoolConfig()
        self.n_exc = n_excitatory
        n_inh = self.config.n_interneurons
        cfg = self.config

        # ── Interneuron membrane state ────────────────────────────────
        self.v_inh: NDArray[np.float32] = np.full(
            n_inh, cfg.v_rest, dtype=np.float32,
        )
        self.spikes_inh: NDArray[np.bool_] = np.zeros(n_inh, dtype=bool)

        # ── Synaptic weights ──────────────────────────────────────────
        # E→I: log-normal-like initialization (broad convergence)
        self.w_ei: NDArray[np.float32] = np.abs(np.random.normal(
            cfg.w_ei_mean, cfg.w_ei_mean * 0.3, (n_excitatory, n_inh),
        )).astype(np.float32)

        # I→E: perisomatic blanket inhibition
        self.w_ie: NDArray[np.float32] = np.abs(np.random.normal(
            cfg.w_ie_mean, cfg.w_ie_mean * 0.2, (n_inh, n_excitatory),
        )).astype(np.float32)

        # ── GABA current traces (current-based, not conductance) ──────
        self.i_gaba_a: NDArray[np.float32] = np.zeros(n_excitatory, dtype=np.float32)
        self.i_gaba_b: NDArray[np.float32] = np.zeros(n_excitatory, dtype=np.float32)

        # ── Precomputed decays from config ────────────────────────────
        syn_cfg = SynapseConfig(ctx=cfg.ctx)
        self._decay_inh: float = cfg.inh_decay
        self._decay_gaba_a: float = syn_cfg.gaba_a_decay
        self._decay_gaba_b: float = syn_cfg.gaba_b_decay

        # ── Inhibitory STDP traces (Woodin et al. 2003) ──────────────
        self._trace_exc: NDArray[np.float32] = np.zeros(n_excitatory, dtype=np.float32)
        self._trace_inh: NDArray[np.float32] = np.zeros(n_inh, dtype=np.float32)
        self._trace_decay: float = cfg.ctx.decay(20.0)  # τ = 20ms

        # ── DA modulation gain ────────────────────────────────────────
        self._ie_gain: float = 1.0

    def step(self, exc_spikes: NDArray[np.float32]) -> NDArray[np.float32]:
        """One timestep of E→I→E inhibition.

        Args:
            exc_spikes: (n_excitatory,) spike vector.

        Returns:
            (n_excitatory,) inhibitory current to subtract from excitatory V.
        """
        exc_f32 = exc_spikes.astype(np.float32)
        cfg = self.config

        # ── E→I drive ─────────────────────────────────────────────────
        i_input = exc_f32 @ self.w_ei  # (n_inh,)

        # ── Interneuron LIF integration ───────────────────────────────
        gain = cfg.ctx.complement(cfg.tau_m_inh)
        self.v_inh = (
            self.v_inh * self._decay_inh
            + (cfg.v_rest + i_input) * gain
        )

        self.spikes_inh = self.v_inh >= cfg.v_thresh
        self.v_inh[self.spikes_inh] = cfg.v_reset

        # ── I→E feedback (dual GABA channels) ────────────────────────
        inh_f32 = self.spikes_inh.astype(np.float32)
        feedback = inh_f32 @ self.w_ie * self._ie_gain  # (n_exc,)

        self.i_gaba_a *= self._decay_gaba_a
        self.i_gaba_a += (1.0 - cfg.gaba_b_ratio) * feedback

        self.i_gaba_b *= self._decay_gaba_b
        self.i_gaba_b += cfg.gaba_b_ratio * feedback

        # ── Inhibitory STDP (Woodin et al. 2003) ─────────────────────
        self._trace_exc *= self._trace_decay
        self._trace_exc += exc_f32
        self._trace_inh *= self._trace_decay
        self._trace_inh += inh_f32

        # E→I: Hebbian — co-activation strengthens
        if np.any(self.spikes_inh):
            dw_ei = cfg.inh_stdp_lr * np.outer(
                self._trace_exc, inh_f32,
            )
            self.w_ei += dw_ei.astype(np.float32)
            np.maximum(self.w_ei, 0.0, out=self.w_ei)  # Dale's law

        # I→E: Anti-Hebbian / homeostatic (maintains E/I balance)
        if np.any(exc_f32 > 0.1):
            dw_ie = -cfg.ei_balance_lr * np.outer(
                self._trace_inh, exc_f32,
            )
            self.w_ie += dw_ie.astype(np.float32)
            np.maximum(self.w_ie, 0.0, out=self.w_ie)  # Inhibitory weights ≥ 0

        return self.i_gaba_a + self.i_gaba_b

    def modulate_by_da(self, da_level: float) -> None:
        """DA D2 modulation on PV+ interneurons (Seamans & Yang 2004)."""
        self._ie_gain = float(0.7 + 0.8 * np.clip(da_level, 0.0, 1.0))

    def reset_state(self) -> None:
        """Reset transient state between episodes. Weights preserved."""
        self.v_inh.fill(self.config.v_rest)
        self.spikes_inh.fill(False)
        self.i_gaba_a.fill(0.0)
        self.i_gaba_b.fill(0.0)
        self._trace_exc.fill(0.0)
        self._trace_inh.fill(0.0)
        self._ie_gain = 1.0

    @property
    def interneuron_rate(self) -> float:
        """Current firing rate of the inhibitory pool."""
        return float(np.mean(self.spikes_inh))
