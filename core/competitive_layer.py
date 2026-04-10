"""
CompetitiveLIFLayer — k-WTA competitive population with derived inhibition.

Changes from legacy:
  1. k_winners derived from ``CompetitiveConfig.target_sparsity × num_neurons``
     (not hardcoded).
  2. Inhibition magnitude derived from biophysics:
     ``i_inh = gap × N/k × strength`` (conductance-scaled).
  3. Uses composable configs (NeuronConfig + STDPConfig + HomeostaticConfig +
     CompetitiveConfig) instead of fragile inheritance chain with flag suppression.
  4. Proactive inhibition *before* spike detection (GABAergic tonic inhibition).
  5. Lateral inhibition at window boundary (oscillator-gated k-WTA).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import (
    NeuronConfig,
    STDPConfig,
    HomeostaticConfig,
    CompetitiveConfig,
)
from .neuron import LIFLayer


class CompetitiveLIFLayer(LIFLayer):
    """k-WTA competitive population built on top of LIFLayer.

    The number of winners ``k`` and the inhibition strength ``i_inh`` are
    computed from ``CompetitiveConfig`` rather than hardcoded.

    The layer manages its *own* homeostatic threshold adaptation
    (using ``HomeostaticConfig``), so the parent ``LIFLayer._update_homeostatic``
    is bypassed by passing ``homeo_cfg=None`` to the parent and handling
    homeostasis locally within the k-WTA evaluation window.
    """

    def __init__(
        self,
        num_inputs: int,
        num_neurons: int = 20,
        neuron_cfg: NeuronConfig | None = None,
        stdp_cfg: STDPConfig | None = None,
        homeo_cfg: HomeostaticConfig | None = None,
        comp_cfg: CompetitiveConfig | None = None,
    ) -> None:
        ncfg = neuron_cfg or NeuronConfig()
        self.comp_cfg = comp_cfg or CompetitiveConfig()
        self._homeo_kwta = homeo_cfg or HomeostaticConfig()

        # Derive k and i_inh from config + population size
        self.k_winners: int = CompetitiveConfig.derive_k(
            self.comp_cfg.target_sparsity, num_neurons,
        )
        self.i_inh: float = CompetitiveConfig.derive_i_inh(
            gap=ncfg.gap,
            num_neurons=num_neurons,
            k_winners=self.k_winners,
            strength=self.comp_cfg.inhibition_strength,
        )

        # Parent gets NO homeostatic config — we manage it ourselves
        super().__init__(
            num_inputs=num_inputs,
            num_neurons=num_neurons,
            neuron_cfg=ncfg,
            stdp_cfg=stdp_cfg or STDPConfig(),
            homeo_cfg=None,   # ← k-WTA manages own homeostasis
            excitatory=True,
        )

        # ── k-WTA window state ────────────────────────────────────────
        self.window_spike_counts: NDArray[np.int32] = np.zeros(
            num_neurons, dtype=np.int32,
        )
        self.last_winners: NDArray[np.int32] = np.array([], dtype=np.int32)
        self._current_window_size: int = 0
        self._phase_reset_pending: bool = False

        # ── Homeostatic state (managed locally, not by parent) ────────
        self.v_thresh_adaptive: NDArray[np.float32] = np.full(
            num_neurons, ncfg.v_thresh, dtype=np.float32,
        )
        self.avg_rate: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )

        # ── Dark matter neurons ───────────────────────────────────────
        self._is_dark_matter: NDArray[np.bool_] = np.zeros(
            num_neurons, dtype=bool,
        )
        n_dark = int(num_neurons * self._homeo_kwta.dark_matter_ratio)
        if n_dark > 0:
            dark_idx = np.random.choice(num_neurons, n_dark, replace=False)
            self._is_dark_matter[dark_idx] = True
            self.v_thresh_adaptive[dark_idx] += (
                self._homeo_kwta.dark_matter_thresh_offset
            )

    # ------------------------------------------------------------------
    # Oscillator interface
    # ------------------------------------------------------------------

    def trigger_phase_reset(self) -> None:
        """Called by NetworkGraph when gamma/oscillator cycle completes."""
        self._phase_reset_pending = True

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, pre_spikes: NDArray[np.float32]) -> NDArray[np.bool_]:
        # Proactive inhibition BEFORE spike detection (tonic GABA)
        self._apply_proactive_inhibition()

        spikes = super().forward(pre_spikes)

        self.window_spike_counts += spikes.astype(np.int32)
        self._current_window_size += 1

        if self._phase_reset_pending:
            self._apply_lateral_inhibition()
            if self._current_window_size > 0:
                self._update_kwta_homeostasis(self._current_window_size)
            self._reset_window()

        return spikes

    # ------------------------------------------------------------------
    # Proactive inhibition (continuous, before spike detection)
    # ------------------------------------------------------------------

    def _apply_proactive_inhibition(self) -> None:
        """GABAergic tonic inhibition: penalise over-active neurons.

        For each neuron, excess = spikes - expected_share. Neurons with
        excess > 0 receive inhibitory current proportional to excess, scaled
        so the per-step force is independent of window length.
        """
        if self._current_window_size == 0 or self.k_winners >= self.num_neurons:
            return
        total = int(np.sum(self.window_spike_counts))
        if total == 0:
            return
        expected = total * self.k_winners / self.num_neurons
        excess = np.maximum(
            0.0,
            self.window_spike_counts.astype(np.float32) - expected,
        )
        inhibition = excess * (self.i_inh / self._current_window_size)
        self.v -= inhibition

    # ------------------------------------------------------------------
    # Lateral inhibition at window boundary (k-WTA evaluation)
    # ------------------------------------------------------------------

    def _apply_lateral_inhibition(self) -> None:
        """End-of-window k-WTA: push losers below rest, zero their traces."""
        if self.k_winners >= self.num_neurons:
            return
        if np.max(self.window_spike_counts) == 0:
            self.last_winners = np.array([], dtype=np.int32)
            return

        sorted_idx = np.argsort(self.window_spike_counts, kind='stable')
        winner_idx = sorted_idx[-self.k_winners:]
        self.last_winners = winner_idx

        losers = np.ones(self.num_neurons, dtype=bool)
        losers[winner_idx] = False
        no_spike = self.window_spike_counts == 0
        losers |= no_spike

        self.v[losers] -= self.i_inh
        self.e[:, losers] = 0.0
        self.x_post[losers] = 0.0
        self.refrac_count[losers] = 0

    # ------------------------------------------------------------------
    # k-WTA homeostatic threshold adaptation (own management)
    # ------------------------------------------------------------------

    def _update_kwta_homeostasis(self, window_steps: int) -> None:
        """Update adaptive threshold using window-averaged firing rate."""
        cfg = self._homeo_kwta
        spikes_f = self.window_spike_counts.astype(np.float32)

        # Only count winners' spikes for rate estimation
        if len(self.last_winners) > 0:
            losers_mask = np.ones(self.num_neurons, dtype=bool)
            losers_mask[self.last_winners] = False
            spikes_f[losers_mask] = 0.0

        spikes_per_step = spikes_f / window_steps
        ctx = cfg.ctx
        decay = ctx.decay(cfg.homeostatic_tau * window_steps)

        self.avg_rate = (
            self.avg_rate * decay
            + spikes_per_step * (1.0 - decay)
        )
        rate_error = self.avg_rate - cfg.target_rate
        self.v_thresh_adaptive += cfg.thresh_adapt_lr * window_steps * rate_error
        np.clip(
            self.v_thresh_adaptive,
            cfg.thresh_min, cfg.thresh_max,
            out=self.v_thresh_adaptive,
        )

    # ------------------------------------------------------------------
    # Override threshold to use our adaptive threshold
    # ------------------------------------------------------------------

    def _effective_threshold(self) -> NDArray[np.float32]:
        ne_drop = self._ne_level * self._homeo_kwta.ne_thresh_drop
        return self.v_thresh_adaptive - ne_drop

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_window(self) -> None:
        self.window_spike_counts.fill(0)
        self._current_window_size = 0
        self._phase_reset_pending = False

    def reset_state(self) -> None:
        super().reset_state()
        self._reset_window()
        self.last_winners = np.array([], dtype=np.int32)
        self.v_thresh_adaptive.fill(self.neuron_cfg.v_thresh)
        if hasattr(self, '_is_dark_matter'):
            self.v_thresh_adaptive[self._is_dark_matter] += (
                self._homeo_kwta.dark_matter_thresh_offset
            )
        self.avg_rate.fill(0.0)
