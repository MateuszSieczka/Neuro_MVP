"""
Spatial Attention System — top-down gain control for columnar architectures.

Biological grounding:
  Attentional modulation in the cortex operates through acetylcholine (ACh)
  released from basal forebrain projections.  Unlike the global tonic ACh
  level tracked by NeuromodulatorSystem, *spatial* attention is column-specific:
  higher cortical areas (e.g. prefrontal, parietal) project back to sensory
  columns and selectively amplify those whose receptive fields overlap with
  the attended location/object.

  Mechanistically, attention acts as multiplicative gain:
    effective_drive = base_drive × (1 + gain × attention_weight)
  This does not change feature selectivity (tuning curves shift vertically,
  not horizontally — Reynolds & Heeger, 2009).

Architecture:
  SpatialAttentionController sits between an association layer (top-down
  source) and a set of columnar layers (gain targets).  Each timestep:
    1. Read the association layer's spike pattern (or firing rate proxy).
    2. Project through learned attention weights → raw saliency per column.
    3. Apply softmax normalization → attention distribution sums to 1.
    4. Scale by global ACh → final per-column gain values.

  The attention weights are updated with a simple Hebbian rule:
  columns that fire strongly when attended get reinforced, creating a
  self-sharpening loop (winner-take-more across columns).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AttentionConfig


class SpatialAttentionController:
    """
    Computes per-column attention gains from a top-down association signal.

    Usage::

        attn = SpatialAttentionController(
            assoc_neurons=32,
            n_columns=4,
        )
        # Each network step:
        gains = attn.compute(assoc_spikes, global_ach=0.6)
        # gains is a dict: {"col_0": 1.45, "col_1": 0.88, ...}
    """

    def __init__(
        self,
        assoc_neurons: int,
        n_columns: int,
        column_names: list[str],
        config: AttentionConfig | None = None,
    ) -> None:
        self.config = config or AttentionConfig()
        self.assoc_neurons = assoc_neurons
        self.n_columns = n_columns
        self.column_names = list(column_names)

        # Learned projection: association activity → per-column saliency
        self.w_attn: np.ndarray = np.random.uniform(
            -0.1, 0.1, (assoc_neurons, n_columns)
        ).astype(np.float32)

        # Smoothed attention distribution (temporal persistence)
        self._attn_weights: np.ndarray = np.full(
            n_columns, 1.0 / n_columns, dtype=np.float32
        )

        # Per-column gain outputs from last compute() call
        self.column_gains: dict[str, float] = {
            name: 1.0 for name in column_names
        }

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        assoc_activity: np.ndarray,
        global_ach: float = 0.5,
    ) -> dict[str, float]:
        """
        Compute per-column attention gains.

        Args:
            assoc_activity:  Spike/rate vector from the association layer
                             (shape: assoc_neurons).
            global_ach:      Global acetylcholine level [0, 1].
                             Scales the overall attention effect.

        Returns:
            Dict mapping column_name → gain (float ≥ 1.0 for attended,
            < 1.0 for suppressed).
        """
        act = assoc_activity.astype(np.float32)

        # Raw saliency per column
        raw = act @ self.w_attn  # (n_columns,)

        # Softmax with temperature
        shifted = raw - np.max(raw)  # numerical stability
        exp_vals = np.exp(shifted / max(self.config.temperature, 1e-6))
        softmax = exp_vals / (np.sum(exp_vals) + 1e-8)

        # Temporal smoothing
        self._attn_weights = (
            self._attn_weights * self.config.decay
            + softmax * (1.0 - self.config.decay)
        )

        # Gain: uniform baseline (1.0) + ACh-scaled attention boost
        # Columns above average get boosted; below average get suppressed
        mean_w = np.mean(self._attn_weights)
        gain_modulation = (self._attn_weights - mean_w) * self.config.gain_strength
        gains = 1.0 + global_ach * gain_modulation

        # Ensure gains don't go negative
        gains = np.maximum(gains, 0.1)

        self.column_gains = {
            name: float(gains[i]) for i, name in enumerate(self.column_names)
        }
        return self.column_gains

    def update(
        self,
        assoc_activity: np.ndarray,
        column_activities: dict[str, np.ndarray],
    ) -> None:
        """
        Hebbian update of attention projection weights.

        Reinforces the connection between association patterns and columns
        that were active when attended.

        Args:
            assoc_activity:     Association layer spikes (assoc_neurons,).
            column_activities:  Dict mapping column_name → spike array.
        """
        act = assoc_activity.astype(np.float32)

        for i, name in enumerate(self.column_names):
            if name in column_activities:
                col_rate = float(np.mean(column_activities[name]))
                gain = self.column_gains.get(name, 1.0)
                # Reinforce: if column was attended (high gain) AND active → strengthen
                signal = col_rate * (gain - 1.0)
                self.w_attn[:, i] += self.config.learning_rate * act * signal
        np.clip(self.w_attn, -2.0, 2.0, out=self.w_attn)

    def reset_state(self) -> None:
        """Reset transient state. Learned weights are preserved."""
        self._attn_weights.fill(1.0 / self.n_columns)
        self.column_gains = {name: 1.0 for name in self.column_names}

    @property
    def attention_distribution(self) -> np.ndarray:
        """Current smoothed attention weights (n_columns,)."""
        return self._attn_weights.copy()
