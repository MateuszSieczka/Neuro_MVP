"""
Spatial Attention — top-down + bottom-up saliency with IOR.

Reference:
  Reynolds & Heeger (2009)  Normalization model of attention
  Posner & Cohen (1984)     Inhibition of return
  Usher & Damasio (2000)    NE inverse-U / locus coeruleus

Changes from legacy:
  1. Bottom-up saliency (prediction error magnitude per column)
  2. Inhibition of Return (IOR): τ_IOR ≈ 400ms inhibitory trace
  3. Adaptive temperature modulated by NE (inverse-U)
  4. Uses AttentionConfig from config.py (derived IOR decay)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import AttentionConfig


class SpatialAttentionController:
    """Per-column attention gains from top-down + bottom-up signals.

    Total saliency = α × bottom_up + (1-α) × top_down
    α modulated by task engagement (tonic DA).
    Temperature modulated by NE (inverse-U).
    IOR suppresses previously attended columns.
    """

    def __init__(
        self,
        assoc_neurons: int,
        n_columns: int,
        column_names: list[str],
        config: AttentionConfig | None = None,
        assoc_name: str = "assoc",
    ) -> None:
        self.config = config or AttentionConfig()
        self.assoc_neurons = assoc_neurons
        self.n_columns = n_columns
        self.column_names = list(column_names)
        self.assoc_name = assoc_name
        cfg = self.config

        # ── Top-down projection weights ───────────────────────────────
        self.w_attn: NDArray[np.float32] = np.random.uniform(
            -0.1, 0.1, (assoc_neurons, n_columns),
        ).astype(np.float32)

        # ── Smoothed attention distribution ───────────────────────────
        self._attn_weights: NDArray[np.float32] = np.full(
            n_columns, 1.0 / n_columns, dtype=np.float32,
        )

        # ── IOR trace per column (Posner & Cohen 1984) ────────────────
        self._ior_trace: NDArray[np.float32] = np.zeros(
            n_columns, dtype=np.float32,
        )
        self._ior_decay: float = cfg.ior_decay

        # ── Bottom-up saliency (prediction error per column) ──────────
        self._bu_saliency: NDArray[np.float32] = np.zeros(
            n_columns, dtype=np.float32,
        )

        # ── Per-column gain outputs ───────────────────────────────────
        self.column_gains: dict[str, float] = {
            name: 1.0 for name in column_names
        }

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        assoc_activity: NDArray[np.float32],
        global_ach: float = 0.5,
        ne_level: float = 0.5,
        bottom_up_errors: NDArray[np.float32] | None = None,
    ) -> dict[str, float]:
        """Compute per-column attention gains.

        Args:
            assoc_activity:    Association layer activity (assoc_neurons,).
            global_ach:        ACh level scales overall attention effect.
            ne_level:          NE level modulates softmax temperature.
            bottom_up_errors:  Per-column prediction error magnitude (n_columns,).
        """
        act = assoc_activity.astype(np.float32)
        cfg = self.config

        # ── Top-down saliency ─────────────────────────────────────────
        td_raw = act @ self.w_attn  # (n_columns,)

        # ── Bottom-up saliency (surprise) ─────────────────────────────
        if bottom_up_errors is not None:
            self._bu_saliency = np.abs(bottom_up_errors).astype(np.float32)
            bu_norm = self._bu_saliency / (np.sum(self._bu_saliency) + 1e-8)
        else:
            bu_norm = np.zeros(self.n_columns, dtype=np.float32)

        # ── Mix: α × bottom_up + (1 - α) × top_down ──────────────────
        alpha = cfg.bottom_up_weight
        td_shifted = td_raw - np.max(td_raw)
        temperature = cfg.ne_modulated_temperature(ne_level)
        td_exp = np.exp(td_shifted / max(temperature, 1e-6))
        td_norm = td_exp / (np.sum(td_exp) + 1e-8)

        combined = alpha * bu_norm + (1.0 - alpha) * td_norm

        # ── Apply IOR suppression ─────────────────────────────────────
        combined = combined * (1.0 - cfg.ior_strength * self._ior_trace)
        combined = np.maximum(combined, 0.0)

        # Re-normalize
        total = np.sum(combined) + 1e-8
        combined = combined / total

        # ── Temporal smoothing ────────────────────────────────────────
        self._attn_weights = (
            self._attn_weights * cfg.decay
            + combined.astype(np.float32) * (1.0 - cfg.decay)
        )

        # ── IOR update: attended columns accumulate inhibitory trace ──
        self._ior_trace *= self._ior_decay
        # Columns above mean attention → accumulate IOR
        mean_w = float(np.mean(self._attn_weights))
        ior_input = np.maximum(self._attn_weights - mean_w, 0.0)
        self._ior_trace += ior_input * (1.0 - self._ior_decay)
        np.clip(self._ior_trace, 0.0, 1.0, out=self._ior_trace)

        # ── Gain: baseline + ACh × attention modulation ───────────────
        mean_a = float(np.mean(self._attn_weights))
        gain_modulation = (self._attn_weights - mean_a) * cfg.gain_strength
        gains = 1.0 + global_ach * gain_modulation
        gains = np.maximum(gains, 0.1).astype(np.float32)

        self.column_gains = {
            name: float(gains[i]) for i, name in enumerate(self.column_names)
        }
        return self.column_gains

    def update(
        self,
        assoc_activity: NDArray[np.float32],
        column_activities: dict[str, NDArray[np.float32]],
    ) -> None:
        """Hebbian update of top-down attention projection weights."""
        act = assoc_activity.astype(np.float32)
        cfg = self.config

        for i, name in enumerate(self.column_names):
            if name in column_activities:
                col_rate = float(np.mean(column_activities[name]))
                gain = self.column_gains.get(name, 1.0)
                signal = col_rate * (gain - 1.0)
                self.w_attn[:, i] += cfg.learning_rate * act * signal
        np.clip(self.w_attn, -2.0, 2.0, out=self.w_attn)

    def reset_state(self) -> None:
        """Reset transient state. Learned weights preserved."""
        self._attn_weights.fill(1.0 / self.n_columns)
        self._ior_trace.fill(0.0)
        self._bu_saliency.fill(0.0)
        self.column_gains = {name: 1.0 for name in self.column_names}

    @property
    def attention_distribution(self) -> NDArray[np.float32]:
        """Current smoothed attention weights (n_columns,)."""
        return self._attn_weights.copy()
