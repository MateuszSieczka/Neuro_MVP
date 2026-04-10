"""
Working Memory — prefrontal attractor dynamics with dual ACh+DA gating.

Reference:
  Goldman-Rakic (1995)  Prefrontal persistent activity
  O'Reilly & Frank (2006)  "Making working memory work"

Changes from legacy:
  1. Dual gating: ACh (sensory) AND DA (update signal) — conjunction gate
  2. Uses WorkingMemoryConfig from config.py (derived mem_decay, content_decay)
  3. Gate opens only when BOTH ACh ≥ threshold AND DA ≥ threshold
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import WorkingMemoryConfig


class WorkingMemoryModule:
    """Persistent WM via recurrent attractor dynamics with dual gating.

    Dual gating (O'Reilly & Frank 2006):
      ACh gates sensory input (bottom-up relevance)
      DA gates update signal (reward PE → new context important)
      Gate OPEN:  ACh ≥ threshold AND DA ≥ threshold (conjunction)
      Gate CLOSED: sustain content through w_lateral alone
    """

    def __init__(
        self,
        num_external_inputs: int,
        num_neurons: int,
        config: WorkingMemoryConfig | None = None,
    ) -> None:
        self.config = config or WorkingMemoryConfig()
        self.num_neurons = num_neurons
        self.num_external_inputs = num_external_inputs
        cfg = self.config

        # ── Membrane state ────────────────────────────────────────────
        self.v: NDArray[np.float32] = np.full(
            num_neurons, cfg.v_rest, dtype=np.float32,
        )
        self.has_spiked: NDArray[np.bool_] = np.zeros(num_neurons, dtype=bool)
        self.refrac_count: NDArray[np.int32] = np.zeros(num_neurons, dtype=np.int32)

        # ── Synaptic weights ──────────────────────────────────────────
        self.w_ff: NDArray[np.float32] = np.random.uniform(
            0.1, 0.5, (num_external_inputs, num_neurons),
        ).astype(np.float32)
        self.w_lateral: NDArray[np.float32] = np.zeros(
            (num_neurons, num_neurons), dtype=np.float32,
        )

        # ── Eligibility traces ────────────────────────────────────────
        self.e: NDArray[np.float32] = np.zeros(
            (num_external_inputs, num_neurons), dtype=np.float32,
        )
        self.x_pre: NDArray[np.float32] = np.zeros(
            num_external_inputs, dtype=np.float32,
        )
        self.x_post: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )

        # ── Precomputed decays from config ────────────────────────────
        self._mem_decay: float = cfg.mem_decay
        self._trace_decay: float = cfg.ctx.decay(cfg.tau_e)
        self._pre_decay: float = cfg.ctx.decay(cfg.tau_pre)
        self._post_decay: float = cfg.ctx.decay(cfg.tau_post)
        self._content_decay: float = cfg.content_decay

        # ── Gate state ────────────────────────────────────────────────
        self.gate_open: bool = False

        # ── Content: low-pass filtered activity (attractor trace) ─────
        self.content: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )

        # ── Prediction error placeholder ──────────────────────────────
        self.prediction_error: NDArray[np.float32] = np.ones(
            num_neurons, dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Dual gating (O'Reilly & Frank 2006)
    # ------------------------------------------------------------------

    def gate(self, ach_level: float, da_level: float = 1.0) -> None:
        """Dual ACh+DA conjunction gate.

        Gate opens only when BOTH ACh ≥ threshold AND DA ≥ threshold.
        ACh gates sensory input, DA gates update signal.
        """
        cfg = self.config
        self.gate_open = (
            float(ach_level) >= cfg.ach_gate_threshold
            and float(da_level) >= cfg.da_gate_threshold
        )

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def forward(self, external_input: NDArray[np.float32]) -> NDArray[np.bool_]:
        """One timestep of WM dynamics.

        OPEN:   integrates external_input through w_ff + recurrent w_lateral.
        CLOSED: ignores external_input; sustains content through w_lateral.
        """
        cfg = self.config

        # ── Trace decay ───────────────────────────────────────────────
        self.x_pre *= self._pre_decay
        self.x_post *= self._post_decay

        # ── Input current ─────────────────────────────────────────────
        if self.gate_open:
            ext_f32 = external_input.astype(np.float32)
            external_current = ext_f32 @ self.w_ff
            self.x_pre += np.clip(ext_f32, 0.0, 1.0)
        else:
            external_current = np.zeros(self.num_neurons, dtype=np.float32)

        # Recurrent contribution always active (attractor maintenance)
        recurrent_current = (
            self.content @ self.w_lateral * cfg.lateral_strength
        )
        total_current = external_current + recurrent_current

        # ── LIF integration ───────────────────────────────────────────
        in_refrac = self.refrac_count > 0
        self.refrac_count[in_refrac] -= 1

        integrated_v = (
            self.v * self._mem_decay
            + (cfg.v_rest + total_current) * (1.0 - self._mem_decay)
        )
        self.v = np.where(in_refrac, cfg.v_reset, integrated_v)
        self.has_spiked = (self.v >= cfg.v_thresh) & ~in_refrac

        self.v[self.has_spiked] = cfg.v_reset
        self.refrac_count[self.has_spiked] = cfg.refrac_period
        self.x_post[self.has_spiked] += 1.0

        # ── Eligibility traces (feedforward, gate-gated) ─────────────
        if self.gate_open:
            self.e *= self._trace_decay
            if np.any(self.has_spiked):
                self.e[:, self.has_spiked] += self.x_pre[:, np.newaxis]
            ext_f32 = external_input.astype(np.float32)
            pre_active = ext_f32 > 0.1
            if np.any(pre_active):
                self.e[pre_active, :] += self.x_post[np.newaxis, :]

        # ── Content update + lateral learning ─────────────────────────
        self.content = (
            self.content * self._content_decay
            + self.has_spiked.astype(np.float32)
        )
        self._update_lateral_weights()

        return self.has_spiked

    # ------------------------------------------------------------------
    # Lateral Hebbian learning
    # ------------------------------------------------------------------

    def _update_lateral_weights(self) -> None:
        """Hebbian co-activation: neurons that fire together wire together."""
        active = self.has_spiked.astype(np.float32)
        if np.sum(active) < 2:
            return

        dw = self.config.lateral_lr * np.outer(active, active)
        np.fill_diagonal(dw, 0.0)
        self.w_lateral += dw

        # Soft normalisation
        row_max = np.max(self.w_lateral, axis=1, keepdims=True)
        scale = np.where(row_max > 1.0, row_max, 1.0)
        self.w_lateral /= scale
        np.fill_diagonal(self.w_lateral, 0.0)

    # ------------------------------------------------------------------
    # Weight update (three-factor rule)
    # ------------------------------------------------------------------

    def update_weights(self, m_t: float, pred_error: NDArray[np.float32]) -> None:
        """Three-factor STDP for feedforward weights."""
        if np.isclose(m_t, 0.0):
            return
        dw = self.config.learning_rate * m_t * self.e * pred_error
        self.w_ff += dw

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset transient state. Learned weights preserved."""
        self.v.fill(self.config.v_rest)
        self.e.fill(0.0)
        self.x_pre.fill(0.0)
        self.x_post.fill(0.0)
        self.refrac_count.fill(0)
        self.has_spiked.fill(False)
        self.content.fill(0.0)
        self.prediction_error.fill(1.0)
        self.gate_open = False
