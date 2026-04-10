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
      Gate signal = σ(gain×(ACh-thresh)) × σ(gain×(DA-thresh))
      Soft gating scales feedforward current, not binary on/off.
    """

    # ── NetworkGraph layer interface ─────────────────────────────

    @property
    def num_inputs(self) -> int:
        return self.num_external_inputs

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

        # ── Gate state (soft sigmoid) ────────────────────────────────
        self._gate_signal: float = 0.0
        self._gate_gain: float = 8.0  # sigmoid steepness
        self._ach_level: float = 0.0
        self._da_level: float = 0.0

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
        """Dual ACh+DA soft conjunction gate (O'Reilly & Frank 2006).

        gate_signal = σ(gain×(ACh-thresh)) × σ(gain×(DA-thresh))
        Smooth sigmoid replaces binary threshold for gradient-friendly dynamics.
        """
        cfg = self.config
        self._ach_level = float(ach_level)
        self._da_level = float(da_level)
        g = self._gate_gain
        ach_sig = 1.0 / (1.0 + np.exp(-g * (ach_level - cfg.ach_gate_threshold)))
        da_sig = 1.0 / (1.0 + np.exp(-g * (da_level - cfg.da_gate_threshold)))
        self._gate_signal = float(ach_sig * da_sig)

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def forward(self, external_input: NDArray[np.float32]) -> NDArray[np.float32]:
        """One timestep of WM dynamics.

        Feedforward current scaled by soft gate signal [0, 1].
        Recurrent attractor always active for content maintenance.

        Returns:
            (num_neurons,) spike array as float32.
        """
        cfg = self.config
        gate = self._gate_signal

        # ── Trace decay ───────────────────────────────────────────────
        self.x_pre *= self._pre_decay
        self.x_post *= self._post_decay

        # ── Input current (scaled by soft gate) ─────────────────────
        ext_f32 = external_input.astype(np.float32)
        external_current = gate * (ext_f32 @ self.w_ff)
        self.x_pre += np.clip(ext_f32, 0.0, 1.0) * gate

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

        # ── Eligibility traces (feedforward, gate-scaled) ───────────
        self.e *= self._trace_decay
        if gate > 0.01:
            if np.any(self.has_spiked):
                self.e[:, self.has_spiked] += gate * self.x_pre[:, np.newaxis]
            pre_active = ext_f32 > 0.1
            if np.any(pre_active):
                self.e[pre_active, :] += gate * self.x_post[np.newaxis, :]

        # ── Content update + lateral learning ─────────────────────────
        self.content = (
            self.content * self._content_decay
            + self.has_spiked.astype(np.float32)
        )
        self._update_lateral_weights()

        return self.has_spiked.astype(np.float32)

    # ------------------------------------------------------------------
    # NetworkGraph-compatible neuromodulator setters
    # ------------------------------------------------------------------

    def set_ach_level(self, ach: float) -> None:
        """ACh level for gating (re-evaluated on next gate() call)."""
        self._ach_level = float(ach)

    def set_ne_level(self, ne: float) -> None:
        """NE level — no direct effect on WM dynamics."""
        pass

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
        self._gate_signal = 0.0
        self._ach_level = 0.0
        self._da_level = 0.0
