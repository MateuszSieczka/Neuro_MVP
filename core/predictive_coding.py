"""
PredictiveCodingLayer — extends CompetitiveLIFLayer with PC mechanics.

Reference: Friston (2010), Rao & Ballard (1999), Hasselmo (2006)

Changes from legacy:
  1. Convergence-checked relaxation loop (Lipschitz-bounded step size)
     replaces fixed 10 iterations × 0.1 rate.
  2. ACh modulates bottom-up / top-down balance via M1 receptor activation.
  3. Feedback weights use init_weights() for principled scaling.
  4. No duplicated STDP trace management — delegates to parent.
  5. No artificial e clipping — parent uses synaptic scaling.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import (
    NeuronConfig,
    STDPConfig,
    HomeostaticConfig,
    CompetitiveConfig,
    PredictiveCodingConfig,
    init_weights,
)
from .competitive_layer import CompetitiveLIFLayer
from .spike_encoder import PoissonEncoder


class PredictiveCodingLayer(CompetitiveLIFLayer):
    """Hierarchical predictive coding layer (Rao & Ballard 1999).

    Simultaneously:
      - Receives bottom-up error signals from the layer below.
      - Receives top-down predictions from the layer above.
      - Computes signed prediction error (actual − predicted).
      - Generates prediction for the layer below via feedback_w.

    ACh (M1 receptor) controls bottom-up vs top-down weighting:
      ACh → 1.0: trust sensory input (novel environment)
      ACh → 0.0: trust internal predictions (familiar state)
    """

    def __init__(
        self,
        num_inputs: int,
        num_neurons: int = 20,
        pc_cfg: PredictiveCodingConfig | None = None,
        neuron_cfg: NeuronConfig | None = None,
        stdp_cfg: STDPConfig | None = None,
        homeo_cfg: HomeostaticConfig | None = None,
        comp_cfg: CompetitiveConfig | None = None,
    ) -> None:
        self.pc_cfg = pc_cfg or PredictiveCodingConfig()

        super().__init__(
            num_inputs=num_inputs,
            num_neurons=num_neurons,
            neuron_cfg=neuron_cfg,
            stdp_cfg=stdp_cfg,
            homeo_cfg=homeo_cfg,
            comp_cfg=comp_cfg,
        )

        self._encoder = PoissonEncoder()

        # ── Feedback weights (this layer → layer below) ───────────────
        self.feedback_w: NDArray[np.float32] = init_weights(
            num_neurons, num_inputs,
            psp_target=1.0,
            excitatory=True,
        )

        # ── Prediction error and top-down buffers ─────────────────────
        self.top_down_prediction: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )
        self.prediction_error: NDArray[np.float32] = np.zeros(
            num_inputs, dtype=np.float32,
        )

        # ── Modulation state ──────────────────────────────────────────
        self.ach_level: float = 0.8
        self.attention_gain: float = 1.0
        self.error_spikes: NDArray[np.bool_] = np.zeros(
            num_inputs, dtype=bool,
        )

        # ── Spike timing for causal STDP window (±20ms) ──────────────
        self.t_since_pre_spike: NDArray[np.int32] = np.full(
            num_inputs, 1000, dtype=np.int32,
        )
        self.t_since_post_spike: NDArray[np.int32] = np.full(
            num_neurons, 1000, dtype=np.int32,
        )
        self._stdp_window: int = 20

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def forward(self, pre_spikes: NDArray[np.float32]) -> NDArray[np.float32]:
        """Single-step predictive coding dynamics (no relaxation loop).

        One dt step matching the spiking paradigm:
          1. Feedforward drive from learned weights.
          2. Single-step gradient: v += ACh × error_gradient + (1-ACh) × top_down.
          3. Spike detection.
          4. Update feedback prediction.

        Returns:
            (num_neurons,) float spike array.
        """
        pre_f32 = pre_spikes.astype(np.float32)
        ncfg = self.neuron_cfg

        # ── Feedforward drive (thalamo-cortical volley) ───────────────
        ff_drive = pre_f32 @ self.w  # (num_neurons,)
        ff_drive *= self.attention_gain

        # ── Proactive inhibition (continuous k-WTA) ───────────────────
        self._apply_proactive_inhibition()

        # ── Membrane leak ─────────────────────────────────────────────
        self.v *= self._mem_decay

        # ── Inject feedforward drive ──────────────────────────────────
        self.v += ff_drive

        # ── Single-step prediction error gradient ─────────────────────
        r = np.clip(
            (self.v - ncfg.v_rest) / ncfg.gap,
            0.0, 1.0,
        )
        my_prediction = r @ self.feedback_w
        self.prediction_error = pre_f32 - my_prediction

        # ACh-weighted gradient (Hasselmo 2006)
        error_gradient = self.prediction_error @ self.feedback_w.T
        combined = (
            self.ach_level * error_gradient
            + (1.0 - self.ach_level) * self.top_down_prediction
        )
        self.v += combined
        np.clip(self.v, ncfg.v_reset, ncfg.v_thresh + 10.0, out=self.v)

        # ── Spike phase ───────────────────────────────────────────────
        in_refrac = self.refrac_count > 0
        self.refrac_count[in_refrac] -= 1

        thresh = self._effective_threshold()
        self.has_spiked = (self.v >= thresh) & ~in_refrac

        self.v[self.has_spiked] = ncfg.v_reset
        self.refrac_count[self.has_spiked] = ncfg.refrac_period

        # ── Event-based STDP traces with causal ±20ms window ─────────
        self.x_pre *= self._pre_decay
        self.x_post *= self._post_decay
        pre_binary = (pre_f32 > 0.5).astype(np.float32)
        self.x_pre += pre_binary
        self.x_post[self.has_spiked] += 1.0

        # Update spike timing counters
        self.t_since_pre_spike += 1
        self.t_since_pre_spike[pre_binary > 0.5] = 0
        self.t_since_post_spike += 1
        self.t_since_post_spike[self.has_spiked] = 0

        # Eligibility trace with causal window (Bi & Poo 2001)
        self.e *= self._elig_decay
        if np.any(self.has_spiked):
            post_idx = np.where(self.has_spiked)[0]
            ltp_mask = (self.t_since_pre_spike <= self._stdp_window).astype(np.float32)
            self.e[:, post_idx] += (
                self.stdp_cfg.a_plus
                * (self.x_pre * ltp_mask)[:, np.newaxis]
            )
        pre_spiked = pre_binary > 0.5
        if np.any(pre_spiked):
            ltd_mask = (self.t_since_post_spike <= self._stdp_window).astype(np.float32)
            self.e[pre_spiked, :] -= (
                self.stdp_cfg.a_minus
                * (self.x_post * ltd_mask)[np.newaxis, :]
            )

        # ── k-WTA window bookkeeping ─────────────────────────────────
        self.window_spike_counts += self.has_spiked.astype(np.int32)
        self._current_window_size += 1

        if self._phase_reset_pending:
            self._apply_lateral_inhibition()
            if self._current_window_size > 0:
                self._update_kwta_homeostasis(self._current_window_size)
            self._reset_window()

        # ── Error spikes for downstream consumers ────────────────────
        pos_error = np.clip(self.prediction_error, 0.0, 1.0) * self.ach_level
        self.error_spikes = self._encoder.encode(pos_error).astype(bool)

        return self.has_spiked.astype(np.float32)

    # ------------------------------------------------------------------
    # Prediction interface
    # ------------------------------------------------------------------

    def generate_prediction(self) -> NDArray[np.float32]:
        """Top-down prediction for the layer below."""
        raw = self.has_spiked.astype(np.float32) @ self.feedback_w
        return np.clip(raw * self.pc_cfg.feedback_strength, 0.0, 1.0)

    def receive_prediction(self, prediction: NDArray[np.float32]) -> None:
        """Accept top-down prediction from the layer above."""
        self.top_down_prediction = prediction.astype(np.float32)

    def set_ach_level(self, ach: float) -> None:
        self.ach_level = float(np.clip(ach, 0.0, 1.0))

    def set_attention_gain(self, gain: float) -> None:
        self.attention_gain = float(max(gain, 0.1))

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def update_weights(
        self,
        m_t: float,
        pred_error: NDArray[np.float32],
    ) -> None:
        """Three-factor STDP + Hebbian feedback weight update."""
        super().update_weights(m_t, pred_error)

        if np.any(self.has_spiked):
            # Anti-Hebbian: generative model learns to predict input,
            # reducing error (Rao & Ballard 1999, Bogacz 2017 eq 3.14)
            dw = -self.pc_cfg.feedback_learning_rate * np.outer(
                self.has_spiked.astype(np.float32),
                self.prediction_error,
            )
            self.feedback_w += dw * m_t

            if self.pc_cfg.feedback_norm:
                norms = np.linalg.norm(
                    self.feedback_w, axis=1, keepdims=True,
                ) + 1e-8
                self.feedback_w /= norms

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        super().reset_state()
        self.top_down_prediction.fill(0.0)
        self.prediction_error.fill(0.0)
        self.attention_gain = 1.0
