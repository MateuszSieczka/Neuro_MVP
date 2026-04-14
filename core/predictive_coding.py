"""
PredictiveCodingLayer — extends CompetitiveLIFLayer with PC mechanics.

Reference: Friston (2010), Rao & Ballard (1999), Hasselmo (2006), Bogacz (2017)

Changes from legacy:
  1. Convergence-checked relaxation loop (Lipschitz-bounded step size)
     iterates belief update 3-5× per dt before spike generation.
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
        """Predictive coding with relaxation loop + AdEx dynamics.

        Per timestep dt:
          1. Feedforward drive (once).
          2. Inner relaxation loop (Bogacz 2017): iteratively refine
             belief (rate proxy r) until prediction error converges.
             Step size bounded by 1/L (Lipschitz, from feedback_w).
          3. Use converged error gradient for I_syn.
          4. AdEx membrane integration via Exponential Euler.
          5. Spike detection + adaptation + STDP traces.

        Returns:
            (num_neurons,) float spike array.
        """
        pre_f32 = pre_spikes.astype(np.float32)
        ncfg = self.neuron_cfg
        ctx = ncfg.ctx

        # ── Feedforward drive (thalamo-cortical volley) ───────────────
        ff_drive = pre_f32 @ self.w  # (num_neurons,)
        ff_drive *= self.attention_gain

        # ── Proactive inhibition (continuous k-WTA) ───────────────────
        self._apply_proactive_inhibition()

        # ── Relaxation loop (Rao & Ballard 1999; Bogacz 2017) ────────
        # Rate proxy from current membrane state
        r = np.clip(
            (self.v - ncfg.v_rest) / ncfg.gap,
            0.0, 1.0,
        )
        # Lipschitz-bounded step size: η = 1 / ||W^T W|| ≈ 1 / ||W||_F²/n
        # Ensures gradient descent on F(r) = 0.5‖x - rW‖² is stable.
        w_norm_sq = max(float(np.sum(self.feedback_w ** 2)), 1e-6)
        step_size = min(float(self.feedback_w.shape[0]) / w_norm_sq, 0.5)

        n_relax = self.pc_cfg.n_relax_steps
        tol = self.pc_cfg.relax_tolerance

        for _ in range(n_relax):
            my_prediction = r @ self.feedback_w
            error = pre_f32 - my_prediction
            error_gradient = error @ self.feedback_w.T
            dr = step_size * (
                self.ach_level * error_gradient
                + (1.0 - self.ach_level) * self.top_down_prediction
            )
            r = np.clip(r + dr, 0.0, 1.0)
            # Convergence check
            if float(np.max(np.abs(dr))) < tol:
                break

        # Final prediction error from converged belief
        self.prediction_error = pre_f32 - r @ self.feedback_w

        # Converged error gradient for membrane drive
        combined = (
            self.ach_level * (self.prediction_error @ self.feedback_w.T)
            + (1.0 - self.ach_level) * self.top_down_prediction
        )

        # ── Total synaptic input ──────────────────────────────────────
        I_syn = ff_drive + combined

        # ── AdEx membrane integration via Exponential Euler ───────────
        # ATP modulation (Krok 1.3)
        if self._astrocyte is not None:
            z = self._zone_idx
            eff_v_thresh = ncfg.v_thresh + self._astrocyte.threshold_shift[z]
            eff_g_L = ncfg.g_L * self._astrocyte.leak_gain[z]
        else:
            eff_v_thresh = ncfg.v_thresh
            eff_g_L = ncfg.g_L

        exp_term = np.exp(
            np.clip((self.v - eff_v_thresh) / ncfg.delta_t, -20.0, 10.0),
        )
        inv_Cm = 1.0 / ncfg.C_m
        F_v = inv_Cm * (
            -eff_g_L * (self.v - ncfg.v_rest)
            + eff_g_L * ncfg.delta_t * exp_term
            + I_syn - self.w_adapt
        )
        J_v = inv_Cm * (-eff_g_L + eff_g_L * exp_term)

        in_refrac = self.refrac_count > 0
        self.refrac_count[in_refrac] -= 1

        integrated_v = ctx.exp_euler_step(self.v, F_v, J_v)
        self.v = np.where(in_refrac, ncfg.v_reset, integrated_v)

        # ── Spike detection ───────────────────────────────────────────
        thresh = self._effective_threshold()
        spike_thresh = np.minimum(
            np.float32(ncfg.v_spike_cutoff), thresh,
        ) if isinstance(thresh, np.ndarray) else np.float32(ncfg.v_spike_cutoff)
        self.has_spiked = (self.v >= spike_thresh) & ~in_refrac

        self.v[self.has_spiked] = ncfg.v_reset
        self.w_adapt[self.has_spiked] += ncfg.b
        self.refrac_count[self.has_spiked] = ncfg.refrac_period

        # ── Subthreshold adaptation ───────────────────────────────────
        self.w_adapt = (
            self.w_adapt * ncfg.w_decay
            + ncfg.a * (self.v - ncfg.v_rest) * ncfg.w_gain
        )

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
