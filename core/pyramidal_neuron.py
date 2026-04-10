"""
PyramidalLayer — Multi-compartment pyramidal neuron with Ca²⁺ spike dynamics.

Reference: Larkum, Zhu & Bhatt (1999), Payeur et al. (2021), Sacramento et al. (2018)

Architecture:
  Basal dendrites  ← feedforward / bottom-up spikes (w, inherited)
  Apical dendrites ← top-down context predictions (w_apical)
  Soma             ← integrates both; generates spikes

Key biophysics:
  - Apical trunk passive cable τ = 100-200ms (electrotonically remote)
  - Ca²⁺ spike: voltage-dependent activation m_Ca = sigmoid((V - V_half) / k)
  - BAC firing: soma spike + apical Ca²⁺ plateau within window → burst
  - Burst → 3-5× STDP eligibility boost (Payeur et al. 2021)

Changes from legacy:
  1. Uses composable configs (PyramidalConfig + NeuronConfig + STDPConfig).
  2. Proper Ca²⁺ voltage-dependent activation (not threshold crossing).
  3. Convergence-checked relaxation (shared logic with PC layer).
  4. No duplicated homeostasis flag chain — manages own homeostatic
     adaptation with BCM-derived HomeostaticConfig.
  5. Principled weight init via init_weights().
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import (
    NeuronConfig,
    STDPConfig,
    HomeostaticConfig,
    CompetitiveConfig,
    PyramidalConfig,
    PredictiveCodingConfig,
    init_weights,
)
from .competitive_layer import CompetitiveLIFLayer


class PyramidalLayer(CompetitiveLIFLayer):
    """Multi-compartment pyramidal neuron layer with burst-dependent plasticity.

    Manages its own homeostatic threshold adaptation independently from
    the CompetitiveLIFLayer k-WTA homeostasis by using a separate
    HomeostaticConfig with BCM-derived learning rate.
    """

    def __init__(
        self,
        num_inputs: int,
        num_neurons: int = 20,
        pyr_cfg: PyramidalConfig | None = None,
        neuron_cfg: NeuronConfig | None = None,
        stdp_cfg: STDPConfig | None = None,
        homeo_cfg: HomeostaticConfig | None = None,
        comp_cfg: CompetitiveConfig | None = None,
        pc_cfg: PredictiveCodingConfig | None = None,
    ) -> None:
        self.pyr_cfg = pyr_cfg or PyramidalConfig()
        self.pc_cfg = pc_cfg or PredictiveCodingConfig()
        ncfg = neuron_cfg or NeuronConfig()
        hcfg = homeo_cfg or HomeostaticConfig()

        # CompetitiveLIFLayer creates k-WTA; we pass homeo_cfg=None
        # to parent and manage homeostasis ourselves.
        super().__init__(
            num_inputs=num_inputs,
            num_neurons=num_neurons,
            neuron_cfg=ncfg,
            stdp_cfg=stdp_cfg or STDPConfig(),
            homeo_cfg=None,   # Managed locally
            comp_cfg=comp_cfg,
        )

        # Store homeo config for local management
        self._pyr_homeo = hcfg

        # ── Apical compartment weights ────────────────────────────────
        self.w_apical: NDArray[np.float32] = init_weights(
            num_inputs, num_neurons,
            psp_target=0.5,
            excitatory=True,
        )

        # ── Apical membrane state ─────────────────────────────────────
        self.v_apical: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )

        # ── Top-down prediction buffer ────────────────────────────────
        self.top_down_prediction: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )

        # ── Burst / Ca²⁺ plateau state ───────────────────────────────
        self.is_burst: NDArray[np.bool_] = np.zeros(num_neurons, dtype=bool)
        self.plateau_timer: NDArray[np.int32] = np.zeros(
            num_neurons, dtype=np.int32,
        )
        self.in_plateau: NDArray[np.bool_] = np.zeros(num_neurons, dtype=bool)

        # ── Prediction error (for network feedback) ───────────────────
        self.prediction_error: NDArray[np.float32] = np.zeros(
            num_inputs, dtype=np.float32,
        )

        # ── Local homeostatic state ───────────────────────────────────
        self.v_thresh_adaptive: NDArray[np.float32] = np.full(
            num_neurons, ncfg.v_thresh, dtype=np.float32,
        )
        self.avg_rate: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )

        # ── ACh modulation ────────────────────────────────────────────
        self._ach_apical_scale: float = 1.0

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def forward(self, pre_spikes: NDArray[np.float32]) -> NDArray[np.bool_]:
        pre_f32 = pre_spikes.astype(np.float32)
        ncfg = self.neuron_cfg
        pyr = self.pyr_cfg
        pc = self.pc_cfg

        # ── STDP trace update ─────────────────────────────────────────
        self.x_pre *= self._pre_decay
        self.x_post *= self._post_decay
        self.x_pre += np.clip(pre_f32, 0.0, 1.0)

        # ── 1. Apical integration (passive, slow) ────────────────────
        apical_current = self.top_down_prediction.astype(np.float32)
        apical_gain = ncfg.ctx.complement(pyr.tau_apical)
        self.v_apical = (
            self.v_apical * pyr.apical_decay
            + apical_current * apical_gain
        )

        # ── 2. Ca²⁺ plateau trigger (voltage-dependent, Larkum 2013) ─
        # m_Ca = sigmoid((V_apical - V_half) / k)
        ca_activation = 1.0 / (
            1.0 + np.exp(-(self.v_apical - pyr.ca_v_half) / pyr.ca_k)
        )
        new_plateaus = (ca_activation > 0.5) & (self.plateau_timer == 0)
        self.plateau_timer[new_plateaus] = pyr.plateau_duration_ms
        self.in_plateau = self.plateau_timer > 0
        self.plateau_timer[self.in_plateau] -= 1

        # ── 3. Feedforward drive ──────────────────────────────────────
        ff_drive = pre_f32 @ self.w  # (num_neurons,)

        # ── 4. Proactive k-WTA inhibition ─────────────────────────────
        self._apply_proactive_inhibition()

        # ── 5. Convergence-checked relaxation loop ────────────────────
        self.v *= self._mem_decay
        self.v += ff_drive  # Inject once

        rate = pc.initial_relaxation_rate
        for _ in range(pc.max_relaxation_steps):
            r = np.clip(
                (self.v - ncfg.v_rest) / ncfg.gap, 0.0, 1.0,
            )
            my_prediction = r @ self.w_apical.T
            self.prediction_error = pre_f32 - my_prediction
            error_gradient = self.prediction_error @ self.w_apical

            combined = error_gradient + ff_drive
            grad_norm = float(np.linalg.norm(combined))
            if grad_norm < pc.relaxation_threshold:
                break

            self.v += rate * combined
            np.clip(self.v, ncfg.v_reset, ncfg.v_thresh + 10.0, out=self.v)

        # ── 6. Effective threshold (apical priming + ACh) ─────────────
        effective_thresh = (
            self.v_thresh_adaptive
            - pyr.apical_boost
            * self.in_plateau.astype(np.float32)
            * self._ach_apical_scale
        )

        # Background cortical noise
        if pyr.background_noise_std > 0:
            noise = np.random.normal(
                0.0, pyr.background_noise_std, self.num_neurons,
            ).astype(np.float32)
            self.v += noise

        # ── 7. Spike detection ────────────────────────────────────────
        in_refrac = self.refrac_count > 0
        self.refrac_count[in_refrac] -= 1

        self.has_spiked = (self.v >= effective_thresh) & ~in_refrac
        self.v[self.has_spiked] = ncfg.v_reset
        self.refrac_count[self.has_spiked] = ncfg.refrac_period
        self.x_post[self.has_spiked] += 1.0

        # ── 8. Eligibility traces (Bi & Poo STDP) ────────────────────
        self.e *= self._elig_decay
        if np.any(self.has_spiked):
            self.e[:, self.has_spiked] += (
                self.stdp_cfg.a_plus * self.x_pre[:, np.newaxis]
            )
        pre_active = pre_f32 > 0.1
        if np.any(pre_active):
            self.e[pre_active, :] -= (
                self.stdp_cfg.a_minus * self.x_post[np.newaxis, :]
            )

        # ── 9. k-WTA window bookkeeping ──────────────────────────────
        self.window_spike_counts += self.has_spiked.astype(np.int32)
        self._current_window_size += 1
        if self._phase_reset_pending:
            self._apply_lateral_inhibition()
            self._reset_window()

        # ── 10. Burst detection (BAC firing, Larkum 1999) ─────────────
        self.is_burst = self.has_spiked & self.in_plateau
        if np.any(self.is_burst):
            burst_f = self.is_burst.astype(np.float32)
            boost = 1.0 + (pyr.burst_stdp_factor - 1.0) * burst_f
            self.e *= boost[np.newaxis, :]

        # ── 11. Homeostatic adaptation ────────────────────────────────
        self._update_adaptive_threshold()

        return self.has_spiked

    # ------------------------------------------------------------------
    # Override threshold for parent's spike detection in competitive
    # ------------------------------------------------------------------

    def _effective_threshold(self) -> NDArray[np.float32]:
        """Not used during forward (we do spike detection inline), but
        provided for API compat with CompetitiveLIFLayer."""
        return self.v_thresh_adaptive

    # ------------------------------------------------------------------
    # Prediction interface
    # ------------------------------------------------------------------

    def receive_prediction(self, prediction: NDArray[np.float32]) -> None:
        self.top_down_prediction = prediction.astype(np.float32)

    def generate_prediction(self) -> NDArray[np.float32]:
        """Top-down prediction via tied weights (w_apical.T)."""
        raw = self.has_spiked.astype(np.float32) @ self.w_apical.T
        return np.clip(raw * self.pc_cfg.feedback_strength, 0.0, 1.0)

    def set_ach_level(self, ach: float) -> None:
        """ACh → 1.0: bottom-up trust (halve apical boost); ACh → 0.0: full apical."""
        self._ach_apical_scale = float(1.0 - 0.5 * np.clip(ach, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def update_weights(
        self,
        m_t: float,
        pred_error: NDArray[np.float32],
    ) -> None:
        """Three-factor STDP (basal) + Hebbian apical learning."""
        super().update_weights(m_t, pred_error)

        if np.any(self.has_spiked):
            pos_err = np.clip(self.prediction_error, 0.0, None)
            if np.any(pos_err > 0):
                dw = self.pyr_cfg.apical_lr * np.outer(
                    pos_err,
                    self.has_spiked.astype(np.float32),
                )
                self.w_apical += dw * m_t

    # ------------------------------------------------------------------
    # Homeostatic threshold (local management)
    # ------------------------------------------------------------------

    def _update_adaptive_threshold(self) -> None:
        cfg = self._pyr_homeo
        self.avg_rate = (
            self.avg_rate * cfg.homeo_decay
            + self.has_spiked.astype(np.float32) * (1.0 - cfg.homeo_decay)
        )
        rate_error = self.avg_rate - cfg.target_rate
        self.v_thresh_adaptive += cfg.thresh_adapt_lr * rate_error
        np.clip(
            self.v_thresh_adaptive, cfg.thresh_min, cfg.thresh_max,
            out=self.v_thresh_adaptive,
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        super().reset_state()
        self.v_apical.fill(0.0)
        self.top_down_prediction.fill(0.0)
        self.prediction_error.fill(0.0)
        self.is_burst.fill(False)
        self.plateau_timer.fill(0)
        self.in_plateau.fill(False)
        self.v_thresh_adaptive.fill(self.neuron_cfg.v_thresh)
        self.avg_rate.fill(0.0)
        self._ach_apical_scale = 1.0
