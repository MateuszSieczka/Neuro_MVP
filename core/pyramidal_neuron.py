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
    init_weights,
)
from .competitive_layer import CompetitiveLIFLayer
from .neuron import HomeostaticState


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
    ) -> None:
        self.pyr_cfg = pyr_cfg or PyramidalConfig()
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

        # ── Apical compartment weights (nS conductance) ──────────────
        self.w_apical: NDArray[np.float32] = init_weights(
            num_inputs, num_neurons,
            psp_target=0.5,
            excitatory=True,
            g_L=ncfg.g_L,
            driving_force=ncfg.driving_force_exc,
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
        self._pyr_homeo_state = HomeostaticState(
            num_neurons, ncfg.v_thresh, hcfg,
        )
        self.v_thresh_adaptive = self._pyr_homeo_state.v_thresh_adaptive
        self.avg_rate = self._pyr_homeo_state.avg_rate

        # ── Spike timing for causal STDP window (±20ms) ──────────────
        self.t_since_pre_spike: NDArray[np.int32] = np.full(
            num_inputs, 1000, dtype=np.int32,
        )
        self.t_since_post_spike: NDArray[np.int32] = np.full(
            num_neurons, 1000, dtype=np.int32,
        )
        self._stdp_window: int = 20

        # ── ACh modulation ────────────────────────────────────────────
        self._ach_apical_scale: float = 1.0

        # ── Apical delay buffer (Stuart & Spruston 1998) ─────────────
        # Ring buffer: stores last `apical_delay_ms` timesteps of
        # top-down prediction.  At each step the oldest entry is
        # consumed as apical input, implementing propagation delay.
        # Buffer size = delay + 1: write-then-read gives N-1 step
        # delay, so +1 yields exact target delay.
        delay_steps = max(1, int(self.pyr_cfg.apical_delay_ms / ncfg.ctx.dt))
        buf_size = delay_steps + 1
        self._apical_delay_len: int = buf_size
        self._apical_delay_steps: int = delay_steps
        self._apical_delay_buf: NDArray[np.float32] = np.zeros(
            (buf_size, num_neurons), dtype=np.float32,
        )
        self._apical_delay_idx: int = 0

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def forward(self, pre_spikes: NDArray[np.float32]) -> NDArray[np.bool_]:
        pre_f32 = pre_spikes.astype(np.float32)
        ncfg = self.neuron_cfg
        pyr = self.pyr_cfg

        # ── Event-based STDP trace update (Bi & Poo 2001) ────────────
        self.x_pre *= self._pre_decay
        self.x_post *= self._post_decay
        pre_binary = (pre_f32 > 0.5).astype(np.float32)
        self.x_pre += pre_binary

        # Update spike timing counters
        self.t_since_pre_spike += 1
        self.t_since_pre_spike[pre_binary > 0.5] = 0
        self.t_since_post_spike += 1

        # ── 1. Apical integration with delay (Stuart & Spruston 1998) ─
        # Write current top-down signal into ring buffer
        self._apical_delay_buf[self._apical_delay_idx] = (
            self.top_down_prediction.astype(np.float32)
        )
        # Read oldest entry (delayed by apical_delay_ms)
        read_idx = (self._apical_delay_idx + 1) % self._apical_delay_len
        apical_current = self._apical_delay_buf[read_idx]
        self._apical_delay_idx = read_idx

        # Passive low-pass cable filter (τ_apical ≈ 150ms)
        apical_gain = ncfg.ctx.complement(pyr.tau_apical)
        self.v_apical = (
            self.v_apical * pyr.apical_decay
            + apical_current * apical_gain
        )

        # ── 2. Ca²⁺ plateau trigger (voltage-dependent, Larkum 2013) ─
        ca_activation = 1.0 / (
            1.0 + np.exp(-(self.v_apical - pyr.ca_v_half) / pyr.ca_k)
        )
        new_plateaus = (ca_activation > 0.5) & (self.plateau_timer == 0)
        self.plateau_timer[new_plateaus] = pyr.plateau_duration_ms
        self.in_plateau = self.plateau_timer > 0
        self.plateau_timer[self.in_plateau] -= 1

        # ── 3. Feedforward drive: conductance-based ─────────────────
        g_ff = pre_f32 @ self.w          # total excitatory conductance (nS)
        ff_drive = g_ff * (ncfg.e_exc - self.v)  # pA = nS × mV

        # ── 4. Proactive k-WTA inhibition ─────────────────────────────
        self._apply_proactive_inhibition()

        # ── 5. Prediction error gradient ──────────────────────────────
        r = np.clip((self.v - ncfg.v_rest) / ncfg.gap, 0.0, 1.0)
        my_prediction = r @ self.w_apical.T
        self.prediction_error = pre_f32 - my_prediction
        error_gradient = self.prediction_error @ self.w_apical

        ach = self._ach_apical_scale
        combined = ach * error_gradient + (1.0 - ach) * self.top_down_prediction

        # ── 6. Total synaptic input ───────────────────────────────────
        I_syn = ff_drive + combined

        # ── 7. AdEx membrane integration via Exponential Euler ────────
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

        ctx = ncfg.ctx
        integrated_v = ctx.exp_euler_step(self.v, F_v, J_v)
        np.clip(integrated_v, None, 50.0, out=integrated_v)  # cap phi1 runaway

        in_refrac = self.refrac_count > 0
        self.refrac_count[in_refrac] -= 1
        self.v = np.where(in_refrac, ncfg.v_reset, integrated_v)

        # ── 8. Effective threshold (apical priming + ACh) ─────────────
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

        # ── 9. Spike detection (AdEx cutoff combined with adaptive) ───
        spike_thresh = np.minimum(np.float32(ncfg.v_spike_cutoff), effective_thresh)
        self.has_spiked = (self.v >= spike_thresh) & ~in_refrac
        self.v[self.has_spiked] = ncfg.v_reset
        self.w_adapt[self.has_spiked] += ncfg.b  # spike-triggered adaptation
        self.refrac_count[self.has_spiked] = ncfg.refrac_period
        self.x_post[self.has_spiked] += 1.0
        self.t_since_post_spike[self.has_spiked] = 0

        # ── 10. Subthreshold adaptation ───────────────────────────────
        self.w_adapt = (
            self.w_adapt * ncfg.w_decay
            + ncfg.a * (self.v - ncfg.v_rest) * ncfg.w_gain
        )

        # ── 11. Eligibility traces — causal ±20ms window (Bi & Poo) ──
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

        # ── 12. k-WTA window bookkeeping ─────────────────────────────
        self.window_spike_counts += self.has_spiked.astype(np.int32)
        self._current_window_size += 1
        if self._phase_reset_pending:
            self._apply_lateral_inhibition()
            self._reset_window()

        # ── 13. Burst detection (BAC firing, Larkum 1999) ─────────────
        #  Synapse-specific burst boost (Payeur et al. 2021):
        #  boost ∝ presynaptic activity × burst factor
        self.is_burst = self.has_spiked & self.in_plateau
        if np.any(self.is_burst):
            burst_f = self.is_burst.astype(np.float32)
            # Synapse-specific: active synapses boosted, silent unchanged
            self.e *= (
                1.0
                + (pyr.burst_stdp_factor - 1.0)
                * burst_f[np.newaxis, :]
                * self.x_pre[:, np.newaxis]
            )

        # ── 14. Homeostatic adaptation ────────────────────────────────
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
        return np.clip(raw * self.pyr_cfg.feedback_strength, 0.0, 1.0)

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
        self._pyr_homeo_state.update(self.has_spiked)

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
        self._apical_delay_buf.fill(0.0)
        self._apical_delay_idx = 0
        self._pyr_homeo_state.reset(self.neuron_cfg.v_thresh)
        self._ach_apical_scale = 1.0
