"""
ErrorNeuronLayer — continuous Predictive Coding via State/Error populations.

Reference: Bogacz (2017), Bastos et al. (2012), Rao & Ballard (1999)

Changes from legacy:
  1. Uses ErrorNeuronConfig from config.py (derived decay factors).
  2. ACh gain via Hill equation dose-response (not linear interpolation).
  3. Correct anti-Hebbian learning for W_td (Rao & Ballard 1999):
     dW_td = -lr × outer(state_rate, error_rate).
  4. Precision broadcasting uses free_energy.py helper.
  5. Principled weight init via init_weights().
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import ErrorNeuronConfig, init_weights
from .free_energy import _broadcast_precision


class ErrorNeuronLayer:
    """Continuous predictive coding: State neurons (belief μ) + Error neurons (ε).

    State neurons (pyramidal L2/3): slow τ ~20ms, maintain belief.
    Error neurons (stellate L4): fast τ ~4ms, encode ε = input − g(μ).

    One forward() = one dt step. No inner relaxation loops.
    """

    # ── NetworkGraph layer interface ─────────────────────────────────

    @property
    def num_inputs(self) -> int:
        return self.n_input

    @property
    def num_neurons(self) -> int:
        return self.n_state

    def __init__(
        self,
        n_input: int,
        config: ErrorNeuronConfig | None = None,
    ) -> None:
        self.config = config or ErrorNeuronConfig()
        cfg = self.config
        self.n_input = n_input
        self.n_state = cfg.n_state
        self.n_error = cfg.n_error

        # ── State neuron membrane ─────────────────────────────────────
        v_rest = -70.0  # mV (cortical pyramidal)
        v_thresh = -55.0
        v_reset = -75.0
        self._v_rest = v_rest
        self._v_thresh = v_thresh
        self._v_reset = v_reset

        self.v_state: NDArray[np.float32] = np.full(
            self.n_state, v_rest, dtype=np.float32,
        )
        self.spikes_state: NDArray[np.bool_] = np.zeros(self.n_state, dtype=bool)
        self.refrac_state: NDArray[np.int32] = np.zeros(self.n_state, dtype=np.int32)

        # ── Error neuron membrane ─────────────────────────────────────
        self.v_error: NDArray[np.float32] = np.full(
            self.n_error, v_rest, dtype=np.float32,
        )
        self.spikes_error: NDArray[np.bool_] = np.zeros(self.n_error, dtype=bool)
        self.refrac_error: NDArray[np.int32] = np.zeros(self.n_error, dtype=np.int32)

        # ── Synaptic weights (principled init) ────────────────────────
        gap = abs(v_thresh - v_rest)

        # W_bu: error → state (bottom-up error correction)
        self.w_bu: NDArray[np.float32] = init_weights(
            self.n_error, self.n_state, psp_target=gap * 0.2,
        )

        # W_td: state → error (top-down generative model)
        self.w_td: NDArray[np.float32] = init_weights(
            self.n_state, self.n_error, psp_target=gap * 0.2,
        )

        # W_in: input → error (feedforward drive)
        self.w_in: NDArray[np.float32] = init_weights(
            n_input, self.n_error, psp_target=gap * 0.15,
        )

        # ── Eligibility traces ────────────────────────────────────────
        self.e_bu: NDArray[np.float32] = np.zeros_like(self.w_bu)
        self.e_td: NDArray[np.float32] = np.zeros_like(self.w_td)

        # ── Spike timing for causal STDP window (±20ms) ──────────────
        self.t_since_error_spike: NDArray[np.int32] = np.full(
            self.n_error, 1000, dtype=np.int32,
        )
        self.t_since_state_spike: NDArray[np.int32] = np.full(
            self.n_state, 1000, dtype=np.int32,
        )
        self._stdp_window: int = 20  # ±20 timesteps

        # ── Rate coding (EMA of spikes) ───────────────────────────────
        self.state_rate: NDArray[np.float32] = np.zeros(
            self.n_state, dtype=np.float32,
        )
        self.error_rate: NDArray[np.float32] = np.zeros(
            self.n_error, dtype=np.float32,
        )
        self._rate_decay: float = cfg.ctx.decay(20.0)  # ~20ms EMA

        # ── ACh gain ──────────────────────────────────────────────────
        self._ach_gain: float = 1.0
        # ── Receptor dose-response modulation (D2) ───────────────────
        self._receptor_gain: float = 1.0
        self._receptor_lr: float = 1.0
    def forward(
        self,
        input_spikes: NDArray[np.float32],
    ) -> NDArray[np.bool_]:
        """One dt step: error neurons compute ε, state neurons update μ.

        Returns:
            (n_state,) boolean spike array.
        """
        inp = input_spikes.astype(np.float32)
        cfg = self.config

        # ── Prediction from state rates (generative model) ───────────
        prediction = self.state_rate @ self.w_td  # (n_error,)

        # ── Error neuron drive: ACh-gated feedforward − prediction ────
        feedforward = inp @ self.w_in  # (n_error,)
        error_input = (self._ach_gain * feedforward - prediction) * self._receptor_gain

        # ── Error neuron LIF (fast τ) ─────────────────────────────────
        in_refrac_e = self.refrac_error > 0
        self.refrac_error[in_refrac_e] -= 1

        gain_e = 1.0 - cfg.error_decay
        leaked_e = self.v_error * cfg.error_decay + self._v_rest * gain_e
        integrated_e = leaked_e + error_input
        self.v_error = np.where(in_refrac_e, self._v_reset, integrated_e)

        self.spikes_error = (self.v_error >= self._v_thresh) & ~in_refrac_e
        self.v_error[self.spikes_error] = self._v_reset
        self.refrac_error[self.spikes_error] = cfg.refrac_period

        # ── State neuron drive: bottom-up error correction ────────────
        error_signal = self.spikes_error.astype(np.float32)
        state_input = error_signal @ self.w_bu  # (n_state,)

        # ── State neuron LIF (slow τ) ─────────────────────────────────
        in_refrac_s = self.refrac_state > 0
        self.refrac_state[in_refrac_s] -= 1

        gain_s = 1.0 - cfg.state_decay
        leaked_s = self.v_state * cfg.state_decay + self._v_rest * gain_s
        integrated_s = leaked_s + state_input
        self.v_state = np.where(in_refrac_s, self._v_reset, integrated_s)

        self.spikes_state = (self.v_state >= self._v_thresh) & ~in_refrac_s
        self.v_state[self.spikes_state] = self._v_reset
        self.refrac_state[self.spikes_state] = cfg.refrac_period

        # ── Rate EMA ──────────────────────────────────────────────────
        rd = self._rate_decay
        self.state_rate = (
            self.state_rate * rd
            + self.spikes_state.astype(np.float32) * (1.0 - rd)
        )
        self.error_rate = (
            self.error_rate * rd
            + self.spikes_error.astype(np.float32) * (1.0 - rd)
        )

        # ── Spike timing updates ─────────────────────────────────────
        self.t_since_error_spike += 1
        self.t_since_error_spike[self.spikes_error] = 0
        self.t_since_state_spike += 1
        self.t_since_state_spike[self.spikes_state] = 0

        # ── Eligibility traces (causal ±20ms window) ─────────────────
        error_f = self.spikes_error.astype(np.float32)
        state_f = self.spikes_state.astype(np.float32)

        # W_bu (error → state): Hebbian with causal window
        self.e_bu *= cfg.state_decay
        if np.any(self.spikes_state):
            # State neuron spiked: accumulate from error neurons that
            # spiked within the causal window
            bu_mask = (self.t_since_error_spike <= self._stdp_window).astype(np.float32)
            self.e_bu[:, self.spikes_state] += (error_f * bu_mask)[:, np.newaxis]

        # W_td (state → error): with causal window
        self.e_td *= cfg.error_decay
        if np.any(self.spikes_error):
            td_mask = (self.t_since_state_spike <= self._stdp_window).astype(np.float32)
            self.e_td[:, self.spikes_error] += (state_f * td_mask)[:, np.newaxis]

        return self.spikes_state

    def update_weights(
        self,
        modulation: float,
        precision: NDArray[np.float32] | None = None,
    ) -> None:
        """Three-factor learning with correct anti-Hebbian for W_td.

        W_bu (error→state): Hebbian — state learns from errors.
        W_td (state→error): Anti-Hebbian — generative model learns to
        predict, reducing error. dW_td = -lr × e_td × modulation.
        """
        if np.isclose(modulation, 0.0):
            return

        cfg = self.config

        # ── Bottom-up: Hebbian ────────────────────────────────────────
        effective_mod = modulation * self._receptor_lr
        dw_bu = cfg.w_bu_lr * effective_mod * self.e_bu
        if precision is not None:
            prec = _broadcast_precision(precision, self.n_error)
            dw_bu *= prec[:, np.newaxis]
        self.w_bu += dw_bu

        # ── Top-down: Anti-Hebbian (Rao & Ballard 1999) ──────────────
        # Negative sign: reduce weights that generate over-prediction
        dw_td = -cfg.w_td_lr * effective_mod * self.e_td
        self.w_td += dw_td

    def set_ach_level(self, ach: float) -> None:
        """ACh via Hill equation dose-response (ErrorNeuronConfig params)."""
        ach = float(np.clip(ach, 0.0, 1.0))
        cfg = self.config
        # Hill equation: gain = min + (max - min) × ACh^n / (ACh^n + EC50^n)
        ach_n = ach ** cfg.ach_hill_n
        ec50_n = cfg.ach_ec50 ** cfg.ach_hill_n
        frac = ach_n / (ach_n + ec50_n + 1e-12)
        self._ach_gain = cfg.ach_gain_min + (cfg.ach_gain_max - cfg.ach_gain_min) * frac

    @property
    def prediction_error_rate(self) -> NDArray[np.float32]:
        """Error neuron firing rates (= biological prediction error)."""
        return self.error_rate.copy()

    @property
    def belief(self) -> NDArray[np.float32]:
        """State neuron firing rates (= current belief μ)."""
        return self.state_rate.copy()

    def reset_state(self) -> None:
        """Reset transient state. Weights preserved."""
        self.v_state.fill(self._v_rest)
        self.v_error.fill(self._v_rest)
        self.spikes_state.fill(False)
        self.spikes_error.fill(False)
        self.refrac_state.fill(0)
        self.refrac_error.fill(0)
        self.state_rate.fill(0.0)
        self.error_rate.fill(0.0)
        self.e_bu.fill(0.0)
        self.e_td.fill(0.0)
        self._ach_gain = 1.0
        self._receptor_gain = 1.0
        self._receptor_lr = 1.0

    def set_receptor_modulation(self, gain_mod: float, lr_mod: float) -> None:
        """Apply receptor dose-response modulation (Hill equation effects)."""
        self._receptor_gain = float(gain_mod)
        self._receptor_lr = float(lr_mod)
