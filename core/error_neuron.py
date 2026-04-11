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

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .config import ErrorNeuronConfig, NeuronConfig, init_weights
from .free_energy import _broadcast_precision

if TYPE_CHECKING:
    from .astrocyte import AstrocyteField


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
        neuron_cfg: NeuronConfig | None = None,
    ) -> None:
        self.config = config or ErrorNeuronConfig()
        cfg = self.config
        # AdEx parameters — use provided NeuronConfig or defaults
        self._ncfg = neuron_cfg or NeuronConfig(ctx=cfg.ctx)
        self.n_input = n_input
        self.n_state = cfg.n_state
        self.n_error = cfg.n_error

        # ── State neuron membrane ─────────────────────────────────────
        v_rest = self._ncfg.v_rest
        v_thresh = self._ncfg.v_thresh
        v_reset = self._ncfg.v_reset
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

        # ── AdEx adaptation currents ──────────────────────────────────
        self.w_adapt_state: NDArray[np.float32] = np.zeros(
            self.n_state, dtype=np.float32,
        )
        self.w_adapt_error: NDArray[np.float32] = np.zeros(
            self.n_error, dtype=np.float32,
        )

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

        # ── Astrocyte ATP modulation (Krok 1.3, optional) ────────────
        self._astrocyte: AstrocyteField | None = None
        self._zone_idx_state: NDArray[np.int32] | None = None
        self._zone_idx_error: NDArray[np.int32] | None = None

    def set_astrocyte(
        self,
        astrocyte: AstrocyteField,
        zone_idx_state: NDArray[np.int32] | None = None,
        zone_idx_error: NDArray[np.int32] | None = None,
    ) -> None:
        """Attach AstrocyteField for ATP V_T / g_L modulation."""
        self._astrocyte = astrocyte
        self._zone_idx_state = (
            zone_idx_state.astype(np.int32) if zone_idx_state is not None
            else np.linspace(0, astrocyte.n_zones - 1, self.n_state).astype(np.int32)
        )
        self._zone_idx_error = (
            zone_idx_error.astype(np.int32) if zone_idx_error is not None
            else np.linspace(0, astrocyte.n_zones - 1, self.n_error).astype(np.int32)
        )

    def _adex_step(
        self,
        v: NDArray[np.float32],
        w_adapt: NDArray[np.float32],
        I_syn: NDArray[np.float32],
        in_refrac: NDArray[np.bool_],
        eff_v_thresh: float | NDArray[np.float32] | None = None,
        eff_g_L: float | NDArray[np.float32] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
        """Single AdEx integration step for either population.

        Args:
            eff_v_thresh: ATP-modulated V_T (or None → use ncfg.v_thresh).
            eff_g_L: ATP-modulated g_L (or None → use ncfg.g_L).

        Returns: (v_new, w_adapt_new, spiked)
        """
        ncfg = self._ncfg
        ctx = ncfg.ctx
        vt = eff_v_thresh if eff_v_thresh is not None else ncfg.v_thresh
        gL = eff_g_L if eff_g_L is not None else ncfg.g_L

        exp_term = np.exp(
            np.clip((v - vt) / ncfg.delta_t, -20.0, 10.0),
        )
        inv_Cm = 1.0 / ncfg.C_m
        F_v = inv_Cm * (
            -gL * (v - ncfg.v_rest)
            + gL * ncfg.delta_t * exp_term
            + I_syn - w_adapt
        )
        J_v = inv_Cm * (-gL + gL * exp_term)
        integrated = ctx.exp_euler_step(v, F_v, J_v)
        v_new = np.where(in_refrac, ncfg.v_reset, integrated)

        spiked = (v_new >= ncfg.v_spike_cutoff) & ~in_refrac
        v_new[spiked] = ncfg.v_reset
        w_adapt[spiked] += ncfg.b

        # Subthreshold adaptation
        w_adapt_new = (
            w_adapt * ncfg.w_decay
            + ncfg.a * (v_new - ncfg.v_rest) * ncfg.w_gain
        )
        return v_new, w_adapt_new, spiked

    def forward(
        self,
        input_spikes: NDArray[np.float32],
    ) -> NDArray[np.bool_]:
        """One dt step: error neurons compute ε, state neurons update μ.

        Both populations use AdEx dynamics.

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

        # ── Error neuron AdEx (fast τ) ────────────────────────────────
        in_refrac_e = self.refrac_error > 0
        self.refrac_error[in_refrac_e] -= 1

        # ATP modulation (Krok 1.3)
        if self._astrocyte is not None:
            ze = self._zone_idx_error
            zs = self._zone_idx_state
            vt_error = self._ncfg.v_thresh + self._astrocyte.threshold_shift[ze]
            gL_error = self._ncfg.g_L * self._astrocyte.leak_gain[ze]
            vt_state = self._ncfg.v_thresh + self._astrocyte.threshold_shift[zs]
            gL_state = self._ncfg.g_L * self._astrocyte.leak_gain[zs]
        else:
            vt_error = vt_state = None
            gL_error = gL_state = None

        self.v_error, self.w_adapt_error, self.spikes_error = self._adex_step(
            self.v_error, self.w_adapt_error, error_input, in_refrac_e,
            eff_v_thresh=vt_error, eff_g_L=gL_error,
        )
        self.refrac_error[self.spikes_error] = cfg.refrac_period

        # ── State neuron drive: bottom-up error correction ────────────
        error_signal = self.spikes_error.astype(np.float32)
        state_input = error_signal @ self.w_bu  # (n_state,)

        # ── State neuron AdEx (slow τ) ────────────────────────────────
        in_refrac_s = self.refrac_state > 0
        self.refrac_state[in_refrac_s] -= 1

        self.v_state, self.w_adapt_state, self.spikes_state = self._adex_step(
            self.v_state, self.w_adapt_state, state_input, in_refrac_s,
            eff_v_thresh=vt_state, eff_g_L=gL_state,
        )
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
        self.w_adapt_state.fill(0.0)
        self.w_adapt_error.fill(0.0)
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
