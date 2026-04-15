"""
AdExLayer — Adaptive Exponential Integrate-and-Fire neuron population.

Reference: Brette & Gerstner (2005) "Adaptive Exponential Integrate-and-Fire
Model as an Effective Description of Neuronal Activity"

Changes from legacy LIFLayer:
  1. AdEx membrane dynamics: exponential spike initiation + adaptation current w.
  2. Exponential Euler integration (A-stable) instead of linear exact decay.
  3. Spike detection at v_spike_cutoff (above V_T) instead of v_thresh.
  4. w_adapt state: spike-triggered + subthreshold adaptation.
  5. Different neuron types from SAME equations (RS, FS, IB, LS via params).
  6. Multiplicative STDP kernel (Bi & Poo 2001) with proper causal asymmetry.
  7. Calcium-based eligibility traces (Graupner & Brunel 2012).
  8. Synaptic scaling (Turrigiano 2008) replaces hard weight clips.
  9. SynapticChannels integration for AMPA/NMDA temporal dynamics.
  10. NE/ACh timescale modulation with explicit compression factors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .config import (
    NeuronConfig,
    STDPConfig,
    HomeostaticConfig,
    SynapseConfig,
    init_weights,
)
from .synapse import SynapticChannels

if TYPE_CHECKING:
    from .astrocyte import AstrocyteField


class HomeostaticState:
    """Shared homeostatic threshold adaptation state.

    Used by LIFLayer, CompetitiveLIFLayer, and PyramidalLayer to manage
    v_thresh_adaptive, avg_rate, and dark matter neurons with identical logic.
    Eliminates ~60 lines of duplicated homeostatic code.
    """

    def __init__(
        self,
        num_neurons: int,
        v_thresh: float,
        config: HomeostaticConfig,
    ) -> None:
        self.config = config
        self.v_thresh_adaptive: NDArray[np.float32] = np.full(
            num_neurons, v_thresh, dtype=np.float32,
        )
        self.avg_rate: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )
        self.is_dark_matter: NDArray[np.bool_] = np.zeros(
            num_neurons, dtype=bool,
        )
        n_dark = int(num_neurons * config.dark_matter_ratio)
        if n_dark > 0:
            dark_idx = np.random.choice(num_neurons, n_dark, replace=False)
            self.is_dark_matter[dark_idx] = True
            self.v_thresh_adaptive[dark_idx] += config.dark_matter_thresh_offset

    def update(self, spikes: NDArray[np.bool_], window_steps: int = 1) -> None:
        """BCM-derived threshold adaptation toward target_rate.

        Args:
            spikes: per-neuron spike counts or binary spikes for the window.
            window_steps: number of steps in evaluation window (1 for per-step).
        """
        cfg = self.config
        spikes_f = spikes.astype(np.float32)
        spikes_per_step = spikes_f / max(window_steps, 1)

        if window_steps > 1:
            decay = cfg.ctx.decay(cfg.homeostatic_tau * window_steps)
        else:
            decay = cfg.homeo_decay

        self.avg_rate = self.avg_rate * decay + spikes_per_step * (1.0 - decay)
        rate_error = self.avg_rate - cfg.target_rate
        self.v_thresh_adaptive += cfg.thresh_adapt_lr * window_steps * rate_error
        np.clip(
            self.v_thresh_adaptive, cfg.thresh_min, cfg.thresh_max,
            out=self.v_thresh_adaptive,
        )

    def effective_threshold(self, ne_level: float) -> NDArray[np.float32]:
        """Return threshold with NE-driven drop."""
        ne_drop = ne_level * self.config.ne_thresh_drop
        return self.v_thresh_adaptive - ne_drop

    def reset(self, v_thresh: float) -> None:
        """Reset to initial state (preserves dark matter offsets)."""
        self.v_thresh_adaptive.fill(v_thresh)
        self.v_thresh_adaptive[self.is_dark_matter] += (
            self.config.dark_matter_thresh_offset
        )
        self.avg_rate.fill(0.0)


class AdExLayer:
    """Vectorised Adaptive Exponential Integrate-and-Fire population.

    The layer composes independent config dataclasses:
      - ``NeuronConfig``      — AdEx membrane dynamics (tau_m, delta_t, tau_w, a, b, ...)
      - ``STDPConfig``        — STDP kernel parameters (Bi & Poo 2001)
      - ``HomeostaticConfig`` — optional BCM-derived threshold adaptation
      - ``SynapseConfig``     — optional AMPA/NMDA channel dynamics

    Membrane equation (Brette & Gerstner 2005):
      C_m dV/dt = -g_L(V - E_L) + g_L Δ_T exp((V - V_T)/Δ_T) + I_syn - w
      τ_w dw/dt = a(V - E_L) - w

    Integration via Exponential Euler (A-stable, handles AdEx + NMDA stiffness).

    Weight update is three-factor (Izhikevich 2007):
      Δw = lr × modulator × eligibility × error_signal
    """

    def __init__(
        self,
        num_inputs: int,
        num_neurons: int = 1,
        neuron_cfg: NeuronConfig | None = None,
        stdp_cfg: STDPConfig | None = None,
        homeo_cfg: HomeostaticConfig | None = None,
        synapse_cfg: SynapseConfig | None = None,
        excitatory: bool = True,
    ) -> None:
        self.neuron_cfg = neuron_cfg or NeuronConfig()
        self.stdp_cfg = stdp_cfg or STDPConfig()
        self.homeo_cfg = homeo_cfg  # None = no homeostatic adaptation
        self.synapse_cfg = synapse_cfg  # None = instantaneous PSP model

        self.num_inputs = num_inputs
        self.num_neurons = num_neurons

        # Backward-compat alias used by downstream layers
        self.config = self.neuron_cfg

        # ── Membrane state ────────────────────────────────────────────
        self.v: NDArray[np.float32] = np.full(
            num_neurons, self.neuron_cfg.v_rest, dtype=np.float32,
        )
        self.has_spiked: NDArray[np.bool_] = np.zeros(num_neurons, dtype=bool)
        self.refrac_count: NDArray[np.int32] = np.zeros(num_neurons, dtype=np.int32)

        # ── AdEx adaptation current (Brette & Gerstner 2005) ─────────
        self.w_adapt: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )

        # ── Synaptic weights (nS conductance) ─────────────────────────
        self.w: NDArray[np.float32] = init_weights(
            num_inputs, num_neurons,
            psp_target=self.neuron_cfg.psp_target,
            excitatory=excitatory,
            g_L=self.neuron_cfg.g_L,
            driving_force=self.neuron_cfg.driving_force_exc,
        )

        # ── STDP traces (Bi & Poo 2001) ──────────────────────────────
        # Pre-synaptic trace: incremented on pre spike, decays with τ_plus
        self.x_pre: NDArray[np.float32] = np.zeros(num_inputs, dtype=np.float32)
        # Post-synaptic trace: incremented on post spike, decays with τ_minus
        self.x_post: NDArray[np.float32] = np.zeros(num_neurons, dtype=np.float32)

        # ── Spike timing for causal STDP window (±20ms, Bi & Poo 2001) ─
        # Time since last spike (in timesteps). Large init = no recent spike.
        self.t_since_pre_spike: NDArray[np.int32] = np.full(
            num_inputs, 1000, dtype=np.int32,
        )
        self.t_since_post_spike: NDArray[np.int32] = np.full(
            num_neurons, 1000, dtype=np.int32,
        )
        self._stdp_window: int = 20  # ±20 timesteps (±20ms at dt=1ms)

        # ── Eligibility trace (three-factor, Graupner & Brunel 2012) ─
        self.e: NDArray[np.float32] = np.zeros(
            (num_inputs, num_neurons), dtype=np.float32,
        )

        # ── Synaptic channels (optional) ──────────────────────────────
        self.channels: SynapticChannels | None = None
        if self.synapse_cfg is not None:
            self.channels = SynapticChannels(
                n_post=num_neurons, config=self.synapse_cfg,
            )

        # ── Homeostatic state ─────────────────────────────────────────
        self._ne_level: float = 0.0
        self._homeo_state: HomeostaticState | None = None
        if self.homeo_cfg is not None:
            self._homeo_state = HomeostaticState(
                num_neurons, self.neuron_cfg.v_thresh, self.homeo_cfg,
            )
            # Backward-compat aliases
            self.v_thresh_adaptive = self._homeo_state.v_thresh_adaptive
            self.avg_rate = self._homeo_state.avg_rate
            self._is_dark_matter = self._homeo_state.is_dark_matter

        # ── Effective decay factors (modulated by NE / ACh) ──────────
        self._mem_decay: float = self.neuron_cfg.mem_decay
        self._mem_gain: float = self.neuron_cfg.mem_gain
        self._pre_decay: float = self.stdp_cfg.pre_decay
        self._post_decay: float = self.stdp_cfg.post_decay
        self._elig_decay: float = self.stdp_cfg.elig_decay

        # ── Synaptic scaling bookkeeping (Turrigiano 2008) ────────────
        self._scaling_counter: int = 0
        self._scaling_interval: int = self.neuron_cfg.scaling_interval

        # ── Astrocyte ATP modulation (Krok 1.3, optional) ────────────
        self._astrocyte: AstrocyteField | None = None
        self._zone_idx: NDArray[np.int32] | None = None

    # ------------------------------------------------------------------
    # Astrocyte coupling
    # ------------------------------------------------------------------

    def set_astrocyte(
        self,
        astrocyte: AstrocyteField,
        zone_idx: NDArray[np.int32] | None = None,
    ) -> None:
        """Attach an AstrocyteField for ATP-based V_T / g_L modulation.

        Args:
            astrocyte: AstrocyteField instance providing per-zone ATP state.
            zone_idx: (num_neurons,) mapping neuron i → zone. If None,
                      neurons are distributed evenly across zones.
        """
        self._astrocyte = astrocyte
        if zone_idx is not None:
            self._zone_idx = zone_idx.astype(np.int32)
        else:
            self._zone_idx = np.linspace(
                0, astrocyte.n_zones - 1, self.num_neurons,
            ).astype(np.int32)

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def forward(self, pre_spikes: NDArray[np.float32]) -> NDArray[np.bool_]:
        """One AdEx integration step: current → exp euler → spike → adapt → traces.

        Membrane (Brette & Gerstner 2005):
          C_m dV/dt = -g_L(V-E_L) + g_L Δ_T exp((V-V_T)/Δ_T) + I_syn - w
        Integrated via Exponential Euler (A-stable).

        Args:
            pre_spikes: (num_inputs,) presynaptic spike vector (0/1 or rate).

        Returns:
            (num_neurons,) boolean spike array.
        """
        pre_f32 = pre_spikes.astype(np.float32)
        ncfg = self.neuron_cfg
        ctx = ncfg.ctx

        # 1. STDP trace decay
        self.x_pre *= self._pre_decay
        self.x_post *= self._post_decay

        # Event-based pre trace: increment only on discrete spikes (Bi & Poo 2001)
        pre_binary = (pre_f32 > 0.5).astype(np.float32)
        self.x_pre += pre_binary

        # Update spike timing counters
        self.t_since_pre_spike += 1
        self.t_since_pre_spike[pre_binary > 0.5] = 0
        self.t_since_post_spike += 1

        # 2. Refractory management
        in_refrac = self.refrac_count > 0
        self.refrac_count[in_refrac] -= 1

        # 3. Compute synaptic drive
        if self.channels is not None:
            # Conductance-based (AMPA + NMDA temporal dynamics)
            self.channels.receive_excitatory(pre_f32, self.w)
            self.channels.decay()
            current = self.channels.compute_current(self.v)
        else:
            # Instantaneous conductance-based model: I = g × (E_exc − V)
            g_exc = pre_f32 @ self.w  # total excitatory conductance (nS)
            current = g_exc * (ncfg.e_exc - self.v)  # pA

        # 4. AdEx membrane integration via Exponential Euler
        # F(V) = (1/C_m) * [-g_L*(V-E_L) + g_L*Δ_T*exp((V-V_T)/Δ_T) + I - w]
        # J(V) = ∂F/∂V = (1/C_m) * [-g_L + g_L*exp((V-V_T)/Δ_T)]
        # ATP modulation (Krok 1.3): effective V_T and g_L per neuron
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
            + current - self.w_adapt
        )
        J_v = inv_Cm * (-eff_g_L + eff_g_L * exp_term)

        integrated_v = ctx.exp_euler_step(self.v, F_v, J_v)
        np.clip(integrated_v, None, 50.0, out=integrated_v)  # cap phi1 runaway
        self.v = np.where(in_refrac, ncfg.v_reset, integrated_v)

        # 5. Spike detection (v >= v_spike_cutoff, combined with adaptive thresh)
        thresh = self._effective_threshold()
        spike_thresh = np.minimum(
            np.float32(ncfg.v_spike_cutoff), thresh,
        ) if isinstance(thresh, np.ndarray) else np.float32(ncfg.v_spike_cutoff)
        self.has_spiked = (self.v >= spike_thresh) & ~in_refrac

        # 6. Reset spiked neurons + spike-triggered adaptation
        self.v[self.has_spiked] = ncfg.v_reset
        self.w_adapt[self.has_spiked] += ncfg.b
        self.refrac_count[self.has_spiked] = ncfg.refrac_period
        # Event-based post trace: increment by 1.0 on spike event
        self.x_post[self.has_spiked] += 1.0
        self.t_since_post_spike[self.has_spiked] = 0

        # 7. Subthreshold adaptation: w = w*decay + a*(V - E_L)*gain
        self.w_adapt = (
            self.w_adapt * ncfg.w_decay
            + ncfg.a * (self.v - ncfg.v_rest) * ncfg.w_gain
        )

        # 8. Eligibility trace — causal STDP window (±20ms, Bi & Poo 2001)
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

        # 9. Homeostatic threshold adaptation (if configured)
        if self.homeo_cfg is not None:
            self._update_homeostatic()

        # 10. Periodic synaptic scaling (Turrigiano 2008)
        self._scaling_counter += 1
        if self._scaling_counter >= self._scaling_interval:
            self._synaptic_scaling()
            self._scaling_counter = 0

        return self.has_spiked

    # ------------------------------------------------------------------
    # Threshold helpers
    # ------------------------------------------------------------------

    def _effective_threshold(self) -> NDArray[np.float32] | float:
        """Return adaptive threshold (with NE drop) or static threshold."""
        if self._homeo_state is not None:
            return self._homeo_state.effective_threshold(self._ne_level)
        if hasattr(self, 'v_thresh_adaptive'):
            # Subclass may have created this directly (e.g. CompetitiveLIFLayer)
            ne_drop = self._ne_level * getattr(
                self.homeo_cfg, 'ne_thresh_drop', 0.0,
            ) if self.homeo_cfg else 0.0
            return self.v_thresh_adaptive - ne_drop
        return np.float32(self.neuron_cfg.v_thresh)

    # ------------------------------------------------------------------
    # Homeostatic plasticity (BCM-derived)
    # ------------------------------------------------------------------

    def _update_homeostatic(self) -> None:
        """BCM-derived threshold adaptation toward target_rate."""
        assert self._homeo_state is not None
        self._homeo_state.update(self.has_spiked)

    # ------------------------------------------------------------------
    # Synaptic scaling (Turrigiano 2008)
    # ------------------------------------------------------------------

    def _synaptic_scaling(self) -> None:
        """Multiplicative synaptic scaling to maintain column-wise weight norms.

        Every ``_scaling_interval`` steps, rescale:
            w_col *= target_norm / actual_norm
        where target_norm = initial column-wise L2 norm (approximated
        from init std × sqrt(fan_in)), in nS conductance units.
        """
        col_norms = np.linalg.norm(self.w, axis=0)
        ncfg = self.neuron_cfg
        target = np.sqrt(float(self.num_inputs)) * (
            ncfg.psp_target * ncfg.g_L
            / (ncfg.driving_force_exc * np.sqrt(max(1.0, self.num_inputs * 0.05)))
        )
        scale = np.where(col_norms > 1e-8, target / col_norms, 1.0)
        # Soft scaling — move 10% toward target per event
        scale = 1.0 + 0.1 * (scale - 1.0)
        self.w *= scale.astype(np.float32)

    # ------------------------------------------------------------------
    # Neuromodulatory interfaces
    # ------------------------------------------------------------------

    def set_ne_level(self, ne: float) -> None:
        """Set noradrenaline level for dark-matter recruitment."""
        self._ne_level = float(np.clip(ne, 0.0, 1.0))

    def set_plasticity_timescales(self, ne: float, ach: float = 0.5) -> None:
        """Modulate trace/membrane time constants via NE and ACh.

        NE → compresses eligibility/STDP traces (explore new associations).
        ACh → compresses membrane τ (prioritise bottom-up input).
        """
        ne = float(np.clip(ne, 0.0, 1.0))
        ach = float(np.clip(ach, 0.0, 1.0))

        ctx = self.neuron_cfg.ctx

        # NE compression on trace time constants
        ne_factor = 1.0 + ne * self.neuron_cfg.ne_trace_compression
        eff_tau_e = self.stdp_cfg.tau_eligibility / ne_factor
        eff_tau_pre = self.stdp_cfg.tau_plus / ne_factor
        eff_tau_post = self.stdp_cfg.tau_minus / ne_factor

        # ACh compression on membrane τ
        ach_factor = 1.0 + ach * self.neuron_cfg.ach_membrane_compression
        eff_tau_m = self.neuron_cfg.tau_m / ach_factor

        self._elig_decay = ctx.decay(eff_tau_e)
        self._pre_decay = ctx.decay(eff_tau_pre)
        self._post_decay = ctx.decay(eff_tau_post)
        self._mem_decay = ctx.decay(eff_tau_m)
        self._mem_gain = ctx.complement(eff_tau_m)

    # ------------------------------------------------------------------
    # Weight update — three-factor STDP (Izhikevich 2007)
    # ------------------------------------------------------------------

    def update_weights(
        self,
        m_t: float,
        pred_error: NDArray[np.float32],
    ) -> None:
        """Three-factor STDP: Δw = lr × m_t × e × error_signal.

        Broadcasting logic:
          - If pred_error matches num_inputs → input-space error (PC).
          - If pred_error matches num_neurons → output-space error (BG).
        """
        if np.isclose(m_t, 0.0):
            return

        # Determine broadcast shape
        if pred_error.shape[0] == self.num_inputs:
            error_signal = pred_error[:, np.newaxis]
        elif pred_error.shape[0] == self.num_neurons:
            error_signal = pred_error[np.newaxis, :]
        else:
            raise ValueError(
                f"pred_error shape {pred_error.shape} incompatible with "
                f"inputs ({self.num_inputs}) or neurons ({self.num_neurons})."
            )

        dw = self.stdp_cfg.a_plus * m_t * self.e * error_signal
        self.w += dw

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset transient state between episodes. Weights preserved."""
        self.v.fill(self.neuron_cfg.v_rest)
        self.w_adapt.fill(0.0)
        self.e.fill(0.0)
        self.x_pre.fill(0.0)
        self.x_post.fill(0.0)
        self.refrac_count.fill(0)
        self.has_spiked.fill(False)

        if self.channels is not None:
            self.channels.reset()

        if self._homeo_state is not None:
            self._homeo_state.reset(self.neuron_cfg.v_thresh)
        self._ne_level = 0.0

        # Reset effective decay factors to base values
        self._mem_decay = self.neuron_cfg.mem_decay
        self._mem_gain = self.neuron_cfg.mem_gain
        self._pre_decay = self.stdp_cfg.pre_decay
        self._post_decay = self.stdp_cfg.post_decay
        self._elig_decay = self.stdp_cfg.elig_decay
        self._scaling_counter = 0


# Backward-compat alias — downstream code imports LIFLayer
LIFLayer = AdExLayer
