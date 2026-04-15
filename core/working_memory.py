"""
Working Memory — prefrontal attractor dynamics with dual ACh+DA gating.

Reference:
  Goldman-Rakic (1995)  Prefrontal persistent activity
  O'Reilly & Frank (2006)  "Making working memory work"
  Durstewitz et al. (2000)  "Neurocomputational models of working memory"
  Compte et al. (2000)  Synaptic mechanisms of persistent activity

Changes from legacy:
  1. Dual gating: ACh (sensory) AND DA (update signal) — conjunction gate
  2. Uses WorkingMemoryConfig from config.py (derived decays)
  3. Gate opens only when BOTH ACh ≥ threshold AND DA ≥ threshold
  4. Content neurons: AdEx with PFC-like slow adaptation (τ_w=300ms)
  5. Conductance-based synapses: I = g × (E_exc - V) (Ohm's law)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .config import WorkingMemoryConfig, InhibitoryPoolConfig, SynapseConfig, init_weights
from .interneuron import InhibitoryPool

if TYPE_CHECKING:
    from .astrocyte import AstrocyteField


class WorkingMemoryModule:
    """Persistent WM via recurrent attractor dynamics with dual gating.

    Dual gating (O'Reilly & Frank 2006):
      ACh gates sensory input (bottom-up relevance)
      DA gates update signal (reward PE → new context important)
      Gate = spiking MSN population: ACh and DA modulate excitability
      via receptor-like dynamics.  Both must exceed threshold for the
      gate neurons to fire → conjunction gate from biophysics.
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
        # AdEx adaptation current (Durstewitz et al. 2000)
        self.w_adapt: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )

        # ── Synaptic weights (nS, conductance-based; Feldmeyer 2002) ─
        # WM receives sparse low-dimensional input (raw state, not
        # population-coded): typically 1-3 out of N inputs active per
        # step.  Each synapse must be strong enough that 2 active inputs
        # approach threshold.  Calibrate PSP to gap/2 so that 2 co-active
        # inputs produce ~gap mV total depolarisation → threshold.
        # This matches thalamic→PFC unitary EPSPs: 5-10 mV
        # (Cruikshank et al. 2012; Gil & Bhatt 1999).
        _wm_psp = cfg.gap / 2.0
        self.w_ff: NDArray[np.float32] = init_weights(
            num_external_inputs, num_neurons,
            psp_target=_wm_psp,
            excitatory=True,
            g_L=cfg.g_L,
            driving_force=cfg.driving_force_exc,
        )
        # Recurrent connectivity for attractor dynamics (Compte et al.
        # 2000; Goldman-Rakic 1995).  PFC persistent activity requires
        # pre-existing recurrent excitation — lateral weights cannot
        # bootstrap from zero because without firing there is no
        # Hebbian learning signal (chicken-and-egg problem).
        # Lateral PSP scaled to gap/3: recurrent alone shouldn't cause
        # runaway firing, but 3 co-active neighbours → threshold.
        _lat_psp = cfg.gap / 3.0
        self.w_lateral: NDArray[np.float32] = init_weights(
            num_neurons, num_neurons,
            psp_target=_lat_psp,
            excitatory=True,
            g_L=cfg.g_L,
            driving_force=cfg.driving_force_exc,
        )
        np.fill_diagonal(self.w_lateral, 0.0)  # No autapses (no self-connections)

        # ── Feedback inhibition (Compte et al. 2000) ──────────────────
        # PFC attractor requires E-I balance for bistability: strong
        # recurrent excitation sustains a bump, while PV+ interneuron
        # feedback prevents runaway.  Without inhibition, lateral_strength
        # > 1.0 would cause seizure-like activity.
        # PFC inhibitory parameters: τ_m slightly slower (~10ms vs 8ms
        # for cortical basket cells, Gonzalez-Burgos et al. 2005).
        _inh_cfg = InhibitoryPoolConfig(
            ctx=cfg.ctx,
            n_interneurons=max(4, num_neurons // 4),
            tau_m_inh=10.0,
            target_sparsity=0.20,
        )
        self.inh_pool = InhibitoryPool(num_neurons, config=_inh_cfg)

        # ── NMDA slow recurrent trace (Wang 2001; Compte et al. 2000) ─
        # PFC persistent activity relies on NMDA-dominated recurrent
        # excitation with τ_NMDA≈100ms (Durstewitz et al. 2000).
        # The slow NMDA trace provides a temporal bridge: even when
        # only 2 neurons fire from the cue, the NMDA conductance
        # accumulates over multiple timesteps and sustains activity
        # after feedforward input ceases.
        self._g_nmda_rec: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )
        self._nmda_decay: float = cfg.ctx.decay(100.0)  # τ_NMDA=100ms

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
        self._trace_decay: float = cfg.ctx.decay(cfg.tau_e)
        self._pre_decay: float = cfg.ctx.decay(cfg.tau_pre)
        self._post_decay: float = cfg.ctx.decay(cfg.tau_post)
        self._content_decay: float = cfg.content_decay

        # ── Gate state: striosomal MSN population (O'Reilly & Frank 2006) ─
        # Population of AdEx MSN-like gate neurons.  ACh and DA each
        # provide excitatory drive; conjunction arises because neither
        # alone exceeds firing threshold.
        # Population size from config (default 32, same as BG action
        # channel — gate is a binary action: open/close).
        ng = cfg.n_gate
        self._n_gate: int = ng
        # Gate MSNs initialise in up-state (Wilson & Kawaguchi 1996;
        # Stern et al. 1998): V ≈ V_T − 2 mV.  In-vivo MSNs
        # alternate between down-state (−80 mV) and up-state
        # (−55 mV); starting in up-state lets the gate respond
        # within 2-3 ms when ACh × DA drive arrives.
        _gate_v_init = cfg.v_thresh - 2.0  # up-state
        self._gate_v: NDArray[np.float32] = np.full(
            ng, _gate_v_init, dtype=np.float32,
        )
        self._gate_spikes: NDArray[np.bool_] = np.zeros(ng, dtype=bool)
        self._gate_refrac: NDArray[np.int32] = np.zeros(ng, dtype=np.int32)
        self._gate_rate: NDArray[np.float32] = np.zeros(ng, dtype=np.float32)
        # AdEx adaptation current for gate neurons
        self._gate_w_adapt: NDArray[np.float32] = np.zeros(
            ng, dtype=np.float32,
        )
        # Gate membrane/rate decays from config (derived from gate_tau)
        self._gate_mem_decay: float = cfg.gate_mem_decay
        self._gate_rate_decay: float = cfg.gate_rate_decay
        # Drive calibration: at both thresholds, total synaptic current
        # should just reach AdEx rheobase + steady-state adaptation.
        # AdEx rheobase ≈ g_L × (V_T - E_L - Δ_T) (Brette & Gerstner 2005)
        # where the -Δ_T accounts for the exponential spike initiation
        # lowering the effective threshold.
        # Adaptation at threshold: w_eq = a × (V_T - E_L) — the steady-
        # state subthreshold adaptation current that builds up as V
        # approaches V_T.  Without this term, the gate current balances
        # against rheobase but the accumulated adaptation absorbs the
        # margin, creating a stable sub-threshold equilibrium where
        # gate neurons can never spike (observed: V stalls at ~-55 mV).
        _g_L_eff = cfg.gate_C_m / cfg.gate_tau
        _gap = cfg.v_thresh - cfg.v_rest  # 15 mV
        _i_rheo = _g_L_eff * (_gap - cfg.gate_delta_t)
        _w_adapt_at_thresh = cfg.gate_a * _gap
        self._gate_drive: float = (_i_rheo + _w_adapt_at_thresh) / (
            max(cfg.ach_gate_threshold, 0.01) * max(cfg.da_gate_threshold, 0.01)
        )
        self._gate_signal: float = 0.0
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

        # ── Receptor dose-response modulation (D2) ───────────────────
        self._receptor_gain: float = 1.0
        self._receptor_lr: float = 1.0

        # ── Astrocyte field (De Pittà et al. 2011) ──────────────────
        self._astrocyte: AstrocyteField | None = None
        self._zone_idx: NDArray[np.int32] | None = None

    # ------------------------------------------------------------------
    # Dual gating (O'Reilly & Frank 2006)
    # ------------------------------------------------------------------

    def gate(self, ach_level: float, da_level: float = 1.0) -> None:
        """AdEx MSN conjunction gate (O'Reilly & Frank 2006).

        Gate neuron population receives ACh and DA as excitatory drives.
        Both must be above threshold for total current to reach rheobase
        and produce spikes → conjunction from biophysics, not sigmoid.
        Gate signal = population firing rate normalised by MSN max rate.

        Uses AdEx neuron model (Brette & Gerstner 2005) consistent with
        the rest of the network — same equations as D1D2Actor MSNs.
        """
        cfg = self.config
        self._ach_level = float(ach_level)
        self._da_level = float(da_level)

        # Multiplicative conjunction: ACh × DA × drive.
        # At thresholds: current ≈ gap → barely fires.
        # Above: fires reliably.  Below: no spikes.
        gate_current = float(ach_level * da_level) * self._gate_drive

        ng = self._n_gate

        # ── AdEx integration for gate neurons (Brette & Gerstner 2005) ─
        in_refrac = self._gate_refrac > 0
        self._gate_refrac[in_refrac] -= 1

        # Effective g_L for MSN Up-state (τ = gate_tau)
        g_L_eff = cfg.gate_C_m / cfg.gate_tau
        inv_Cm = 1.0 / cfg.gate_C_m

        # Current noise: g_L × σ_V (Destexhe et al. 2003)
        noise = np.random.normal(
            0, g_L_eff * cfg.gate_noise_std, ng,
        ).astype(np.float32)

        # AdEx membrane dynamics:
        # C dV/dt = -g_L(V-E_L) + g_L Δ_T exp((V-V_T)/Δ_T) + I - w
        exp_term = np.exp(np.clip(
            (self._gate_v - cfg.v_thresh) / cfg.gate_delta_t,
            -20.0, 10.0,
        ))
        F = inv_Cm * (
            -g_L_eff * (self._gate_v - cfg.v_rest)
            + g_L_eff * cfg.gate_delta_t * exp_term
            + gate_current + noise - self._gate_w_adapt
        )
        J = inv_Cm * (-g_L_eff + g_L_eff * exp_term)
        # Exponential Euler step (same integrator as D1D2Actor)
        # Clip J×dt to prevent overflow in exp(): |J×dt| > 20 means
        # the linearisation has broken down and we fall back to forward Euler.
        dt = cfg.ctx.dt
        Jdt = J * dt
        Jdt_clipped = np.clip(Jdt, -20.0, 20.0)
        integrated = np.where(
            np.abs(Jdt) > 1e-6,
            self._gate_v + F / J * (np.exp(Jdt_clipped) - 1.0),
            self._gate_v + F * dt,
        )
        self._gate_v = np.where(in_refrac, cfg.v_reset, integrated)

        # Spike detection at v_spike_cutoff (above V_T)
        self._gate_spikes = (
            (self._gate_v >= cfg.gate_v_spike_cutoff) & ~in_refrac
        )
        self._gate_v[self._gate_spikes] = cfg.v_reset
        self._gate_refrac[self._gate_spikes] = cfg.refrac_period

        # Adaptation current (w): τ_w dw/dt = a(V-E_L) - w; w += b on spike
        self._gate_w_adapt[self._gate_spikes] += cfg.gate_b
        self._gate_w_adapt = (
            self._gate_w_adapt * cfg.gate_w_decay
            + cfg.gate_a * (self._gate_v - cfg.v_rest) * cfg.gate_w_gain
        )

        # Rate EMA → smooth gate signal
        rc = 1.0 - self._gate_rate_decay
        self._gate_rate = (
            self._gate_rate * self._gate_rate_decay
            + self._gate_spikes.astype(np.float32) * rc
        )
        # Normalise: population mean rate / max sustained MSN rate
        # (Humphries et al. 2006; Planert et al. 2010: up-state MSN 40 Hz)
        raw_signal = float(np.mean(self._gate_rate))
        self._gate_signal = float(np.clip(
            raw_signal / cfg.gate_max_rate_per_step, 0.0, 1.0,
        ))

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def forward(self, external_input: NDArray[np.float32]) -> NDArray[np.float32]:
        """One timestep of WM dynamics with AdEx integration.

        Feedforward conductance scaled by soft gate signal [0, 1].
        Recurrent attractor always active for content maintenance.
        Conductance-based: I = g × (E_exc - V) (Ohm's law).
        AdEx: exponential spike initiation + spike-frequency adaptation
        with PFC-like slow τ_w=300ms (Durstewitz et al. 2000).

        Returns:
            (num_neurons,) spike array as float32.
        """
        cfg = self.config
        gate = self._gate_signal
        ctx = cfg.ctx

        # ── Trace decay ───────────────────────────────────────────────
        self.x_pre *= self._pre_decay
        self.x_post *= self._post_decay

        # ── Conductance-based input (scaled by gate + receptor gain) ──
        ext_f32 = external_input.astype(np.float32)
        g_ff = gate * self._receptor_gain * (ext_f32 @ self.w_ff)  # nS
        I_ff = g_ff * (cfg.e_exc - self.v)                         # pA
        self.x_pre += np.clip(ext_f32, 0.0, 1.0) * gate

        # Recurrent contribution always active (attractor maintenance)
        # Lateral weights are Hebbian [0, 1]; treat as conductance gain
        # AMPA component: fast, direct.
        g_rec_ampa = self.content @ self.w_lateral * cfg.lateral_strength  # nS
        # NMDA component: slow τ=100ms trace with Mg²⁺ block (Wang 2001)
        # Provides temporal bridge for persistent activity across delays.
        _nmda_compl = 1.0 - self._nmda_decay
        self._g_nmda_rec = self._g_nmda_rec * self._nmda_decay + _nmda_compl * g_rec_ampa
        mg_block = SynapseConfig.nmda_mg_block(self.v)
        # AMPA:NMDA ratio from config (Wang 2001: PFC recurrent is NMDA-dominated)
        ampa_frac = 1.0 - cfg.nmda_recurrent_ratio
        g_rec = ampa_frac * g_rec_ampa + cfg.nmda_recurrent_ratio * self._g_nmda_rec * mg_block
        I_rec = g_rec * (cfg.e_exc - self.v)                         # pA

        I_syn = I_ff + I_rec

        # ── AdEx membrane integration (Brette & Gerstner 2005) ────────
        in_refrac = self.refrac_count > 0
        self.refrac_count[in_refrac] -= 1

        # Astrocyte ATP modulation: threshold rises + leak increases
        # as ATP depletes (Na⁺/K⁺-ATPase slowdown, Kann & Kovács 2007).
        eff_v_thresh = cfg.v_thresh
        eff_g_L = cfg.g_L
        if self._astrocyte is not None:
            zc = self._zone_idx
            eff_v_thresh = cfg.v_thresh + self._astrocyte.threshold_shift[zc]
            eff_g_L = cfg.g_L * self._astrocyte.leak_gain[zc]

        inv_Cm = 1.0 / cfg.C_m
        exp_term = np.exp(np.clip(
            (self.v - eff_v_thresh) / cfg.delta_t,
            -20.0, 10.0,
        ))
        F = inv_Cm * (
            -eff_g_L * (self.v - cfg.v_rest)
            + eff_g_L * cfg.delta_t * exp_term
            + I_syn - self.w_adapt
        )
        J = inv_Cm * (-eff_g_L + eff_g_L * exp_term)

        # Exponential Euler step (same integrator as rest of network)
        dt = ctx.dt
        Jdt = J * dt
        Jdt_clipped = np.clip(Jdt, -20.0, 20.0)
        integrated = np.where(
            np.abs(Jdt) > 1e-6,
            self.v + F / J * (np.exp(Jdt_clipped) - 1.0),
            self.v + F * dt,
        )
        self.v = np.where(in_refrac, cfg.v_reset, integrated)

        # ── Spike detection at v_spike_cutoff ─────────────────────────
        self.has_spiked = (self.v >= cfg.v_spike_cutoff) & ~in_refrac
        self.v[self.has_spiked] = cfg.v_reset
        self.refrac_count[self.has_spiked] = cfg.refrac_period
        self.x_post[self.has_spiked] += 1.0

        # ── Feedback inhibition (Compte et al. 2000) ─────────────────
        # PV+ interneuron pool provides E→I→E feedback inhibition.
        # Prevents runaway excitation from strong recurrent connectivity.
        # Conductance-based: I_inh = g_inh × (E_inh − V), self-limiting
        # as V approaches GABA-A reversal (Brunel & Wang 2003).
        inh_current = self.inh_pool.step(
            self.has_spiked.astype(np.float32), v_exc=self.v,
        )
        self.v -= inh_current
        np.clip(self.v, -90.0, None, out=self.v)  # K+ reversal floor

        # ── Adaptation current w: τ_w dw/dt = a(V-E_L) - w; w += b ──
        self.w_adapt[self.has_spiked] += cfg.b
        self.w_adapt = (
            self.w_adapt * cfg.w_decay
            + cfg.a * (self.v - cfg.v_rest) * cfg.w_gain
        )

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

    def set_astrocyte(
        self,
        astrocyte: AstrocyteField,
        zone_idx: NDArray[np.int32] | None = None,
    ) -> None:
        """Attach astrocyte for ATP-based threshold/leak modulation."""
        self._astrocyte = astrocyte
        if zone_idx is not None:
            self._zone_idx = zone_idx
        else:
            self._zone_idx = np.linspace(
                0, astrocyte.n_zones - 1, self.num_neurons,
            ).astype(np.int32)

    def set_ne_level(self, ne: float) -> None:
        """NE level — no direct effect on WM dynamics."""
        pass

    def set_receptor_modulation(self, gain_mod: float, lr_mod: float) -> None:
        """Apply receptor dose-response modulation (Hill equation effects)."""
        self._receptor_gain = float(gain_mod)
        self._receptor_lr = float(lr_mod)

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
        dw = self.config.learning_rate * m_t * self._receptor_lr * self.e * pred_error
        self.w_ff += dw

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset transient state. Learned weights preserved."""
        self.v.fill(self.config.v_rest)
        self.w_adapt.fill(0.0)
        self.e.fill(0.0)
        self.x_pre.fill(0.0)
        self.x_post.fill(0.0)
        self.refrac_count.fill(0)
        self.has_spiked.fill(False)
        self.content.fill(0.0)
        self._g_nmda_rec.fill(0.0)
        self.prediction_error.fill(1.0)
        self._gate_signal = 0.0
        self._ach_level = 0.0
        self._da_level = 0.0
        # Reset gate neuron state — up-state init (Wilson & Kawaguchi 1996)
        _gate_v_init = self.config.v_thresh - 2.0
        self._gate_v.fill(_gate_v_init)
        self._gate_spikes.fill(False)
        self._gate_refrac.fill(0)
        self._gate_rate.fill(0.0)
        self._gate_w_adapt.fill(0.0)
        # Reset inhibitory pool state
        self.inh_pool.reset_state()
