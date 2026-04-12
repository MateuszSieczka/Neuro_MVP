"""
InhibitoryPool — GABAergic E→I→E feedback with inhibitory STDP.

Reference: Brunel & Wang (2003), Woodin et al. (2003), Isaacson & Scanziani (2011)

Changes from legacy:
  1. Uses InhibitoryPoolConfig from config.py (derived tau decays).
  2. Inhibitory STDP (Woodin et al. 2003): E→I Hebbian, I→E anti-Hebbian.
  3. E/I balance homeostatic rate — slow correction factor.
  4. DA modulation via D2 receptors on PV+ interneurons.
  5. Proper current naming (i_gaba_a, not g_gaba_a for current-based model).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .config import InhibitoryPoolConfig, SynapseConfig, NeuronConfig

# GABA reversal potential (Isaacson & Scanziani 2011)
_E_INH: float = -75.0  # mV — Cl⁻ reversal, from SynapseConfig.e_inh

if TYPE_CHECKING:
    from .astrocyte import AstrocyteField


class InhibitoryPool:
    """Fast-spiking PV+ interneuron pool for E→I→E lateral inhibition.

    Replaces algorithmic k-WTA with biophysical competition.
    GABA-A (fast, ~70-80%) + GABA-B (slow, ~20-30%) dual channels.
    """

    def __init__(
        self,
        n_excitatory: int,
        config: InhibitoryPoolConfig | None = None,
        neuron_cfg: NeuronConfig | None = None,
    ) -> None:
        self.config = config or InhibitoryPoolConfig()
        self.n_exc = n_excitatory
        n_inh = self.config.n_interneurons
        cfg = self.config

        # AdEx parameters — Fast Spiking PV+ interneurons (a=0, b=0)
        # Biophysical consistency: derive g_L from C_m / tau_m_inh
        # so that the AdEx integrator uses the correct leak for the
        # specified membrane time constant (Brunel & Wang 2003).
        _C_m = 281.0  # NeuronConfig default (Brette & Gerstner 2005)
        _g_L_inh = _C_m / cfg.tau_m_inh  # = 35.125 nS for tau=8ms
        self._ncfg = neuron_cfg or NeuronConfig(
            ctx=cfg.ctx,
            v_rest=cfg.v_rest,
            v_thresh=cfg.v_thresh,
            v_reset=cfg.v_reset,
            tau_m=cfg.tau_m_inh,
            g_L=_g_L_inh,
            a=0.0,   # FS: no subthreshold adaptation
            b=0.0,   # FS: no spike-triggered adaptation
        )

        # ── Interneuron membrane state ────────────────────────────────
        self.v_inh: NDArray[np.float32] = np.full(
            n_inh, cfg.v_rest, dtype=np.float32,
        )
        self.spikes_inh: NDArray[np.bool_] = np.zeros(n_inh, dtype=bool)

        # ── AdEx adaptation current (zero for FS, but state exists) ───
        self.w_adapt_inh: NDArray[np.float32] = np.zeros(
            n_inh, dtype=np.float32,
        )

        # ── Synaptic weights ──────────────────────────────────────────
        # E→I: log-normal-like initialization (broad convergence)
        self.w_ei: NDArray[np.float32] = np.abs(np.random.normal(
            cfg.w_ei_mean, cfg.w_ei_mean * 0.3, (n_excitatory, n_inh),
        )).astype(np.float32)

        # I→E: perisomatic blanket inhibition
        self.w_ie: NDArray[np.float32] = np.abs(np.random.normal(
            cfg.w_ie_mean, cfg.w_ie_mean * 0.2, (n_inh, n_excitatory),
        )).astype(np.float32)

        # ── GABA current traces (current-based, not conductance) ──────
        self.i_gaba_a: NDArray[np.float32] = np.zeros(n_excitatory, dtype=np.float32)
        self.i_gaba_b: NDArray[np.float32] = np.zeros(n_excitatory, dtype=np.float32)

        # ── Precomputed decays from config ────────────────────────────
        syn_cfg = SynapseConfig(ctx=cfg.ctx)
        self._decay_inh: float = cfg.inh_decay
        self._decay_gaba_a: float = syn_cfg.gaba_a_decay
        self._decay_gaba_b: float = syn_cfg.gaba_b_decay

        # ── Inhibitory STDP traces (Woodin et al. 2003) ──────────────
        self._trace_exc: NDArray[np.float32] = np.zeros(n_excitatory, dtype=np.float32)
        self._trace_inh: NDArray[np.float32] = np.zeros(n_inh, dtype=np.float32)
        self._trace_decay: float = cfg.ctx.decay(20.0)  # τ = 20ms

        # ── E→I input gain (biophysics: AdEx rheobase scaling) ────────
        # PV+ interneurons receive convergent excitatory input and need
        # sufficient current to reach threshold.  Without gain scaling,
        # the raw E→I weight product is ~2 pA per step vs ~350 pA
        # rheobase — a 175× mismatch (Brunel & Wang 2003).
        #
        # Gain is derived identically to the main pathway:
        #   I_rheo = g_L × (V_T - E_L - Δ_T)
        #   gain = I_rheo / (expected_active_inputs × mean_weight)
        #
        # Pessimistic estimate: PV+ interneurons must fire even with
        # very sparse excitatory input.  In vivo, single cortical
        # spikes can trigger PV+ firing (Gabernet, Jadhav & Feldman
        # 2005).  Use min(3, expected) to ensure sensitivity.
        ncfg = self._ncfg
        gap_inh = abs(ncfg.v_thresh - ncfg.v_rest)
        i_rheo_inh = ncfg.g_L * (gap_inh - ncfg.delta_t)
        expected_active_exc = max(1.0, min(3.0, n_excitatory * cfg.target_sparsity))
        self._input_gain: float = float(
            i_rheo_inh / (expected_active_exc * cfg.w_ei_mean)
        )

        # ── DA modulation gain ────────────────────────────────────────
        self._ie_gain: float = 1.0

        # ── Astrocyte ATP coupling (Krok 1.3) ─────────────────────────
        self._astrocyte: AstrocyteField | None = None
        self._zone_idx: NDArray[np.int32] | None = None

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
            n = self.config.n_interneurons
            self._zone_idx = np.linspace(
                0, astrocyte.config.n_zones - 1, n,
            ).astype(np.int32)

    def step(self, exc_spikes: NDArray[np.float32],
             v_exc: NDArray[np.float32] | None = None) -> NDArray[np.float32]:
        """One timestep of E→I→E inhibition (conductance-based).

        Args:
            exc_spikes: (n_excitatory,) spike vector.
            v_exc: (n_excitatory,) membrane voltage of excitatory neurons.
                   Required for conductance-based driving force computation.
                   If None, falls back to v_rest (conservative estimate).

        Returns:
            (n_excitatory,) inhibitory current (pA-scale) to subtract from
            excitatory I_syn.  Conductance-based: I = g_GABA × (V - E_inh).
            Self-limiting: as V → E_inh, driving force → 0.
        """
        exc_f32 = exc_spikes.astype(np.float32)
        cfg = self.config

        # ── E→I drive (scaled to AdEx rheobase) ──────────────────────
        i_input = exc_f32 @ self.w_ei * self._input_gain

        # ── Interneuron AdEx integration ──────────────────────────────
        ncfg = self._ncfg
        ctx = ncfg.ctx

        # ATP modulation (Krok 1.3)
        if self._astrocyte is not None:
            zi = self._zone_idx
            eff_v_thresh = ncfg.v_thresh + self._astrocyte.threshold_shift[zi]
            eff_g_L = ncfg.g_L * self._astrocyte.leak_gain[zi]
        else:
            eff_v_thresh = ncfg.v_thresh
            eff_g_L = ncfg.g_L

        exp_term = np.exp(
            np.clip((self.v_inh - eff_v_thresh) / ncfg.delta_t, -20.0, 10.0),
        )
        inv_Cm = 1.0 / ncfg.C_m
        F_v = inv_Cm * (
            -eff_g_L * (self.v_inh - ncfg.v_rest)
            + eff_g_L * ncfg.delta_t * exp_term
            + i_input - self.w_adapt_inh
        )
        J_v = inv_Cm * (-eff_g_L + eff_g_L * exp_term)
        self.v_inh = ctx.exp_euler_step(self.v_inh, F_v, J_v)

        self.spikes_inh = self.v_inh >= ncfg.v_spike_cutoff
        self.v_inh[self.spikes_inh] = ncfg.v_reset
        self.w_adapt_inh[self.spikes_inh] += ncfg.b

        # Subthreshold adaptation
        self.w_adapt_inh = (
            self.w_adapt_inh * ncfg.w_decay
            + ncfg.a * (self.v_inh - ncfg.v_rest) * ncfg.w_gain
        )

        # ── I→E feedback (dual GABA channels) ────────────────────────
        inh_f32 = self.spikes_inh.astype(np.float32)
        feedback = inh_f32 @ self.w_ie * self._ie_gain  # (n_exc,)

        self.i_gaba_a *= self._decay_gaba_a
        self.i_gaba_a += (1.0 - cfg.gaba_b_ratio) * feedback

        self.i_gaba_b *= self._decay_gaba_b
        self.i_gaba_b += cfg.gaba_b_ratio * feedback

        # ── Inhibitory STDP (Woodin et al. 2003) ─────────────────────
        self._trace_exc *= self._trace_decay
        self._trace_exc += exc_f32
        self._trace_inh *= self._trace_decay
        self._trace_inh += inh_f32

        # E→I: Hebbian — co-activation strengthens
        if np.any(self.spikes_inh):
            dw_ei = cfg.inh_stdp_lr * np.outer(
                self._trace_exc, inh_f32,
            )
            self.w_ei += dw_ei.astype(np.float32)
            np.maximum(self.w_ei, 0.0, out=self.w_ei)  # Dale's law

        # I→E: Anti-Hebbian / homeostatic (maintains E/I balance)
        if np.any(exc_f32 > 0.1):
            dw_ie = -cfg.ei_balance_lr * np.outer(
                self._trace_inh, exc_f32,
            )
            self.w_ie += dw_ie.astype(np.float32)
            np.maximum(self.w_ie, 0.0, out=self.w_ie)  # Inhibitory weights ≥ 0

        # ── Conductance-based inhibition (Isaacson & Scanziani 2011) ──
        # I_inh = g_GABA × (V_exc - E_inh) / (V_rest - E_inh)
        # When V_exc ≈ E_inh (-75 mV), driving force ≈ 0 → self-limiting.
        # Normalised by the THRESHOLD driving force so that at spike
        # threshold (where inhibition matters most), the effective
        # inhibition matches the original current-based model.  Below
        # threshold, inhibition is reduced (self-limiting property).
        g_total = self.i_gaba_a + self.i_gaba_b  # conductance (arb. units)
        ref_drive = self.config.v_thresh - _E_INH  # V_thresh - E_inh ≈ 25 mV
        if v_exc is not None:
            driving_force = np.clip((v_exc - _E_INH) / max(ref_drive, 1.0), 0.0, None)
        else:
            driving_force = (self.config.v_rest - _E_INH) / max(ref_drive, 1.0)
        return g_total * driving_force

    def modulate_by_da(self, da_level: float) -> None:
        """DA D2 modulation on PV+ interneurons (Seamans & Yang 2004).

        Uses Hill equation (consistent with receptor.py):
        D2 on PV+ with density ~0.4 (less than MSN), EC50=0.3, n=1.2.
        ie_gain = 1.0 + density × hill_response(DA, EC50, n)
        """
        da = float(np.clip(da_level, 0.0, 1.0))
        # Hill equation: response = da^n / (da^n + ec50^n)
        _ec50 = 0.3
        _n = 1.2
        _density = 0.4
        da_n = da ** _n
        response = da_n / (da_n + _ec50 ** _n)
        self._ie_gain = float(1.0 + _density * response)

    def enter_sws(self, gain_multiplier: float = 2.5) -> None:
        """Elevate I→E gain during SWS (GABA surge).

        During slow-wave sleep, GABAergic inhibition is globally elevated
        (2-3× baseline) to enforce Down state hyperpolarization.
        """
        self._ie_gain *= gain_multiplier

    def exit_sws(self) -> None:
        """Restore normal I→E gain after SWS."""
        self._ie_gain = 1.0

    def reset_state(self) -> None:
        """Reset transient state between episodes. Weights preserved."""
        self.v_inh.fill(self.config.v_rest)
        self.w_adapt_inh.fill(0.0)
        self.spikes_inh.fill(False)
        self.i_gaba_a.fill(0.0)
        self.i_gaba_b.fill(0.0)
        self._trace_exc.fill(0.0)
        self._trace_inh.fill(0.0)
        self._ie_gain = 1.0

    @property
    def interneuron_rate(self) -> float:
        """Current firing rate of the inhibitory pool."""
        return float(np.mean(self.spikes_inh))
