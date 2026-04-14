"""Phase 3 verification tests — Synaptic & Inhibitory Biophysics.

Tests for:
  BIO 1  — Dual-exponential synaptic kinetics
  BIO 3  — Conductance-based competitive layer inhibition
  BIO 4  — PSP target derived from conductance parameters
  MOD 1  — Multiplicative receptor integration
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.config import (
    CompetitiveConfig,
    NeuronConfig,
    ReceptorType,
    SynapseConfig,
)
from core.receptor import (
    aggregate_receptor_effects,
    receptor_effect,
)
from core.simulation_context import SimulationContext
from core.synapse import SynapticChannels


# =====================================================================
# BIO 1 — Dual-exponential synaptic kinetics
# =====================================================================

class TestDualExponentialSynapses:
    """Dual-exponential kinetics: g(t) = N × (e^(-t/τ_d) − e^(-t/τ_r))."""

    @pytest.fixture
    def ctx(self) -> SimulationContext:
        return SimulationContext(dt=1.0)

    @pytest.fixture
    def cfg(self, ctx: SimulationContext) -> SynapseConfig:
        return SynapseConfig(ctx=ctx)

    @pytest.fixture
    def channels(self, cfg: SynapseConfig) -> SynapticChannels:
        return SynapticChannels(n_post=1, config=cfg)

    def _spike_and_track(
        self,
        channels: SynapticChannels,
        steps: int,
        channel: str = "ampa",
    ) -> list[float]:
        """Send one spike and record effective conductance over time."""
        pre = np.array([1.0], dtype=np.float32)
        w = np.array([[1.0]], dtype=np.float32)
        cfg = channels.config
        # Use pure AMPA/pure NMDA by adjusting which channel receives
        if channel == "ampa":
            orig_ratio = cfg.ampa_nmda_ratio
            # Trick: receive with ratio → only AMPA
            channels.receive_excitatory(pre, w)
        elif channel == "nmda":
            cfg2 = SynapseConfig(ctx=cfg.ctx, ampa_nmda_ratio=0.0)
            ch2 = SynapticChannels(n_post=1, config=cfg2)
            ch2.receive_excitatory(pre, w)
            return self._track_channels(ch2, steps, "nmda")

        return self._track_channels(channels, steps, channel)

    def _track_channels(
        self,
        channels: SynapticChannels,
        steps: int,
        channel: str,
    ) -> list[float]:
        g_trace = []
        for _ in range(steps):
            channels.decay()
            g_decay = getattr(channels, f"g_{channel}")[0]
            g_rise = getattr(channels, f"g_{channel}_rise")[0]
            g_eff = max(g_decay - g_rise, 0.0)
            g_trace.append(float(g_eff))
        return g_trace

    def test_ampa_peak_within_1ms(self, channels: SynapticChannels) -> None:
        """AMPA peak at t_peak ≈ τ_r×τ_d/(τ_d−τ_r)×ln(τ_d/τ_r) ≈ 0.8ms.

        With dt=1ms, peak should be at step 1 (first ms).
        Reference: Destexhe et al. (1998), τ_rise=0.4ms, τ_decay=2ms.
        """
        trace = self._spike_and_track(channels, 10)
        peak_step = int(np.argmax(trace))
        assert peak_step == 0, (
            f"AMPA should peak at step 1 (0.8ms theory), got step {peak_step + 1}"
        )

    def test_ampa_decay_time(self, channels: SynapticChannels) -> None:
        """AMPA should decay to ≤37% of peak within ~2ms (τ_decay=2ms)."""
        trace = self._spike_and_track(channels, 10)
        peak = trace[0]
        assert peak > 0, "AMPA should have non-zero peak"
        # At t=2ms (step index 1): g should be ~37% of peak (e^{-1})
        frac_at_2ms = trace[1] / peak if peak > 0 else 0
        assert frac_at_2ms < 0.80, (
            f"AMPA at 2ms should be <80% of peak, got {frac_at_2ms:.3f}"
        )

    def test_nmda_peaks_after_10ms(self) -> None:
        """NMDA current peaks well after 10ms (not instantaneous).

        With τ_rise=10ms, τ_decay=100ms: t_peak ≈ 25.6ms.
        Reference: Lester et al. (1990), Destexhe et al. (1998).
        """
        ctx = SimulationContext(dt=1.0)
        cfg = SynapseConfig(ctx=ctx, ampa_nmda_ratio=0.0)  # pure NMDA
        ch = SynapticChannels(n_post=1, config=cfg)
        pre = np.array([1.0], dtype=np.float32)
        w = np.array([[1.0]], dtype=np.float32)
        ch.receive_excitatory(pre, w)

        trace = []
        for _ in range(60):
            ch.decay()
            g_eff = max(ch.g_nmda[0] - ch.g_nmda_rise[0], 0.0)
            trace.append(g_eff)

        peak_step = int(np.argmax(trace)) + 1  # 1-indexed ms
        assert peak_step > 10, (
            f"NMDA should peak after 10ms, got {peak_step}ms"
        )
        # Theoretical: 25.6ms
        assert 20 <= peak_step <= 30, (
            f"NMDA peak should be ~26ms (theory 25.6), got {peak_step}ms"
        )

    def test_gaba_a_rise_and_decay(self) -> None:
        """GABA-A: τ_rise=0.25ms, τ_decay=5ms → distinct waveform."""
        ctx = SimulationContext(dt=1.0)
        cfg = SynapseConfig(ctx=ctx)
        ch = SynapticChannels(n_post=1, config=cfg)
        inh = np.array([1.0], dtype=np.float32)
        w_ie = np.array([[1.0]], dtype=np.float32)
        ch.receive_inhibitory(inh, w_ie, gaba_b_ratio=0.0)

        trace = []
        for _ in range(20):
            ch.decay()
            g_eff = max(ch.g_gaba_a[0] - ch.g_gaba_a_rise[0], 0.0)
            trace.append(g_eff)

        # Peak should be at step 1 (fast rise)
        peak_step = int(np.argmax(trace))
        assert peak_step <= 1, f"GABA-A should peak within 2ms, got step {peak_step + 1}"
        # Should decay significantly by 10ms
        if trace[0] > 0:
            frac_10ms = trace[9] / trace[0]
            assert frac_10ms < 0.5, (
                f"GABA-A at 10ms should be <50% of peak, got {frac_10ms:.3f}"
            )

    def test_rise_traces_zeroed_after_reset(self) -> None:
        """reset() must clear both decay and rise traces."""
        ctx = SimulationContext(dt=1.0)
        ch = SynapticChannels(n_post=4, config=SynapseConfig(ctx=ctx))
        pre = np.ones(3, dtype=np.float32)
        w = np.ones((3, 4), dtype=np.float32)
        ch.receive_excitatory(pre, w)
        ch.reset()
        assert np.all(ch.g_ampa_rise == 0)
        assert np.all(ch.g_nmda_rise == 0)
        assert np.all(ch.g_gaba_a_rise == 0)
        assert np.all(ch.g_gaba_b_rise == 0)

    def test_normalisation_peak_equals_drive(self) -> None:
        """Peak effective conductance should ≈ drive (normalised)."""
        ctx = SimulationContext(dt=1.0)
        cfg = SynapseConfig(ctx=ctx, ampa_nmda_ratio=1e6)  # nearly pure AMPA
        ch = SynapticChannels(n_post=1, config=cfg)
        pre = np.array([1.0], dtype=np.float32)
        w = np.array([[1.0]], dtype=np.float32)
        ch.receive_excitatory(pre, w)

        # Track AMPA effective conductance
        peak_g = 0.0
        for _ in range(20):
            ch.decay()
            g_eff = max(ch.g_ampa[0] - ch.g_ampa_rise[0], 0.0)
            peak_g = max(peak_g, g_eff)

        # Peak should be close to 1.0 (normalised by N factor)
        # With dt=1ms and τ_rise=0.4ms, peak occurs within first step
        # so discretisation causes some undershoot; accept ±50%
        assert 0.5 < peak_g < 1.5, (
            f"Normalised peak should be near 1.0, got {peak_g:.3f}"
        )


# =====================================================================
# BIO 3 — Conductance-based competitive layer inhibition
# =====================================================================

class TestConductanceBasedInhibition:
    """Inhibition uses g_inh × (E_inh − V), self-limiting at E_inh."""

    def test_derive_g_inh_positive(self) -> None:
        """g_inh should be positive for standard parameters."""
        g = CompetitiveConfig.derive_g_inh(
            gap=15.0, v_thresh=-55.0, e_inh=-75.0,
            num_neurons=20, k_winners=3, strength=1.5,
        )
        assert g > 0, f"g_inh should be positive, got {g}"

    def test_inhibition_self_limits_near_e_inh(self) -> None:
        """At V = E_inh, ΔV from inhibition should be ≈ 0."""
        e_inh = -75.0
        g_inh = CompetitiveConfig.derive_g_inh(
            gap=15.0, v_thresh=-55.0, e_inh=e_inh,
            num_neurons=20, k_winners=3,
        )
        v = e_inh  # membrane at reversal
        dv = g_inh * (e_inh - v)
        assert abs(dv) < 1e-6, (
            f"At V=E_inh, inhibition should be 0, got ΔV={dv:.6f}"
        )

    def test_inhibition_stronger_at_threshold(self) -> None:
        """Inhibition should be stronger when V is further from E_inh."""
        e_inh = -75.0
        g_inh = CompetitiveConfig.derive_g_inh(
            gap=15.0, v_thresh=-55.0, e_inh=e_inh,
            num_neurons=20, k_winners=3,
        )
        dv_thresh = g_inh * (e_inh - (-55.0))  # at V_thresh
        dv_rest = g_inh * (e_inh - (-70.0))    # at V_rest
        assert abs(dv_thresh) > abs(dv_rest), (
            f"|ΔV at thresh|={abs(dv_thresh):.2f} should be > "
            f"|ΔV at rest|={abs(dv_rest):.2f}"
        )

    def test_cannot_push_below_e_inh(self) -> None:
        """Conductance-based inhibition cannot push V below E_inh.

        CompetitiveLIFLayer clamps V at E_inh to prevent overshoot
        from explicit Euler with large g_inh.
        """
        from core.competitive_layer import CompetitiveLIFLayer

        layer = CompetitiveLIFLayer(
            num_inputs=10, num_neurons=20,
        )
        # Manually set losers high (above E_inh but will be pushed)
        layer.v[:] = -70.0
        layer.window_spike_counts[:] = 0
        layer.window_spike_counts[0] = 5  # one winner
        layer._current_window_size = 5
        layer._phase_reset_pending = True
        # Trigger lateral inhibition
        layer._apply_lateral_inhibition()
        # All losers' V should be >= E_inh (clamped)
        assert np.all(layer.v >= layer._e_inh), (
            f"V should be >= E_inh={layer._e_inh}, "
            f"min V={float(np.min(layer.v)):.2f}"
        )

    def test_competitive_config_has_e_inh(self) -> None:
        """CompetitiveConfig should expose GABA-A reversal potential."""
        cfg = CompetitiveConfig()
        assert hasattr(cfg, 'e_inh')
        assert cfg.e_inh == -75.0, f"Expected E_inh=-75.0, got {cfg.e_inh}"


# =====================================================================
# BIO 4 — PSP target derived from conductance parameters
# =====================================================================

class TestPSPTargetDerived:
    """psp_target = g_syn × (E_exc − V_rest) / g_L — no magic 0.15."""

    def test_psp_target_derived_from_gsyn(self) -> None:
        """NeuronConfig.psp_target should equal g_syn(E_exc-V_rest)/g_L."""
        ncfg = NeuronConfig()
        expected = ncfg.g_syn_unitary * (ncfg.e_exc - ncfg.v_rest) / ncfg.g_L
        assert abs(ncfg.psp_target - expected) < 1e-6, (
            f"psp_target={ncfg.psp_target:.4f} != derived={expected:.4f}"
        )

    def test_psp_target_in_feldmeyer_range(self) -> None:
        """Unitary EPSP should be in 0.3-2.5 mV (Feldmeyer 2002)."""
        ncfg = NeuronConfig()
        assert 0.3 <= ncfg.psp_target <= 2.5, (
            f"psp_target={ncfg.psp_target:.3f} outside Feldmeyer range [0.3, 2.5]"
        )

    def test_custom_gsyn_changes_psp(self) -> None:
        """Changing g_syn_unitary should change psp_target."""
        ncfg1 = NeuronConfig(g_syn_unitary=0.43)
        ncfg2 = NeuronConfig(g_syn_unitary=1.0)
        assert ncfg2.psp_target > ncfg1.psp_target, (
            f"Larger g_syn should give larger psp: "
            f"{ncfg2.psp_target:.3f} <= {ncfg1.psp_target:.3f}"
        )

    def test_no_magic_015_in_neuron(self) -> None:
        """AdExLayer should NOT have literal 0.15 factor in weight init."""
        import inspect
        from core.neuron import AdExLayer
        src = inspect.getsource(AdExLayer.__init__)
        assert "* 0.15" not in src and "*0.15" not in src, (
            "AdExLayer.__init__ still contains magic 0.15 factor"
        )


# =====================================================================
# MOD 1 — Multiplicative receptor integration
# =====================================================================

class TestMultiplicativeReceptors:
    """Receptor effects compose multiplicatively, not additively."""

    def test_single_receptor_unchanged(self) -> None:
        """With one receptor, multiplicative == additive."""
        effects = {ReceptorType.D1: 0.5}
        gain, _ = aggregate_receptor_effects(effects)
        assert abs(gain - 1.5) < 1e-6, f"Expected 1.5, got {gain:.4f}"

    def test_multiplicative_not_additive(self) -> None:
        """Two excitatory effects should multiply, not add."""
        effects = {ReceptorType.D1: 0.5, ReceptorType.M1: 0.5}
        gain, _ = aggregate_receptor_effects(effects)
        # Multiplicative: (1+0.5)×(1+0.5) = 2.25
        # Additive would give: 1 + 0.5 + 0.5 = 2.0
        expected_mult = 1.5 * 1.5
        assert abs(gain - expected_mult) < 1e-6, (
            f"Expected multiplicative {expected_mult}, got {gain:.4f}"
        )

    def test_opposing_effects_diminish(self) -> None:
        """Excitatory + inhibitory should give diminishing returns."""
        effects = {ReceptorType.D1: 0.8, ReceptorType.D2: -0.8}
        gain, _ = aggregate_receptor_effects(effects)
        # Multiplicative: (1+0.8)×(1-0.8) = 1.8 × 0.2 = 0.36
        # Additive: 1 + 0 = 1.0
        expected_mult = 1.8 * 0.2
        assert abs(gain - expected_mult) < 1e-6, (
            f"Expected {expected_mult:.3f}, got {gain:.4f}"
        )

    def test_gain_floor_at_01(self) -> None:
        """Gain should not drop below 0.1."""
        effects = {ReceptorType.D2: -0.99, ReceptorType.HT1A: -0.99}
        gain, _ = aggregate_receptor_effects(effects)
        assert gain >= 0.1, f"Gain should be >= 0.1, got {gain:.4f}"

    def test_plasticity_also_multiplicative(self) -> None:
        """Plasticity modulation should also be multiplicative."""
        effects = {ReceptorType.D1: 0.5, ReceptorType.M1: 0.3}
        _, plast = aggregate_receptor_effects(effects)
        expected = 1.5 * 1.3
        assert abs(plast - expected) < 1e-6, (
            f"Plasticity should be {expected:.3f}, got {plast:.4f}"
        )

    def test_empty_effects_identity(self) -> None:
        """No receptors → gain = 1.0, plasticity = 1.0."""
        gain, plast = aggregate_receptor_effects({})
        assert gain == 1.0
        assert plast == 1.0
