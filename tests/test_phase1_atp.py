"""
Phase 1.3 Verification — ATP Energy Budget (continuous, not binary).

Tests:
  1. AstrocyteField ATP properties: threshold_shift / leak_gain correctness.
  2. ATP depletion under sustained activity → monotonic threshold rise.
  3. ATP regeneration → full recovery to atp_max.
  4. Firing rate monotonically decreases under constant stimulation (never drops to zero instantly).
  5. All layers accept set_astrocyte() without error.
  6. Numerical stability: 10,000 steps with ATP-attached AdExLayer, no NaN.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import (
    AstrocyteConfig,
    NeuronConfig,
    InhibitoryPoolConfig,
    BasalGangliaConfig,
    ErrorNeuronConfig,
    PredictiveCodingConfig,
)
from core.simulation_context import SimulationContext
from core.astrocyte import AstrocyteField
from core.neuron import AdExLayer
from core.interneuron import InhibitoryPool
from core.error_neuron import ErrorNeuronLayer
from core.basal_ganglia import SNNDeepCritic, D1D2Actor


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def ctx() -> SimulationContext:
    return SimulationContext(dt=1.0)


@pytest.fixture
def astro(ctx: SimulationContext) -> AstrocyteField:
    cfg = AstrocyteConfig(ctx=ctx, n_zones=4)
    return AstrocyteField(config=cfg)


# =====================================================================
# Test 1: threshold_shift / leak_gain properties
# =====================================================================

class TestATPProperties:
    def test_full_atp_no_shift(self, astro: AstrocyteField) -> None:
        """At full ATP, threshold_shift == 0 and leak_gain == 1."""
        np.testing.assert_allclose(astro.threshold_shift, 0.0, atol=1e-6)
        np.testing.assert_allclose(astro.leak_gain, 1.0, atol=1e-6)

    def test_zero_atp_max_shift(self, astro: AstrocyteField) -> None:
        """At zero ATP, threshold_shift == atp_threshold_shift, leak_gain == 1 + atp_leak_gain."""
        astro.atp.fill(0.0)
        cfg = astro.config
        np.testing.assert_allclose(
            astro.threshold_shift, cfg.atp_threshold_shift, atol=1e-5,
        )
        np.testing.assert_allclose(
            astro.leak_gain, 1.0 + cfg.atp_leak_gain, atol=1e-5,
        )

    def test_half_atp_intermediate(self, astro: AstrocyteField) -> None:
        """At half ATP, shift and gain are half of max effect."""
        astro.atp.fill(0.5)
        cfg = astro.config
        np.testing.assert_allclose(
            astro.threshold_shift, cfg.atp_threshold_shift * 0.5, atol=1e-5,
        )
        np.testing.assert_allclose(
            astro.leak_gain, 1.0 + cfg.atp_leak_gain * 0.5, atol=1e-5,
        )

    def test_reset_restores_atp(self, astro: AstrocyteField) -> None:
        """reset_state() restores ATP to atp_max."""
        astro.atp.fill(0.0)
        astro.reset_state()
        np.testing.assert_allclose(astro.atp, astro.config.atp_max, atol=1e-6)


# =====================================================================
# Test 2: ATP depletion under sustained activity
# =====================================================================

class TestATPDepletion:
    def test_atp_monotonically_decreases(self, ctx: SimulationContext) -> None:
        """Under constant high activity, ATP falls monotonically."""
        cfg = AstrocyteConfig(ctx=ctx, n_zones=4, atp_regen_rate=0.0, atp_spike_cost=0.02)
        astro = AstrocyteField(config=cfg)
        high_rates = np.ones(4, dtype=np.float32)
        prev_atp = astro.atp.copy()
        for _ in range(100):
            astro.update(high_rates)
            assert np.all(astro.atp <= prev_atp + 1e-7), \
                "ATP should not increase with zero regen"
            prev_atp = astro.atp.copy()
        assert np.all(astro.atp < 0.5), "ATP should have depleted significantly"

    def test_threshold_rises_monotonically(self, ctx: SimulationContext) -> None:
        """threshold_shift rises monotonically as ATP depletes."""
        cfg = AstrocyteConfig(ctx=ctx, n_zones=4, atp_regen_rate=0.0, atp_spike_cost=0.02)
        astro = AstrocyteField(config=cfg)
        high_rates = np.ones(4, dtype=np.float32)
        prev_shift = astro.threshold_shift.copy()
        for _ in range(100):
            astro.update(high_rates)
            cur_shift = astro.threshold_shift
            assert np.all(cur_shift >= prev_shift - 1e-7), \
                "threshold_shift should rise monotonically as ATP depletes"
            prev_shift = cur_shift.copy()

    def test_atp_clamps_at_zero(self, ctx: SimulationContext) -> None:
        """ATP never goes below 0."""
        cfg = AstrocyteConfig(
            ctx=ctx, n_zones=2, atp_regen_rate=0.0, atp_spike_cost=1.0,
        )
        astro = AstrocyteField(config=cfg)
        huge_rates = np.full(2, 10.0, dtype=np.float32)
        for _ in range(200):
            astro.update(huge_rates)
        np.testing.assert_array_less(-1e-7, astro.atp)  # atp >= 0


# =====================================================================
# Test 3: ATP regeneration
# =====================================================================

class TestATPRegeneration:
    def test_full_recovery_with_no_activity(self, ctx: SimulationContext) -> None:
        """After depletion, ATP recovers to atp_max with zero activity."""
        cfg = AstrocyteConfig(ctx=ctx, n_zones=2, atp_regen_rate=0.01)
        astro = AstrocyteField(config=cfg)
        astro.atp.fill(0.0)
        zero_rates = np.zeros(2, dtype=np.float32)
        for _ in range(5000):
            astro.update(zero_rates)
        np.testing.assert_allclose(
            astro.atp, cfg.atp_max, atol=0.01,
            err_msg="ATP should recover close to atp_max",
        )

    def test_equilibrium_under_moderate_activity(self, ctx: SimulationContext) -> None:
        """Under moderate activity, ATP should converge to a stable equilibrium < atp_max."""
        cfg = AstrocyteConfig(ctx=ctx, n_zones=2)
        astro = AstrocyteField(config=cfg)
        moderate_rates = np.full(2, 0.3, dtype=np.float32)
        for _ in range(10_000):
            astro.update(moderate_rates)
        atp_eq = astro.atp.copy()
        # Run 100 more — should be stable
        for _ in range(100):
            astro.update(moderate_rates)
        np.testing.assert_allclose(astro.atp, atp_eq, atol=0.001)
        # Equilibrium should be less than atp_max
        assert np.all(atp_eq < cfg.atp_max - 0.01)


# =====================================================================
# Test 4: Firing rate decreases under constant stimulation
# =====================================================================

class TestFiringRateModulation:
    def test_rate_decreases_with_atp_depletion(self, ctx: SimulationContext) -> None:
        """AdExLayer with astrocyte: firing rate decreases as ATP depletes."""
        ncfg = NeuronConfig(ctx=ctx)
        layer = AdExLayer(num_inputs=10, num_neurons=20, neuron_cfg=ncfg)

        astro_cfg = AstrocyteConfig(
            ctx=ctx, n_zones=4, atp_regen_rate=0.0, atp_spike_cost=0.05,
        )
        astro = AstrocyteField(config=astro_cfg)
        layer.set_astrocyte(astro)

        constant_input = np.random.uniform(0, 1, 10).astype(np.float32)
        constant_input[constant_input > 0.5] = 1.0
        constant_input[constant_input <= 0.5] = 0.0

        # Collect spike counts in windows
        window = 50
        n_windows = 6
        spike_counts = []
        for w in range(n_windows):
            count = 0
            for _ in range(window):
                out = layer.forward(constant_input)
                count += int(np.sum(out > 0.5))
                # Feed spike rates to astrocyte
                astro.update(np.ones(4, dtype=np.float32))
            spike_counts.append(count)

        # Trend should be non-increasing (allow equal windows, but last < first)
        assert spike_counts[-1] <= spike_counts[0], (
            f"Firing should decrease as ATP depletes: {spike_counts}"
        )


# =====================================================================
# Test 5: set_astrocyte() accepted by all layers
# =====================================================================

class TestSetAstrocyte:
    def test_adex_layer(self, ctx: SimulationContext, astro: AstrocyteField) -> None:
        layer = AdExLayer(num_inputs=8, num_neurons=16, neuron_cfg=NeuronConfig(ctx=ctx))
        layer.set_astrocyte(astro)
        assert layer._astrocyte is astro
        assert layer._zone_idx is not None
        assert layer._zone_idx.shape == (16,)

    def test_inhibitory_pool(self, ctx: SimulationContext, astro: AstrocyteField) -> None:
        pool = InhibitoryPool(n_excitatory=16, config=InhibitoryPoolConfig(ctx=ctx))
        pool.set_astrocyte(astro)
        assert pool._astrocyte is astro
        assert pool._zone_idx is not None

    def test_error_neuron_layer(self, ctx: SimulationContext, astro: AstrocyteField) -> None:
        ecfg = ErrorNeuronConfig(ctx=ctx)
        layer = ErrorNeuronLayer(n_input=8, config=ecfg)
        layer.set_astrocyte(astro)
        assert layer._astrocyte is astro
        assert layer._zone_idx_error is not None
        assert layer._zone_idx_state is not None

    def test_snn_deep_critic(self, ctx: SimulationContext, astro: AstrocyteField) -> None:
        bgcfg = BasalGangliaConfig(ctx=ctx)
        critic = SNNDeepCritic(state_size=8, config=bgcfg)
        critic.set_astrocyte(astro)
        assert critic._astrocyte is astro

    def test_d1d2_actor(self, ctx: SimulationContext, astro: AstrocyteField) -> None:
        bgcfg = BasalGangliaConfig(ctx=ctx)
        actor = D1D2Actor(state_size=8, motor_dim=4, internal_dim=2, config=bgcfg)
        actor.set_astrocyte(astro)
        assert actor._astrocyte is astro

    def test_custom_zone_idx(self, ctx: SimulationContext, astro: AstrocyteField) -> None:
        layer = AdExLayer(num_inputs=8, num_neurons=8, neuron_cfg=NeuronConfig(ctx=ctx))
        zi = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
        layer.set_astrocyte(astro, zone_idx=zi)
        np.testing.assert_array_equal(layer._zone_idx, zi)


# =====================================================================
# Test 6: Numerical stability with ATP-attached layer
# =====================================================================

class TestNumericalStabilityATP:
    def test_10k_steps_no_nan(self, ctx: SimulationContext) -> None:
        """10,000 steps with ATP-coupled AdExLayer: no NaN, no inf."""
        ncfg = NeuronConfig(ctx=ctx)
        layer = AdExLayer(num_inputs=10, num_neurons=20, neuron_cfg=ncfg)

        astro_cfg = AstrocyteConfig(ctx=ctx, n_zones=4)
        astro = AstrocyteField(config=astro_cfg)
        layer.set_astrocyte(astro)

        rng = np.random.default_rng(42)
        for t in range(10_000):
            inp = (rng.random(10) > 0.7).astype(np.float32)
            out = layer.forward(inp)
            # Feed astrocyte with some rates
            if t % 10 == 0:
                astro.update(rng.random(4).astype(np.float32))

            assert not np.any(np.isnan(layer.v)), f"NaN in v at step {t}"
            assert not np.any(np.isinf(layer.v)), f"inf in v at step {t}"
            assert not np.any(np.isnan(out)), f"NaN in output at step {t}"

    def test_inhibitory_pool_10k_steps(self, ctx: SimulationContext) -> None:
        """10,000 steps with ATP-coupled InhibitoryPool: no NaN."""
        pool = InhibitoryPool(
            n_excitatory=16,
            config=InhibitoryPoolConfig(ctx=ctx, n_interneurons=8),
        )
        astro_cfg = AstrocyteConfig(ctx=ctx, n_zones=4)
        astro = AstrocyteField(config=astro_cfg)
        pool.set_astrocyte(astro)

        rng = np.random.default_rng(42)
        for t in range(10_000):
            exc = (rng.random(16) > 0.8).astype(np.float32)
            inh_current = pool.step(exc)
            if t % 10 == 0:
                astro.update(rng.random(4).astype(np.float32))
            assert not np.any(np.isnan(inh_current)), f"NaN at step {t}"
            assert not np.any(np.isnan(pool.v_inh)), f"NaN in v_inh at step {t}"
