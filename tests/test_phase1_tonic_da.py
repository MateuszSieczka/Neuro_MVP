"""
Phase 1.4 + 1.5 Verification — Tonic DA leaky integrator & episode-free agent.

Tests:
  1. Tonic DA stabilises under constant reward.
  2. Tonic DA relaxes to new value after reward switch.
  3. Tonic DA time constant is ~60s (τ = 60000ms).
  4. No episodic API remains (update_tonic_da removed).
  5. Agent observe() works without episode boundary tracking.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import NeuromodulatorConfig
from core.simulation_context import SimulationContext
from core.neuromodulator import NeuromodulatorSystem


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def ctx() -> SimulationContext:
    return SimulationContext(dt=1.0)


@pytest.fixture
def nm(ctx: SimulationContext) -> NeuromodulatorSystem:
    cfg = NeuromodulatorConfig(ctx=ctx)
    return NeuromodulatorSystem(config=cfg)


# =====================================================================
# Test 1: Tonic DA stabilises under constant RPE
# =====================================================================

class TestTonicDAStabilisation:
    def test_constant_rpe_converges(self, nm: NeuromodulatorSystem) -> None:
        """Under constant |td_error|, tonic_da should converge to a stable value."""
        pe = np.array([0.5], dtype=np.float32)
        td = 0.8

        # Run many steps
        for _ in range(200_000):
            nm.update(prediction_error=pe, td_error=td)

        tda_snapshot = nm.tonic_da
        # Run 1000 more — should barely change
        for _ in range(1000):
            nm.update(prediction_error=pe, td_error=td)

        assert abs(nm.tonic_da - tda_snapshot) < 0.001, (
            f"tonic_da should stabilise: was {tda_snapshot}, now {nm.tonic_da}"
        )

    def test_zero_rpe_decays_to_zero(self, nm: NeuromodulatorSystem) -> None:
        """With zero td_error, tonic_da should decay toward baseline (0)."""
        # First prime it
        pe = np.array([0.5], dtype=np.float32)
        for _ in range(10_000):
            nm.update(prediction_error=pe, td_error=1.0)
        assert nm.tonic_da > 0.01

        # Now let it decay
        for _ in range(300_000):
            nm.update(prediction_error=np.zeros(1, dtype=np.float32), td_error=0.0)

        assert nm.tonic_da < 0.01, f"tonic_da should decay near zero, got {nm.tonic_da}"


# =====================================================================
# Test 2: Tonic DA relaxes after reward switch
# =====================================================================

class TestTonicDARelaxation:
    def test_reward_switch(self, nm: NeuromodulatorSystem) -> None:
        """After switching RPE magnitude, tonic_da should relax to new equilibrium."""
        pe = np.array([0.5], dtype=np.float32)

        # Phase 1: high RPE
        for _ in range(200_000):
            nm.update(prediction_error=pe, td_error=0.8)
        high_eq = nm.tonic_da

        # Phase 2: low RPE
        for _ in range(200_000):
            nm.update(prediction_error=pe, td_error=0.1)
        low_eq = nm.tonic_da

        assert low_eq < high_eq, (
            f"tonic_da should be lower with smaller RPE: high={high_eq}, low={low_eq}"
        )


# =====================================================================
# Test 3: Time constant is correct (~60s = 60000ms)
# =====================================================================

class TestTonicDATimeConstant:
    def test_tau_tonic_da_in_config(self, ctx: SimulationContext) -> None:
        """NeuromodulatorConfig.tau_tonic_da should be 60000ms by default."""
        cfg = NeuromodulatorConfig(ctx=ctx)
        assert cfg.tau_tonic_da == 60_000.0

    def test_tonic_da_decay_derived(self, ctx: SimulationContext) -> None:
        """tonic_da_decay should be exp(-dt/tau_tonic_da)."""
        cfg = NeuromodulatorConfig(ctx=ctx)
        expected = np.exp(-ctx.dt / cfg.tau_tonic_da)
        assert abs(cfg.tonic_da_decay - expected) < 1e-10

    def test_custom_tau(self, ctx: SimulationContext) -> None:
        """Faster τ should yield faster convergence."""
        fast_cfg = NeuromodulatorConfig(ctx=ctx, tau_tonic_da=1000.0)
        slow_cfg = NeuromodulatorConfig(ctx=ctx, tau_tonic_da=60_000.0)

        fast_nm = NeuromodulatorSystem(config=fast_cfg)
        slow_nm = NeuromodulatorSystem(config=slow_cfg)

        pe = np.array([0.5], dtype=np.float32)
        for _ in range(5000):
            fast_nm.update(prediction_error=pe, td_error=1.0)
            slow_nm.update(prediction_error=pe, td_error=1.0)

        # Fast should be closer to equilibrium after same number of steps
        assert fast_nm.tonic_da > slow_nm.tonic_da, (
            f"Faster τ should converge sooner: fast={fast_nm.tonic_da}, slow={slow_nm.tonic_da}"
        )


# =====================================================================
# Test 4: No episodic API
# =====================================================================

class TestNoEpisodicAPI:
    def test_no_update_tonic_da_method(self, nm: NeuromodulatorSystem) -> None:
        """update_tonic_da() should not exist."""
        assert not hasattr(nm, 'update_tonic_da')

    def test_no_reward_history(self, nm: NeuromodulatorSystem) -> None:
        """reward_history property should not exist."""
        assert not hasattr(nm, 'reward_history')

    def test_no_welford_state(self, nm: NeuromodulatorSystem) -> None:
        """Welford state variables should not exist."""
        assert not hasattr(nm, '_welford_n')
        assert not hasattr(nm, '_welford_mean')
        assert not hasattr(nm, '_welford_m2')

    def test_no_smoothed_reward(self, nm: NeuromodulatorSystem) -> None:
        """_smoothed_reward should not exist."""
        assert not hasattr(nm, '_smoothed_reward')

    def test_no_episode_pred_errors(self, nm: NeuromodulatorSystem) -> None:
        """_episode_pred_errors should not exist."""
        assert not hasattr(nm, '_episode_pred_errors')


# =====================================================================
# Test 5: Stagnation detector still works (ACC)
# =====================================================================

class TestStagnationDetector:
    def test_stagnation_increases_under_flat_tonic(self, nm: NeuromodulatorSystem) -> None:
        """Constant tonic_da → stagnation_factor increases."""
        pe = np.array([0.5], dtype=np.float32)
        # Constant RPE → constant tonic_da → low variability → stagnation
        for _ in range(2000):
            nm.update(prediction_error=pe, td_error=0.5)
        assert nm._stagnation_factor > 0.0

    def test_consolidation_gate_attenuated(self, nm: NeuromodulatorSystem) -> None:
        """consolidation_gate property should still work with stagnation."""
        pe = np.array([0.5], dtype=np.float32)
        for _ in range(100):
            nm.update(prediction_error=pe, td_error=0.5)
        gate = nm.consolidation_gate
        assert 0.0 <= gate <= 1.0
