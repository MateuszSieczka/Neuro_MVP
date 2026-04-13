"""
Phase 5 tests: Structural Cleanup & Consistency.

Verifies:
  5.1  All graph connections use sum aggregation (no concat)
  5.2  Homeostasis is continuous (no counter/modulo pattern)
  5.3  No hard column-norm clipping in BG update paths
  5.4  PSP target derived from Feldmeyer et al. (2002) biophysics
  5.5  Dead code removed (_step_count, solve_rate_threshold, unused TrainResult)
  5.6  CartPole removed from REGISTRY; AGI-appropriate envs present
  5.7  feature_rms EMA removed (Phase 3; verified here for completeness)
"""

from __future__ import annotations

import inspect
import textwrap

import numpy as np
import pytest

from core.config import (
    BasalGangliaConfig,
    NeuronConfig,
    _PSP_TARGET_DEFAULT,
    _UNITARY_EPSP_MV,
    compute_weight_std,
    init_weights,
)
from core.basal_ganglia import SNNDeepCritic, D1D2Actor
from core.simulation_context import SimulationContext
from arena.core import TrainResult
from arena.benchmark import BenchmarkConfig
from arena.task_config import REGISTRY, TaskConfig


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def ctx():
    return SimulationContext(dt=1.0)


@pytest.fixture
def bg_cfg(ctx):
    return BasalGangliaConfig(ctx=ctx, hidden_size=32, neurons_per_action=8)


@pytest.fixture
def critic(bg_cfg):
    np.random.seed(42)
    return SNNDeepCritic(state_size=16, config=bg_cfg)


@pytest.fixture
def actor(bg_cfg):
    np.random.seed(42)
    return D1D2Actor(state_size=16, motor_dim=2, internal_dim=0, config=bg_cfg)


# =====================================================================
# 5.1  Sum aggregation (no concat)
# =====================================================================

class TestSumAggregation:
    """All graph connections use sum mode — synaptic currents sum at dendrite."""

    def test_snn_agent_no_concat_in_build_graph(self):
        """_build_graph source should not contain aggregation_mode='concat'."""
        from arena.snn_agent import SNNAgent
        source = inspect.getsource(SNNAgent._build_graph)
        assert '"concat"' not in source, (
            "_build_graph still uses concat aggregation"
        )

    def test_snn_agent_graph_connections_all_sum(self):
        """All connections in a constructed agent use sum aggregation."""
        from arena.snn_agent import SNNAgent
        agent = SNNAgent(
            state_size=4, n_actions=2,
            use_world_model=True, use_working_memory=True,
        )
        for conn in agent.network._connections:
            assert conn.aggregation_mode == "sum", (
                f"Connection {conn.source}→{conn.target} uses "
                f"'{conn.aggregation_mode}', expected 'sum'"
            )

    def test_no_wm_size_in_bg_input(self):
        """bg_input_size should NOT include wm_num_neurons."""
        from arena.snn_agent import SNNAgent
        agent = SNNAgent(
            state_size=4, n_actions=2,
            use_world_model=True, use_working_memory=True,
        )
        # Critic input size should equal encoded_size (pop encoder output),
        # NOT encoded_size + wm_num_neurons.
        encoded_size = agent._encoded_size
        assert agent.critic.num_inputs == encoded_size, (
            f"Critic num_inputs={agent.critic.num_inputs} but "
            f"encoded_size={encoded_size}. "
            f"WM size should not inflate BG input."
        )


# =====================================================================
# 5.2  Continuous homeostasis (no counter modulo)
# =====================================================================

class TestContinuousHomeostasis:
    """Homeostatic scaling runs every step, not every N steps."""

    def test_critic_no_homeo_interval_check(self):
        """SNNDeepCritic.update() should not contain 'homeo_interval'."""
        source = inspect.getsource(SNNDeepCritic.update)
        assert "homeo_interval" not in source, (
            "Critic update still uses periodic homeo_interval check"
        )

    def test_actor_no_homeo_interval_check(self):
        """D1D2Actor.update() should not contain 'homeo_interval'."""
        source = inspect.getsource(D1D2Actor.update)
        assert "homeo_interval" not in source, (
            "Actor update still uses periodic homeo_interval check"
        )

    def test_config_no_homeo_interval(self):
        """BasalGangliaConfig should not have homeo_interval."""
        assert not hasattr(BasalGangliaConfig(), "homeo_interval"), (
            "homeo_interval still exists in config"
        )

    def test_config_no_homeo_max_change(self):
        """BasalGangliaConfig should not have homeo_max_change."""
        assert not hasattr(BasalGangliaConfig(), "homeo_max_change"), (
            "homeo_max_change still exists in config"
        )

    def test_continuous_correction_uses_homeo_tau(self):
        """update() source should reference homeo_tau for per-step alpha."""
        source = inspect.getsource(SNNDeepCritic.update)
        assert "homeo_tau" in source, (
            "Continuous homeostasis should derive alpha from homeo_tau"
        )

    def test_homeostasis_modifies_weights_after_few_steps(
        self, critic: SNNDeepCritic,
    ):
        """Weights should change after just 10 steps (continuous, no interval)."""
        state = np.full(16, 0.5, dtype=np.float32)
        w_before = critic.w_h.copy()
        for _ in range(10):
            spikes = (np.random.random(16) < 0.5).astype(np.float32)
            critic.forward(spikes)
            critic.update(td_error=0.0)
        # Even with zero TD, homeostatic correction should modify weights
        assert not np.allclose(critic.w_h, w_before, atol=1e-8), (
            "Continuous homeostasis should change weights within 10 steps"
        )


# =====================================================================
# 5.3  No hard column-norm clipping
# =====================================================================

class TestNoHardClipping:
    """Hard column-norm clipping removed; continuous homeostasis handles stability."""

    def test_critic_update_no_col_norm_clip(self):
        """SNNDeepCritic.update should not contain column norm clipping loop."""
        source = inspect.getsource(SNNDeepCritic.update)
        assert "col_norm" not in source, (
            "Critic update still has column norm clipping"
        )

    def test_actor_update_no_col_norm_clip(self):
        """D1D2Actor.update should not contain column norm clipping loop."""
        source = inspect.getsource(D1D2Actor.update)
        assert "col_norm" not in source, (
            "Actor update still has column norm clipping"
        )

    def test_dales_law_preserved_after_updates(self, actor: D1D2Actor):
        """Dale's law (w >= 0) maintained without hard clips."""
        state = np.random.random(16).astype(np.float32)
        for _ in range(50):
            spikes = (np.random.random(16) < 0.5).astype(np.float32)
            actor.forward(spikes)
            td = np.random.choice([-2.0, -0.5, 0.0, 0.5, 2.0])
            actor.update(td_error=td)
        assert np.all(actor.w_d1 >= 0.0), "D1 violates Dale's law"
        assert np.all(actor.w_d2 >= 0.0), "D2 violates Dale's law"


# =====================================================================
# 5.4  PSP target from Feldmeyer biophysics
# =====================================================================

class TestPSPBiophysics:
    """PSP target derived from unitary EPSP conductance biophysics."""

    def test_unitary_epsp_in_feldmeyer_range(self):
        """_UNITARY_EPSP_MV should fall within 0.15–5.5 mV (Feldmeyer 2002)."""
        assert 0.15 <= _UNITARY_EPSP_MV <= 5.5, (
            f"Unitary EPSP {_UNITARY_EPSP_MV:.3f} mV outside Feldmeyer range"
        )

    def test_psp_target_derived_from_conductance(self):
        """_PSP_TARGET_DEFAULT should be EPSP / sqrt(2/π), not arbitrary."""
        expected = _UNITARY_EPSP_MV / np.sqrt(2.0 / np.pi)
        np.testing.assert_allclose(
            _PSP_TARGET_DEFAULT, expected, rtol=1e-6,
            err_msg="PSP target not derived from unitary EPSP biophysics",
        )

    def test_init_weights_default_uses_biophysical_psp(self):
        """init_weights default psp_target should be the derived biophysical value."""
        sig = inspect.signature(init_weights)
        default_psp = sig.parameters["psp_target"].default
        np.testing.assert_allclose(
            default_psp, _PSP_TARGET_DEFAULT, rtol=1e-6,
            err_msg="init_weights default psp_target differs from biophysical derivation",
        )

    def test_compute_weight_std_default_uses_biophysical_psp(self):
        """compute_weight_std default psp_target should be biophysical."""
        sig = inspect.signature(compute_weight_std)
        default_psp = sig.parameters["psp_target"].default
        np.testing.assert_allclose(
            default_psp, _PSP_TARGET_DEFAULT, rtol=1e-6,
        )

    def test_single_synapse_epsp_reasonable(self):
        """With a single active synapse, expected weight ≈ unitary EPSP."""
        std = compute_weight_std(fan_in=1, fan_out=1)
        expected_single_w = std * np.sqrt(2.0 / np.pi)  # E[|N(0,σ)|]
        # Should be close to the unitary EPSP
        np.testing.assert_allclose(
            expected_single_w, _UNITARY_EPSP_MV, rtol=0.01,
        )


# =====================================================================
# 5.5  Dead code removed
# =====================================================================

class TestDeadCodeRemoved:
    """Previously dead code should no longer exist."""

    def test_no_step_count_in_agent(self):
        """SNNAgent should not have _step_count attribute."""
        from arena.snn_agent import SNNAgent
        agent = SNNAgent(state_size=4, n_actions=2, use_world_model=False)
        assert not hasattr(agent, "_step_count"), (
            "_step_count still exists in SNNAgent"
        )

    def test_no_solve_rate_threshold(self):
        """BenchmarkConfig should not have solve_rate_threshold."""
        assert not hasattr(BenchmarkConfig(), "solve_rate_threshold"), (
            "solve_rate_threshold still in BenchmarkConfig"
        )

    def test_no_learning_curve_method(self):
        """TrainResult should not have learning_curve()."""
        assert not hasattr(TrainResult(), "learning_curve"), (
            "Dead method learning_curve still exists"
        )

    def test_no_is_improving_method(self):
        """TrainResult should not have is_improving()."""
        assert not hasattr(TrainResult(), "is_improving"), (
            "Dead method is_improving still exists"
        )

    def test_no_action_distribution_method(self):
        """TrainResult should not have action_distribution()."""
        assert not hasattr(TrainResult(), "action_distribution"), (
            "Dead method action_distribution still exists"
        )


# =====================================================================
# 5.6  CartPole removed; AGI-appropriate environments
# =====================================================================

class TestTaskRegistry:
    """REGISTRY should have AGI-appropriate tasks, not CartPole."""

    def test_no_cartpole(self):
        """CartPole-v1 should be removed from REGISTRY."""
        assert "CartPole-v1" not in REGISTRY, (
            "CartPole-v1 still in REGISTRY — dense +1 reward is anti-AGI"
        )

    def test_shifting_bandit_present(self):
        """ShiftingBandit environment tests continual plasticity."""
        assert "ShiftingBandit" in REGISTRY

    def test_tmaze_present(self):
        """TMaze environment tests working memory."""
        assert "TMaze" in REGISTRY

    def test_punishment_avoidance_present(self):
        """PunishmentAvoidance tests D2/NoGo pathway."""
        assert "PunishmentAvoidance" in REGISTRY

    def test_mountain_car_still_present(self):
        """MountainCar-v0 (sparse reward) should remain."""
        assert "MountainCar-v0" in REGISTRY

    def test_task_config_has_env_class(self):
        """TaskConfig should support custom env_class field."""
        cfg = REGISTRY["ShiftingBandit"]
        assert cfg.env_class is not None, (
            "Custom env tasks should have env_class set"
        )

    def test_custom_env_instantiation(self):
        """Custom env_class should be instantiable."""
        for name, cfg in REGISTRY.items():
            if cfg.env_class is not None:
                env = cfg.env_class()
                obs = env.reset()
                assert obs.shape == (env.state_size,), (
                    f"{name}: reset() shape mismatch"
                )


# =====================================================================
# 5.7  No feature_rms EMA (verified from Phase 3)
# =====================================================================

class TestNoFeatureRms:
    """_feature_rms_ema was removed in Phase 3."""

    def test_critic_no_feature_rms_ema(self, critic: SNNDeepCritic):
        """Critic should not have _feature_rms_ema attribute."""
        assert not hasattr(critic, "_feature_rms_ema"), (
            "_feature_rms_ema still present in critic"
        )


# =====================================================================
# Integration: codebase-wide grep checks
# =====================================================================

class TestCodebaseGreps:
    """Verify no algorithmic hacks remain in source files."""

    def test_no_concat_aggregation_in_agent(self):
        """No 'concat' aggregation mode in snn_agent source."""
        source = inspect.getsource(
            __import__("arena.snn_agent", fromlist=["SNNAgent"]).SNNAgent
        )
        # exclude comments
        lines = [
            l for l in source.split("\n")
            if not l.strip().startswith("#")
        ]
        code = "\n".join(lines)
        assert 'aggregation_mode="concat"' not in code
        assert "aggregation_mode='concat'" not in code

    def test_no_homeo_interval_in_basal_ganglia(self):
        """No homeo_interval reference in basal_ganglia update paths."""
        import core.basal_ganglia as bg
        critic_src = inspect.getsource(bg.SNNDeepCritic.update)
        actor_src = inspect.getsource(bg.D1D2Actor.update)
        combined = critic_src + actor_src
        assert "homeo_interval" not in combined

    def test_no_col_norm_in_bg_update(self):
        """No column norm clipping in BG update methods."""
        import core.basal_ganglia as bg
        critic_src = inspect.getsource(bg.SNNDeepCritic.update)
        actor_src = inspect.getsource(bg.D1D2Actor.update)
        combined = critic_src + actor_src
        assert "col_norm" not in combined
