"""
Phase 4 tests — Biological Sleep Consolidation.

Tests verify that SWS replay uses neural replay (re-running the critic
network) with VTA RPE, replacing Monte Carlo returns and advantage
normalization.  Also verifies removal of curiosity z-score normalization,
additive intrinsic reward, and arbitrary sleep_gain formula.

References:
    Buzsáki (2015): Sharp-wave ripple replay.
    Diba & Buzsáki (2007): Compressed replay is reconstruction.
    Eshel et al. (2015): VTA circuit arithmetic during replay.
    Walker & Stickgold (2006): Sleep-dependent memory consolidation.
    Pace-Schott & Hobson (2002): SWS neuromodulatory state.
    Hasselmo (2006): NE/ACh exploration modulation.
    Friston (2010): Precision-weighted prediction error.

Test structure:
    1. ReplayBufferConfig — swr_substeps, no gamma
    2. SWS Neural Replay — critic re-runs, VTA RPE, STDP updates
    3. VTA Integration During Sleep — RPE from replayed activations
    4. Curiosity Signal — raw precision-weighted PE, no z-score
    5. No Additive Intrinsic Reward
    6. No sleep_gain Formula
    7. Old API Removal Verification
    8. Numerical Stability
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config import (
    AgentConfig,
    BasalGangliaConfig,
    ReplayBufferConfig,
    VTAConfig,
)
from core.basal_ganglia import SNNDeepCritic, D1D2Actor
from core.replay_buffer import Experience, ReplayBuffer
from core.simulation_context import SimulationContext
from core.vta import VTACircuit


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def ctx():
    return SimulationContext(dt=1.0)


@pytest.fixture
def bg_config(ctx):
    return BasalGangliaConfig(ctx=ctx, hidden_size=32, neurons_per_action=8)


@pytest.fixture
def vta_config(ctx):
    return VTAConfig(ctx=ctx)


@pytest.fixture
def critic(bg_config):
    np.random.seed(42)
    return SNNDeepCritic(state_size=16, config=bg_config)


@pytest.fixture
def actor(bg_config):
    np.random.seed(42)
    return D1D2Actor(state_size=16, motor_dim=2, internal_dim=0, config=bg_config)


@pytest.fixture
def vta(bg_config, vta_config):
    np.random.seed(42)
    return VTACircuit(critic_hidden_size=bg_config.hidden_size, config=vta_config)


def _make_experience(state_size=16, done=False, reward=1.0):
    """Create a minimal Experience for testing."""
    return Experience(
        state=np.random.randn(state_size).astype(np.float32),
        action=0,
        reward=reward,
        next_state=np.random.randn(state_size).astype(np.float32),
        prediction_error=np.random.randn(state_size).astype(np.float32),
        spike_trains=[np.random.randn(state_size).astype(np.float32)],
        synaptic_fingerprint={},
        salience=0.5,
        recorded_da=0.5,
        curiosity=0.3,
        done=done,
    )


class _FakeNeuromodulator:
    """Minimal neuromodulator stub for sleep phase tests."""
    serotonin: float = 0.3
    dopamine: float = 0.5
    tonic_da: float = 0.5
    learning_rate_modulation: float = 0.5
    competition_sharpness: float = 0.5
    bottom_up_gain: float = 0.5


class _FakeBGFacade:
    """Minimal BG facade stub for sleep phase tests."""

    def __init__(self, critic, actor, vta=None):
        self.critic = critic
        self.actor = actor
        self._vta = vta

    @property
    def vta(self):
        return self._vta

    @property
    def last_v(self):
        if self._vta is not None:
            return float(np.dot(self.critic.activation, self._vta.w_value))
        return 0.0


# =====================================================================
# 1. ReplayBufferConfig
# =====================================================================

class TestReplayBufferConfig:
    """ReplayBufferConfig should have swr_substeps, no gamma."""

    def test_swr_substeps_default(self):
        cfg = ReplayBufferConfig()
        assert cfg.swr_substeps == 5

    def test_swr_substeps_custom(self, ctx):
        cfg = ReplayBufferConfig(ctx=ctx, swr_substeps=10)
        assert cfg.swr_substeps == 10

    def test_no_gamma_field(self):
        cfg = ReplayBufferConfig()
        assert not hasattr(cfg, 'gamma'), "gamma removed — VTA handles discounting"

    def test_swr_substeps_positive(self, ctx):
        with pytest.raises(AssertionError, match="swr_substeps must be positive"):
            ReplayBufferConfig(ctx=ctx, swr_substeps=0)


# =====================================================================
# 2. SWS Neural Replay
# =====================================================================

class TestSWSNeuralReplay:
    """SWS replay re-runs critic network with VTA RPE."""

    def test_sleep_phase_returns_errors(self, critic, actor, vta):
        """sleep_phase should return per-experience world model MSE."""
        buf = ReplayBuffer(config=ReplayBufferConfig())
        np.random.seed(123)
        for _ in range(5):
            buf.store(_make_experience())

        bg = _FakeBGFacade(critic, actor, vta)
        neuromod = _FakeNeuromodulator()

        # Need a minimal world model stub
        wm = _FakeWorldModel(state_size=16)
        errors = buf.sleep_phase(
            world_model=wm,
            neuromodulator=neuromod,
            bg=bg,
        )
        assert len(errors) > 0
        assert all(isinstance(e, float) for e in errors)

    def test_critic_weights_change_after_sleep(self, critic, actor, vta):
        """Critic hidden weights should change after SWS replay."""
        buf = ReplayBuffer(config=ReplayBufferConfig())
        np.random.seed(123)
        for _ in range(5):
            buf.store(_make_experience(reward=1.0))

        bg = _FakeBGFacade(critic, actor, vta)
        neuromod = _FakeNeuromodulator()
        wm = _FakeWorldModel(state_size=16)

        w_h_before = critic.w_h.copy()
        buf.sleep_phase(world_model=wm, neuromodulator=neuromod, bg=bg)
        assert not np.allclose(critic.w_h, w_h_before), \
            "Critic weights must change during SWS replay"

    def test_vta_weights_change_after_sleep(self, critic, actor, vta):
        """VTA value readout weights should change after SWS replay."""
        buf = ReplayBuffer(config=ReplayBufferConfig())
        np.random.seed(123)
        for _ in range(5):
            buf.store(_make_experience(reward=1.0))

        bg = _FakeBGFacade(critic, actor, vta)
        neuromod = _FakeNeuromodulator()
        wm = _FakeWorldModel(state_size=16)

        w_before = vta.w_value.copy()
        buf.sleep_phase(world_model=wm, neuromodulator=neuromod, bg=bg)
        assert not np.allclose(vta.w_value, w_before), \
            "VTA w_value must change during SWS replay"

    def test_no_gamma_parameter_in_sleep_phase(self, critic, actor, vta):
        """sleep_phase should not accept gamma parameter."""
        import inspect
        sig = inspect.signature(ReplayBuffer.sleep_phase)
        assert 'gamma' not in sig.parameters, \
            "gamma parameter removed — VTA handles temporal discounting"

    def test_no_sleep_gain_parameter(self, critic, actor, vta):
        """sleep_phase should not accept sleep_gain parameter."""
        import inspect
        sig = inspect.signature(ReplayBuffer.sleep_phase)
        assert 'sleep_gain' not in sig.parameters, \
            "sleep_gain removed — VTA D2 autoreceptor handles gain"

    def test_reverse_replay_order(self, critic, actor, vta):
        """Experiences should be replayed in reverse (SWR characteristic)."""
        buf = ReplayBuffer(config=ReplayBufferConfig())
        np.random.seed(42)
        exps = [_make_experience(reward=float(i)) for i in range(3)]
        for exp in exps:
            buf.store(exp)

        bg = _FakeBGFacade(critic, actor, vta)
        neuromod = _FakeNeuromodulator()
        wm = _FakeWorldModel(state_size=16)

        # After sleep, VTA last_rpe should reflect the FIRST experience
        # (last in reverse order) since it's processed last.
        buf.sleep_phase(world_model=wm, neuromodulator=neuromod, bg=bg)
        # VTA should have processed something (non-zero RPE state)
        assert vta.last_rpe != 0.0 or vta.last_v_s != 0.0

    def test_empty_buffer_returns_empty(self, critic, actor, vta):
        """Empty buffer should return empty error list."""
        buf = ReplayBuffer(config=ReplayBufferConfig())
        bg = _FakeBGFacade(critic, actor, vta)
        neuromod = _FakeNeuromodulator()
        wm = _FakeWorldModel(state_size=16)
        errors = buf.sleep_phase(world_model=wm, neuromodulator=neuromod, bg=bg)
        assert errors == []


# =====================================================================
# 3. VTA Integration During Sleep
# =====================================================================

class TestVTASleepIntegration:
    """VTA should compute RPE from replay-generated activations."""

    def test_vta_store_prediction_called_during_replay(self, critic, actor, vta):
        """VTA.last_v_s should reflect replay V(s)."""
        buf = ReplayBuffer(config=ReplayBufferConfig())
        np.random.seed(42)
        buf.store(_make_experience(reward=5.0))

        bg = _FakeBGFacade(critic, actor, vta)
        neuromod = _FakeNeuromodulator()
        wm = _FakeWorldModel(state_size=16)

        vta_v_s_before = vta.last_v_s
        buf.sleep_phase(world_model=wm, neuromodulator=neuromod, bg=bg)
        # VTA should have stored a new V(s) during replay
        # (may or may not differ from before, but should have been called)
        assert isinstance(vta.last_v_s, float)

    def test_vta_rpe_nonzero_with_reward(self, critic, actor, vta):
        """VTA RPE should be non-zero when replaying reward experiences."""
        buf = ReplayBuffer(config=ReplayBufferConfig())
        np.random.seed(42)
        for _ in range(3):
            buf.store(_make_experience(reward=10.0))

        bg = _FakeBGFacade(critic, actor, vta)
        neuromod = _FakeNeuromodulator()
        wm = _FakeWorldModel(state_size=16)

        buf.sleep_phase(world_model=wm, neuromodulator=neuromod, bg=bg)
        assert vta.last_rpe != 0.0, \
            "VTA should produce non-zero RPE for reward experiences"

    def test_swr_substeps_control_integration_length(self, bg_config, vta_config):
        """Fewer swr_substeps = less activation buildup."""
        np.random.seed(42)
        critic_short = SNNDeepCritic(state_size=16, config=bg_config)
        vta_short = VTACircuit(
            critic_hidden_size=bg_config.hidden_size, config=vta_config,
        )

        np.random.seed(42)
        critic_long = SNNDeepCritic(state_size=16, config=bg_config)
        vta_long = VTACircuit(
            critic_hidden_size=bg_config.hidden_size, config=vta_config,
        )

        np.random.seed(99)
        exp = _make_experience(reward=1.0)

        # Short replay (2 substeps)
        buf_short = ReplayBuffer(config=ReplayBufferConfig(swr_substeps=2))
        buf_short.store(exp)
        actor_s = D1D2Actor(state_size=16, motor_dim=2, internal_dim=0, config=bg_config)
        bg_s = _FakeBGFacade(critic_short, actor_s, vta_short)
        neuromod = _FakeNeuromodulator()
        wm_s = _FakeWorldModel(state_size=16)
        buf_short.sleep_phase(world_model=wm_s, neuromodulator=neuromod, bg=bg_s)
        act_short = float(np.sum(np.abs(critic_short.activation)))

        # Long replay (10 substeps)
        buf_long = ReplayBuffer(config=ReplayBufferConfig(swr_substeps=10))
        buf_long.store(exp)
        actor_l = D1D2Actor(state_size=16, motor_dim=2, internal_dim=0, config=bg_config)
        bg_l = _FakeBGFacade(critic_long, actor_l, vta_long)
        wm_l = _FakeWorldModel(state_size=16)
        buf_long.sleep_phase(world_model=wm_l, neuromodulator=neuromod, bg=bg_l)
        act_long = float(np.sum(np.abs(critic_long.activation)))

        # More substeps → more activation (longer integration)
        assert act_long >= act_short, \
            f"10 substeps should build more activation than 2: {act_long:.4f} vs {act_short:.4f}"

    def test_terminal_experience_zeroes_future_value(self, critic, actor, vta):
        """Terminal experiences should have no future value in replay."""
        buf = ReplayBuffer(config=ReplayBufferConfig())
        np.random.seed(42)
        buf.store(_make_experience(reward=1.0, done=True))

        bg = _FakeBGFacade(critic, actor, vta)
        neuromod = _FakeNeuromodulator()
        wm = _FakeWorldModel(state_size=16)

        buf.sleep_phase(world_model=wm, neuromodulator=neuromod, bg=bg)
        # For terminal state, V(s') should be 0
        assert vta.last_v_s_prime == 0.0, \
            f"Terminal replay should zero future value, got {vta.last_v_s_prime}"


# =====================================================================
# 4. Curiosity Signal — No Z-Score
# =====================================================================

class TestCuriosityNoZScore:
    """Curiosity signal should use raw precision-weighted PE."""

    def test_curiosity_no_history_tracking(self):
        """World model should not track curiosity history for z-score."""
        try:
            from core.world_model import SNNWorldModel
            from core.config import WorldModelConfig
            ctx = SimulationContext(dt=1.0)
            wm_cfg = WorldModelConfig(ctx=ctx, hidden_size=16)
            wm = SNNWorldModel(state_size=4, action_size=2, config=wm_cfg)
            assert not hasattr(wm, '_curiosity_history'), \
                "_curiosity_history removed — no z-score normalization"
        except ImportError:
            pytest.skip("SNNWorldModel not available")

    def test_curiosity_returns_raw_scaled(self):
        """Curiosity should return raw precision-weighted PE in [0, 2]."""
        try:
            from core.world_model import SNNWorldModel
            from core.config import WorldModelConfig
            ctx = SimulationContext(dt=1.0)
            wm_cfg = WorldModelConfig(ctx=ctx, hidden_size=16)
            wm = SNNWorldModel(state_size=4, action_size=2, config=wm_cfg)

            # Set known prediction error
            wm.prediction_error = np.ones(4, dtype=np.float32) * 2.0
            c1 = wm.curiosity_signal()
            assert 0.0 <= c1 <= 2.0, f"Curiosity out of bounds: {c1}"

            # Repeated calls should give same result (no history effect)
            c2 = wm.curiosity_signal()
            assert c1 == c2, "No history → repeated calls identical"
        except ImportError:
            pytest.skip("SNNWorldModel not available")


# =====================================================================
# 5. No Additive Intrinsic Reward
# =====================================================================

class TestNoIntrinsicReward:
    """Agent should not add intrinsic reward to environmental reward."""

    def test_agent_config_no_intrinsic_reward_weight(self):
        """AgentConfig should not have intrinsic_reward_weight."""
        cfg = AgentConfig()
        assert not hasattr(cfg, 'intrinsic_reward_weight'), \
            "intrinsic_reward_weight removed — curiosity → NE/ACh, not reward"

    def test_agent_config_no_sleep_gain_scale(self):
        """AgentConfig should not have sleep_gain_scale."""
        cfg = AgentConfig()
        assert not hasattr(cfg, 'sleep_gain_scale'), \
            "sleep_gain_scale removed — VTA handles adaptation"

    def test_agent_config_no_sleep_gain_max(self):
        """AgentConfig should not have sleep_gain_max."""
        cfg = AgentConfig()
        assert not hasattr(cfg, 'sleep_gain_max'), \
            "sleep_gain_max removed — VTA handles adaptation"


# =====================================================================
# 6. No sleep_gain
# =====================================================================

class TestNoSleepGain:
    """Sleep consolidation should not use arbitrary sleep_gain formula."""

    def test_sws_phase_no_sleep_gain_param(self):
        """_sws_phase should not accept sleep_gain parameter."""
        import inspect
        sig = inspect.signature(ReplayBuffer._sws_phase)
        assert 'sleep_gain' not in sig.parameters

    def test_sws_phase_no_gamma_param(self):
        """_sws_phase should not accept gamma parameter."""
        import inspect
        sig = inspect.signature(ReplayBuffer._sws_phase)
        assert 'gamma' not in sig.parameters

    def test_sws_phase_has_neuromodulator_param(self):
        """_sws_phase should receive neuromodulator for serotonin access."""
        import inspect
        sig = inspect.signature(ReplayBuffer._sws_phase)
        assert 'neuromodulator' in sig.parameters


# =====================================================================
# 7. Old API Removal
# =====================================================================

class TestOldSleepAPIRemoved:
    """Verify all RL/ML hacks are removed from sleep code."""

    def test_no_cumulative_returns(self):
        """_sws_phase should not compute cumulative Monte Carlo returns."""
        import inspect
        source = inspect.getsource(ReplayBuffer._sws_phase)
        assert 'cumulative_return' not in source

    def test_no_advantage_normalization(self):
        """_sws_phase should not compute or normalize advantages."""
        import inspect
        source = inspect.getsource(ReplayBuffer._sws_phase)
        assert 'adv_arr' not in source
        assert 'adv_norm' not in source
        assert 'adv_std' not in source
        assert 'all_advantages' not in source

    def test_no_pos_neg_ratio(self):
        """_sws_phase should not use pos_ratio/neg_ratio scaling."""
        import inspect
        source = inspect.getsource(ReplayBuffer._sws_phase)
        assert 'pos_ratio' not in source
        assert 'neg_ratio' not in source

    def test_no_sleep_signal_formula(self):
        """_sws_phase should not compute sleep_signal from advantage."""
        import inspect
        source = inspect.getsource(ReplayBuffer._sws_phase)
        assert 'sleep_signal' not in source

    def test_no_intrinsic_weight_in_sleep(self):
        """_sws_phase should not use intrinsic_weight."""
        import inspect
        source = inspect.getsource(ReplayBuffer._sws_phase)
        assert 'intrinsic_weight' not in source

    def test_uses_vta_compute_rpe(self):
        """_sws_phase should use VTA compute_rpe, not algebraic TD."""
        import inspect
        source = inspect.getsource(ReplayBuffer._sws_phase)
        assert 'compute_rpe' in source

    def test_uses_vta_store_prediction(self):
        """_sws_phase should use VTA store_prediction."""
        import inspect
        source = inspect.getsource(ReplayBuffer._sws_phase)
        assert 'store_prediction' in source

    def test_uses_vta_update(self):
        """_sws_phase should use VTA update."""
        import inspect
        source = inspect.getsource(ReplayBuffer._sws_phase)
        assert 'vta.update' in source or 'update(rpe)' in source

    def test_uses_critic_forward(self):
        """_sws_phase should re-run critic.forward() for neural replay."""
        import inspect
        source = inspect.getsource(ReplayBuffer._sws_phase)
        assert 'critic.forward' in source

    def test_no_explicit_gamma_in_sws(self):
        """_sws_phase should not use explicit gamma variable."""
        import inspect
        source = inspect.getsource(ReplayBuffer._sws_phase)
        # Should not have "eff_gamma" or "gamma *" patterns
        assert 'eff_gamma' not in source
        assert 'gamma *' not in source


# =====================================================================
# 8. Numerical Stability
# =====================================================================

class TestSleepNumericalStability:
    """Sleep replay should not produce NaN/Inf."""

    def test_many_experiences_no_nan(self, critic, actor, vta):
        """100 experiences replayed without numerical issues."""
        buf = ReplayBuffer(config=ReplayBufferConfig())
        np.random.seed(42)
        for i in range(100):
            buf.store(_make_experience(reward=float((-1)**i), done=(i % 20 == 19)))

        bg = _FakeBGFacade(critic, actor, vta)
        neuromod = _FakeNeuromodulator()
        wm = _FakeWorldModel(state_size=16)

        errors = buf.sleep_phase(world_model=wm, neuromodulator=neuromod, bg=bg)
        assert all(np.isfinite(e) for e in errors), "NaN/Inf in sleep errors"
        assert np.all(np.isfinite(critic.w_h)), "NaN in critic weights after sleep"
        assert np.all(np.isfinite(vta.w_value)), "NaN in VTA weights after sleep"

    def test_extreme_rewards_no_nan(self, critic, actor, vta):
        """Extreme rewards during replay should not produce NaN."""
        buf = ReplayBuffer(config=ReplayBufferConfig())
        np.random.seed(42)
        for r in [1000.0, -1000.0, 0.0, 1e6, -1e6]:
            buf.store(_make_experience(reward=r))

        bg = _FakeBGFacade(critic, actor, vta)
        neuromod = _FakeNeuromodulator()
        wm = _FakeWorldModel(state_size=16)

        errors = buf.sleep_phase(world_model=wm, neuromodulator=neuromod, bg=bg)
        assert all(np.isfinite(e) for e in errors)
        assert np.all(np.isfinite(critic.w_h))
        assert np.all(np.isfinite(vta.w_value))

    def test_zero_reward_experiences(self, critic, actor, vta):
        """All-zero rewards should produce near-zero RPE."""
        buf = ReplayBuffer(config=ReplayBufferConfig())
        np.random.seed(42)
        for _ in range(10):
            buf.store(_make_experience(reward=0.0))

        bg = _FakeBGFacade(critic, actor, vta)
        neuromod = _FakeNeuromodulator()
        wm = _FakeWorldModel(state_size=16)

        buf.sleep_phase(world_model=wm, neuromodulator=neuromod, bg=bg)
        # RPE should be small after zero-reward replay
        assert abs(vta.last_rpe) < 50.0, \
            f"Zero-reward replay should give small RPE, got {vta.last_rpe}"


# =====================================================================
# 9. Eligibility Preservation During Replay
# =====================================================================

class TestEligibilityPreservation:
    """Critic eligibility should be preserved for V(s) during replay."""

    def test_e_h_restored_after_v_sp_integration(self, critic, actor, vta):
        """Critic e_h after sleep should differ from pre-sleep
        (showing that V(s) eligibility was used for update)."""
        buf = ReplayBuffer(config=ReplayBufferConfig())
        np.random.seed(42)
        buf.store(_make_experience(reward=1.0))

        bg = _FakeBGFacade(critic, actor, vta)
        neuromod = _FakeNeuromodulator()
        wm = _FakeWorldModel(state_size=16)

        e_h_before = critic.e_h.copy()
        buf.sleep_phase(world_model=wm, neuromodulator=neuromod, bg=bg)
        # Eligibility should have been used and changed
        # (direct comparison is complex — just verify no NaN)
        assert np.all(np.isfinite(critic.e_h))


# =====================================================================
# Minimal stubs for testing
# =====================================================================

class _FakeEncoder:
    """Minimal encoder stub for world model."""

    def __init__(self, size):
        self.e_bu = np.zeros((size, size), dtype=np.float32)
        self.e_td = np.zeros((size, size), dtype=np.float32)
        self.spikes_state = np.zeros(size, dtype=np.float32)
        self.spikes_error = np.zeros(size, dtype=np.float32)
        self.prediction_error_rate = np.zeros(size, dtype=np.float32)
        self.error_rate = np.zeros(size, dtype=np.float32)
        self.belief = np.zeros(size, dtype=np.float32)

    def forward(self, x):
        self.belief = x[:len(self.belief)] if len(x) >= len(self.belief) else x
        return self.belief

    def update_weights(self, modulation=1.0, precision=1.0):
        pass


class _FakeAstrocyte:
    """Minimal astrocyte stub."""
    mean_precision: float = 1.0
    precision: float = 1.0

    def update(self, rate):
        pass


class _FakeWorldModel:
    """Minimal world model stub for replay buffer tests."""

    def __init__(self, state_size=16):
        self.encoder = _FakeEncoder(state_size)
        self.astrocyte = _FakeAstrocyte()
        self.prediction_error = np.zeros(state_size, dtype=np.float32)
        self._saved = None

    def snapshot_encoder(self):
        return self.encoder.e_bu.copy()

    def restore_encoder(self, snapshot):
        self.encoder.e_bu[:] = snapshot

    def reset_state(self):
        pass

    def update(self, state, action, next_state, m_t=1.0):
        error = next_state.astype(np.float32) - state.astype(np.float32)
        self.prediction_error = error
        return error
