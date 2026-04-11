"""Phase 0 verification tests — validate all hygiene fixes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.simulation_context import SimulationContext
from core.config import (
    AgentConfig,
    CompetitiveConfig,
    EpisodicMemoryConfig,
    HomeostaticConfig,
    NeuronConfig,
    OscillatorConfig,
    PredictiveCodingConfig,
    PyramidalConfig,
    ReplayBufferConfig,
    SequenceMemoryConfig,
    WorldModelConfig,
    ActiveInferenceConfig,
    BasalGangliaConfig,
)
from core.neuron import HomeostaticState


# =====================================================================
# Krok 0.1 — Magic numbers extracted to AgentConfig
# =====================================================================

class TestAgentConfig:
    def test_defaults_match_old_magic_numbers(self) -> None:
        cfg = AgentConfig()
        assert cfg.intrinsic_reward_weight == 0.1
        assert cfg.da_offset == 0.0
        assert cfg.td_clip == 50.0
        assert cfg.consolidation_midpoint == 0.7
        assert cfg.consolidation_steepness == 8.0
        assert cfg.consolidation_floor == 0.8
        assert cfg.noise_smoothing == 0.8
        assert cfg.min_exploration == 0.15
        assert cfg.sleep_gain_scale == 0.5
        assert cfg.sleep_gain_max == 2.5

    def test_frozen(self) -> None:
        cfg = AgentConfig()
        with pytest.raises(AttributeError):
            cfg.td_clip = 100.0  # type: ignore[misc]


# =====================================================================
# Krok 0.2 — No private attribute access from arena/
# =====================================================================

class TestNoPrivateAccess:
    def test_tonic_da_updated_per_step(self) -> None:
        """Tonic DA is updated as part of per-step update(), no episodic API."""
        from core.neuromodulator import NeuromodulatorSystem
        nm = NeuromodulatorSystem()
        assert nm.tonic_da == 0.0
        # Feed constant positive td_error — tonic_da should rise
        pe = np.array([0.5], dtype=np.float32)
        for _ in range(1000):
            nm.update(prediction_error=pe, td_error=1.0)
        assert nm.tonic_da > 0.01, "tonic_da should rise under constant RPE"

    def test_tonic_da_reset(self) -> None:
        """reset() restores tonic_da to baseline."""
        from core.neuromodulator import NeuromodulatorSystem
        nm = NeuromodulatorSystem()
        pe = np.array([0.5], dtype=np.float32)
        for _ in range(100):
            nm.update(prediction_error=pe, td_error=1.0)
        nm.reset()
        assert nm.tonic_da == nm.config.baseline_tonic_da


# =====================================================================
# Krok 0.3 — HomeostaticState shared logic
# =====================================================================

class TestHomeostaticState:
    def test_init_shapes(self) -> None:
        cfg = HomeostaticConfig()
        hs = HomeostaticState(num_neurons=10, v_thresh=-55.0, config=cfg)
        assert hs.v_thresh_adaptive.shape == (10,)
        assert hs.avg_rate.shape == (10,)
        assert hs.is_dark_matter.shape == (10,)

    def test_dark_matter_higher_threshold(self) -> None:
        cfg = HomeostaticConfig(dark_matter_ratio=0.5, dark_matter_thresh_offset=5.0)
        hs = HomeostaticState(num_neurons=100, v_thresh=-55.0, config=cfg)
        dm = hs.is_dark_matter
        assert dm.sum() == 50
        assert np.allclose(hs.v_thresh_adaptive[dm], -50.0)
        assert np.allclose(hs.v_thresh_adaptive[~dm], -55.0)

    def test_update_raises_threshold_for_overactive(self) -> None:
        cfg = HomeostaticConfig(target_rate=0.05, dark_matter_ratio=0.0)
        hs = HomeostaticState(num_neurons=5, v_thresh=-55.0, config=cfg)
        always_firing = np.ones(5, dtype=bool)
        for _ in range(200):
            hs.update(always_firing)
        # Threshold should increase for overactive neurons (rate 1.0 >> target 0.05)
        assert np.all(hs.v_thresh_adaptive > -55.0)

    def test_effective_threshold_ne_drop(self) -> None:
        cfg = HomeostaticConfig(ne_thresh_drop=5.0, dark_matter_ratio=0.0)
        hs = HomeostaticState(num_neurons=3, v_thresh=-55.0, config=cfg)
        eff = hs.effective_threshold(ne_level=1.0)
        assert np.allclose(eff, -60.0)

    def test_reset_preserves_dark_matter(self) -> None:
        cfg = HomeostaticConfig(dark_matter_ratio=0.5, dark_matter_thresh_offset=5.0)
        np.random.seed(42)
        hs = HomeostaticState(num_neurons=10, v_thresh=-55.0, config=cfg)
        dm_before = hs.is_dark_matter.copy()
        # Perturb, then reset
        hs.v_thresh_adaptive.fill(-40.0)
        hs.reset(v_thresh=-55.0)
        assert np.array_equal(hs.is_dark_matter, dm_before)
        assert np.allclose(hs.v_thresh_adaptive[dm_before], -50.0)
        assert np.allclose(hs.v_thresh_adaptive[~dm_before], -55.0)


# =====================================================================
# Krok 0.4 — Config validation rejects bad values
# =====================================================================

class TestConfigValidation:
    def test_neuron_config_rejects_neg_tau(self) -> None:
        with pytest.raises(AssertionError, match="tau_m"):
            NeuronConfig(tau_m=-1.0)

    def test_competitive_config_rejects_bad_sparsity(self) -> None:
        with pytest.raises(AssertionError, match="target_sparsity"):
            CompetitiveConfig(target_sparsity=0.0)
        with pytest.raises(AssertionError, match="target_sparsity"):
            CompetitiveConfig(target_sparsity=1.0)

    def test_oscillator_config_rejects_bad_freq_range(self) -> None:
        with pytest.raises(AssertionError, match="theta"):
            OscillatorConfig(theta_freq_hz=2.0, theta_min_hz=4.0)

    def test_oscillator_config_rejects_bad_pac(self) -> None:
        with pytest.raises(AssertionError, match="pac_depth"):
            OscillatorConfig(pac_depth=1.5)

    def test_episodic_config_rejects_zero_capacity(self) -> None:
        with pytest.raises(AssertionError, match="capacity"):
            EpisodicMemoryConfig(capacity=0)

    def test_sequence_config_rejects_neg_lr(self) -> None:
        with pytest.raises(AssertionError, match="learning_rate"):
            SequenceMemoryConfig(learning_rate=-0.01)

    def test_replay_buffer_rejects_bad_fractions(self) -> None:
        with pytest.raises(AssertionError, match="replay fractions"):
            ReplayBufferConfig(sws_replay_fraction=0.8, rem_replay_fraction=0.5)

    def test_world_model_rejects_zero_hidden(self) -> None:
        with pytest.raises(AssertionError, match="hidden_size"):
            WorldModelConfig(hidden_size=0)

    def test_active_inference_rejects_bad_method(self) -> None:
        with pytest.raises(AssertionError, match="uncertainty_method"):
            ActiveInferenceConfig(uncertainty_method="wrong")

    def test_active_inference_accepts_valid_methods(self) -> None:
        for m in ("novelty", "entropy", "variance"):
            cfg = ActiveInferenceConfig(uncertainty_method=m)
            assert cfg.uncertainty_method == m

    def test_agent_config_rejects_bad_floor(self) -> None:
        with pytest.raises(AssertionError, match="consolidation_floor"):
            AgentConfig(consolidation_floor=0.0)

    def test_basal_ganglia_rejects_bad_gamma(self) -> None:
        with pytest.raises(AssertionError, match="gamma"):
            BasalGangliaConfig(gamma=0.0)

    def test_predictive_coding_rejects_neg_lr(self) -> None:
        with pytest.raises(AssertionError, match="feedback_learning_rate"):
            PredictiveCodingConfig(feedback_learning_rate=-0.01)

    def test_pyramidal_config_valid_defaults(self) -> None:
        cfg = PyramidalConfig()
        assert cfg.feedback_strength == 0.5

    def test_all_configs_accept_defaults(self) -> None:
        """Every config class should instantiate successfully with defaults."""
        for cls in [
            NeuronConfig, CompetitiveConfig, PredictiveCodingConfig,
            PyramidalConfig, OscillatorConfig, EpisodicMemoryConfig,
            SequenceMemoryConfig, ReplayBufferConfig, WorldModelConfig,
            ActiveInferenceConfig, AgentConfig, BasalGangliaConfig,
        ]:
            cfg = cls()
            assert cfg is not None


# =====================================================================
# Krok 0.6 — Welford's algorithm (Bessel correction)
# =====================================================================

class TestWelford:
    def test_welford_unbiased_variance(self) -> None:
        """After feeding known data, running variance should match np.var(ddof=1)."""
        from arena.gym_env import GymEnv

        # Create a simple env — we just need _update_running_stats
        env = GymEnv("CartPole-v1", normalize=True)
        # Reset to init stats
        assert env._obs_count == 0

        rng = np.random.default_rng(42)
        data = rng.normal(loc=2.0, scale=3.0, size=(200, env._obs_dim))

        for obs in data:
            env._update_running_stats(obs.astype(np.float64))

        expected_mean = data.mean(axis=0)
        expected_std = data.std(axis=0, ddof=1)

        np.testing.assert_allclose(env._obs_mean, expected_mean, atol=1e-10)
        np.testing.assert_allclose(env._obs_std, expected_std, atol=1e-10)

    def test_welford_single_sample_no_nan(self) -> None:
        """A single observation should not produce NaN."""
        from arena.gym_env import GymEnv

        env = GymEnv("CartPole-v1", normalize=True)
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        env._update_running_stats(obs)
        assert not np.any(np.isnan(env._obs_std))
        # std should still be 1.0 (initial) since we need >= 2 samples
        np.testing.assert_allclose(env._obs_std, np.ones(4))


# =====================================================================
# Krok 0.7 — PyramidalLayer.generate_prediction uses config
# =====================================================================

class TestPyramidalFeedbackStrength:
    def test_generate_prediction_uses_config_strength(self) -> None:
        from core.pyramidal_neuron import PyramidalLayer

        cfg = PyramidalConfig(feedback_strength=0.0)
        layer = PyramidalLayer(
            num_inputs=4,
            num_neurons=4,
            pyr_cfg=cfg,
        )
        # generate_prediction uses has_spiked @ w_apical.T * feedback_strength
        layer.has_spiked = np.ones(4, dtype=bool)  # Force spikes
        pred = layer.generate_prediction()
        # With feedback_strength=0 the prediction should be zero
        np.testing.assert_allclose(pred, 0.0, atol=1e-7)

    def test_nonzero_feedback_strength(self) -> None:
        from core.pyramidal_neuron import PyramidalLayer

        cfg = PyramidalConfig(feedback_strength=1.0)
        layer = PyramidalLayer(
            num_inputs=4,
            num_neurons=4,
            pyr_cfg=cfg,
        )
        layer.has_spiked = np.ones(4, dtype=bool)  # Force spikes
        pred = layer.generate_prediction()
        # With strength=1.0 and spiking activity, prediction should be non-zero
        assert np.any(np.abs(pred) > 0)
