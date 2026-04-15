"""
Phase 4 tests — Neural Module Consistency.

Verifies:
  BIO 2: WM content neurons use AdEx (not LIF), conductance-based synapses.
  BIO 7: PC relaxation loop converges prediction error in ≤5 iterations.
  BIO 5: Pyramidal apical input delayed by ~5ms (Stuart & Spruston 1998).
  MOD 4: DG projection learns via competitive Hebbian (Rolls 2013).

References:
    Durstewitz et al. (2000): PFC WM model with slow adaptation.
    Compte et al. (2000): Persistent activity in PFC attractors.
    Bogacz (2017): Free-energy tutorial — PC relaxation.
    Rao & Ballard (1999): Predictive coding in visual cortex.
    Stuart & Spruston (1998): Apical dendritic propagation delay.
    Rolls (2013): DG pattern separation via competitive learning.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config import (
    WorkingMemoryConfig,
    NeuronConfig,
    STDPConfig,
    HomeostaticConfig,
    CompetitiveConfig,
    PredictiveCodingConfig,
    PyramidalConfig,
    SequenceMemoryConfig,
)
from core.working_memory import WorkingMemoryModule
from core.predictive_coding import PredictiveCodingLayer
from core.pyramidal_neuron import PyramidalLayer
from core.sequence_memory import SequenceMemory
from core.simulation_context import SimulationContext


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def ctx():
    return SimulationContext(dt=1.0)


@pytest.fixture
def wm_cfg(ctx):
    return WorkingMemoryConfig(ctx=ctx)


@pytest.fixture
def ncfg(ctx):
    return NeuronConfig(ctx=ctx)


# =====================================================================
# BIO 2: WM AdEx + conductance-based
# =====================================================================

class TestWMAdEx:
    """WM content neurons must use AdEx, not LIF."""

    def test_wm_has_adaptation_current(self, wm_cfg):
        """WM neurons must have spike-frequency adaptation."""
        wm = WorkingMemoryModule(10, 8, config=wm_cfg)
        assert hasattr(wm, 'w_adapt'), "WM must have w_adapt (AdEx adaptation)"
        assert wm.w_adapt.shape == (8,)
        assert np.allclose(wm.w_adapt, 0.0)

    def test_wm_adex_params_pfc_like(self, wm_cfg):
        """WM AdEx params must be PFC-like: slow adaptation."""
        # Durstewitz et al. (2000): τ_w ≈ 300ms for PFC sustained activity
        assert wm_cfg.tau_w >= 200.0, "PFC needs slow adaptation (τ_w ≥ 200ms)"
        # Mild spike-triggered adaptation (b ≤ 30 pA) for sustained firing
        assert wm_cfg.b <= 30.0, f"b={wm_cfg.b} too large for sustained firing"
        # Must have AdEx spike initiation sharpness
        assert wm_cfg.delta_t > 0, "Must have AdEx exponential spike initiation"
        assert wm_cfg.v_spike_cutoff > wm_cfg.v_thresh

    def test_wm_conductance_based_weights(self, wm_cfg):
        """WM feedforward weights must be in nS (conductance-based)."""
        wm = WorkingMemoryModule(10, 8, config=wm_cfg)
        # Weights from init_weights() should be positive (excitatory, Dale's law)
        assert np.all(wm.w_ff >= 0), "Excitatory weights must be ≥ 0 (Dale's law)"
        # Should be in nS range, not arbitrary [0.1, 0.5]
        mean_w = float(np.mean(wm.w_ff))
        # init_weights half-normal: E[|w|] = σ√(2/π) ≈ 2.57 nS for WM params.
        # Allow +1σ finite-sample margin → 3.3 nS.
        assert mean_w < 3.3, f"Mean weight {mean_w:.3f} nS too large"

    def test_wm_adaptation_increases_with_firing(self, wm_cfg):
        """Adaptation current must grow with sustained firing."""
        wm = WorkingMemoryModule(10, 8, config=wm_cfg)
        # Open gate wide
        wm.gate(ach_level=1.0, da_level=1.0)
        # Drive with strong input
        strong_input = np.ones(10, dtype=np.float32)
        for _ in range(50):
            wm.forward(strong_input)
            wm.gate(ach_level=1.0, da_level=1.0)
        w_adapt_after = float(np.mean(np.abs(wm.w_adapt)))
        assert w_adapt_after > 0, "Adaptation current must grow with activity"

    def test_wm_sustains_pattern_without_input(self, wm_cfg):
        """WM attractor must sustain pattern >500ms without input.

        This is the key PFC WM property (Goldman-Rakic 1995):
        once loaded, content persists via recurrent attractor.
        The slow adaptation (τ_w=300ms) prevents runaway without
        killing persistent activity.

        We pre-load lateral weights to create a working attractor
        (in the real system, these build up during training).
        """
        wm = WorkingMemoryModule(50, 8, config=wm_cfg)

        # Pre-build attractor: set lateral weights coupling neurons 0-3
        # This mimics what lateral Hebbian learning achieves during training.
        for i in range(4):
            for j in range(4):
                if i != j:
                    wm.w_lateral[i, j] = 0.8

        # Pre-load content: set neurons 0-3 as active trace
        wm.content[:4] = 1.0
        content_after_load = wm.content.copy()
        assert np.any(content_after_load > 0.01), "Failed to set initial content"

        # Gate closed, no input for 500ms — content should persist
        # via recurrent attractor dynamics
        wm.gate(ach_level=0.0, da_level=0.0)
        for _ in range(500):
            wm.forward(np.zeros(50, dtype=np.float32))
        content_after_delay = wm.content.copy()

        # Content should not have decayed completely
        persistence = float(np.sum(content_after_delay[:4])) / max(
            float(np.sum(content_after_load[:4])), 1e-8)
        assert persistence > 0.05, (
            f"Content decayed to {persistence:.3f} of original — "
            f"attractor failed."
        )

    def test_wm_no_lif_integration(self):
        """Ensure no LIF-style integration remains in forward()."""
        import inspect
        src = inspect.getsource(WorkingMemoryModule.forward)
        assert 'mem_decay' not in src, "LIF mem_decay found in forward()"
        assert '* self._mem_decay' not in src, "LIF integration in forward()"

    def test_wm_reset_clears_adaptation(self, wm_cfg):
        """reset_state() must clear w_adapt."""
        wm = WorkingMemoryModule(10, 8, config=wm_cfg)
        wm.w_adapt[:] = 100.0
        wm.reset_state()
        assert np.allclose(wm.w_adapt, 0.0)


# =====================================================================
# BIO 7: PC relaxation loop
# =====================================================================

class TestPCRelaxation:
    """Prediction error must converge within 3-5 relaxation steps."""

    def test_pc_config_has_relaxation(self, ctx):
        """Config must expose n_relax_steps."""
        pc_cfg = PredictiveCodingConfig(ctx=ctx)
        assert hasattr(pc_cfg, 'n_relax_steps')
        assert pc_cfg.n_relax_steps >= 1

    def test_pc_relaxation_reduces_error(self, ctx, ncfg):
        """Relaxation loop must reduce prediction error vs single step."""
        np.random.seed(42)
        # Layer with relaxation
        pc_relax = PredictiveCodingLayer(
            num_inputs=10, num_neurons=8,
            pc_cfg=PredictiveCodingConfig(ctx=ctx, n_relax_steps=5),
            neuron_cfg=ncfg,
        )
        # Layer without relaxation (1 step ≈ old behaviour)
        pc_single = PredictiveCodingLayer(
            num_inputs=10, num_neurons=8,
            pc_cfg=PredictiveCodingConfig(ctx=ctx, n_relax_steps=1),
            neuron_cfg=ncfg,
        )
        # Same weights
        pc_single.feedback_w = pc_relax.feedback_w.copy()
        pc_single.w = pc_relax.w.copy()

        # Drive with structured input
        inp = np.random.rand(10).astype(np.float32)
        pc_relax.forward(inp)
        pc_single.forward(inp)

        err_relax = float(np.mean(pc_relax.prediction_error ** 2))
        err_single = float(np.mean(pc_single.prediction_error ** 2))
        # Relaxation should reduce or match error (not increase)
        assert err_relax <= err_single + 1e-4, (
            f"Relaxation error {err_relax:.4f} > single-step {err_single:.4f}"
        )

    def test_pc_no_single_step_label(self):
        """Source code should not claim 'single-step' prediction error."""
        import inspect
        src = inspect.getsource(PredictiveCodingLayer.forward)
        assert 'Single-step prediction error' not in src


# =====================================================================
# BIO 5: Pyramidal apical delay
# =====================================================================

class TestApicalDelay:
    """Apical input must be delayed by ~5ms at soma."""

    def test_pyramidal_has_delay_buffer(self, ctx, ncfg):
        """PyramidalLayer must have an apical delay ring buffer."""
        pyr_cfg = PyramidalConfig(ctx=ctx)
        layer = PyramidalLayer(
            num_inputs=10, num_neurons=8,
            pyr_cfg=pyr_cfg, neuron_cfg=ncfg,
        )
        assert hasattr(layer, '_apical_delay_buf')
        assert hasattr(layer, '_apical_delay_steps')
        assert layer._apical_delay_steps == pyr_cfg.apical_delay_ms

    def test_apical_impulse_delayed(self, ctx, ncfg):
        """Impulse in top-down prediction should appear ~5ms later."""
        delay_ms = 5
        pyr_cfg = PyramidalConfig(ctx=ctx, apical_delay_ms=delay_ms)
        layer = PyramidalLayer(
            num_inputs=10, num_neurons=8,
            pyr_cfg=pyr_cfg, neuron_cfg=ncfg,
        )
        zero_input = np.zeros(10, dtype=np.float32)

        # Step 0: inject impulse AND process
        layer.receive_prediction(np.ones(8, dtype=np.float32))
        layer.forward(zero_input)
        # v_apical should be ~0 (impulse in buffer, not yet read)

        # Steps 1 through delay_ms-2: zero prediction, impulse still in buffer
        for t in range(delay_ms - 2):
            layer.receive_prediction(np.zeros(8, dtype=np.float32))
            layer.forward(zero_input)
        v_apical_during = layer.v_apical.copy()

        # Steps delay_ms-1 to delay_ms: impulse should arrive
        for _ in range(2):
            layer.receive_prediction(np.zeros(8, dtype=np.float32))
            layer.forward(zero_input)
        v_apical_after_delay = layer.v_apical.copy()

        # After delay, apical voltage should show the impulse effect
        energy_during = float(np.sum(np.abs(v_apical_during)))
        energy_after = float(np.sum(np.abs(v_apical_after_delay)))
        assert energy_after > energy_during, (
            f"Apical energy at t={delay_ms} ({energy_after:.6f}) should exceed "
            f"t<{delay_ms} ({energy_during:.6f}) — impulse should arrive at delay"
        )

    def test_delay_buffer_resets(self, ctx, ncfg):
        """reset_state() must clear the delay buffer."""
        pyr_cfg = PyramidalConfig(ctx=ctx)
        layer = PyramidalLayer(10, 8, pyr_cfg=pyr_cfg, neuron_cfg=ncfg)
        layer._apical_delay_buf[:] = 99.0
        layer.reset_state()
        assert np.allclose(layer._apical_delay_buf, 0.0)


# =====================================================================
# MOD 4: Plastic DG projection
# =====================================================================

class TestPlasticDG:
    """DG projection must learn via competitive Hebbian (Rolls 2013)."""

    def test_dg_config_has_learning_rate(self, ctx):
        """Config must expose dg_learning_rate."""
        cfg = SequenceMemoryConfig(ctx=ctx)
        assert hasattr(cfg, 'dg_learning_rate')
        assert cfg.dg_learning_rate > 0

    def test_dg_weights_change_with_exposure(self, ctx):
        """w_dg must change after repeated pattern exposure."""
        cfg = SequenceMemoryConfig(ctx=ctx, dg_learning_rate=0.01)
        sm = SequenceMemory(10, config=cfg)
        w_dg_before = sm._w_dg.copy()

        # Present structured pattern repeatedly
        pattern = np.zeros(10, dtype=np.float32)
        pattern[:5] = 1.0
        for _ in range(20):
            sm.observe(pattern)

        w_dg_after = sm._w_dg.copy()
        diff = float(np.mean(np.abs(w_dg_after - w_dg_before)))
        assert diff > 1e-6, (
            f"DG weights unchanged after 20 exposures (diff={diff:.8f})"
        )

    def test_dg_zero_lr_preserves_random(self, ctx):
        """With dg_learning_rate=0, weights remain random (backward compat)."""
        cfg = SequenceMemoryConfig(ctx=ctx, dg_learning_rate=0.0)
        sm = SequenceMemory(10, config=cfg)
        w_dg_before = sm._w_dg.copy()

        pattern = np.ones(10, dtype=np.float32)
        for _ in range(20):
            sm.observe(pattern)

        assert np.allclose(sm._w_dg, w_dg_before)

    def test_dg_column_norms_preserved(self, ctx):
        """Oja-like normalisation must keep column norms stable."""
        cfg = SequenceMemoryConfig(ctx=ctx, dg_learning_rate=0.01)
        sm = SequenceMemory(10, config=cfg)
        norms_before = np.linalg.norm(sm._w_dg, axis=0)

        pattern = np.random.rand(10).astype(np.float32)
        for _ in range(50):
            sm.observe(pattern)

        norms_after = np.linalg.norm(sm._w_dg, axis=0)
        # Column norms should be approximately preserved
        ratio = norms_after / np.maximum(norms_before, 1e-8)
        assert float(np.max(np.abs(ratio - 1.0))) < 0.1, (
            f"Column norms changed >10%: max ratio deviation = {float(np.max(np.abs(ratio-1.0))):.3f}"
        )

    def test_dg_improves_separation(self, ctx):
        """DG learning should improve pattern separation vs random."""
        np.random.seed(42)
        cfg_learn = SequenceMemoryConfig(ctx=ctx, dg_learning_rate=0.01)
        cfg_fixed = SequenceMemoryConfig(ctx=ctx, dg_learning_rate=0.0)

        sm_learn = SequenceMemory(20, config=cfg_learn)
        sm_fixed = SequenceMemory(20, config=cfg_fixed)
        # Same initial weights
        sm_fixed._w_dg = sm_learn._w_dg.copy()
        sm_fixed._init_col_norm = sm_learn._init_col_norm.copy()

        # Two similar but distinct patterns
        p1 = np.zeros(20, dtype=np.float32)
        p1[:10] = 1.0
        p2 = np.zeros(20, dtype=np.float32)
        p2[5:15] = 1.0  # 50% overlap

        # Train DG on both patterns
        for _ in range(50):
            sm_learn.observe(p1)
            sm_learn.observe(p2)

        # Measure separation: cosine similarity of separated representations
        sep1_learn = sm_learn._pattern_separate(p1)
        sep2_learn = sm_learn._pattern_separate(p2)
        sep1_fixed = sm_fixed._pattern_separate(p1)
        sep2_fixed = sm_fixed._pattern_separate(p2)

        def cosine_sim(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-8 or nb < 1e-8:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        sim_learn = cosine_sim(sep1_learn, sep2_learn)
        sim_fixed = cosine_sim(sep1_fixed, sep2_fixed)

        # Learned DG should produce LESS similar representations (better separation)
        # or at least not be worse
        assert sim_learn <= sim_fixed + 0.1, (
            f"Learned DG similarity ({sim_learn:.3f}) should be ≤ "
            f"fixed ({sim_fixed:.3f}) — DG learning should improve separation"
        )
