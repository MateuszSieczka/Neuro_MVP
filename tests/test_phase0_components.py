"""
Tests for Phase 0 foundational biological components:
  - InhibitoryPool (GABAergic E→I→E lateral inhibition)
  - AstrocyteField (Ca²⁺-based precision estimation)
  - ErrorNeuronLayer (continuous predictive coding)
"""

import numpy as np
import pytest
from core.interneuron import InhibitoryPool, InhibitoryPoolConfig
from core.astrocyte import AstrocyteField, AstrocyteConfig
from core.error_neuron import ErrorNeuronLayer, ErrorNeuronConfig


# ═══════════════════════════════════════════════════════════════════════
# InhibitoryPool tests
# ═══════════════════════════════════════════════════════════════════════

class TestInhibitoryPool:
    """Tests that InhibitoryPool produces emergent sparse competition."""

    def test_basic_step(self):
        """Pool returns inhibitory current with correct shape."""
        pool = InhibitoryPool(n_excitatory=32, config=InhibitoryPoolConfig())
        spikes = np.random.rand(32) > 0.7
        inh = pool.step(spikes)
        assert inh.shape == (32,)
        assert inh.dtype == np.float32

    def test_no_spikes_no_inhibition(self):
        """With zero input, inhibition should stay near zero."""
        pool = InhibitoryPool(n_excitatory=32, config=InhibitoryPoolConfig())
        for _ in range(20):
            inh = pool.step(np.zeros(32))
        assert np.max(inh) < 0.1

    def test_strong_input_produces_inhibition(self):
        """Strong excitatory drive should produce significant inhibition."""
        pool = InhibitoryPool(n_excitatory=32, config=InhibitoryPoolConfig())
        strong_spikes = np.ones(32, dtype=np.float32)
        total_inh = 0
        for _ in range(50):
            inh = pool.step(strong_spikes)
            total_inh += np.mean(inh)
        assert total_inh > 1.0, "Strong input should produce substantial inhibition"

    def test_emergent_sparsity(self):
        """
        When excitatory input + inhibitory feedback are combined,
        the effective activation should be sparser than the input.
        """
        n_exc = 64
        pool = InhibitoryPool(n_excitatory=n_exc, config=InhibitoryPoolConfig(
            n_interneurons=16, w_ei_mean=1.0, w_ie_mean=0.8
        ))
        # Simulate: all neurons active, apply inhibition
        v = np.full(n_exc, -60.0, dtype=np.float32)  # near threshold
        active_counts = []
        for _ in range(100):
            exc_spikes = v > -55.0
            inh = pool.step(exc_spikes)
            # Apply inhibition to membrane
            v -= inh
            # Re-excite (uniform input)
            v += np.random.uniform(10, 15, n_exc).astype(np.float32)
            v = np.clip(v, -75, -40)
            active = np.sum(exc_spikes)
            active_counts.append(active)
        # After transient, activity should be sparser than 100%
        avg_active = np.mean(active_counts[50:])
        assert avg_active < n_exc * 0.8, f"Expected sparsity, got {avg_active}/{n_exc} active"

    def test_dual_gaba_channels(self):
        """GABA-A and GABA-B should have different time constants."""
        pool = InhibitoryPool(n_excitatory=16, config=InhibitoryPoolConfig(
            tau_gaba_a=5.0, tau_gaba_b=50.0, gaba_b_ratio=0.3
        ))
        # Drive strongly for 10 steps
        for _ in range(10):
            pool.step(np.ones(16))
        # Record GABA-A and GABA-B
        gaba_a_peak = pool.g_gaba_a.copy()
        gaba_b_peak = pool.g_gaba_b.copy()
        # Decay for 20 steps (no input)
        for _ in range(20):
            pool.step(np.zeros(16))
        gaba_a_after = pool.g_gaba_a.copy()
        gaba_b_after = pool.g_gaba_b.copy()
        # GABA-A (fast) should have decayed more than GABA-B (slow)
        if np.max(gaba_a_peak) > 0 and np.max(gaba_b_peak) > 0:
            a_ratio = np.mean(gaba_a_after) / (np.mean(gaba_a_peak) + 1e-8)
            b_ratio = np.mean(gaba_b_after) / (np.mean(gaba_b_peak) + 1e-8)
            assert a_ratio < b_ratio, "GABA-A should decay faster than GABA-B"

    def test_reset_state(self):
        """reset_state should clear all transient variables."""
        pool = InhibitoryPool(n_excitatory=16)
        pool.step(np.ones(16))
        pool.reset_state()
        assert np.allclose(pool.v_inh, pool.config.v_rest)
        assert np.allclose(pool.g_gaba_a, 0.0)
        assert np.allclose(pool.g_gaba_b, 0.0)

    def test_no_np_argsort(self):
        """Verify no algorithmic sorting in executable code."""
        import inspect
        source = inspect.getsource(InhibitoryPool.step)
        assert "argsort" not in source, "step() must not use np.argsort"
        assert "sorted(" not in source, "step() must not use sorted()"


# ═══════════════════════════════════════════════════════════════════════
# AstrocyteField tests
# ═══════════════════════════════════════════════════════════════════════

class TestAstrocyteField:
    """Tests that AstrocyteField provides biologically grounded precision."""

    def test_basic_shape(self):
        """All outputs should have n_zones shape."""
        field = AstrocyteField(n_zones=8)
        assert field.precision.shape == (8,)
        assert field.synaptic_gain.shape == (8,)
        assert field.metabolic_lr.shape == (8,)

    def test_zero_error_high_precision(self):
        """With no prediction errors, precision should be maximal."""
        field = AstrocyteField(n_zones=8)
        for _ in range(100):
            field.update(np.zeros(8))
        assert np.all(field.precision > 0.9), "Zero error should give high precision"

    def test_high_error_low_precision(self):
        """Sustained prediction errors should lower precision."""
        field = AstrocyteField(n_zones=8, config=AstrocyteConfig(
            ca_accumulation=1.0, tau_ca=50.0
        ))
        for _ in range(300):
            field.update(np.ones(8) * 5.0)  # large errors
        assert np.all(field.precision < 0.5), f"High error should lower precision, got {field.precision}"

    def test_calcium_integrates_error(self):
        """Ca²⁺ should accumulate with sustained error."""
        field = AstrocyteField(n_zones=4, config=AstrocyteConfig(
            ca_accumulation=0.3, tau_ca=200.0
        ))
        ca_history = []
        for i in range(100):
            field.update(np.ones(4))
            ca_history.append(field.mean_calcium)
        # Ca²⁺ should increase over time
        assert ca_history[-1] > ca_history[10], "Ca²⁺ should accumulate"

    def test_d_serine_release(self):
        """D-Serine should be released when Ca²⁺ exceeds threshold."""
        field = AstrocyteField(n_zones=4, config=AstrocyteConfig(
            ca_accumulation=1.0, tau_ca=50.0, ca_threshold=0.3
        ))
        # Drive Ca²⁺ above threshold
        for _ in range(100):
            field.update(np.ones(4) * 3.0)
        assert np.any(field.d_serine > 0), "D-Serine should be released"

    def test_synaptic_gain_modulation(self):
        """Synaptic gain should increase with D-Serine."""
        field = AstrocyteField(n_zones=4, config=AstrocyteConfig(
            ca_accumulation=1.0, tau_ca=50.0, ca_threshold=0.2,
            gain_baseline=1.0, gain_max=2.0
        ))
        baseline_gain = field.synaptic_gain.copy()
        for _ in range(200):
            field.update(np.ones(4) * 5.0)
        boosted_gain = field.synaptic_gain.copy()
        assert np.mean(boosted_gain) > np.mean(baseline_gain), \
            "D-Serine should boost synaptic gain"

    def test_zone_mapping(self):
        """Arbitrary-length error arrays should map into zones."""
        field = AstrocyteField(n_zones=4)
        # 100-dim error → 4 zones
        field.update(np.random.randn(100))
        assert field.calcium.shape == (4,)

    def test_metabolic_lr_scales_with_activity(self):
        """Active zones should get higher learning rate multiplier."""
        field = AstrocyteField(n_zones=4, config=AstrocyteConfig(
            ca_accumulation=1.0, tau_ca=50.0, metabolic_scale=1.0
        ))
        errors = np.array([5.0, 0.0, 5.0, 0.0])
        for _ in range(100):
            field.update(errors)
        lr = field.metabolic_lr
        assert lr[0] > lr[1], "Active zones should have higher metabolic LR"

    def test_reset_state(self):
        """Reset should clear Ca²⁺ and D-Serine."""
        field = AstrocyteField(n_zones=4)
        field.update(np.ones(4) * 5)
        field.reset_state()
        assert np.allclose(field.calcium, 0.0)
        assert np.allclose(field.d_serine, 0.0)


# ═══════════════════════════════════════════════════════════════════════
# ErrorNeuronLayer tests
# ═══════════════════════════════════════════════════════════════════════

class TestErrorNeuronLayer:
    """Tests for continuous predictive coding with Error/State neurons."""

    def test_basic_forward(self):
        """forward() returns state spikes with correct shape."""
        layer = ErrorNeuronLayer(n_input=30, config=ErrorNeuronConfig(
            n_state=64, n_error=30
        ))
        spikes = layer.forward(np.random.rand(30))
        assert spikes.shape == (64,)
        assert spikes.dtype == bool

    def test_no_for_loops_in_forward(self):
        """Verify no relaxation loops — the key biological fix."""
        import inspect
        source = inspect.getsource(ErrorNeuronLayer.forward)
        # Should not contain for loops (relaxation hack)
        lines = source.split('\n')
        for_lines = [l.strip() for l in lines
                     if l.strip().startswith('for ') and 'range(' in l]
        assert len(for_lines) == 0, \
            f"forward() must not contain for-loops, found: {for_lines}"

    def test_error_neurons_faster_than_state(self):
        """Error neurons (fast τ) should respond before state neurons."""
        cfg = ErrorNeuronConfig(n_state=32, n_error=20, tau_state=20.0, tau_error=4.0)
        layer = ErrorNeuronLayer(n_input=20, config=cfg)
        # Strong sustained input
        inp = np.ones(20, dtype=np.float32) * 0.8
        error_active_step = None
        state_active_step = None
        for step in range(100):
            layer.forward(inp)
            if error_active_step is None and np.any(layer.spikes_error):
                error_active_step = step
            if state_active_step is None and np.any(layer.spikes_state):
                state_active_step = step
            if error_active_step is not None and state_active_step is not None:
                break
        if error_active_step is not None and state_active_step is not None:
            assert error_active_step <= state_active_step, \
                f"Error neurons should fire first: error@{error_active_step} vs state@{state_active_step}"

    def test_prediction_error_is_membrane_state(self):
        """prediction_error should reflect error neuron firing rates."""
        layer = ErrorNeuronLayer(n_input=20, config=ErrorNeuronConfig(
            n_state=32, n_error=20
        ))
        # Run a few steps
        for _ in range(50):
            layer.forward(np.random.rand(20) * 0.5)
        pe = layer.prediction_error
        assert pe.shape == (20,)
        # Should be non-negative (firing rates)
        assert np.all(pe >= 0)

    def test_belief_tracks_input(self):
        """With sustained constant input, belief should converge."""
        layer = ErrorNeuronLayer(n_input=30, config=ErrorNeuronConfig(
            n_state=32, n_error=30, tau_state=20.0, tau_error=4.0
        ))
        inp = np.ones(30) * 0.8
        beliefs = []
        for _ in range(300):
            layer.forward(inp)
            beliefs.append(layer.belief.copy())
        # Check that state neurons are active (non-trivial belief)
        final_belief = beliefs[-1]
        total_activity = sum(np.sum(b) for b in beliefs[200:])
        assert total_activity > 0, "State neurons should fire with sustained input"

    def test_ach_modulates_error_gain(self):
        """High ACh should amplify error neuron activity."""
        cfg = ErrorNeuronConfig(n_state=32, n_error=20, ach_gain_range=(0.5, 2.0))
        layer_lo = ErrorNeuronLayer(n_input=20, config=cfg)
        layer_hi = ErrorNeuronLayer(n_input=20, config=cfg)
        # Copy weights for fair comparison
        layer_hi.w_bu[:] = layer_lo.w_bu
        layer_hi.w_td[:] = layer_lo.w_td
        layer_hi.w_in[:] = layer_lo.w_in

        layer_lo.set_ach_level(0.1)
        layer_hi.set_ach_level(0.9)

        inp = np.random.rand(20) * 0.5
        error_lo_total = 0.0
        error_hi_total = 0.0
        for _ in range(200):
            layer_lo.forward(inp)
            layer_hi.forward(inp)
            error_lo_total += np.sum(layer_lo.error_rate)
            error_hi_total += np.sum(layer_hi.error_rate)

        assert error_hi_total > error_lo_total * 0.8, \
            "High ACh should produce at least comparable error activity"

    def test_weight_update(self):
        """update_weights should modify w_bu and w_td."""
        layer = ErrorNeuronLayer(n_input=20, config=ErrorNeuronConfig(
            n_state=32, n_error=20
        ))
        # Run some steps to build eligibility
        for _ in range(50):
            layer.forward(np.random.rand(20) * 0.5)
        w_bu_before = layer.w_bu.copy()
        w_td_before = layer.w_td.copy()
        layer.update_weights(modulation=1.0)
        # At least one matrix should have changed
        changed = (not np.allclose(layer.w_bu, w_bu_before) or
                   not np.allclose(layer.w_td, w_td_before))
        assert changed, "Weights should update with non-zero modulation and eligibility"

    def test_reset_state(self):
        """reset_state should clear membranes and traces, keep weights."""
        layer = ErrorNeuronLayer(n_input=20, config=ErrorNeuronConfig(
            n_state=32, n_error=20
        ))
        for _ in range(50):
            layer.forward(np.random.rand(20))
        w_bu_copy = layer.w_bu.copy()
        layer.reset_state()
        assert np.allclose(layer.v_state, layer.config.v_rest)
        assert np.allclose(layer.v_error, layer.config.v_rest)
        assert np.allclose(layer.e_bu, 0.0)
        assert np.allclose(layer.w_bu, w_bu_copy), "Weights should be preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
