"""
Phase 7 — Cleanup & Derivation tests.

Verifies:
  CLN 1: Dead code removed (encode_value, dead _ne_level)
  CLN 3: Astrocyte ATP cost uses raw rates (no sqrt(rate²) roundabout)
  CLN 4: Synapse sign convention (positive = depolarizing)
  MOD 2: ACC stagnation via error neuron persistence (no CV)
  MOD 3: Seizure brake uses astrocyte ATP pathway (no hard v reset)
  Magic numbers: all traceable to biophysical parameters or cited papers
"""

from __future__ import annotations

import inspect
import textwrap

import numpy as np
import pytest

from core.config import (
    AstrocyteConfig,
    NeuronConfig,
    NeuromodulatorConfig,
    OscillatorConfig,
    SimulationContext,
    SynapseConfig,
)


# Shared context
CTX = SimulationContext(dt=1.0)


# =====================================================================
# CLN 1: Dead code removal
# =====================================================================

class TestCLN1_DeadCode:

    def test_no_encode_value_in_spike_encoder(self):
        """encode_value() removed from PoissonEncoder."""
        from core.spike_encoder import PoissonEncoder
        assert not hasattr(PoissonEncoder, 'encode_value')

    def test_d1d2actor_no_ne_level_state(self):
        """D1D2Actor no longer stores _ne_level (dead state)."""
        from core.basal_ganglia import D1D2Actor
        from core.config import BasalGangliaConfig
        cfg = BasalGangliaConfig(ctx=CTX)
        actor = D1D2Actor(state_size=4, motor_dim=2, internal_dim=0, config=cfg)
        assert not hasattr(actor, '_ne_level')

    def test_d1d2actor_set_ne_level_is_noop(self):
        """D1D2Actor.set_ne_level() is a pass stub (interface contract)."""
        from core.basal_ganglia import D1D2Actor
        from core.config import BasalGangliaConfig
        cfg = BasalGangliaConfig(ctx=CTX)
        actor = D1D2Actor(state_size=4, motor_dim=2, internal_dim=0, config=cfg)
        # Should not raise, but also should not store anything
        actor.set_ne_level(0.8)

    def test_wm_set_ne_level_is_noop(self):
        """WorkingMemory.set_ne_level() is a pass stub."""
        from core.working_memory import WorkingMemoryModule
        from core.config import WorkingMemoryConfig
        cfg = WorkingMemoryConfig(ctx=CTX)
        wm = WorkingMemoryModule(num_external_inputs=4, num_neurons=8, config=cfg)
        wm.set_ne_level(0.9)  # Should not raise


# =====================================================================
# CLN 3: Astrocyte ATP cost — raw rates, no sqrt roundabout
# =====================================================================

class TestCLN3_AstrocyteATPCost:

    def test_to_zones_returns_tuple(self):
        """_to_zones() returns (raw_rates, rates_squared)."""
        from core.astrocyte import AstrocyteField
        cfg = AstrocyteConfig(ctx=CTX, n_zones=4)
        astro = AstrocyteField(config=cfg)
        rates = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
        result = astro._to_zones(rates)
        assert isinstance(result, tuple)
        assert len(result) == 2
        raw, sq = result
        np.testing.assert_allclose(raw, [0.2, 0.4, 0.6, 0.8], atol=1e-6)
        np.testing.assert_allclose(sq, [0.04, 0.16, 0.36, 0.64], atol=1e-5)

    def test_no_sqrt_in_update(self):
        """update() does not call np.sqrt (removed roundabout)."""
        from core.astrocyte import AstrocyteField
        src = inspect.getsource(AstrocyteField.update)
        assert 'np.sqrt' not in src, "np.sqrt found in update — CLN 3 not applied"

    def test_atp_depletes_proportional_to_raw_rate(self):
        """ATP cost is proportional to raw spike rate, not rate²."""
        from core.astrocyte import AstrocyteField
        cfg = AstrocyteConfig(ctx=CTX, n_zones=2, atp_regen_rate=0.0)
        astro = AstrocyteField(config=cfg)
        # Feed constant rates
        rates = np.array([0.5, 0.0], dtype=np.float32)
        initial_atp = astro.atp.copy()
        astro.update(rates)
        # Zone 0 should deplete by atp_spike_cost * 0.5 * dt
        expected_drop = cfg.atp_spike_cost * 0.5 * CTX.dt
        actual_drop = float(initial_atp[0] - astro.atp[0])
        assert abs(actual_drop - expected_drop) < 1e-5
        # Zone 1 (zero rate) should not deplete
        assert abs(float(initial_atp[1] - astro.atp[1])) < 1e-6

    def test_calcium_uses_squared_rates(self):
        """Ca²⁺ accumulation still uses rate² (energy proxy)."""
        from core.astrocyte import AstrocyteField
        cfg = AstrocyteConfig(ctx=CTX, n_zones=2)
        astro = AstrocyteField(config=cfg)
        rates = np.array([0.5, 0.0], dtype=np.float32)
        astro.update(rates)
        # Ca²⁺ for zone 0 should be nonzero (rate² = 0.25)
        assert astro.calcium[0] > 0.0
        # Ca²⁺ for zone 1 should be zero
        assert abs(astro.calcium[1]) < 1e-8


# =====================================================================
# CLN 4: Synapse sign convention
# =====================================================================

class TestCLN4_SynapseSign:

    def test_excitatory_positive_at_rest(self):
        """Excitatory current positive (depolarizing) at rest."""
        from core.synapse import SynapticChannels
        cfg = SynapseConfig(ctx=CTX)
        syn = SynapticChannels(n_post=1, config=cfg)
        # Inject AMPA spike: pre_spikes (1,), weights (1, 1)
        pre = np.array([1.0], dtype=np.float32)
        w = np.array([[5.0]], dtype=np.float32)  # 5 nS weight
        syn.receive_excitatory(pre, w)
        syn.decay()
        v_rest = np.array([-65.0], dtype=np.float32)
        current = syn.compute_current(v_rest)
        assert float(current[0]) > 0.0, "Excitatory current should be positive at rest"

    def test_inhibitory_negative_at_rest(self):
        """Inhibitory current negative (hyperpolarizing) at rest."""
        from core.synapse import SynapticChannels
        cfg = SynapseConfig(ctx=CTX)
        syn = SynapticChannels(n_post=1, config=cfg)
        # Inject GABA spike: inh_spikes (1,), weights (1, 1)
        inh = np.array([1.0], dtype=np.float32)
        w_ie = np.array([[5.0]], dtype=np.float32)
        syn.receive_inhibitory(inh, w_ie)
        syn.decay()
        v_rest = np.array([-65.0], dtype=np.float32)
        current = syn.compute_current(v_rest)
        assert float(current[0]) < 0.0, "Inhibitory current should be negative at rest"


# =====================================================================
# MOD 2: ACC stagnation — error neuron persistence, no CV
# =====================================================================

class TestMOD2_ErrorNeuronPersistence:

    @pytest.fixture
    def nm(self):
        cfg = NeuromodulatorConfig(ctx=CTX)
        return __import__('core.neuromodulator', fromlist=['NeuromodulatorSystem']).NeuromodulatorSystem(config=cfg)

    def test_no_cv_in_stagnation(self, nm):
        """Stagnation tracker has no np.std / np.mean CV computation."""
        from core.neuromodulator import NeuromodulatorSystem
        src = inspect.getsource(NeuromodulatorSystem.update)
        assert 'np.std' not in src, "CV-based stagnation (np.std) still present"
        assert 'td_cv' not in src, "td_cv variable still present"

    def test_no_tda_history(self, nm):
        """_tda_history deque removed (was only for CV stagnation)."""
        assert not hasattr(nm, '_tda_history')

    def test_acc_pe_trace_exists(self, nm):
        """ACC error neuron persistence trace present."""
        assert hasattr(nm, '_acc_pe_trace')
        assert nm._acc_pe_trace == 0.0

    def test_stagnation_rises_under_sustained_error(self, nm):
        """Sustained high PE → stagnation_factor increases."""
        pe = np.array([0.8], dtype=np.float32)
        for _ in range(5000):
            nm.update(prediction_error=pe, td_error=0.5)
        assert nm._stagnation_factor > 0.1

    def test_stagnation_low_under_zero_error(self, nm):
        """Zero PE → stagnation_factor stays near zero."""
        pe = np.zeros(1, dtype=np.float32)
        for _ in range(500):
            nm.update(prediction_error=pe, td_error=0.0)
        assert nm._stagnation_factor < 0.1

    def test_acc_pe_decay_from_config(self, nm):
        """acc_pe_decay derived from tau_acc in config."""
        cfg = nm.config
        expected = cfg.ctx.decay(cfg.tau_acc)
        assert abs(cfg.acc_pe_decay - expected) < 1e-8

    def test_consolidation_gate_attenuated_by_stagnation(self, nm):
        """consolidation_gate still works with the new stagnation."""
        pe = np.array([0.6], dtype=np.float32)
        for _ in range(2000):
            nm.update(prediction_error=pe, td_error=0.5, reward=0.5)
        gate = nm.consolidation_gate
        assert 0.0 <= gate <= 1.0


# =====================================================================
# MOD 3: Seizure brake — astrocyte ATP pathway, no hard reset
# =====================================================================

class TestMOD3_ATPSeizurePathway:

    def test_no_hard_voltage_reset_in_seizure(self):
        """check_and_handle_seizure() uses ATP pathway — no voltage-reset fallback."""
        from core.network import NetworkGraph
        src = inspect.getsource(NetworkGraph.check_and_handle_seizure)
        # Primary path uses astrocyte ATP depletion
        assert 'astro.atp' in src, "ATP depletion path missing"
        assert 'atp_spike_cost' in src, "ATP spike cost missing"
        # No hard voltage-reset fallback — astrocytes are mandatory
        assert 'v_rest' not in src and 'v_reset' not in src, (
            "Hard voltage-reset fallback should be removed"
        )

    def test_seizure_depletes_astrocyte_atp(self):
        """Seizure detection triggers astrocyte ATP depletion."""
        from core.astrocyte import AstrocyteField
        from core.network import NetworkGraph

        net = NetworkGraph(ctx=CTX)

        # Create a mock layer with an astrocyte
        class MockLayer:
            def __init__(self):
                self.num_inputs = 4
                self.num_neurons = 4
                self._astrocyte = AstrocyteField(
                    config=AstrocyteConfig(ctx=CTX, n_zones=4),
                )
                self.v_hidden = np.full(4, -65.0, dtype=np.float32)

            def forward(self, x):
                return np.ones(4, dtype=np.float32)

        layer = MockLayer()
        net.add_layer("test", layer)

        # Record initial ATP
        initial_atp = layer._astrocyte.atp.copy()

        # Simulate seizure: very high firing rate
        outputs = {"test": np.ones(100, dtype=np.float32)}
        # mean_rate = 100/100 = 1.0, baseline 0.05, threshold = 3*0.05 = 0.15
        result = net.check_and_handle_seizure(outputs, baseline_rate=0.05)

        assert result is True, "Seizure should be detected"
        # ATP should be depleted
        assert float(np.mean(layer._astrocyte.atp)) < float(np.mean(initial_atp))

    def test_seizure_atp_depletion_proportional_to_severity(self):
        """Higher firing rate → more ATP depletion."""
        from core.astrocyte import AstrocyteField
        from core.network import NetworkGraph

        class MockLayerWithAstro:
            def __init__(self):
                self.num_inputs = 4
                self.num_neurons = 4
                self._astrocyte = AstrocyteField(
                    config=AstrocyteConfig(
                        ctx=CTX, n_zones=4,
                        atp_spike_cost=0.001,  # lower cost so both don't zero out
                    ),
                )
            def forward(self, x):
                return np.ones(4, dtype=np.float32)

        # Mild seizure: 20% spiking
        net1 = NetworkGraph(ctx=CTX)
        layer1 = MockLayerWithAstro()
        net1.add_layer("test", layer1)
        out1 = {"test": np.concatenate([
            np.ones(20, dtype=np.float32),
            np.zeros(80, dtype=np.float32),
        ])}
        net1.check_and_handle_seizure(out1, baseline_rate=0.05)
        atp_after_mild = float(np.mean(layer1._astrocyte.atp))

        # Severe seizure: 90% spiking
        net2 = NetworkGraph(ctx=CTX)
        layer2 = MockLayerWithAstro()
        net2.add_layer("test", layer2)
        out2 = {"test": np.concatenate([
            np.ones(90, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
        ])}
        net2.check_and_handle_seizure(out2, baseline_rate=0.05)
        atp_after_severe = float(np.mean(layer2._astrocyte.atp))

        assert atp_after_severe < atp_after_mild, (
            f"Severe seizure should deplete more ATP: "
            f"severe={atp_after_severe:.4f} vs mild={atp_after_mild:.4f}"
        )


# =====================================================================
# Magic numbers: all traceable to citations
# =====================================================================

class TestMagicNumbers:

    def test_no_undocumented_075_in_network(self):
        """No undocumented -75.0 voltage reset in seizure handler."""
        from core.network import NetworkGraph
        src = inspect.getsource(NetworkGraph.check_and_handle_seizure)
        # -75.0 fallback was removed — astrocytes handle seizure via ATP
        assert '-75.0' not in src, "Undocumented -75.0 still in seizure handler"

    def test_ipsp_conductances_cited(self):
        """iPSP conductance values have Taverna citation."""
        import core.basal_ganglia as bg
        src = inspect.getsource(bg.D1D2Actor.forward)
        assert 'Taverna' in src, "iPSP conductances missing Taverna citation"

    def test_homeo_clip_cited(self):
        """_HOMEO_CLIP has Turrigiano citation."""
        import core.basal_ganglia as bg
        # Check both SNNDeepCritic and D1D2Actor update methods
        critic_src = inspect.getsource(bg.SNNDeepCritic.update)
        actor_src = inspect.getsource(bg.D1D2Actor.update)
        assert 'Turrigiano' in critic_src or 'Turrigiano' in actor_src

    def test_novelty_weights_cited(self):
        """Multi-scale novelty weights have Lisman & Jensen citation."""
        import core.sequence_memory as sm
        src = inspect.getsource(sm.HierarchicalSequenceMemory.novelty_signal)
        assert 'Lisman' in src

    def test_vesicle_p_cited(self):
        """Vesicle release probability has Markram citation."""
        import core.world_model as wm
        src = inspect.getsource(wm.SNNWorldModel.__init__)
        assert 'Markram' in src

    def test_stagnation_no_magic_0995(self):
        """No hardcoded 0.995 stagnation decay — uses config tau_acc."""
        from core.neuromodulator import NeuromodulatorSystem
        src = inspect.getsource(NeuromodulatorSystem.update)
        assert '0.995' not in src, "Hardcoded stagnation decay 0.995 still present"
