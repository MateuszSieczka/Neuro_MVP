import unittest
import numpy as np

from core.config import PredictiveCodingConfig, NeuromodulatorConfig, SequenceMemoryConfig
from core.predictive_coding import PredictiveCodingLayer
from core.neuromodulator import NeuromodulatorSystem
from core.sequence_memory import SequenceMemory, HierarchicalSequenceMemory
from core.network import NetworkGraph


class TestNetworkGraphRegistration(unittest.TestCase):
    """Tests for layer registration, topology sorting, and cycle detection."""

    def setUp(self) -> None:
        self.net = NetworkGraph(dt=1.0)

    def test_add_layer_registers_name(self) -> None:
        layer = PredictiveCodingLayer(num_inputs=4, num_neurons=6)
        self.net.add_layer("L1", layer)
        self.assertIn("L1", self.net.layer_names)

    def test_add_duplicate_name_raises(self) -> None:
        layer = PredictiveCodingLayer(num_inputs=4, num_neurons=6)
        self.net.add_layer("L1", layer)
        with self.assertRaises(ValueError):
            self.net.add_layer("L1", layer)

    def test_connect_unknown_layer_raises(self) -> None:
        layer = PredictiveCodingLayer(num_inputs=4, num_neurons=6)
        self.net.add_layer("L1", layer)
        with self.assertRaises(ValueError):
            self.net.connect("L1", "UNKNOWN")

    def test_connect_invalid_type_raises(self) -> None:
        self.net.add_layer("L1", PredictiveCodingLayer(4, 4))
        self.net.add_layer("L2", PredictiveCodingLayer(4, 4))
        with self.assertRaises(ValueError):
            self.net.connect("L1", "L2", "invalid_type")

    def test_topological_sort_feedforward(self) -> None:
        """Graph should sort layers strictly bottom-to-top regardless of add order."""
        self.net.add_layer("Top", PredictiveCodingLayer(4, 4))
        self.net.add_layer("Bottom", PredictiveCodingLayer(4, 4))
        self.net.add_layer("Middle", PredictiveCodingLayer(4, 4))

        self.net.connect("Bottom", "Middle", "feedforward")
        self.net.connect("Middle", "Top", "feedforward")

        # Accessing layer_names triggers the _refresh_order()
        ordered = self.net.layer_names
        self.assertEqual(ordered, ["Bottom", "Middle", "Top"])

    def test_topological_sort_ignores_feedback_cycles(self) -> None:
        """Feedback connections should not create cycles in the topological sort."""
        self.net.add_layer("L1", PredictiveCodingLayer(4, 4))
        self.net.add_layer("L2", PredictiveCodingLayer(4, 4))

        self.net.connect("L1", "L2", "feedforward")
        self.net.connect("L2", "L1", "feedback")

        self.assertEqual(self.net.layer_names, ["L1", "L2"])


class TestNetworkGraphAggregation(unittest.TestCase):
    """Tests for multi-source feedforward aggregation (sum vs concat)."""

    def setUp(self) -> None:
        self.net = NetworkGraph(dt=1.0)

    def test_sum_aggregation_mode(self) -> None:
        self.net.add_layer("L1", PredictiveCodingLayer(2, 2))
        self.net.add_layer("L2", PredictiveCodingLayer(2, 2))
        self.net.add_layer("Target", PredictiveCodingLayer(2, 2))

        # Both feed into Target with sum mode and weights
        self.net.connect("L1", "Target", "feedforward", "sum", weight=1.0)
        self.net.connect("L2", "Target", "feedforward", "sum", weight=0.5)

        # Mocking delay buffer history manually to bypass forward passes
        outputs = {
            "L1": np.array([1.0, 1.0], dtype=np.float32),
            "L2": np.array([1.0, 1.0], dtype=np.float32),
        }
        self.net._output_history["L1"] = [outputs["L1"]]
        self.net._output_history["L2"] = [outputs["L2"]]

        result = self.net._aggregate_feedforward_inputs("Target", outputs)
        np.testing.assert_array_almost_equal(result, np.array([1.5, 1.5]))

    def test_concat_aggregation_mode(self) -> None:
        # L1 produces 2 features, L2 produces 3 features
        self.net.add_layer("L1", PredictiveCodingLayer(2, 2))
        self.net.add_layer("L2", PredictiveCodingLayer(3, 3))
        # Target accepts exactly 5 features (2 + 3)
        self.net.add_layer("Target", PredictiveCodingLayer(5, 5))

        self.net.connect("L1", "Target", "feedforward", "concat")
        self.net.connect("L2", "Target", "feedforward", "concat")

        outputs = {
            "L1": np.array([1.0, 2.0], dtype=np.float32),
            "L2": np.array([3.0, 4.0, 5.0], dtype=np.float32),
        }
        self.net._output_history["L1"] = [outputs["L1"]]
        self.net._output_history["L2"] = [outputs["L2"]]

        # Accessing layer_names triggers _refresh_order and offset precalculation
        _ = self.net.layer_names

        result = self.net._aggregate_feedforward_inputs("Target", outputs)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


class TestNetworkGraphDelays(unittest.TestCase):
    """Tests for synaptic delay buffers."""

    def test_delayed_feedforward_connection(self) -> None:
        net = NetworkGraph()
        net.add_layer("L1", PredictiveCodingLayer(2, 2))
        net.add_layer("L2", PredictiveCodingLayer(2, 2))

        # Delay of 1 means L2 receives L1's output from timestep (t-1)
        net.connect("L1", "L2", "feedforward", delay=1)

        # Timestep 0: L1 input
        sensory_0 = {"L1": np.array([1.0, 1.0], dtype=np.float32)}
        out_0 = net.step(sensory_0)

        # In step 0, L2 should have received zero feedforward drive
        ff_input_t0 = net._aggregate_feedforward_inputs("L2", out_0)
        np.testing.assert_array_equal(ff_input_t0, np.zeros(2))

        # Timestep 1: L1 is silent. L2 should now receive exactly what L1 fired at t=0.
        sensory_1 = {"L1": np.array([0.0, 0.0], dtype=np.float32)}
        out_1 = net.step(sensory_1)

        ff_input_t1 = net._aggregate_feedforward_inputs("L2", out_1)

        # NAPRAWA: Zamiast hardcodować [1.0, 1.0], sprawdzamy opóźnienie względem realnego wyjścia.
        np.testing.assert_array_equal(ff_input_t1, out_0["L1"])

class TestNetworkGraphNeuromodulation(unittest.TestCase):
    """Tests for global distribution of ACh, NE, and 5-HT."""

    def test_step_distributes_neuromodulators(self) -> None:
        net = NetworkGraph()
        l1 = PredictiveCodingLayer(4, 4)
        net.add_layer("L1", l1)

        nm = NeuromodulatorSystem(NeuromodulatorConfig(
            baseline_ach=0.8, baseline_ne=0.7, baseline_sero=0.6
        ))

        net.step({"L1": np.zeros(4)}, neuromodulator=nm)

        # Check if layer received the signals (assuming mock/duck typing behavior)
        self.assertAlmostEqual(l1.ach_level, 0.8)
        # Note: In the codebase, NE and 5-HT are passed via `set_neuromodulators`
        # if the layer supports it (e.g. PyramidalLayer, but PC layer may just ignore them
        # or we assume NetworkGraph correctly attempted to call it).


class TestNetworkGraphSequenceMemory(unittest.TestCase):
    """Tests for attaching and observing Sequence Memory."""

    def test_attach_sequence_memory_to_unknown_layer_raises(self) -> None:
        net = NetworkGraph()
        sm = SequenceMemory(6)
        with self.assertRaises(ValueError):
            net.attach_sequence_memory("unknown", sm)

    def test_step_updates_attached_sequence_memory(self) -> None:
        net = NetworkGraph()
        # refrac_period=0 ensures the layer fires on every step with strong weights,
        # providing consecutive non-zero spike patterns for transition_w learning.
        config = PredictiveCodingConfig(k_winners=4, window_ms=5, refrac_period=0)
        layer = PredictiveCodingLayer(4, 4, config=config)
        # Force weights to guarantee spikes
        layer.w.fill(50.0)
        layer.set_ach_level(1.0)
        net.add_layer("L1", layer)

        sm = SequenceMemory(4, SequenceMemoryConfig(learning_rate=0.1))
        net.attach_sequence_memory("L1", sm)

        for _ in range(5):
            net.step({"L1": np.ones(4, dtype=np.float32)})

        # forward() returns has_spiked (num_neurons). With refrac_period=0 the
        # layer fires every step, so SequenceMemory observes consecutive non-zero
        # patterns and transition_w learns temporal associations.
        self.assertTrue(
            np.any(sm.transition_w > 0),
            "SequenceMemory nie wyuczyła wag przejść po kolejnych impulsach."
        )


class TestNetworkGraphReset(unittest.TestCase):
    """Tests for state wiping across episodes."""

    def test_reset_clears_timestep_and_history(self) -> None:
        net = NetworkGraph()
        layer = PredictiveCodingLayer(num_inputs=4, num_neurons=4)
        net.add_layer("L1", layer)

        net.step({"L1": np.ones(4)})
        net.step({"L1": np.ones(4)})

        self.assertEqual(net.timestep, 2)
        self.assertIn("L1", net._output_history)

        # Maxlen for zero-delay network is 1, so the buffer should hold exactly 1 element
        self.assertEqual(len(net._output_history["L1"]), 1)

        net.reset_state()

        self.assertEqual(net.timestep, 0)
        self.assertEqual(len(net._output_history), 0)

    def test_reset_clears_layer_state(self) -> None:
        net = NetworkGraph()
        layer = PredictiveCodingLayer(num_inputs=4, num_neurons=4)
        layer.w.fill(50.0)
        net.add_layer("L1", layer)

        net.step({"L1": np.ones(4, dtype=np.float32)})

        # SNN check: v is different from v_rest (either charging up, or resting at v_reset after a spike)
        self.assertTrue(np.any(layer.v != layer.config.v_rest) or np.any(layer.has_spiked))

        net.reset_state()
        np.testing.assert_array_equal(layer.v, np.full(4, layer.config.v_rest))


if __name__ == "__main__":
    unittest.main(verbosity=2)

if __name__ == "__main__":
    unittest.main(verbosity=2)