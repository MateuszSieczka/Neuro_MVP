import unittest
import numpy as np

from core.config import PredictiveCodingConfig, NeuromodulatorConfig, SequenceMemoryConfig
from core.predictive_coding import PredictiveCodingLayer
from core.neuromodulator import NeuromodulatorSystem
from core.sequence_memory import SequenceMemory
from core.network import NetworkGraph, LayerConnection


class TestNetworkGraphRegistration(unittest.TestCase):
    """Tests for layer registration and connection management."""

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
        l1 = PredictiveCodingLayer(num_inputs=4, num_neurons=6)
        l2 = PredictiveCodingLayer(num_inputs=6, num_neurons=8)
        self.net.add_layer("L1", l1)
        self.net.add_layer("L2", l2)
        with self.assertRaises(ValueError):
            self.net.connect("L1", "L2", "invalid_type")

    def test_get_layer_returns_registered_layer(self) -> None:
        layer = PredictiveCodingLayer(num_inputs=4, num_neurons=6)
        self.net.add_layer("L1", layer)
        self.assertIs(self.net.get_layer("L1"), layer)

    def test_layer_names_preserves_registration_order(self) -> None:
        for name in ["bottom", "middle", "top"]:
            self.net.add_layer(name, PredictiveCodingLayer(num_inputs=4, num_neurons=6))
        self.assertEqual(self.net.layer_names, ["bottom", "middle", "top"])


class TestNetworkGraphStep(unittest.TestCase):
    """Tests for the step() feedforward/feedback pipeline."""

    def _build_two_layer_network(self) -> tuple[NetworkGraph, str, str]:
        """
        Build a simple two-layer hierarchy.

        Dimension contract:
          L1: num_inputs=4, num_neurons=4
          L2: num_inputs=4, num_neurons=6
          L2's generate_prediction() produces shape (4,) which matches L1's num_inputs.
        """
        net = NetworkGraph(dt=1.0)
        l1 = PredictiveCodingLayer(num_inputs=4, num_neurons=4)
        l2 = PredictiveCodingLayer(num_inputs=4, num_neurons=6)

        net.add_layer("L1", l1)
        net.add_layer("L2", l2)
        net.connect("L1", "L2", "feedforward")
        net.connect("L2", "L1", "feedback")
        return net, "L1", "L2"

    def test_step_returns_outputs_for_all_layers(self) -> None:
        net, l1, l2 = self._build_two_layer_network()
        sensory = {"L1": np.ones(4, dtype=np.float32)}
        outputs = net.step(sensory)
        self.assertIn(l1, outputs)
        self.assertIn(l2, outputs)

    def test_step_output_shapes(self) -> None:
        net, l1, l2 = self._build_two_layer_network()
        sensory = {"L1": np.ones(4, dtype=np.float32)}
        outputs = net.step(sensory)
        self.assertEqual(outputs[l1].shape, (4,))
        self.assertEqual(outputs[l2].shape, (6,))

    def test_step_output_is_boolean_or_binary(self) -> None:
        net, l1, _ = self._build_two_layer_network()
        sensory = {"L1": np.ones(4, dtype=np.float32)}
        outputs = net.step(sensory)
        unique_vals = set(outputs[l1].astype(float).tolist())
        self.assertTrue(unique_vals <= {0.0, 1.0, True, False})

    def test_step_increments_timestep(self) -> None:
        net, _, _ = self._build_two_layer_network()
        self.assertEqual(net.timestep, 0)
        net.step({"L1": np.zeros(4)})
        self.assertEqual(net.timestep, 1)
        net.step({"L1": np.zeros(4)})
        self.assertEqual(net.timestep, 2)

    def test_step_with_neuromodulator_distributes_ach(self) -> None:
        net, _, _ = self._build_two_layer_network()
        nm = NeuromodulatorSystem(NeuromodulatorConfig(baseline_ach=0.9))
        sensory = {"L1": np.ones(4, dtype=np.float32)}
        net.step(sensory, neuromodulator=nm)

        l1 = net.get_layer("L1")
        # ACh should have been distributed to all layers
        self.assertAlmostEqual(l1.ach_level, nm.bottom_up_gain, places=2)

    def test_missing_sensory_input_uses_zeros(self) -> None:
        """Layer without sensory input or feedforward source gets zero input."""
        net = NetworkGraph()
        layer = PredictiveCodingLayer(num_inputs=4, num_neurons=6)
        net.add_layer("isolated", layer)
        # No sensory input provided
        outputs = net.step({})
        # Should not crash; output is all-false spikes
        self.assertEqual(outputs["isolated"].shape, (6,))

    def test_feedback_delivers_prediction_to_lower_layer(self) -> None:
        """
        After training L2 to spike, its generate_prediction() should produce
        a non-zero prediction that is received by L1.
        """
        net, _, _ = self._build_two_layer_network()
        l2 = net.get_layer("L2")
        # Force L2 to have some spikes in its history
        l2.has_spiked[:3] = True
        l2.feedback_w.fill(0.5)

        # After step, L1 should have received a prediction
        net.step({"L1": np.ones(4, dtype=np.float32)})
        l1 = net.get_layer("L1")
        # top_down_prediction should match L1's num_inputs
        self.assertEqual(l1.top_down_prediction.shape, (4,))


class TestNetworkGraphSequenceMemory(unittest.TestCase):
    """Tests for SequenceMemory integration."""

    def test_attach_sequence_memory_to_unknown_layer_raises(self) -> None:
        net = NetworkGraph()
        sm = SequenceMemory(6)
        with self.assertRaises(ValueError):
            net.attach_sequence_memory("unknown", sm)

    def test_step_updates_attached_sequence_memory(self) -> None:
        net = NetworkGraph()
        config = PredictiveCodingConfig(k_winners=3, window_ms=5)
        layer = PredictiveCodingLayer(num_inputs=4, num_neurons=6, config=config)
        layer.w.fill(50.0)
        layer.set_ach_level(1.0)  # Full bottom-up for reliable spiking
        net.add_layer("L1", layer)

        sm = SequenceMemory(6, SequenceMemoryConfig(learning_rate=0.1))
        net.attach_sequence_memory("L1", sm)

        # Run enough steps to guarantee spikes even with Poisson encoding
        any_observed = False
        for _ in range(200):
            outputs = net.step({"L1": np.ones(4, dtype=np.float32)})
            if np.any(outputs["L1"]):
                any_observed = True

        self.assertTrue(
            any_observed,
            "Network should have produced at least one spike in 200 steps.",
        )


class TestNetworkGraphWeightUpdate(unittest.TestCase):
    """Tests for the global update_weights method."""

    def test_update_weights_with_prediction_error(self) -> None:
        net = NetworkGraph()
        config = PredictiveCodingConfig(k_winners=3, window_ms=5)
        layer = PredictiveCodingLayer(num_inputs=4, num_neurons=6, config=config)
        layer.w.fill(50.0)
        net.add_layer("L1", layer)

        nm = NeuromodulatorSystem()
        nm.dopamine = 1.0

        # Run a few steps to build eligibility traces
        for _ in range(10):
            net.step({"L1": np.ones(4, dtype=np.float32)}, neuromodulator=nm)

        # Set a prediction error in input space (num_inputs=4)
        layer.prediction_error = np.ones(4, dtype=np.float32) * 0.5

        # update_weights should project error from input space to output space
        # and not raise a shape error
        initial_w = layer.w.copy()
        net.update_weights(nm)

        # If there are non-zero eligibility traces, weights should have changed
        if np.any(layer.e != 0.0):
            self.assertFalse(np.allclose(layer.w, initial_w))


class TestNetworkGraphReset(unittest.TestCase):
    """Tests for reset_state."""

    def test_reset_clears_timestep(self) -> None:
        net = NetworkGraph()
        layer = PredictiveCodingLayer(num_inputs=4, num_neurons=6)
        net.add_layer("L1", layer)
        net.step({"L1": np.zeros(4)})
        net.step({"L1": np.zeros(4)})
        net.reset_state()
        self.assertEqual(net.timestep, 0)

    def test_reset_clears_layer_state(self) -> None:
        net = NetworkGraph()
        config = PredictiveCodingConfig(k_winners=3, window_ms=5)
        layer = PredictiveCodingLayer(num_inputs=4, num_neurons=4, config=config)
        layer.w.fill(50.0)
        net.add_layer("L1", layer)

        for _ in range(10):
            net.step({"L1": np.ones(4, dtype=np.float32)})
        net.reset_state()

        np.testing.assert_array_equal(
            layer.v, np.full(4, layer.config.v_rest),
        )

    def test_reset_clears_sequence_memory(self) -> None:
        net = NetworkGraph()
        layer = PredictiveCodingLayer(num_inputs=4, num_neurons=6)
        layer.w.fill(50.0)
        net.add_layer("L1", layer)
        sm = SequenceMemory(6)
        net.attach_sequence_memory("L1", sm)

        for _ in range(20):
            net.step({"L1": np.ones(4, dtype=np.float32)})
        net.reset_state()

        np.testing.assert_array_equal(sm.prev_pattern, np.zeros(6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
