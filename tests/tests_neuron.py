import unittest
import numpy as np
from neuron import LIFLayer


class TestLIFLayer(unittest.TestCase):
    """
    Unit test suite for the vectorized LIFLayer.
    Validates matrix operations, trace routing, and plasticity constraints.
    """

    def setUp(self) -> None:
        self.num_inputs = 2
        self.num_neurons = 3
        self.layer = LIFLayer(num_inputs=self.num_inputs, num_neurons=self.num_neurons)

    def test_vectorized_integration_and_spike(self) -> None:
        """
        Verify that continuous input spikes eventually drive target neurons
        over the threshold and trigger a vectorized reset.
        """
        self.layer.w.fill(50.0)

        pre_spikes = np.array([1.0, 0.0])
        spiked = np.zeros(self.num_neurons, dtype=bool)

        for _ in range(100):
            current_spikes = self.layer.forward(pre_spikes)

            if np.any(current_spikes):
                # Every neuron that spikes in this timestep must be reset immediately
                np.testing.assert_allclose(
                    self.layer.v[current_spikes],
                    self.layer.config.v_reset,
                    err_msg="Spiking neurons were not reset to v_reset."
                )

            spiked |= current_spikes

            if np.all(spiked):
                break

        self.assertTrue(np.all(spiked), "Neurons failed to spike under continuous input.")

    def test_trace_decay_matrix(self) -> None:
        """
        Verify the exponential decay of the eligibility trace matrix.
        """
        self.layer.w.fill(100.0)
        pre_spikes = np.array([1.0, 1.0])

        # Integrate until spike occurs
        spiked = False
        for _ in range(20):
            if np.any(self.layer.forward(pre_spikes)):
                spiked = True
                break

        self.assertTrue(spiked, "Failed to force a spike.")

        peak_traces = self.layer.e.copy()
        self.assertTrue(np.all(peak_traces > 0.9), "Eligibility traces failed to initialize.")

        # Let it decay for tau_e steps
        for _ in range(int(self.layer.config.tau_e)):
            self.layer.forward(np.array([0.0, 0.0]))

        decayed_traces = self.layer.e
        expected_decay = peak_traces * np.exp(-1.0)

        np.testing.assert_allclose(
            decayed_traces,
            expected_decay,
            rtol=1e-2,
            err_msg="Trace matrix did not decay according to tau_e."
        )

    def test_plasticity_lock(self) -> None:
        """
        Weights MUST remain strictly unchanged if the modulator M(t) is zero.
        """
        self.layer.w.fill(100.0)
        pre_spikes = np.array([1.0, 1.0])

        for _ in range(20):
            if np.any(self.layer.forward(pre_spikes)):
                break

        initial_weights = self.layer.w.copy()
        pred_error = np.ones(self.num_neurons)

        self.layer.update_weights(m_t=0.0, pred_error=pred_error)

        np.testing.assert_array_equal(
            initial_weights,
            self.layer.w,
            err_msg="Weights mutated despite M(t)=0."
        )

    def test_three_factor_routing(self) -> None:
        """
        Verify that weight updates are strictly routed only to synapses
        connecting a firing presynaptic neuron to a firing postsynaptic neuron.
        """
        self.layer.w = np.zeros((self.num_inputs, self.num_neurons), dtype=np.float32)
        # Strong connection specifically for 0->0
        self.layer.w[0, 0] = 100.0

        initial_weights = self.layer.w.copy()
        pre_spikes = np.array([1.0, 0.0])

        # Integrate until neuron 0 spikes
        for _ in range(20):
            if np.any(self.layer.forward(pre_spikes)):
                break

        pred_error = np.ones(self.num_neurons)
        self.layer.update_weights(m_t=1.0, pred_error=pred_error)

        # Assert synapse 0->0 grew
        self.assertGreater(
            self.layer.w[0, 0],
            initial_weights[0, 0],
            "Active synapse (0->0) failed to grow."
        )

        # Assert all other synapses remained exactly the same (no cross-talk)
        self.assertEqual(self.layer.w[0, 1], initial_weights[0, 1])
        self.assertEqual(self.layer.w[1, 0], initial_weights[1, 0])
        self.assertEqual(self.layer.w[1, 1], initial_weights[1, 1])
        self.assertEqual(self.layer.w[0, 2], initial_weights[0, 2], "Synapse 0->2 mutated.")
        self.assertEqual(self.layer.w[1, 2], initial_weights[1, 2], "Synapse 1->2 mutated.")

if __name__ == '__main__':
    unittest.main(verbosity=2)