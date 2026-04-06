import unittest
import numpy as np

from core.config import HomeostaticLIFConfig
from core.neuron import LIFLayer


class TestLIFLayer(unittest.TestCase):
    """
    Unit test suite for the vectorized LIFLayer.
    Validates matrix operations, trace routing, plasticity constraints,
    refractory periods, and homeostatic plasticity.
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
        # Force high weights to guarantee quick spiking
        self.layer.w.fill(50.0)

        pre_spikes = np.array([1.0, 0.0], dtype=np.float32)
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

    def test_refractory_period(self) -> None:
        """
        Neurons must not integrate voltage or spike while in the absolute refractory period.
        """
        # Force neuron 0 to spike
        self.layer.v[0] = 100.0
        self.layer.forward(np.array([0.0, 0.0], dtype=np.float32))

        # Neuron 0 should have spiked and entered refractory period
        self.assertTrue(self.layer.has_spiked[0])
        self.assertEqual(self.layer.refrac_count[0], self.layer.config.refrac_period)

        # Attempt to force it again immediately
        self.layer.v[0] = 100.0
        self.layer.forward(np.array([0.0, 0.0], dtype=np.float32))

        # Must be blocked by refractory
        self.assertFalse(self.layer.has_spiked[0], "Neuron spiked during refractory period.")
        # Voltage must be strictly held at v_reset
        self.assertEqual(self.layer.v[0], self.layer.config.v_reset)

    def test_trace_decay_matrix(self) -> None:
        """
        Verify the exponential decay of the eligibility trace matrix.
        """
        self.layer.w.fill(50.0)
        pre_spikes = np.array([1.0, 1.0], dtype=np.float32)

        # Integrate until spike occurs
        spiked = False
        for _ in range(20):
            if np.any(self.layer.forward(pre_spikes)):
                spiked = True
                break

        self.assertTrue(spiked, "Failed to force a spike.")

        peak_traces = self.layer.e.copy()
        self.assertTrue(np.all(peak_traces > 0.0), "Eligibility traces failed to initialize.")

        # Let it decay for tau_e steps
        for _ in range(int(self.layer.config.tau_e)):
            self.layer.forward(np.array([0.0, 0.0], dtype=np.float32))

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
        self.layer.w.fill(0.5)
        pre_spikes = np.array([1.0, 1.0], dtype=np.float32)

        self.layer.v[:] = 100.0
        self.layer.forward(pre_spikes)

        initial_weights = self.layer.w.copy()
        pred_error = np.ones(self.num_neurons, dtype=np.float32)

        # Apply update with M(t) = 0.0
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
        # Start weights at 0.5 so they are strictly within the [0, 1] clip window
        self.layer.w = np.full((self.num_inputs, self.num_neurons), 0.5, dtype=np.float32)
        initial_weights = self.layer.w.copy()

        # pre[0] is active, pre[1] is silent
        pre_spikes = np.array([1.0, 0.0], dtype=np.float32)

        # Force neuron 0 to spike, keep others low
        self.layer.v[0] = 100.0
        self.layer.v[1:] = self.layer.config.v_rest

        # Forward pass establishes the traces
        self.layer.forward(pre_spikes)

        # Update weights (positive prediction error and max dopamine)
        pred_error = np.ones(self.num_neurons, dtype=np.float32)
        self.layer.update_weights(m_t=1.0, pred_error=pred_error)

        # Assert synapse 0->0 grew (Active Pre + Active Post)
        self.assertGreater(
            self.layer.w[0, 0],
            initial_weights[0, 0],
            "Active synapse (0->0) failed to grow."
        )

        # Assert all other synapses remained exactly the same (no cross-talk)
        self.assertEqual(self.layer.w[0, 1], initial_weights[0, 1])
        self.assertEqual(self.layer.w[1, 0], initial_weights[1, 0])
        self.assertEqual(self.layer.w[1, 1], initial_weights[1, 1])

    def test_update_weights_broadcasting_shapes(self) -> None:
        """
        update_weights must correctly handle prediction_error of either
        shape (num_inputs,) or (num_neurons,).
        """
        self.layer.w.fill(0.5)
        self.layer.e.fill(1.0)  # fake a massive trace

        # 1. Error shape matching num_inputs (Predictive Coding)
        err_in = np.ones(self.num_inputs, dtype=np.float32)
        try:
            self.layer.update_weights(m_t=1.0, pred_error=err_in)
        except ValueError:
            self.fail("update_weights crashed on pred_error of shape (num_inputs,).")

        # 2. Error shape matching num_neurons (Standard LIF / Basal Ganglia)
        err_out = np.ones(self.num_neurons, dtype=np.float32)
        try:
            self.layer.update_weights(m_t=1.0, pred_error=err_out)
        except ValueError:
            self.fail("update_weights crashed on pred_error of shape (num_neurons,).")

        # 3. Invalid shape must raise ValueError
        err_invalid = np.ones(999, dtype=np.float32)
        with self.assertRaises(ValueError):
            self.layer.update_weights(m_t=1.0, pred_error=err_invalid)

    def test_homeostatic_plasticity(self) -> None:
        """
        Jeśli HomeostaticLIFConfig jest podany, nadmierne strzelanie powinno
        zwiększać adaptacyjny próg, aby wyciszyć neuron.
        """
        # NAPRAWA: Ustawiamy bardzo krótkie homeostatic_tau (np. 2.0),
        # aby średnia częstotliwość (EMA) reagowała natychmiastowo.
        config = HomeostaticLIFConfig(
            target_rate=0.01,
            thresh_adapt_lr=1.0,
            homeostatic_tau=2.0
        )
        layer = LIFLayer(num_inputs=2, num_neurons=3, config=config)

        initial_thresh = layer.v_thresh_adaptive.copy()

        # Wielokrotnie wymuszamy impuls
        # Z homeostatic_tau=2.0, jeden impuls podbije avg_rate do ~0.39,
        # co jest znacznie powyżej target_rate=0.01.
        for _ in range(20):
            layer.v[:] = 100.0
            layer.forward(np.array([0.0, 0.0], dtype=np.float32))

        # Próg musi zaadaptować się w górę
        self.assertTrue(
            np.all(layer.v_thresh_adaptive > initial_thresh),
            f"Próg nie wzrósł. Przed: {initial_thresh}, Po: {layer.v_thresh_adaptive}"
        )

    def test_reset_state(self) -> None:
        """
        reset_state() must clear all transient traces, voltages, and refractory counters
        without deleting the learned weights.
        """
        self.layer.w.fill(0.99)
        self.layer.v.fill(100.0)
        self.layer.e.fill(0.5)
        self.layer.refrac_count.fill(2)
        self.layer.has_spiked.fill(True)

        self.layer.reset_state()

        np.testing.assert_array_equal(self.layer.v, np.full(self.num_neurons, self.layer.config.v_rest))
        np.testing.assert_array_equal(self.layer.e, np.zeros((self.num_inputs, self.num_neurons)))
        np.testing.assert_array_equal(self.layer.refrac_count, np.zeros(self.num_neurons))
        self.assertFalse(np.any(self.layer.has_spiked))

        # Weights must be preserved
        np.testing.assert_array_equal(self.layer.w,
                                      np.full((self.num_inputs, self.num_neurons), 0.99, dtype=np.float32))


if __name__ == '__main__':
    unittest.main(verbosity=2)