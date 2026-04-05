import unittest
import numpy as np

from core.config import SequenceMemoryConfig
from core.sequence_memory import SequenceMemory


class TestSequenceMemory(unittest.TestCase):
    """
    Unit tests for SequenceMemory.

    Validates:
      A) Temporal transition learning.
      B) Prediction and prediction error.
      C) Concept cluster discovery.
      D) Novelty signal.
      E) State management.
    """

    def setUp(self) -> None:
        self.num_neurons = 10
        self.config = SequenceMemoryConfig(
            learning_rate=0.1,
            decay=0.999,
            max_weight=1.0,
        )
        self.sm = SequenceMemory(self.num_neurons, self.config)

    # ──────────────────────────────────────────────────────────────────
    # A. Temporal transition learning
    # ──────────────────────────────────────────────────────────────────

    def test_transition_weights_initialized_to_zero(self) -> None:
        np.testing.assert_array_equal(
            self.sm.transition_w,
            np.zeros((self.num_neurons, self.num_neurons)),
        )

    def test_single_observe_does_not_learn_without_previous(self) -> None:
        """First observation has no predecessor → no weight change."""
        pattern = np.zeros(self.num_neurons, dtype=np.float32)
        pattern[0] = 1.0
        self.sm.observe(pattern)
        np.testing.assert_array_equal(
            self.sm.transition_w,
            np.zeros((self.num_neurons, self.num_neurons)),
        )

    def test_sequential_observation_strengthens_transition(self) -> None:
        """Observing A then B should create a positive transition_w[B, A]."""
        a = np.zeros(self.num_neurons, dtype=np.float32)
        b = np.zeros(self.num_neurons, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0

        self.sm.observe(a)
        self.sm.observe(b)

        # transition_w[1, 0] should be positive (neuron 0 predicts neuron 1)
        self.assertGreater(
            self.sm.transition_w[1, 0], 0.0,
            "Transition weight from neuron 0 → neuron 1 should be positive.",
        )

    def test_repeated_sequence_strengthens_transition(self) -> None:
        """Repeating A→B multiple times should increase the transition weight."""
        a = np.zeros(self.num_neurons, dtype=np.float32)
        b = np.zeros(self.num_neurons, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0

        self.sm.observe(a)
        self.sm.observe(b)
        w_after_one = self.sm.transition_w[1, 0]

        for _ in range(10):
            self.sm.observe(a)
            self.sm.observe(b)

        self.assertGreater(
            self.sm.transition_w[1, 0], w_after_one,
            "Repeated A→B should strengthen the transition weight.",
        )

    def test_transition_weights_stay_within_bounds(self) -> None:
        """Weights must stay in [0, max_weight] after many updates."""
        a = np.ones(self.num_neurons, dtype=np.float32)
        for _ in range(500):
            self.sm.observe(a)

        self.assertTrue(np.all(self.sm.transition_w >= 0.0))
        self.assertTrue(np.all(self.sm.transition_w <= self.config.max_weight + 1e-6))

    # ──────────────────────────────────────────────────────────────────
    # B. Prediction and prediction error
    # ──────────────────────────────────────────────────────────────────

    def test_predict_next_shape(self) -> None:
        pred = self.sm.predict_next()
        self.assertEqual(pred.shape, (self.num_neurons,))

    def test_predict_next_values_in_unit_interval(self) -> None:
        # Train a sequence first
        a = np.zeros(self.num_neurons, dtype=np.float32)
        a[0] = 1.0
        for _ in range(10):
            self.sm.observe(a)
        pred = self.sm.predict_next()
        self.assertTrue(np.all(pred >= 0.0) and np.all(pred <= 1.0))

    def test_temporal_error_shape(self) -> None:
        pattern = np.zeros(self.num_neurons, dtype=np.float32)
        pattern[0] = 1.0
        error = self.sm.observe(pattern)
        self.assertEqual(error.shape, (self.num_neurons,))

    def test_prediction_error_reflects_surprise(self) -> None:
        """
        After learning A→B, presenting A→C (unexpected) should produce a
        larger prediction error than A→B (expected).
        """
        a = np.zeros(self.num_neurons, dtype=np.float32)
        b = np.zeros(self.num_neurons, dtype=np.float32)
        c = np.zeros(self.num_neurons, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        c[2] = 1.0

        # Learn A→B
        for _ in range(50):
            self.sm.observe(a)
            self.sm.observe(b)

        # Measure expected transition A→B
        self.sm.observe(a)
        error_expected = self.sm.observe(b)

        # Measure unexpected transition A→C
        self.sm.observe(a)
        error_unexpected = self.sm.observe(c)

        self.assertGreater(
            float(np.mean(np.abs(error_unexpected))),
            float(np.mean(np.abs(error_expected))),
            "Unexpected transition should have larger prediction error.",
        )

    # ──────────────────────────────────────────────────────────────────
    # C. Concept cluster discovery
    # ──────────────────────────────────────────────────────────────────

    def test_no_clusters_before_training(self) -> None:
        clusters = self.sm.get_temporal_clusters(threshold=0.01)
        self.assertEqual(len(clusters), 0)

    def test_bidirectional_sequence_forms_cluster(self) -> None:
        """A→B→A repeatedly should form a mutual cluster {A, B}."""
        a = np.zeros(self.num_neurons, dtype=np.float32)
        b = np.zeros(self.num_neurons, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0

        for _ in range(50):
            self.sm.observe(a)
            self.sm.observe(b)

        clusters = self.sm.get_temporal_clusters(threshold=0.01)
        # Should find a cluster containing neurons 0 and 1
        found = any(0 in c and 1 in c for c in clusters)
        self.assertTrue(found, f"Expected cluster {{0, 1}}, got: {clusters}")

    def test_get_associated_neurons_returns_correct_targets(self) -> None:
        a = np.zeros(self.num_neurons, dtype=np.float32)
        b = np.zeros(self.num_neurons, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0

        for _ in range(20):
            self.sm.observe(a)
            self.sm.observe(b)

        associated = self.sm.get_associated_neurons(0, threshold=0.01)
        self.assertIn(1, associated, "Neuron 1 should be associated with neuron 0.")

    # ──────────────────────────────────────────────────────────────────
    # D. Novelty signal
    # ──────────────────────────────────────────────────────────────────

    def test_novelty_signal_in_unit_interval(self) -> None:
        pattern = np.random.rand(self.num_neurons).astype(np.float32)
        self.sm.observe(pattern)
        novelty = self.sm.novelty_signal()
        self.assertGreaterEqual(novelty, 0.0)
        self.assertLessEqual(novelty, 1.0)

    def test_novelty_zero_when_no_error(self) -> None:
        """Before any observation, temporal_error is zero → novelty is zero."""
        self.assertAlmostEqual(self.sm.novelty_signal(), 0.0)

    # ──────────────────────────────────────────────────────────────────
    # E. State management
    # ──────────────────────────────────────────────────────────────────

    def test_reset_state_clears_transient_but_preserves_weights(self) -> None:
        a = np.ones(self.num_neurons, dtype=np.float32)
        for _ in range(10):
            self.sm.observe(a)

        weights_before = self.sm.transition_w.copy()
        self.sm.reset_state()

        np.testing.assert_array_equal(self.sm.prev_pattern, np.zeros(self.num_neurons))
        np.testing.assert_array_equal(self.sm.predicted_next, np.zeros(self.num_neurons))
        np.testing.assert_array_equal(self.sm.temporal_error, np.zeros(self.num_neurons))
        np.testing.assert_array_equal(self.sm.transition_w, weights_before)

    def test_reset_all_clears_everything(self) -> None:
        a = np.ones(self.num_neurons, dtype=np.float32)
        for _ in range(10):
            self.sm.observe(a)

        self.sm.reset_all()

        np.testing.assert_array_equal(
            self.sm.transition_w,
            np.zeros((self.num_neurons, self.num_neurons)),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
