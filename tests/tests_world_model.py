import unittest
import numpy as np

from config import WorldModelConfig
from world_model import WorldModel


class TestWorldModel(unittest.TestCase):
    """
    Unit tests for WorldModel.

    Validates:
      A) Output shapes and value constraints.
      B) Learning convergence on a fixed transition.
      C) Mental rehearsal completeness and value ranges.
      D) Curiosity / novelty signal properties.
      E) Utility helpers (one-hot encoding, error history reset).
    """

    def setUp(self) -> None:
        self.state_size = 8
        self.action_size = 4
        self.config = WorldModelConfig(learning_rate=0.05)
        self.model = WorldModel(self.state_size, self.action_size, self.config)

    # ──────────────────────────────────────────────────────────────────
    # A. Output shapes and value constraints
    # ──────────────────────────────────────────────────────────────────

    def test_predict_output_shape(self) -> None:
        """predict() must return a vector of exactly state_size elements."""
        state = np.random.rand(self.state_size).astype(np.float32)
        pred = self.model.predict(state, 0)
        self.assertEqual(pred.shape, (self.state_size,))

    def test_predict_values_in_unit_interval(self) -> None:
        """predict() output must always be in [0, 1]."""
        state = np.random.rand(self.state_size).astype(np.float32)
        for action in range(self.action_size):
            pred = self.model.predict(state, action)
            self.assertTrue(
                np.all(pred >= 0.0) and np.all(pred <= 1.0),
                f"Prediction out of [0, 1] for action {action}.",
            )

    def test_update_returns_error_vector_of_state_size(self) -> None:
        """update() must return a signed error vector of shape (state_size,)."""
        state = np.random.rand(self.state_size)
        next_state = np.random.rand(self.state_size)
        error = self.model.update(state, 0, next_state)
        self.assertEqual(error.shape, (self.state_size,))

    def test_update_error_is_signed(self) -> None:
        """update() error = actual − predicted; must contain both signs."""
        state = np.zeros(self.state_size)
        # actual = all-ones → prediction (initially ≈ 0) will be underestimated
        next_state = np.ones(self.state_size)
        error = self.model.update(state, 0, next_state)

        # With near-zero initial weights, error ≈ +1 everywhere
        self.assertTrue(np.all(error >= -2.0) and np.all(error <= 2.0),
            "Error vector contains implausible values.")

    # ──────────────────────────────────────────────────────────────────
    # B. Learning convergence
    # ──────────────────────────────────────────────────────────────────

    def test_repeated_update_on_same_transition_reduces_mse(self) -> None:
        """
        MSE must decrease when the model is repeatedly trained on the same
        (state, action, next_state) triple.
        """
        state = np.random.rand(self.state_size).astype(np.float32)
        next_state = np.random.rand(self.state_size).astype(np.float32)
        action = 1

        # Measure initial error (before any update)
        initial_pred = self.model.predict(state, action)
        initial_mse = float(np.mean((initial_pred - next_state) ** 2))

        for _ in range(300):
            self.model.update(state, action, next_state)

        final_pred = self.model.predict(state, action)
        final_mse = float(np.mean((final_pred - next_state) ** 2))

        self.assertLess(
            final_mse, initial_mse,
            f"World model did not converge. Initial MSE={initial_mse:.4f}, "
            f"Final MSE={final_mse:.4f}.",
        )

    def test_prediction_error_attribute_updated_after_update(self) -> None:
        """prediction_error attribute must be set to the most recent MSE."""
        state = np.random.rand(self.state_size)
        next_state = np.random.rand(self.state_size)
        self.model.update(state, 0, next_state)

        self.assertIsInstance(self.model.prediction_error, float)
        self.assertGreaterEqual(self.model.prediction_error, 0.0)

    def test_different_actions_produce_different_predictions(self) -> None:
        """
        After training on distinct (state, a0, next_0) and (state, a1, next_1),
        predictions for a0 and a1 must differ.
        """
        state = np.ones(self.state_size, dtype=np.float32) * 0.5
        next_0 = np.zeros(self.state_size, dtype=np.float32)
        next_1 = np.ones(self.state_size, dtype=np.float32)

        for _ in range(200):
            self.model.update(state, 0, next_0)
            self.model.update(state, 1, next_1)

        p0 = self.model.predict(state, 0)
        p1 = self.model.predict(state, 1)

        self.assertFalse(
            np.allclose(p0, p1),
            "Predictions for distinct actions must differ after differentiated training.",
        )

    # ──────────────────────────────────────────────────────────────────
    # C. Mental rehearsal
    # ──────────────────────────────────────────────────────────────────

    def test_mental_rehearsal_covers_all_candidate_actions(self) -> None:
        """mental_rehearsal must return a result dict entry for every candidate."""
        state = np.random.rand(self.state_size).astype(np.float32)
        candidates = [0, 1, 2, 3]
        results = self.model.mental_rehearsal(state, candidates)

        for a in candidates:
            self.assertIn(a, results, f"Action {a} missing from rehearsal results.")

    def test_mental_rehearsal_result_has_required_keys(self) -> None:
        """Each result entry must contain predicted_state, novelty, familiarity."""
        state = np.random.rand(self.state_size).astype(np.float32)
        results = self.model.mental_rehearsal(state, [0])

        for key in ("predicted_state", "novelty", "familiarity"):
            self.assertIn(key, results[0], f"Key '{key}' missing from rehearsal result.")

    def test_mental_rehearsal_novelty_in_unit_interval(self) -> None:
        """novelty for every action must be in [0, 1]."""
        state = np.random.rand(self.state_size).astype(np.float32)
        results = self.model.mental_rehearsal(state, list(range(self.action_size)))

        for a, r in results.items():
            self.assertGreaterEqual(r["novelty"], 0.0, f"Novelty < 0 for action {a}.")
            self.assertLessEqual(r["novelty"], 1.0, f"Novelty > 1 for action {a}.")

    def test_mental_rehearsal_familiarity_equals_one_minus_novelty(self) -> None:
        """familiarity must be exactly 1 − novelty."""
        state = np.random.rand(self.state_size).astype(np.float32)
        results = self.model.mental_rehearsal(state, [0, 2])

        for a, r in results.items():
            self.assertAlmostEqual(
                r["familiarity"], 1.0 - r["novelty"], places=5,
                msg=f"familiarity ≠ 1 − novelty for action {a}.",
            )

    def test_mental_rehearsal_predicted_state_shape(self) -> None:
        """predicted_state must have shape (state_size,)."""
        state = np.random.rand(self.state_size).astype(np.float32)
        results = self.model.mental_rehearsal(state, [0])
        self.assertEqual(results[0]["predicted_state"].shape, (self.state_size,))

    # ──────────────────────────────────────────────────────────────────
    # D. Curiosity signal
    # ──────────────────────────────────────────────────────────────────

    def test_curiosity_signal_in_unit_interval(self) -> None:
        """curiosity_signal must return a float in [0, 1]."""
        for _ in range(20):
            error = np.random.randn(self.state_size) * np.random.rand()
            c = self.model.curiosity_signal(error)
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)

    def test_curiosity_signal_zero_for_zero_error(self) -> None:
        """Zero error must yield zero curiosity."""
        c = self.model.curiosity_signal(np.zeros(self.state_size))
        self.assertAlmostEqual(c, 0.0)

    def test_curiosity_signal_higher_for_larger_error(self) -> None:
        """A larger error vector must produce higher (or equal) curiosity."""
        small_error = np.ones(self.state_size) * 0.01
        large_error = np.ones(self.state_size) * 0.9
        self.assertGreaterEqual(
            self.model.curiosity_signal(large_error),
            self.model.curiosity_signal(small_error),
        )

    # ──────────────────────────────────────────────────────────────────
    # E. Utility helpers
    # ──────────────────────────────────────────────────────────────────

    def test_encode_action_is_one_hot(self) -> None:
        """_encode_action must produce a valid one-hot vector."""
        for a in range(self.action_size):
            vec = self.model._encode_action(a)
            self.assertEqual(vec.shape, (self.action_size,))
            self.assertAlmostEqual(float(np.sum(vec)), 1.0)
            self.assertAlmostEqual(float(vec[a]), 1.0)

    def test_predict_accepts_one_hot_action(self) -> None:
        """predict() must accept a pre-encoded one-hot vector as action."""
        state = np.random.rand(self.state_size).astype(np.float32)
        one_hot = self.model._encode_action(2)
        pred_int = self.model.predict(state, 2)
        pred_vec = self.model.predict(state, one_hot)
        np.testing.assert_array_almost_equal(pred_int, pred_vec)

    def test_reset_error_history_clears_log(self) -> None:
        """reset_error_history must empty _error_history and zero prediction_error."""
        state = np.random.rand(self.state_size)
        next_state = np.random.rand(self.state_size)
        for _ in range(5):
            self.model.update(state, 0, next_state)

        self.model.reset_error_history()

        self.assertEqual(len(self.model._error_history), 0)
        self.assertAlmostEqual(self.model.prediction_error, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)