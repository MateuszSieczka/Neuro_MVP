import unittest
import numpy as np

from core.config import PredictiveCodingConfig
from core.predictive_coding import PredictiveCodingLayer


class TestPredictiveCodingLayer(unittest.TestCase):
    """
    Unit tests for PredictiveCodingLayer.

    Validates three neuro_mvp contracts:
      A) Prediction error arithmetic.
      B) ACh-gated bottom-up / top-down blending.
      C) Feedback weight shape, update, and state management.
    """

    def setUp(self) -> None:
        self.num_inputs = 6
        self.num_neurons = 12
        self.config = PredictiveCodingConfig(
            k_winners=3,
            window_ms=20,
            feedback_learning_rate=0.05,
        )
        self.layer = PredictiveCodingLayer(
            num_inputs=self.num_inputs,
            num_neurons=self.num_neurons,
            config=self.config,
        )

    # ──────────────────────────────────────────────────────────────────
    # A. Prediction error
    # ──────────────────────────────────────────────────────────────────

    def test_prediction_error_is_actual_minus_prediction(self) -> None:
        """
        prediction_error must equal pre_spikes − top_down_prediction,
        computed elementwise, before any non-linearity.
        """
        actual = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        pred = np.array([0.3, 0.7, 0.5, 0.1, 0.9, 0.2], dtype=np.float32)
        self.layer.receive_prediction(pred)
        self.layer.forward(actual)

        np.testing.assert_allclose(
            self.layer.prediction_error,
            actual - pred,
            rtol=1e-5,
            err_msg="prediction_error ≠ actual − top_down_prediction",
        )

    def test_zero_prediction_error_when_input_equals_prediction(self) -> None:
        """
        When top-down prediction exactly matches input, error must be zero.
        """
        pattern = np.ones(self.num_inputs, dtype=np.float32) * 0.5
        self.layer.receive_prediction(pattern)
        self.layer.forward(pattern)

        np.testing.assert_allclose(
            self.layer.prediction_error,
            np.zeros(self.num_inputs),
            atol=1e-6,
            err_msg="Prediction error should be zero when input == prediction.",
        )

    def test_negative_prediction_error_is_preserved(self) -> None:
        """
        Signed negative errors (actual < prediction) must be preserved in
        prediction_error, even though only the positive part drives weight updates.
        """
        actual = np.zeros(self.num_inputs, dtype=np.float32)
        pred = np.ones(self.num_inputs, dtype=np.float32)
        self.layer.receive_prediction(pred)
        self.layer.forward(actual)

        self.assertTrue(
            np.all(self.layer.prediction_error <= 0.0),
            "Negative prediction errors should not be rectified in the error buffer.",
        )

    # ──────────────────────────────────────────────────────────────────
    # B. ACh modulation
    # ──────────────────────────────────────────────────────────────────

    def test_ach_1_ignores_top_down(self) -> None:
        """
        ACh = 1.0 means pure bottom-up: effective_input should equal pre_spikes.
        Verify by checking that layer responds when bottom-up input is strong and
        top-down prediction is zero.
        """
        self.layer.w.fill(50.0)
        self.layer.set_ach_level(1.0)
        self.layer.receive_prediction(np.zeros(self.num_inputs))

        pre_spikes = np.ones(self.num_inputs, dtype=np.float32)
        any_spike = False
        for _ in range(50):
            if np.any(self.layer.forward(pre_spikes)):
                any_spike = True
                break

        self.assertTrue(any_spike, "With ACh=1.0 and strong input, neurons should spike.")

    def test_ach_0_suppresses_bottom_up(self) -> None:
        """
        ACh = 0.0 means pure top-down: bottom-up signal is completely ignored.
        With zero top-down prediction and ACh=0, effective input is zero regardless
        of pre_spikes, so neurons should not spike (absent recurrent drive).
        """
        self.layer.set_ach_level(0.0)
        self.layer.receive_prediction(np.zeros(self.num_inputs))  # zero top-down

        pre_spikes = np.ones(self.num_inputs, dtype=np.float32) * 100.0
        total_spikes = 0
        for _ in range(50):
            total_spikes += np.sum(self.layer.forward(pre_spikes))

        self.assertEqual(
            total_spikes, 0,
            "With ACh=0 and zero top-down prediction, no spikes should occur.",
        )

    def test_set_ach_level_clamps_to_unit_interval(self) -> None:
        """set_ach_level must clamp values outside [0, 1]."""
        self.layer.set_ach_level(5.0)
        self.assertAlmostEqual(self.layer.ach_level, 1.0)
        self.layer.set_ach_level(-3.0)
        self.assertAlmostEqual(self.layer.ach_level, 0.0)

    # ──────────────────────────────────────────────────────────────────
    # C. Feedback weights
    # ──────────────────────────────────────────────────────────────────

    def test_feedback_weight_shape(self) -> None:
        """feedback_w must be (num_neurons, num_inputs)."""
        self.assertEqual(
            self.layer.feedback_w.shape, (self.num_neurons, self.num_inputs)
        )

    def test_generate_prediction_shape(self) -> None:
        """generate_prediction() must return a vector of length num_inputs."""
        pred = self.layer.generate_prediction()
        self.assertEqual(pred.shape, (self.num_inputs,))

    def test_generate_prediction_values_in_unit_interval(self) -> None:
        """All prediction values must lie in [0, 1]."""
        pred = self.layer.generate_prediction()
        self.assertTrue(
            np.all(pred >= 0.0) and np.all(pred <= 1.0),
            "Prediction values out of [0, 1].",
        )

    def test_receive_prediction_updates_buffer(self) -> None:
        """receive_prediction must overwrite top_down_prediction exactly."""
        pattern = np.random.rand(self.num_inputs).astype(np.float32)
        self.layer.receive_prediction(pattern)
        np.testing.assert_array_almost_equal(
            self.layer.top_down_prediction, pattern
        )

    def test_feedback_weights_grow_on_positive_error_with_spike(self) -> None:
        """
        Feedback weights must increase when neurons spike AND there is a
        positive prediction error (actual > prediction).
        """
        self.layer.w.fill(100.0)
        self.layer.set_ach_level(1.0)
        # Set prediction below input so error is positive
        self.layer.receive_prediction(np.zeros(self.num_inputs))

        initial_fw = self.layer.feedback_w.copy()
        pre_spikes = np.ones(self.num_inputs, dtype=np.float32)

        spiked = False
        for _ in range(20):
            spikes = self.layer.forward(pre_spikes)
            if np.any(spikes):
                spiked = True
                break

        self.assertTrue(spiked, "Neurons did not spike; cannot validate weight update.")
        self.assertFalse(
            np.allclose(self.layer.feedback_w, initial_fw),
            "Feedback weights must grow when spike + positive error occurs.",
        )

    def test_feedback_weights_clamped_to_unit_interval(self) -> None:
        """Feedback weights must never exceed 1.0 regardless of feedback_lr."""
        self.layer.w.fill(200.0)
        # Initialise feedback_w at the ceiling; run hard to try to push above
        self.layer.feedback_w.fill(0.99)
        pre_spikes = np.ones(self.num_inputs, dtype=np.float32)
        for _ in range(100):
            self.layer.forward(pre_spikes)

        self.assertTrue(
            np.all(self.layer.feedback_w <= 1.0),
            "Feedback weights exceeded the 1.0 upper bound.",
        )

    def test_reset_state_clears_prediction_buffers(self) -> None:
        """reset_state must zero both top_down_prediction and prediction_error."""
        self.layer.receive_prediction(np.ones(self.num_inputs) * 0.7)
        self.layer.forward(np.ones(self.num_inputs))
        self.layer.reset_state()

        np.testing.assert_array_equal(
            self.layer.top_down_prediction,
            np.zeros(self.num_inputs),
            err_msg="top_down_prediction not cleared by reset_state().",
        )
        np.testing.assert_array_equal(
            self.layer.prediction_error,
            np.zeros(self.num_inputs),
            err_msg="prediction_error not cleared by reset_state().",
        )

    def test_reset_state_preserves_feedback_weights(self) -> None:
        """reset_state must NOT reset learned feedback_w."""
        self.layer.feedback_w.fill(0.42)
        self.layer.reset_state()
        np.testing.assert_array_almost_equal(
            self.layer.feedback_w,
            np.full_like(self.layer.feedback_w, 0.42),
            err_msg="reset_state() must preserve learned feedback weights.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)