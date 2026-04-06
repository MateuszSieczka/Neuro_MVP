import unittest
import numpy as np
from core.config import PredictiveCodingConfig
from core.predictive_coding import PredictiveCodingLayer


class TestPredictiveCodingLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.num_inputs = 4
        self.num_neurons = 6
        self.config = PredictiveCodingConfig(
            relaxation_steps=10,
            relaxation_rate=0.2,
            feedback_learning_rate=0.1,
            k_winners=2
        )
        self.layer = PredictiveCodingLayer(
            self.num_inputs, self.num_neurons, self.config
        )

    def test_initialization_shapes(self) -> None:
        """Weryfikuje poprawność wymiarów macierzy wag i buforów."""
        self.assertEqual(self.layer.feedback_w.shape, (self.num_neurons, self.num_inputs))
        self.assertEqual(self.layer.prediction_error.shape, (self.num_inputs,))
        self.assertEqual(self.layer.top_down_prediction.shape, (self.num_neurons,))

    def test_stdp_trace_update_bug1(self) -> None:
        """Sprawdza, czy ślady STDP są aktualizowane (naprawa Bug 1)."""
        pre_spikes = np.array([1, 0, 1, 0], dtype=np.float32)

        # Przed forwardem ślady powinny być zerowe (po inicjalizacji)
        self.assertEqual(np.sum(self.layer.x_pre), 0)

        self.layer.forward(pre_spikes)

        # Po forwardzie x_pre musi odzwierciedlać wejście
        self.assertGreater(self.layer.x_pre[0], 0)
        self.assertEqual(self.layer.x_pre[1], 0)

    def test_ff_drive_influence_bug_a(self) -> None:
        """Weryfikuje wpływ wag feedforward na dynamikę (naprawa Bug A)."""
        pre_spikes = np.ones(self.num_inputs, dtype=np.float32)

        # Test 1: Niskie wagi ff
        self.layer.w.fill(0.01)
        self.layer.v.fill(-70.0)
        self.layer.forward(pre_spikes)
        v_low_w = self.layer.v.copy()

        # Test 2: Wysokie wagi ff
        self.layer.reset_state()
        self.layer.w.fill(1.0)
        self.layer.v.fill(-70.0)
        self.layer.forward(pre_spikes)
        v_high_w = self.layer.v.copy()

        # Potencjał przy wysokich wagach ff powinien być wyższy (silniejszy drive)
        self.assertTrue(np.all(v_high_w >= v_low_w))

    def test_ach_modulation_on_errors(self) -> None:
        """Sprawdza, czy ACh poprawnie skaluje wyjściowe impulsy błędu.

        forward() zwraca has_spiked (num_neurons) — wzorce impulsów warstwy.
        error_spikes (num_inputs) są dostępne jako atrybut warstwy i
        skalowane przez ACh: ACh=1 → pełne transmitowanie błędów,
        ACh=0 → błędy stłumione.
        """
        pre_spikes = np.ones(self.num_inputs, dtype=np.float32)
        self.layer.prediction_error.fill(1.0)  # Sztuczny błąd

        # Pełne zaufanie oddolne (ACh = 1.0)
        self.layer.set_ach_level(1.0)
        self.layer.forward(pre_spikes)
        errors_high_ach = self.layer.error_spikes.copy()

        # Brak zaufania oddolnego (ACh = 0.0) -> brak impulsów błędu
        self.layer.reset_state()
        self.layer.set_ach_level(0.0)
        self.layer.forward(pre_spikes)
        errors_low_ach = self.layer.error_spikes.copy()

        self.assertEqual(np.sum(errors_low_ach), 0, "Przy ACh=0 błędy powinny być tłumione.")
        self.assertGreater(np.sum(errors_high_ach), 0, "Przy ACh=1 błędy powinny być transmitowane.")

    def test_feedback_weight_learning(self) -> None:
        """Weryfikuje Hebbowskie uczenie wag predykcji odgórnej."""
        self.layer.feedback_w.fill(0.1)
        initial_fb_w = self.layer.feedback_w.copy()

        # Symulacja stanu: neurony wystrzeliły, wystąpił błąd predykcji
        self.layer.has_spiked.fill(True)
        self.layer.prediction_error.fill(0.5)

        # m_t = 1.0 (wysoka dopamina/modulacja)
        self.layer.update_weights(m_t=1.0, pred_error=np.zeros(self.num_neurons))

        # Wagi feedback powinny wzrosnąć
        self.assertFalse(np.allclose(self.layer.feedback_w, initial_fb_w))
        self.assertTrue(np.all(self.layer.feedback_w >= initial_fb_w))

    def test_reset_state(self) -> None:
        """Sprawdza, czy bufor predykcji jest czyszczony."""
        self.layer.top_down_prediction.fill(0.5)
        self.layer.prediction_error.fill(0.5)

        self.layer.reset_state()

        self.assertEqual(np.sum(self.layer.top_down_prediction), 0)
        self.assertEqual(np.sum(self.layer.prediction_error), 0)


if __name__ == '__main__':
    unittest.main()