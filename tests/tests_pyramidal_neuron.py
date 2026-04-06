import unittest
import numpy as np
from core.pyramidal_neuron import PyramidalLayer
from core.config import PyramidalConfig


class TestPyramidalLayer(unittest.TestCase):
    """
    Zestaw testów dla warstwy neuronów piramidalnych.
    Weryfikuje:
      - Wieloprzedziałową integrację (bazalną i apikalną).
      - Mechanizm plateau wapniowego (BAC firing).
      - Modulację progu wyładowania.
      - Plastyczność zależną od burstów (serii wyładowań).
    """

    def setUp(self) -> None:
        self.num_inputs = 10
        self.num_neurons = 5
        self.config = PyramidalConfig(
            apical_threshold=0.3,
            apical_boost=15.0,
            burst_stdp_factor=4.0,
            relaxation_steps=5,
            relaxation_rate=0.1,
            v_thresh=-55.0,
            v_rest=-70.0
        )
        self.layer = PyramidalLayer(self.num_inputs, self.num_neurons, self.config)

    def test_initialization(self) -> None:
        """Sprawdza poprawność inicjalizacji wag i stanu przedziałów."""
        # Wagi bazalne i apikalne powinny mieć te same wymiary
        self.assertEqual(self.layer.w.shape, (self.num_inputs, self.num_neurons))
        self.assertEqual(self.layer.w_apical.shape, (self.num_inputs, self.num_neurons))
        # Stan przedziału apikalnego
        self.assertEqual(self.layer.v_apical.shape, (self.num_neurons,))
        self.assertEqual(self.layer.top_down_prediction.shape, (self.num_neurons,))

    def test_apical_integration_and_decay(self) -> None:
        """Weryfikuje integrację sygnałów odgórnych w przedziale apikalnym."""
        # Podanie silnej predykcji odgórnej
        prediction = np.ones(self.num_neurons, dtype=np.float32)
        self.layer.receive_prediction(prediction)

        # Wykonanie kroku forward (integracja sygnału)
        self.layer.forward(np.zeros(self.num_inputs))
        self.assertTrue(np.all(self.layer.v_apical > 0.0), "v_apical nie wzrósł po otrzymaniu predykcji.")

        # Sprawdzenie zaniku (decay) w kolejnym kroku bez wejścia
        v_old = self.layer.v_apical.copy()
        self.layer.receive_prediction(np.zeros(self.num_neurons))
        self.layer.forward(np.zeros(self.num_inputs))
        self.assertTrue(np.all(self.layer.v_apical < v_old), "v_apical nie uległ osłabieniu po braku sygnału.")

    def test_calcium_plateau_and_threshold_boost(self) -> None:
        """Sprawdza, czy przekroczenie progu apikalnego obniża próg somy."""
        # 1. Uzbrojenie plateau
        self.layer.v_apical.fill(1.0)
        self.layer.forward(np.zeros(self.num_inputs))
        self.assertTrue(np.all(self.layer.in_plateau), "Plateau powinno być aktywne.")

        # NAPRAWA: Pełne czyszczenie stanu konkurencji i refrakcji po kroku uzbrajającym.
        # Neuron strzelił w poprzednim kroku, więc musimy wyczyścić licznik refrakcji,
        # aby pozwolić mu strzelić ponownie we właściwym teście.
        self.layer.window_spike_counts.fill(0)
        self.layer._current_window_size = 0
        self.layer.refrac_count.fill(0)
        self.layer.has_spiked.fill(False)

        # 2. Właściwy test progu
        # Ustawiamy v na -56 (poniżej normy -55, powyżej boosta -70)
        self.layer.v.fill(-56.0)
        self.layer.w.fill(1.0)  # ff_drive neutralny

        # Wyłączamy relaksację, by sprawdzić statyczny próg (zmieniamy zamrożony config)
        from dataclasses import replace
        self.layer.pyr_config = replace(self.layer.pyr_config, relaxation_steps=0)

        # Teraz neuron powinien strzelić, bo -56 >= -70 i nie jest w refrakcji
        spikes = self.layer.forward(np.zeros(self.num_inputs))
        self.assertTrue(np.all(spikes), "Neuron powinien strzelić dzięki apical_boost.")


    def test_burst_detection_and_stdp_boost(self) -> None:
        """Weryfikuje wykrywanie burstów i wzmocnienie śladów STDP."""
        self.layer.v_apical.fill(1.0)  # Gwarantuje plateau
        self.layer.e.fill(0.1)

        # Gwarantujemy spike: silny ff_drive
        self.layer.w.fill(100.0)
        pre_spikes = np.ones(self.num_inputs, dtype=np.float32)

        self.layer.forward(pre_spikes)

        # is_burst wymaga jednoczesnego spike'a i plateau
        self.assertTrue(np.any(self.layer.is_burst), "Nie wykryto serii wyładowań (burst).")
        # Ślady e powinny być > 0.1 (0.1 * boost * decay)
        self.assertTrue(np.any(self.layer.e > 0.1), "Ślady STDP nie zostały wzmocnione.")


    def test_ach_modulation_logic(self) -> None:
        """Sprawdza wpływ poziomu ACh na wagę sygnału apikalnego."""
        # Wysoki poziom ACh -> mniejsze zaufanie do predykcji odgórnych
        self.layer.set_ach_level(1.0)
        self.assertAlmostEqual(self.layer._ach_apical_scale, 0.5)

        # Niski poziom ACh -> maksymalne zaufanie do predykcji (marzenia senne/halucynacje)
        self.layer.set_ach_level(0.0)
        self.assertAlmostEqual(self.layer._ach_apical_scale, 1.0)

    def test_predictive_coding_relaxation_bug_a(self) -> None:
        """Weryfikuje wpływ wag feedforward na dynamikę v (Bug A fix)."""
        # NAPRAWA: Używamy słabszych sygnałów, aby uniknąć clippingu potencjału
        pre_spikes = np.ones(self.num_inputs, dtype=np.float32) * 0.1

        # 1. Brak wag
        self.layer.w.fill(0.0)
        self.layer.v.fill(-70.0)
        self.layer.forward(pre_spikes)
        v_low = self.layer.v.copy()

        # 2. Obecność wag
        self.layer.reset_state()
        self.layer.w.fill(2.0)
        self.layer.v.fill(-70.0)
        self.layer.forward(pre_spikes)
        v_high = self.layer.v.copy()

        # Po naprawie Bug A, ff_drive (pre @ w) musi podnieść gradient v
        self.assertTrue(np.any(v_high > v_low), "Wagi w nie wpłynęły na potencjał.")
    def test_generate_prediction_TiedWeights(self) -> None:

        """Weryfikuje generowanie predykcji odgórnej przy użyciu wag sprzężonych."""
        # Symulacja spike'a w neuronie 0
        self.layer.has_spiked.fill(False)
        self.layer.has_spiked[0] = True

        pred = self.layer.generate_prediction()
        # Predykcja powinna mieć rozmiar wejścia
        self.assertEqual(pred.shape, (self.num_inputs,))
        # Powinna być rzutem przez transponowaną macierz w_apical
        expected = np.clip(self.layer.w_apical[:, 0] * self.layer.pyr_config.feedback_strength, 0.0, 1.0)
        np.testing.assert_array_almost_equal(pred, expected)


if __name__ == '__main__':
    unittest.main()