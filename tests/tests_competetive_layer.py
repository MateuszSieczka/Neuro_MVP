import unittest
import numpy as np
from core.config import KWTAConfig
from core.competitive_layer import CompetitiveLIFLayer


class TestCompetitiveLIFLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.num_inputs = 5
        self.num_neurons = 10
        # Zwiększamy okno czasowe, aby ułatwić testowanie proaktywnej inhibicji
        self.config = KWTAConfig(k_winners=2, i_inh=50.0, window_ms=100)
        self.layer = CompetitiveLIFLayer(self.num_inputs, self.num_neurons, self.config)

    def test_k_winners_selection(self) -> None:
        """Weryfikuje, czy po zakończeniu fazy wyłaniani są tylko zwycięzcy."""
        # Symulacja aktywności w oknie
        self.layer.window_spike_counts[0] = 5
        self.layer.window_spike_counts[1] = 10
        self.layer.window_spike_counts[2] = 2

        self.layer.trigger_phase_reset()
        # Pusty forward, aby wyzwolić logikę resetu fazy
        self.layer.forward(np.zeros(self.num_inputs))

        winners = self.layer.last_winners
        self.assertEqual(len(winners), self.config.k_winners)
        self.assertIn(1, winners)
        self.assertIn(0, winners)

    def test_lateral_inhibition_impact(self) -> None:
        """Sprawdza, czy przegrani dostają karę do potencjału v, a zwycięzca reset."""
        # NAPRAWA: Zwiększamy w, aby wymusić spike'a (musi przebić próg -55.0)
        # Przy w=200, V_new ≈ -60*0.95 + (-70+200)*0.05 ≈ -57 + 6.5 = -50.5 (SPIKE!)
        self.layer.v.fill(-60.0)
        self.layer.window_spike_counts[9] = 10  # Ustawiamy 9 jako zwycięzcę

        self.layer.w.fill(200.0)
        pre_spikes = np.zeros(self.num_inputs)
        pre_spikes[0] = 1.0

        self.layer.trigger_phase_reset()
        self.layer.forward(pre_spikes)

        # Zwycięzca (9) powinien wystrzelić i zostać zresetowany
        self.assertEqual(self.layer.v[9], self.layer.config.v_reset)

        # Przegrani (np. 0) nie strzelili i dostali karę i_inh (50.0)
        # Ich v powinno być w okolicy -110 (start -60, kara -50, plus lekki leak)
        self.assertLess(self.layer.v[0], -105.0)

    def test_proactive_inhibition(self) -> None:
        """Weryfikuje, czy nadaktywność obniża potencjał przed detekcją spike'a."""
        self.layer.v.fill(-60.0)
        self.layer.window_spike_counts[5] = 100
        self.layer._current_window_size = 10  # Musi być > 0, by zadziałał dzielnik

        # Zerujemy wagi, by sprawdzić czystą inhibicję bez wpływu wejścia
        self.layer.w.fill(0.0)

        v_before = self.layer.v[5]
        # Wykonujemy krok, w którym zadziała _apply_proactive_inhibition
        self.layer.forward(np.zeros(self.num_inputs))

        # Potencjał powinien spaść (kara za nadaktywność w bieżącym oknie)
        self.assertLess(self.layer.v[5], v_before)

    def test_reset_state_clears_winners(self) -> None:
        self.layer.last_winners = np.array([1, 2])
        self.layer.window_spike_counts.fill(5)

        self.layer.reset_state()

        self.assertEqual(len(self.layer.last_winners), 0)
        self.assertEqual(np.sum(self.layer.window_spike_counts), 0)
        self.assertEqual(self.layer._current_window_size, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)