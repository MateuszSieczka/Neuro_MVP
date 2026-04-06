import unittest
import numpy as np
from core.basal_ganglia import ContinuousBGConfig, SNNDeepCritic, SNNContinuousActor, BasalGangliaAGISystem


class TestBasalGanglia(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        self.state_size = 10
        self.motor_dim = 2
        self.internal_dim = 1
        self.config = ContinuousBGConfig(
            exploration_noise=0.1,
            critic_lr=0.05,  # Wyższy LR dla wyraźniejszych zmian w testach
            actor_lr=0.01,
            gamma=0.95
        )
        self.bg_system = BasalGangliaAGISystem(
            self.state_size,
            self.motor_dim,
            self.internal_dim,
            self.config
        )

    # ──────────────────────────────────────────────────────────────────
    # A. SNNDeepCritic Tests
    # ──────────────────────────────────────────────────────────────────

    def test_critic_forward_output_range(self) -> None:
        state = np.random.randint(0, 2, self.state_size)
        v_s = self.bg_system.critic.forward(state)
        self.assertIsInstance(v_s, float)
        self.assertLess(abs(v_s), 2.0)

    def test_critic_update_reduces_error(self) -> None:
        """NAPRAWA: Wymuszenie aktywności neuronów, by ślady STDP nie były zerowe."""
        critic = self.bg_system.critic
        state = np.ones(self.state_size)

        # Wymuszamy wysokie wagi, aby na pewno wystąpiły impulsy (spikes)
        critic.w_h.fill(1.0)
        critic.w_v.fill(1.0)

        # Wykonujemy forward, aby zbudować ślad e_v i e_h
        critic.forward(state)
        self.assertTrue(np.any(critic.e_v > 0), "Ślad kwalifikowalności krytyka jest zerowy.")

        w_v_before = critic.w_v.copy()
        critic.update(td_error=1.0)

        self.assertFalse(np.array_equal(critic.w_v, w_v_before), "Wagi Krytyka nie uległy aktualizacji.")

    # ──────────────────────────────────────────────────────────────────
    # B. SNNContinuousActor Tests
    # ──────────────────────────────────────────────────────────────────

    def test_actor_action_dimensions_and_ranges(self) -> None:
        state = np.random.randint(0, 2, self.state_size)
        motor, internal = self.bg_system.actor.forward(state)
        self.assertEqual(len(motor), self.motor_dim)
        self.assertEqual(len(internal), self.internal_dim)
        self.assertTrue(np.all(motor >= -1.0) and np.all(motor <= 1.0))
        self.assertTrue(np.all(internal >= 0.0) and np.all(internal <= 1.0))

    # ──────────────────────────────────────────────────────────────────
    # C. BasalGangliaAGISystem (Integration) Tests
    # ──────────────────────────────────────────────────────────────────

    def test_step_calculates_correct_td_error(self) -> None:
        """
        Weryfikuje matematyczną poprawność błędu TD: r + gamma*V(s') - V(s).
        """
        test_cfg = ContinuousBGConfig(critic_lr=0.0, actor_lr=0.0, gamma=0.9)
        sys = BasalGangliaAGISystem(self.state_size, self.motor_dim, 1, test_cfg)
        state = np.ones(self.state_size)

        # Krok 1: Ustalamy stały punkt odniesienia dla V(s)
        # Resetujemy stan, by wynik forward był przewidywalny
        sys.critic.v_hidden.fill(0.0)
        v_old = sys.critic.forward(state)
        sys.last_v = v_old

        # Krok 2: Wykonujemy krok systemowy
        # sys.step() wywoła critic.forward() raz, co zmieni v_hidden
        reward = 1.0
        _, _, td_error = sys.step(state, reward=reward)

        # Krok 3: Ręcznie obliczamy oczekiwany błąd korzystając z faktu,
        # że znamy r i gamma, a v_new to wartość V(s') z momentu wywołania step().
        # Ponieważ LR=0, wagi są stałe, ale musimy uwzględnić stan v_hidden
        # po PIERWSZYM wywołaniu wewnątrz step().

        # Aby test był stabilny, sprawdzamy czy td_error pasuje do komponentów:
        # td_error = reward + gamma * current_v - last_v
        # Wartość current_v została obliczona wewnątrz sys.step().
        # Ponieważ v_old było ręcznie ustawione, sprawdzamy logikę sumowania:
        self.assertAlmostEqual(td_error, reward + test_cfg.gamma * sys.last_v - v_old, places=5)
    def test_terminal_state_handling(self) -> None:
        state = np.ones(self.state_size)
        self.bg_system.step(state, reward=0.0)
        last_v = self.bg_system.last_v

        reward = 5.0
        _, _, td_error = self.bg_system.step(state, reward=reward, is_terminal=True)

        self.assertAlmostEqual(td_error, reward - last_v, places=5)
        self.assertEqual(self.bg_system.last_v, 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)