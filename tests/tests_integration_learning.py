import unittest
import numpy as np
from core.network import NetworkGraph
from core.predictive_coding import PredictiveCodingLayer
from core.basal_ganglia import BasalGangliaAGISystem, ContinuousBGConfig
from core.world_model import SNNWorldModel
from core.neuromodulator import NeuromodulatorSystem
from core.replay_buffer import ReplayBuffer


class TestSystemLearning(unittest.TestCase):
    def setUp(self):
        self.state_dim = 4
        self.action_dim = 2
        self.hidden_dim = 8

        self.net = NetworkGraph()
        self.layer = PredictiveCodingLayer(num_inputs=self.state_dim, num_neurons=self.hidden_dim)

        # Szybsza inicjalizacja membrany (within the weight clip range [−1, 2])
        self.layer.w = np.random.uniform(0.5, 1.0, (self.state_dim, self.hidden_dim)).astype(np.float32)
        self.net.add_layer("L1", self.layer)

        # Ustawienia stabilne dla testu
        bg_config = ContinuousBGConfig(critic_lr=0.05, tau_hidden=5.0)
        self.bg = BasalGangliaAGISystem(state_size=self.hidden_dim, motor_dim=self.action_dim, config=bg_config)
        self.bg.critic.w_h = np.random.uniform(0.1, 0.5, (self.hidden_dim, bg_config.hidden_size)).astype(np.float32)

        self.wm = SNNWorldModel(state_size=self.hidden_dim, action_size=self.action_dim)
        self.nm = NeuromodulatorSystem()
        self.buffer = ReplayBuffer(capacity=500)

    def _warmup_and_get_repr(self, target_state, steps=20):
        """Rozgrzewka SNN do stabilnej reprezentacji. Zwraca l1_repr."""
        for _ in range(steps):
            self.net.step({"L1": target_state}, neuromodulator=self.nm)
        return self.layer.has_spiked.astype(np.float32)

    def _eval_critic_value(self, l1_repr, warmup_steps=10):
        """Rozgrzewka ukrytego stanu Krytyka i odczyt V(s) BEZ aktualizacji wag."""
        for _ in range(warmup_steps):
            self.bg.critic.forward(l1_repr)
        return self.bg.critic.forward(l1_repr)

    def test_end_to_end_learning(self):
        target_state = np.array([5.0, 5.0, 0, 0], dtype=np.float32)

        # Fixed evaluation representation — isolates critic learning from
        # SNN layer weight drift. Uses a plausible sparse spike pattern
        # matching the hidden_dim (8 neurons, 4 active).
        eval_repr = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)

        # 1. Ewaluacja początkowa: V(s) „ślepego" Krytyka
        initial_v = self._eval_critic_value(eval_repr)
        initial_td_err = 1.0 - initial_v

        print(f"\nPoczątkowy V(s): {initial_v:.4f}, TD error: {abs(initial_td_err):.4f}")

        # Pełny reset przed treningiem (ślady z ewaluacji nie powinny wpływać na naukę)
        self.net.reset_state()
        self.nm.reset()
        self.bg.reset_state()

        # 2. Uczenie w 3 epizodach (TD bootstrapping potrzebuje powtórzeń)
        for episode in range(3):
            self.net.reset_state()
            self.bg.reset_state()

            for i in range(100):
                self.net.step({"L1": target_state}, neuromodulator=self.nm)
                l1_repr = self.layer.has_spiked.astype(np.float32)

                is_terminal = (i == 39)
                reward = 1.0 if is_terminal else 0.0
                
                # Zamiast zewnętrznej flagi końca epizodu, symulujemy potężny skok 
                # "zaskoczenia/wagi sytuacji" dokładnie w momencie otrzymania nagrody.
                current_salience = 1.0 if is_terminal else 0.0

                motor, _, td_err = self.bg.step(l1_repr, reward=reward, is_terminal=is_terminal)
                self.wm.update(l1_repr, motor, l1_repr, m_t=self.nm.learning_rate_modulation)

                self.buffer.store(
                    state=l1_repr, action=motor, reward=reward, next_state=l1_repr,
                    layer_traces={"L1": self.layer.e.copy()},
                    layer_outputs={"L1": l1_repr},
                    prediction_error=self.layer.prediction_error,
                    layer_errors={"L1": self.layer.prediction_error},
                    salience=current_salience
                )
                self.net.update_weights(self.nm)
                self.nm.update(self.layer.prediction_error, td_error=td_err)

        # 3. Konsolidacja (wzmacnia wagi warstw SNN)
        self.buffer.sleep_phase({"L1": self.layer}, self.wm, self.nm)

        # 4. Ewaluacja końcowa: V(s) po nauce (ta sama repr co na starcie)
        self.bg.critic.reset_state()
        final_v = self._eval_critic_value(eval_repr)
        final_td_err = 1.0 - final_v

        print(f"Końcowy V(s): {final_v:.4f}, TD error: {abs(final_td_err):.4f}")
        self.assertLess(abs(final_td_err), abs(initial_td_err), "Błąd TD nie spadł.")