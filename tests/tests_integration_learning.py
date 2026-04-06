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

        # Szybsza inicjalizacja membrany
        self.layer.w = np.random.uniform(1.0, 2.0, (self.state_dim, self.hidden_dim)).astype(np.float32)
        self.net.add_layer("L1", self.layer)

        # Ustawienia stabilne dla testu
        bg_config = ContinuousBGConfig(critic_lr=0.05, tau_hidden=5.0)
        self.bg = BasalGangliaAGISystem(state_size=self.hidden_dim, motor_dim=self.action_dim, config=bg_config)
        self.bg.critic.w_h = np.random.uniform(0.1, 0.5, (self.hidden_dim, bg_config.hidden_size)).astype(np.float32)

        self.wm = SNNWorldModel(state_size=self.hidden_dim, action_size=self.action_dim)
        self.nm = NeuromodulatorSystem()
        self.buffer = ReplayBuffer(capacity=500)

    def test_end_to_end_learning(self):
        target_state = np.array([5.0, 5.0, 0, 0], dtype=np.float32)

        # 1. Ewaluacja początkowa (tylko 1 uderzenie, aby sprawdzić "ślepego" Krytyka)
        self.net.step({"L1": target_state}, neuromodulator=self.nm)
        l1_repr = self.layer.has_spiked.astype(np.float32)
        _, _, initial_td_err = self.bg.step(l1_repr, reward=1.0, is_terminal=True)

        print(f"\nPoczątkowy błąd TD: {abs(initial_td_err):.4f}")

        # Reset przed startem
        self.net.reset_state()
        self.nm.reset()
        self.bg.last_v = 0.0

        # 2. Uczenie w jednym, stabilnym epizodzie (np. 100 kroków)
        # Nagroda = 1.0 i is_terminal = True TYLKO w ostatnim kroku!
        for i in range(100):
            self.net.step({"L1": target_state}, neuromodulator=self.nm)
            l1_repr = self.layer.has_spiked.astype(np.float32)

            is_terminal = (i == 99)
            reward = 1.0 if is_terminal else 0.0

            motor, _, td_err = self.bg.step(l1_repr, reward=reward, is_terminal=is_terminal)
            self.wm.update(l1_repr, motor, l1_repr, m_t=self.nm.learning_rate_modulation)

            self.buffer.store(
                state=l1_repr, action=motor, reward=reward, next_state=l1_repr,
                layer_traces={"L1": self.layer.e.copy()},
                layer_outputs={"L1": l1_repr},
                prediction_error=self.layer.prediction_error,
                layer_errors={"L1": self.layer.prediction_error},
                done=is_terminal  # TUTAJ PRZEKAZUJESZ FLAGĘ
            )
            self.net.update_weights(self.nm)
            self.nm.update(self.layer.prediction_error, td_error=td_err)

        # 3. Konsolidacja
        self.buffer.sleep_phase({"L1": self.layer}, self.wm, self.nm)

        # 4. Ewaluacja końcowa po konsolidacji
        self.net.reset_state()
        self.nm.reset()
        self.bg.last_v = 0.0

        # Przewijamy warstwę L1 do stabilnego stanu reprezentacji
        for i in range(100):
            self.net.step({"L1": target_state}, neuromodulator=self.nm)
            l1_repr = self.layer.has_spiked.astype(np.float32)

        # Zastrzyk nagrody w terminal state - sprawdzamy, jak bardzo Krytyk ją przewidział
        _, _, final_td_err = self.bg.step(l1_repr, reward=1.0, is_terminal=True)

        print(f"Końcowy błąd TD: {abs(final_td_err):.4f}")
        self.assertLess(abs(final_td_err), abs(initial_td_err), "Błąd TD nie spadł.")