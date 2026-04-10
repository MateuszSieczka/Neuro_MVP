import unittest
import numpy as np
from core.network import NetworkGraph
from core.predictive_coding import PredictiveCodingLayer
from core.basal_ganglia import BasalGangliaAGISystem, ContinuousBGConfig
from core.world_model import SNNWorldModel
from core.neuromodulator import NeuromodulatorSystem
from core.replay_buffer import ReplayBuffer, Experience


class TestSystemLearning(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.state_dim = 4
        self.action_dim = 2
        self.hidden_dim = 8

        self.net = NetworkGraph()
        self.layer = PredictiveCodingLayer(num_inputs=self.state_dim, num_neurons=self.hidden_dim)

        self.layer.w = np.random.uniform(0.5, 1.0, (self.state_dim, self.hidden_dim)).astype(np.float32)
        self.net.add_layer("L1", self.layer)

        bg_config = ContinuousBGConfig(critic_lr=0.05, hidden_size=64)
        self.bg = BasalGangliaAGISystem(state_size=self.hidden_dim, motor_dim=self.action_dim, config=bg_config)
        self.bg.critic.w_h = np.random.uniform(0.1, 0.5, (self.hidden_dim, bg_config.hidden_size)).astype(np.float32)

        self.wm = SNNWorldModel(state_size=self.hidden_dim, action_size=self.action_dim)
        self.nm = NeuromodulatorSystem()
        self.buffer = ReplayBuffer(capacity=500)

    def _eval_critic_value(self, l1_repr, warmup_steps=10):
        self.bg.critic.reset_state()
        for _ in range(warmup_steps):
            self.bg.critic.forward(l1_repr)
        return self.bg.critic.peek(l1_repr)

    def test_end_to_end_learning(self):
        target_state = np.array([5.0, 5.0, 0, 0], dtype=np.float32)
        eval_repr = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)

        initial_v = self._eval_critic_value(eval_repr)
        initial_td_err = 1.0 - initial_v

        print(f"\nPoczątkowy V(s): {initial_v:.4f}, TD error: {abs(initial_td_err):.4f}")

        self.net.reset_state()
        self.nm.reset()
        self.bg.reset_state()

        for episode in range(3):
            self.net.reset_state()
            self.bg.reset_state()

            for i in range(100):
                self.net.step({"L1": target_state}, neuromodulator=self.nm)
                l1_repr = self.layer.has_spiked.astype(np.float32)

                is_terminal = (i == 39)
                reward = 1.0 if is_terminal else 0.0

                motor, _, td_err = self.bg.step(l1_repr, reward=reward, is_terminal=is_terminal)
                self.wm.update(l1_repr, motor, l1_repr, m_t=self.nm.learning_rate_modulation)

                self.buffer.store(Experience(
                    state=l1_repr, action=motor, reward=reward, next_state=l1_repr,
                    prediction_error=self.layer.prediction_error,
                    encoder_e_bu=self.wm.encoder.e_bu.copy(),
                    encoder_spikes=self.wm.encoder.spikes_state.astype(np.float32),
                    bg_snapshot=self.bg.snapshot_traces(),
                    salience=1.0 if is_terminal else 0.0,
                ))
                self.net.update_weights(self.nm)
                self.nm.update(self.layer.prediction_error, td_error=td_err)

                if is_terminal:
                    break

        self.buffer.sleep_phase(self.wm, self.nm, self.bg)

        self.bg.critic.reset_state()
        final_v = self._eval_critic_value(eval_repr)
        final_td_err = 1.0 - final_v

        print(f"Końcowy V(s): {final_v:.4f}, TD error: {abs(final_td_err):.4f}")
        self.assertLess(abs(final_td_err), abs(initial_td_err), "Błąd TD nie spadł.")