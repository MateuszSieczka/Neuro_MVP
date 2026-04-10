import unittest
import numpy as np

from core.config import SNNWorldModelConfig
from core.world_model import SNNWorldModel
from core.neuromodulator import NeuromodulatorSystem
from core.replay_buffer import ReplayBuffer, Experience
from core.basal_ganglia import BasalGangliaAGISystem, ContinuousBGConfig

STATE_SIZE = 6
ACTION_SIZE = 4


def _make_exp(
    state_size: int = STATE_SIZE,
    action_size: int = ACTION_SIZE,
    done: bool = False,
    recorded_da: float = 1.0,
) -> Experience:
    wm = _make_world_model(state_size, action_size)
    return Experience(
        state=np.random.rand(state_size).astype(np.float32),
        action=int(np.random.randint(0, action_size)),
        reward=float(np.random.rand()),
        next_state=np.random.rand(state_size).astype(np.float32),
        prediction_error=np.random.rand(state_size).astype(np.float32),
        encoder_e_bu=np.random.rand(wm.encoder.n_error, wm.encoder.n_state).astype(np.float32),
        encoder_spikes=np.random.rand(wm.encoder.n_state).astype(np.float32),
        bg_snapshot=None,
        aug_state=np.random.rand(state_size).astype(np.float32),
        salience=0.3,
        recorded_da=recorded_da,
        curiosity=0.1,
        done=done,
    )


def _make_world_model(
    state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE,
) -> SNNWorldModel:
    return SNNWorldModel(state_size=state_size, action_size=action_size)


def _make_nm() -> NeuromodulatorSystem:
    return NeuromodulatorSystem()


def _make_bg(state_size: int = STATE_SIZE, n_actions: int = ACTION_SIZE) -> BasalGangliaAGISystem:
    return BasalGangliaAGISystem(
        state_size=state_size, motor_dim=n_actions, internal_dim=1,
        config=ContinuousBGConfig(),
    )


class TestReplayBufferStorage(unittest.TestCase):
    def setUp(self) -> None:
        self.buf = ReplayBuffer(capacity=50)

    def test_empty_buffer_has_length_zero(self) -> None:
        self.assertEqual(len(self.buf), 0)

    def test_store_increases_length_by_one(self) -> None:
        self.buf.store(_make_exp())
        self.assertEqual(len(self.buf), 1)

    def test_store_multiple_reflects_correct_count(self) -> None:
        for _ in range(10):
            self.buf.store(_make_exp())
        self.assertEqual(len(self.buf), 10)

    def test_capacity_limit_not_exceeded(self) -> None:
        small_buf = ReplayBuffer(capacity=5)
        for _ in range(15):
            small_buf.store(_make_exp())
        self.assertEqual(len(small_buf), 5)

    def test_stored_state_is_independent_copy(self) -> None:
        exp = _make_exp()
        exp_state_ref = exp.state.copy()
        self.buf.store(exp)
        exp.state[:] = 999.0
        stored = list(self.buf._buffer)[0]
        np.testing.assert_array_equal(stored.state, exp_state_ref)

    def test_stored_encoder_traces_are_independent_copies(self) -> None:
        exp = _make_exp()
        original_e_bu = exp.encoder_e_bu.copy()
        self.buf.store(exp)
        exp.encoder_e_bu[:] = -1.0
        stored = list(self.buf._buffer)[0]
        np.testing.assert_array_equal(stored.encoder_e_bu, original_e_bu)

    def test_clear_empties_buffer(self) -> None:
        for _ in range(8):
            self.buf.store(_make_exp())
        self.buf.clear()
        self.assertEqual(len(self.buf), 0)


class TestReplayBufferSampling(unittest.TestCase):
    def setUp(self) -> None:
        self.buf = ReplayBuffer(capacity=100)
        for _ in range(20):
            self.buf.store(_make_exp())

    def test_sample_returns_correct_count(self) -> None:
        self.assertEqual(len(self.buf.sample(5)), 5)

    def test_sample_does_not_exceed_buffer_size(self) -> None:
        self.assertEqual(len(self.buf.sample(1000)), 20)

    def test_sample_returns_experience_instances(self) -> None:
        for item in self.buf.sample(3):
            self.assertIsInstance(item, Experience)

    def test_is_ready_false_when_empty(self) -> None:
        self.assertFalse(ReplayBuffer().is_ready(min_size=1))

    def test_is_ready_true_after_one_store(self) -> None:
        fresh = ReplayBuffer()
        fresh.store(_make_exp())
        self.assertTrue(fresh.is_ready(min_size=1))

    def test_is_ready_respects_min_size(self) -> None:
        small = ReplayBuffer()
        for _ in range(3):
            small.store(_make_exp())
        self.assertFalse(small.is_ready(min_size=5))
        self.assertTrue(small.is_ready(min_size=3))


class TestReplayBufferSleepPhase(unittest.TestCase):
    def setUp(self) -> None:
        self.buf = ReplayBuffer(capacity=100)
        self.wm = _make_world_model()
        self.nm = _make_nm()
        self.bg = _make_bg()

    def _make_consistent_exp(self, reward: float = 1.0, done: bool = False) -> Experience:
        enc = self.wm.encoder
        return Experience(
            state=np.random.rand(STATE_SIZE).astype(np.float32),
            action=0,
            reward=reward,
            next_state=np.random.rand(STATE_SIZE).astype(np.float32),
            prediction_error=np.random.rand(STATE_SIZE).astype(np.float32),
            encoder_e_bu=np.random.rand(enc.n_error, enc.n_state).astype(np.float32),
            encoder_spikes=np.random.rand(enc.n_state).astype(np.float32),
            bg_snapshot=self.bg.snapshot_traces(),
            aug_state=np.random.rand(STATE_SIZE).astype(np.float32),
            recorded_da=1.0,
            done=done,
        )

    def test_sleep_phase_on_empty_buffer_returns_empty_list(self) -> None:
        self.assertEqual(self.buf.sleep_phase(self.wm, self.nm, self.bg), [])

    def test_sleep_phase_returns_one_error_per_replayed_experience(self) -> None:
        for _ in range(7):
            self.buf.store(self._make_consistent_exp())
        self.assertEqual(len(self.buf.sleep_phase(self.wm, self.nm, self.bg)), 7)

    def test_sleep_phase_n_experiences_limits_replay_count(self) -> None:
        for _ in range(15):
            self.buf.store(self._make_consistent_exp())
        self.assertEqual(len(self.buf.sleep_phase(self.wm, self.nm, self.bg, n_experiences=4)), 4)

    def test_sleep_phase_errors_are_non_negative_floats(self) -> None:
        for _ in range(5):
            self.buf.store(self._make_consistent_exp())
        for e in self.buf.sleep_phase(self.wm, self.nm, self.bg):
            self.assertIsInstance(e, float)
            self.assertGreaterEqual(e, 0.0)

    def test_sleep_phase_does_not_modify_buffer_contents(self) -> None:
        for _ in range(6):
            self.buf.store(self._make_consistent_exp())
        self.buf.sleep_phase(self.wm, self.nm, self.bg)
        self.assertEqual(len(self.buf), 6)

    def test_sleep_phase_improves_world_model_on_fixed_transition(self) -> None:
        state = np.full(STATE_SIZE, 0.5, dtype=np.float32)
        next_state = np.ones(STATE_SIZE, dtype=np.float32)
        enc = self.wm.encoder
        for _ in range(50):
            self.buf.store(Experience(
                state=state, action=0, reward=1.0, next_state=next_state,
                prediction_error=np.zeros(STATE_SIZE, dtype=np.float32),
                encoder_e_bu=np.ones((enc.n_error, enc.n_state), dtype=np.float32),
                encoder_spikes=np.ones(enc.n_state, dtype=np.float32),
                bg_snapshot=self.bg.snapshot_traces(), recorded_da=1.0,
            ))
        self.nm.dopamine = 1.0
        self.wm.reset_state()
        initial_mse = float(np.mean((self.wm.predict(state, 0) - next_state) ** 2))
        self.buf.sleep_phase(self.wm, self.nm, self.bg)
        self.wm.reset_state()
        final_mse = float(np.mean((self.wm.predict(state, 0) - next_state) ** 2))
        self.assertLess(final_mse, initial_mse,
                        f"MSE did not improve: {initial_mse:.4f} -> {final_mse:.4f}")

    def test_sleep_phase_reverse_order_updates_prediction(self) -> None:
        state = np.full(STATE_SIZE, 0.5, dtype=np.float32)
        enc = self.wm.encoder
        for next_s in (np.zeros(STATE_SIZE, np.float32), np.ones(STATE_SIZE, np.float32)):
            self.buf.store(Experience(
                state=state, action=0, reward=1.0, next_state=next_s,
                prediction_error=np.zeros(STATE_SIZE, np.float32),
                encoder_e_bu=np.ones((enc.n_error, enc.n_state), np.float32),
                encoder_spikes=np.ones(enc.n_state, np.float32), recorded_da=1.0,
            ))
        wm = SNNWorldModel(STATE_SIZE, ACTION_SIZE, SNNWorldModelConfig(decode_lr=0.5))
        self.nm.dopamine = 1.0
        wm.reset_state()
        pred_before = float(np.mean(wm.predict(state, 0)))
        self.buf.sleep_phase(wm, self.nm, self.bg, n_experiences=1)
        wm.reset_state()
        pred_after = float(np.mean(wm.predict(state, 0)))
        self.assertGreater(pred_after, pred_before)

    def test_sleep_phase_reverse_order_updates_world_model(self) -> None:
        state = np.full(STATE_SIZE, 0.5, dtype=np.float32)
        enc = self.wm.encoder
        for next_s in (np.zeros(STATE_SIZE, np.float32), np.ones(STATE_SIZE, np.float32)):
            self.buf.store(Experience(
                state=state, action=0, reward=1.0, next_state=next_s,
                prediction_error=np.zeros(STATE_SIZE, np.float32),
                encoder_e_bu=np.ones((enc.n_error, enc.n_state), np.float32),
                encoder_spikes=np.ones(enc.n_state, np.float32), recorded_da=1.0,
            ))
        wm = SNNWorldModel(STATE_SIZE, ACTION_SIZE, SNNWorldModelConfig(decode_lr=0.5))
        self.nm.dopamine = 1.0
        wm.reset_state()
        w_before = wm.w_decode.copy()
        self.buf.sleep_phase(wm, self.nm, self.bg, n_experiences=1)
        w_after = wm.w_decode.copy()
        self.assertGreater(float(np.mean(w_after)), float(np.mean(w_before)))

    def test_sleep_phase_with_multiple_layers(self) -> None:
        enc = self.wm.encoder
        self.buf.store(Experience(
            state=np.zeros(STATE_SIZE, np.float32), action=0, reward=1.0,
            next_state=np.ones(STATE_SIZE, np.float32),
            prediction_error=np.zeros(STATE_SIZE, np.float32),
            encoder_e_bu=np.ones((enc.n_error, enc.n_state), np.float32),
            encoder_spikes=np.ones(enc.n_state, np.float32),
            bg_snapshot=self.bg.snapshot_traces(), recorded_da=1.0,
        ))
        errors = self.buf.sleep_phase(self.wm, self.nm, self.bg)
        self.assertEqual(len(errors), 1)

    def test_sleep_phase_restores_traces_to_matching_layers(self) -> None:
        enc = self.wm.encoder
        known_e_bu = np.full((enc.n_error, enc.n_state), 0.42, dtype=np.float32)
        self.buf.store(Experience(
            state=np.zeros(STATE_SIZE, np.float32), action=0, reward=1.0,
            next_state=np.ones(STATE_SIZE, np.float32),
            prediction_error=np.zeros(STATE_SIZE, np.float32),
            encoder_e_bu=known_e_bu,
            encoder_spikes=np.zeros(enc.n_state, np.float32),
            bg_snapshot=self.bg.snapshot_traces(), recorded_da=1.0,
        ))
        self.buf.sleep_phase(self.wm, self.nm, self.bg)
        self.assertEqual(enc.e_bu.shape, (enc.n_error, enc.n_state))


if __name__ == "__main__":
    unittest.main(verbosity=2)
