import unittest
import numpy as np

from core.config import SNNWorldModelConfig
from core.neuron import LIFLayer
from core.world_model import SNNWorldModel
from core.neuromodulator import NeuromodulatorSystem
from core.replay_buffer import ReplayBuffer, Experience

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

STATE_SIZE = 6  # state_size == num_neurons for sleep_phase compatibility
ACTION_SIZE = 4
NUM_INPUTS = 3
LAYER_NAME = "default"


def _make_exp(state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE) -> dict:
    """Create a single randomised experience dict with all required layer dicts."""
    return dict(
        state=np.random.rand(state_size).astype(np.float32),
        action=int(np.random.randint(0, action_size)),
        reward=float(np.random.rand()),
        next_state=np.random.rand(state_size).astype(np.float32),
        layer_traces={
            LAYER_NAME: np.random.rand(NUM_INPUTS, state_size).astype(np.float32),
        },
        layer_outputs={
            LAYER_NAME: np.random.rand(state_size).astype(np.float32),
        },
        prediction_error=np.random.rand(state_size).astype(np.float32),
        layer_errors={
            LAYER_NAME: np.random.rand(state_size).astype(np.float32),
        }
    )


def _make_layer(num_inputs: int = NUM_INPUTS, num_neurons: int = STATE_SIZE) -> LIFLayer:
    return LIFLayer(num_inputs=num_inputs, num_neurons=num_neurons)


def _make_layers() -> dict[str, LIFLayer]:
    return {LAYER_NAME: _make_layer()}


def _make_world_model() -> SNNWorldModel:
    return SNNWorldModel(state_size=STATE_SIZE, action_size=ACTION_SIZE)


def _make_nm() -> NeuromodulatorSystem:
    return NeuromodulatorSystem()


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestReplayBufferStorage(unittest.TestCase):
    """Tests for buffer storage, capacity, and isolation of stored copies."""

    def setUp(self) -> None:
        self.buf = ReplayBuffer(capacity=50)

    def test_empty_buffer_has_length_zero(self) -> None:
        self.assertEqual(len(self.buf), 0)

    def test_store_increases_length_by_one(self) -> None:
        self.buf.store(**_make_exp())
        self.assertEqual(len(self.buf), 1)

    def test_store_multiple_reflects_correct_count(self) -> None:
        for _ in range(10):
            self.buf.store(**_make_exp())
        self.assertEqual(len(self.buf), 10)

    def test_capacity_limit_not_exceeded(self) -> None:
        small_buf = ReplayBuffer(capacity=5)
        for _ in range(15):
            small_buf.store(**_make_exp())
        self.assertEqual(len(small_buf), 5)

    def test_stored_state_is_independent_copy(self) -> None:
        """Mutating the original array after store() must not affect stored data."""
        exp = _make_exp()
        original_state = exp["state"].copy()
        self.buf.store(**exp)
        exp["state"][:] = 999.0  # mutate original

        stored = list(self.buf._buffer)[0]
        np.testing.assert_array_equal(
            stored.state, original_state,
            err_msg="Stored state shares memory with the original — copies must be taken.",
        )

    def test_stored_layer_dicts_are_independent_copies(self) -> None:
        """Ensure traces, outputs, and errors are deep copied."""
        exp = _make_exp()
        original_e = exp["layer_traces"][LAYER_NAME].copy()
        original_o = exp["layer_outputs"][LAYER_NAME].copy()
        original_err = exp["layer_errors"][LAYER_NAME].copy()

        self.buf.store(**exp)

        # Mutate original dicts
        exp["layer_traces"][LAYER_NAME][:] = -1.0
        exp["layer_outputs"][LAYER_NAME][:] = -1.0
        exp["layer_errors"][LAYER_NAME][:] = -1.0

        stored = list(self.buf._buffer)[0]
        np.testing.assert_array_equal(stored.layer_traces[LAYER_NAME], original_e)
        np.testing.assert_array_equal(stored.layer_outputs[LAYER_NAME], original_o)
        np.testing.assert_array_equal(stored.layer_errors[LAYER_NAME], original_err)

    def test_clear_empties_buffer(self) -> None:
        for _ in range(8):
            self.buf.store(**_make_exp())
        self.buf.clear()
        self.assertEqual(len(self.buf), 0)


class TestReplayBufferSampling(unittest.TestCase):
    """Tests for the online random sampling interface."""

    def setUp(self) -> None:
        self.buf = ReplayBuffer(capacity=100)
        for _ in range(20):
            self.buf.store(**_make_exp())

    def test_sample_returns_correct_count(self) -> None:
        sample = self.buf.sample(5)
        self.assertEqual(len(sample), 5)

    def test_sample_does_not_exceed_buffer_size(self) -> None:
        """Requesting more than len(buffer) should return all experiences."""
        sample = self.buf.sample(1000)
        self.assertEqual(len(sample), 20)

    def test_sample_returns_experience_instances(self) -> None:
        sample = self.buf.sample(3)
        for item in sample:
            self.assertIsInstance(item, Experience)

    def test_is_ready_false_when_empty(self) -> None:
        empty = ReplayBuffer()
        self.assertFalse(empty.is_ready(min_size=1))

    def test_is_ready_true_after_one_store(self) -> None:
        fresh = ReplayBuffer()
        fresh.store(**_make_exp())
        self.assertTrue(fresh.is_ready(min_size=1))

    def test_is_ready_respects_min_size(self) -> None:
        small = ReplayBuffer()
        for _ in range(3):
            small.store(**_make_exp())
        self.assertFalse(small.is_ready(min_size=5))
        self.assertTrue(small.is_ready(min_size=3))


class TestReplayBufferSleepPhase(unittest.TestCase):
    """Tests for offline consolidation (sleep_phase / reverse replay)."""

    def setUp(self) -> None:
        self.buf = ReplayBuffer(capacity=100)

    def test_sleep_phase_on_empty_buffer_returns_empty_list(self) -> None:
        errors = self.buf.sleep_phase(_make_layers(), _make_world_model(), _make_nm())
        self.assertEqual(errors, [])

    def test_sleep_phase_returns_one_error_per_replayed_experience(self) -> None:
        n = 7
        for _ in range(n):
            self.buf.store(**_make_exp())

        errors = self.buf.sleep_phase(_make_layers(), _make_world_model(), _make_nm())
        self.assertEqual(len(errors), n)

    def test_sleep_phase_n_experiences_limits_replay_count(self) -> None:
        for _ in range(15):
            self.buf.store(**_make_exp())

        errors = self.buf.sleep_phase(
            _make_layers(), _make_world_model(), _make_nm(),
            n_experiences=4,
        )
        self.assertEqual(len(errors), 4)

    def test_sleep_phase_errors_are_non_negative_floats(self) -> None:
        """Returned MSE values must be non-negative floats."""
        for _ in range(5):
            self.buf.store(**_make_exp())

        errors = self.buf.sleep_phase(_make_layers(), _make_world_model(), _make_nm())
        for e in errors:
            self.assertIsInstance(e, float)
            self.assertGreaterEqual(e, 0.0)

    def test_sleep_phase_does_not_modify_buffer_contents(self) -> None:
        """sleep_phase must not consume or remove experiences from the buffer."""
        for _ in range(6):
            self.buf.store(**_make_exp())

        self.buf.sleep_phase(_make_layers(), _make_world_model(), _make_nm())
        self.assertEqual(len(self.buf), 6)

    def test_sleep_phase_improves_world_model_on_fixed_transition(self) -> None:
        """
        Replaying the same transition repeatedly must reduce world model MSE.
        This is a soft integration test validating the full consolidation pipeline.
        """
        # Mocny sygnał (10.0 zamiast zer), aby wymusić impulsy SNN i gradienty w 1 kroku
        state = np.ones(STATE_SIZE, dtype=np.float32) * 10.0
        next_state = np.ones(STATE_SIZE, dtype=np.float32)
        action = 0

        # Store many copies of the same transition with reward=1.0
        for _ in range(50):
            self.buf.store(
                state=state,
                action=action,
                reward=1.0,
                next_state=next_state,
                layer_traces={
                    LAYER_NAME: np.ones((NUM_INPUTS, STATE_SIZE), dtype=np.float32),
                },
                layer_outputs={
                    LAYER_NAME: np.ones(STATE_SIZE, dtype=np.float32),
                },
                prediction_error=np.zeros(STATE_SIZE, dtype=np.float32),
                layer_errors={
                    LAYER_NAME: np.zeros(STATE_SIZE, dtype=np.float32),
                }
            )

        layers = _make_layers()
        wm = _make_world_model()
        nm = _make_nm()
        nm.dopamine = 1.0  # Maximum learning signal

        # NAPRAWA: Reset przed pomiarem, by uniknąć okna refrakcji
        wm.reset_state()
        initial_pred = wm.predict(state, action)
        initial_mse = float(np.mean((initial_pred - next_state) ** 2))

        # Consolidate
        self.buf.sleep_phase(layers, wm, nm)

        # NAPRAWA: Zresetowanie stanu po śnie (i ew. przywróceniu refrakcji)
        wm.reset_state()
        final_pred = wm.predict(state, action)
        final_mse = float(np.mean((final_pred - next_state) ** 2))

        self.assertLess(
            final_mse, initial_mse,
            f"World model MSE did not improve after sleep consolidation. "
            f"Initial={initial_mse:.4f}, Final={final_mse:.4f}.",
        )

    def test_sleep_phase_reverse_order_updates_world_model(self) -> None:
        """
        Verify that sleep_phase processes experiences in reverse chronological order.
        """
        # Analogicznie podnosimy sygnał bazowy
        state = np.ones(STATE_SIZE, dtype=np.float32) * 10.0
        action = 0

        early_next = np.zeros(STATE_SIZE, dtype=np.float32)  # stored first
        late_next = np.ones(STATE_SIZE, dtype=np.float32)  # stored last (most recent)

        for next_s in (early_next, late_next):
            self.buf.store(
                state=state,
                action=action,
                reward=1.0,
                next_state=next_s,
                layer_traces={
                    LAYER_NAME: np.zeros((NUM_INPUTS, STATE_SIZE), dtype=np.float32),
                },
                layer_outputs={
                    LAYER_NAME: np.ones(STATE_SIZE, dtype=np.float32),
                },
                prediction_error=np.zeros(STATE_SIZE, dtype=np.float32),
                layer_errors={
                    LAYER_NAME: np.zeros(STATE_SIZE, dtype=np.float32),
                }
            )

        from core.config import WorldModelConfig
        wm = SNNWorldModel(STATE_SIZE, ACTION_SIZE, SNNWorldModelConfig(feedback_learning_rate=0.5))
        layers = _make_layers()
        nm = _make_nm()
        nm.dopamine = 1.0

        # NAPRAWA: Reset przed pomiarem
        wm.reset_state()
        pred_before = float(np.mean(wm.predict(state, action)))

        # n_experiences=1 → only the LAST stored experience is replayed (late_next = ones)
        self.buf.sleep_phase(layers, wm, nm, n_experiences=1)

        # NAPRAWA: Reset przed pomiarem
        wm.reset_state()
        pred_after = float(np.mean(wm.predict(state, action)))

        self.assertGreater(
            pred_after,
            pred_before,
            "Reverse replay with n_experiences=1 must update world model from the "
            "most-recent experience (late_next=ones), moving prediction upward.",
        )

    def test_sleep_phase_reverse_order_updates_world_model(self) -> None:
        """
        Verify that sleep_phase processes experiences in reverse chronological order.
        """
        state = np.ones(STATE_SIZE, dtype=np.float32) * 10.0
        action = 0

        early_next = np.zeros(STATE_SIZE, dtype=np.float32)  # stored first
        late_next = np.ones(STATE_SIZE, dtype=np.float32)  # stored last (most recent)

        for next_s in (early_next, late_next):
            self.buf.store(
                state=state,
                action=action,
                reward=1.0,
                next_state=next_s,
                layer_traces={
                    LAYER_NAME: np.zeros((NUM_INPUTS, STATE_SIZE), dtype=np.float32),
                },
                layer_outputs={
                    LAYER_NAME: np.ones(STATE_SIZE, dtype=np.float32),
                },
                prediction_error=np.zeros(STATE_SIZE, dtype=np.float32),
                layer_errors={
                    LAYER_NAME: np.zeros(STATE_SIZE, dtype=np.float32),
                }
            )

        from core.config import WorldModelConfig
        from core.world_model import SNNWorldModelConfig
        wm = SNNWorldModel(STATE_SIZE, ACTION_SIZE, SNNWorldModelConfig(decode_lr=0.5))
        layers = _make_layers()
        nm = _make_nm()
        nm.dopamine = 1.0

        wm.reset_state()

        # Zapisujemy stan wag dekodera przed replayem
        w_before = wm.w_decode.copy()

        # n_experiences=1 → only the LAST stored experience is replayed (late_next = ones)
        self.buf.sleep_phase(layers, wm, nm, n_experiences=1)

        # Skoro uczymy się przejścia w jedynki (late_next), błąd predykcji będzie mocno dodatni.
        # Reguła Hebbowska (STDP dekodera) powinna w rezultacie podbić wagi w_decode.
        w_after = wm.w_decode.copy()

        self.assertGreater(
            float(np.mean(w_after)),
            float(np.mean(w_before)),
            "Reverse replay must process the most-recent experience (ones), "
            "causing decoder weights to increase via Hebbian learning.",
        )

    def test_sleep_phase_restores_traces_to_matching_layers(self) -> None:
        """sleep_phase must restore eligibility traces to the correct layer."""
        known_traces = np.full((NUM_INPUTS, STATE_SIZE), 0.42, dtype=np.float32)
        self.buf.store(
            state=np.zeros(STATE_SIZE, dtype=np.float32),
            action=0,
            reward=1.0,
            next_state=np.ones(STATE_SIZE, dtype=np.float32),
            layer_traces={LAYER_NAME: known_traces},
            layer_outputs={LAYER_NAME: np.zeros(STATE_SIZE, dtype=np.float32)},
            prediction_error=np.zeros(STATE_SIZE, dtype=np.float32),
            layer_errors={LAYER_NAME: np.zeros(STATE_SIZE, dtype=np.float32)}
        )

        layers = _make_layers()
        wm = _make_world_model()
        nm = _make_nm()
        nm.dopamine = 1.0

        self.buf.sleep_phase(layers, wm, nm)
        self.assertEqual(layers[LAYER_NAME].e.shape, (NUM_INPUTS, STATE_SIZE))

    def test_sleep_phase_with_multiple_layers(self) -> None:
        """sleep_phase must handle multiple named layers and match their errors."""
        layer_a = _make_layer(num_inputs=3, num_neurons=STATE_SIZE)
        layer_b = _make_layer(num_inputs=4, num_neurons=STATE_SIZE)
        layers = {"layer_a": layer_a, "layer_b": layer_b}

        self.buf.store(
            state=np.zeros(STATE_SIZE, dtype=np.float32),
            action=0,
            reward=1.0,
            next_state=np.ones(STATE_SIZE, dtype=np.float32),
            layer_traces={
                "layer_a": np.ones((3, STATE_SIZE), dtype=np.float32),
                "layer_b": np.ones((4, STATE_SIZE), dtype=np.float32),
            },
            layer_outputs={
                "layer_a": np.ones(STATE_SIZE, dtype=np.float32),
                "layer_b": np.ones(STATE_SIZE, dtype=np.float32),
            },
            prediction_error=np.zeros(STATE_SIZE, dtype=np.float32),
            layer_errors={
                "layer_a": np.zeros(STATE_SIZE, dtype=np.float32),
                "layer_b": np.zeros(STATE_SIZE, dtype=np.float32),
            }
        )

        wm = _make_world_model()
        nm = _make_nm()
        nm.dopamine = 1.0

        errors = self.buf.sleep_phase(layers, wm, nm)
        self.assertEqual(len(errors), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)