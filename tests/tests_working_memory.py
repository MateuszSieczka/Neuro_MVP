import unittest
import numpy as np

from core.config import WorkingMemoryConfig
from core.working_memory import WorkingMemoryModule


class TestWorkingMemoryModule(unittest.TestCase):
    """
    Unit tests for WorkingMemoryModule.

    Validates four neuro_mvp contracts:
      A) ACh-controlled gate (open / closed routing).
      B) Hebbian lateral weight learning and no-self-connection invariant.
      C) Content tracking and state management.
      D) Three-factor weight update via update_weights().
    """

    def setUp(self) -> None:
        self.num_inputs = 5
        self.num_neurons = 10
        self.config = WorkingMemoryConfig(
            tau_m=300.0,
            gate_threshold=0.5,
            lateral_strength=1.0,
            lateral_lr=0.05,
            v_thresh=-55.0,
            v_rest=-70.0,
            v_reset=-75.0,
        )
        self.wm = WorkingMemoryModule(
            num_external_inputs=self.num_inputs,
            num_neurons=self.num_neurons,
            config=self.config,
        )

    # ──────────────────────────────────────────────────────────────────
    # A. Gate behaviour
    # ──────────────────────────────────────────────────────────────────

    def test_gate_opens_above_threshold(self) -> None:
        """ACh >= gate_threshold must open the gate."""
        self.wm.gate(0.6)
        self.assertTrue(self.wm.gate_open, "Gate should be open when ACh ≥ threshold.")

    def test_gate_closes_below_threshold(self) -> None:
        """ACh < gate_threshold must close the gate."""
        self.wm.gate(0.4)
        self.assertFalse(self.wm.gate_open, "Gate should be closed when ACh < threshold.")

    def test_gate_open_allows_external_input_to_drive_spikes(self) -> None:
        """With gate open and strong feedforward weights, external input must cause spikes."""
        self.wm.w_ff.fill(100.0)
        self.wm.gate(1.0)  # force open

        pre_spikes = np.ones(self.num_inputs, dtype=np.float32)
        spiked_any = False
        for _ in range(60):
            if np.any(self.wm.forward(pre_spikes)):
                spiked_any = True
                break

        self.assertTrue(spiked_any, "Gate open: external input failed to drive any spikes.")

    def test_gate_closed_with_no_content_blocks_external_input(self) -> None:
        """
        With gate closed and no prior content / lateral connections,
        even very strong external input must produce no spikes.
        """
        self.wm.gate(0.0)           # force closed
        self.wm.content.fill(0.0)   # no content to recirculate
        self.wm.w_lateral.fill(0.0) # no lateral connectivity

        pre_spikes = np.ones(self.num_inputs, dtype=np.float32) * 100.0
        total_spikes = 0
        for _ in range(60):
            total_spikes += int(np.sum(self.wm.forward(pre_spikes)))

        self.assertEqual(
            total_spikes, 0,
            "Gate closed with empty content: external input should produce no spikes.",
        )

    def test_gate_open_closed_transition_blocks_new_content(self) -> None:
        """
        After writing content with gate open, closing it should prevent new
        (different) external input from overwriting the held pattern.
        The lateral path may allow residual activity, but the direct ff path is off.
        """
        # Prime with input on neuron group 0
        self.wm.w_ff.fill(0.0)
        self.wm.w_ff[:, :3] = 100.0   # neurons 0-2 driven by all inputs
        self.wm.gate(1.0)
        for _ in range(30):
            self.wm.forward(np.ones(self.num_inputs))

        # Now close gate and silence input to those neurons; excite a different group
        self.wm.gate(0.0)
        self.wm.w_lateral.fill(0.0)   # kill recurrence to isolate the test
        self.wm.content.fill(0.0)

        # Strong input that would normally drive neurons 3-9
        self.wm.w_ff[: , 3:] = 200.0
        new_input = np.ones(self.num_inputs, dtype=np.float32) * 100.0
        total_spikes = sum(int(np.sum(self.wm.forward(new_input))) for _ in range(30))

        self.assertEqual(total_spikes, 0,
            "Gate closed with zero content: no spikes despite new strong input.")

    # ──────────────────────────────────────────────────────────────────
    # B. Lateral weight learning
    # ──────────────────────────────────────────────────────────────────

    def test_co_active_neurons_strengthen_lateral_connections(self) -> None:
        """
        _update_lateral_weights() must increase w_lateral[i,j] when
        both neuron i and neuron j fired simultaneously.
        """
        self.wm.has_spiked[:] = False
        self.wm.has_spiked[0] = True
        self.wm.has_spiked[2] = True

        initial = self.wm.w_lateral[0, 2]
        self.wm._update_lateral_weights()

        self.assertGreater(
            self.wm.w_lateral[0, 2],
            initial,
            "Lateral weight between co-active neurons (0, 2) did not increase.",
        )

    def test_inactive_neuron_pairs_do_not_gain_weight(self) -> None:
        """Pairs where at least one neuron was silent must NOT receive potentiation."""
        self.wm.has_spiked[:] = False
        self.wm.has_spiked[0] = True   # only neuron 0 fires
        # neuron 5 is silent

        initial_weight = self.wm.w_lateral[0, 5]
        self.wm._update_lateral_weights()

        self.assertAlmostEqual(
            self.wm.w_lateral[0, 5],
            initial_weight,
            msg="Inactive pair (0, 5) must not be potentiated.",
        )

    def test_no_self_connections_invariant(self) -> None:
        """
        Diagonal of w_lateral must remain exactly zero after any number of
        Hebbian updates, even when all neurons fire simultaneously.
        """
        self.wm.has_spiked[:] = True

        for _ in range(20):
            self.wm._update_lateral_weights()

        np.testing.assert_array_equal(
            np.diag(self.wm.w_lateral),
            np.zeros(self.num_neurons),
            err_msg="Diagonal (self-connections) must always stay at 0.",
        )

    def test_lateral_weights_do_not_exceed_one_after_many_updates(self) -> None:
        """L∞ normalisation must keep all lateral weights ≤ 1."""
        self.wm.has_spiked[:] = True
        for _ in range(200):
            self.wm._update_lateral_weights()

        off_diag = self.wm.w_lateral.copy()
        np.fill_diagonal(off_diag, 0.0)
        self.assertLessEqual(
            float(np.max(off_diag)),
            1.0 + 1e-6,
            "Lateral weights exceeded the normalisation ceiling of 1.0.",
        )

    # ──────────────────────────────────────────────────────────────────
    # C. Content tracking and state management
    # ──────────────────────────────────────────────────────────────────

    def test_content_reflects_last_spike_pattern(self) -> None:
        """After forward(), content must equal has_spiked cast to float."""
        self.wm.w_ff.fill(100.0)
        self.wm.gate(1.0)
        for _ in range(20):
            self.wm.forward(np.ones(self.num_inputs))

        np.testing.assert_array_equal(
            self.wm.content,
            self.wm.has_spiked.astype(np.float32),
            err_msg="content must equal the current has_spiked pattern.",
        )

    def test_reset_state_clears_transient_variables(self) -> None:
        """reset_state must zero v, e, traces, refrac, has_spiked, and content."""
        self.wm.w_ff.fill(100.0)
        self.wm.gate(1.0)
        for _ in range(30):
            self.wm.forward(np.ones(self.num_inputs))

        self.wm.reset_state()

        np.testing.assert_array_almost_equal(
            self.wm.v, np.full(self.num_neurons, self.config.v_rest),
            err_msg="v must be reset to v_rest.",
        )
        np.testing.assert_array_equal(self.wm.has_spiked, np.zeros(self.num_neurons, dtype=bool))
        np.testing.assert_array_equal(self.wm.content, np.zeros(self.num_neurons))
        np.testing.assert_array_equal(self.wm.e, np.zeros((self.num_inputs, self.num_neurons)))

    def test_reset_state_preserves_learned_weights(self) -> None:
        """reset_state must NOT clear w_ff or w_lateral."""
        self.wm.w_ff.fill(7.0)
        self.wm.w_lateral[0, 1] = 0.5
        self.wm.reset_state()

        np.testing.assert_array_almost_equal(
            self.wm.w_ff, np.full_like(self.wm.w_ff, 7.0),
            err_msg="w_ff must survive reset_state().",
        )
        self.assertAlmostEqual(self.wm.w_lateral[0, 1], 0.5,
            msg="w_lateral must survive reset_state().")

    # ──────────────────────────────────────────────────────────────────
    # D. Feedforward weight update
    # ──────────────────────────────────────────────────────────────────

    def test_update_weights_zero_modulator_leaves_weights_unchanged(self) -> None:
        """update_weights(m_t=0) must be a strict no-op."""
        self.wm.w_ff.fill(100.0)
        self.wm.gate(1.0)
        for _ in range(20):
            self.wm.forward(np.ones(self.num_inputs))

        initial_w = self.wm.w_ff.copy()
        self.wm.update_weights(m_t=0.0, pred_error=np.ones(self.num_neurons))

        np.testing.assert_array_equal(
            self.wm.w_ff, initial_w,
            err_msg="w_ff must not change when m_t=0.",
        )

    def test_update_weights_nonzero_modulator_changes_weights(self) -> None:
        """update_weights with m_t > 0 and non-zero eligibility traces must modify w_ff."""
        self.wm.w_ff.fill(100.0)
        self.wm.gate(1.0)
        for _ in range(30):
            self.wm.forward(np.ones(self.num_inputs))

        initial_w = self.wm.w_ff.copy()
        self.wm.update_weights(m_t=1.0, pred_error=np.ones(self.num_neurons))

        self.assertFalse(
            np.allclose(self.wm.w_ff, initial_w),
            "w_ff should change when m_t > 0 and eligibility traces are non-zero.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)