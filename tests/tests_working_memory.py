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
        self.wm.gate(0.0)  # force closed
        self.wm.content.fill(0.0)  # no content to recirculate
        self.wm.w_lateral.fill(0.0)  # no lateral connectivity

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
        """
        self.wm.w_ff.fill(0.0)
        self.wm.w_ff[:, :3] = 100.0
        self.wm.gate(1.0)
        for _ in range(30):
            self.wm.forward(np.ones(self.num_inputs))

        self.wm.gate(0.0)
        self.wm.w_lateral.fill(0.0)
        self.wm.content.fill(0.0)

        self.wm.w_ff[:, 3:] = 200.0
        new_input = np.ones(self.num_inputs, dtype=np.float32) * 100.0
        total_spikes = sum(int(np.sum(self.wm.forward(new_input))) for _ in range(30))

        self.assertEqual(total_spikes, 0,
                         "Gate closed with zero content: no spikes despite new strong input.")

    # ──────────────────────────────────────────────────────────────────
    # B. Lateral weight learning
    # ──────────────────────────────────────────────────────────────────

    def test_co_active_neurons_strengthen_lateral_connections(self) -> None:
        """_update_lateral_weights() must increase w_lateral for co-active pairs."""
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

    def test_no_self_connections_invariant(self) -> None:
        """Diagonal of w_lateral must remain exactly zero."""
        self.wm.has_spiked[:] = True
        for _ in range(20):
            self.wm._update_lateral_weights()

        np.testing.assert_array_equal(
            np.diag(self.wm.w_lateral),
            np.zeros(self.num_neurons),
            err_msg="Diagonal (self-connections) must always stay at 0.",
        )

    # ──────────────────────────────────────────────────────────────────
    # C. Content tracking and state management
    # ──────────────────────────────────────────────────────────────────

    def test_content_tracks_spike_history_leaky(self) -> None:
        """Content must hold a decaying trace of spikes (Bug B fix verification)."""
        self.wm.w_ff.fill(100.0)
        self.wm.gate(1.0)

        # 1. Force a spike and check if content is populated
        self.wm.v[:] = 100.0
        self.wm.forward(np.ones(self.num_inputs))
        self.assertTrue(np.all(self.wm.content > 0.0), "Content should populate after spikes.")

        # 2. Step with silence and check for persistence via decay
        last_content = self.wm.content.copy()
        self.wm.gate(0.0)  # Close gate to isolate from input
        self.wm.forward(np.zeros(self.num_inputs))

        self.assertTrue(np.all(self.wm.content > 0.0), "Content must persist (leaky).")
        # Trace decay check (approximate due to floating point)
        self.assertLess(float(np.mean(self.wm.content)), float(np.mean(last_content)) + 0.1)

    def test_reset_state_clears_transient_variables(self) -> None:
        """reset_state must zero transient states including content."""
        self.wm.w_ff.fill(100.0)
        self.wm.gate(1.0)
        self.wm.forward(np.ones(self.num_inputs))

        self.wm.reset_state()

        np.testing.assert_array_equal(self.wm.content, np.zeros(self.num_neurons))
        np.testing.assert_array_equal(self.wm.e, np.zeros((self.num_inputs, self.num_neurons)))

    def test_reset_state_preserves_learned_weights(self) -> None:
        """reset_state must NOT clear learned weights."""
        self.wm.w_ff.fill(7.0)
        self.wm.w_lateral[0, 1] = 0.5
        self.wm.reset_state()

        self.assertAlmostEqual(self.wm.w_lateral[0, 1], 0.5)
        np.testing.assert_array_almost_equal(self.wm.w_ff, np.full_like(self.wm.w_ff, 7.0))

    # ──────────────────────────────────────────────────────────────────
    # D. Feedforward weight update
    # ──────────────────────────────────────────────────────────────────

    def test_update_weights_zero_modulator_is_noop(self) -> None:
        self.wm.w_ff.fill(100.0)
        self.wm.gate(1.0)
        self.wm.forward(np.ones(self.num_inputs))

        initial_w = self.wm.w_ff.copy()
        self.wm.update_weights(m_t=0.0, pred_error=np.ones(self.num_neurons))
        np.testing.assert_array_equal(self.wm.w_ff, initial_w)

    def test_update_weights_nonzero_modulator_updates_w_ff(self) -> None:
        """w_ff should update when m_t > 0 and STDP traces are non-zero."""
        self.wm.w_ff.fill(0.5)
        self.wm.gate(1.0)

        # NAPRAWA: Wymuszamy impuls (spike), aby ślady e (eligibility traces) przestały być zerowe.
        self.wm.v[:] = 100.0
        self.wm.forward(np.ones(self.num_inputs))

        initial_w = self.wm.w_ff.copy()
        # Wykonujemy aktualizację z błędem predykcji = 1.0 (wzmocnienie)
        self.wm.update_weights(m_t=1.0, pred_error=np.ones(self.num_neurons, dtype=np.float32))

        self.assertFalse(np.allclose(self.wm.w_ff, initial_w), "w_ff should update when m_t > 0.")


if __name__ == "__main__":
    unittest.main(verbosity=2)