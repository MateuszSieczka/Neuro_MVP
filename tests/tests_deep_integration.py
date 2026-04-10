"""
Deep integration tests for the Neuro_MVP SNN framework.

These tests verify emergent properties that arise from the *interaction*
of multiple subsystems, not from any single component in isolation:

  1. Concept Emergence (SequenceMemory + temporal Hebbian)
     Repeated stimulus sequences produce bidirectional transition clusters
     — unnamed but real "concepts" — and temporal prediction errors decrease.

  2. Predictive Coding Free Energy Minimisation (PredictiveCodingLayer +
     two-layer hierarchy + feedback)
     A layer learns to predict its own input via feedback_w; residual
     prediction error drops as the generative model converges.

  3. Reward-Gated Scalability (BG + WorldModel + NM + ReplayBuffer)
     Full pipeline: the BG critic's value estimate improves, the world
     model's prediction error decreases, and no NaN/Inf appears.
"""

import unittest
import numpy as np

from core.network import NetworkGraph
from core.predictive_coding import PredictiveCodingLayer
from core.config import (
    PredictiveCodingConfig, SequenceMemoryConfig,
)
from core.sequence_memory import SequenceMemory
from core.basal_ganglia import BasalGangliaAGISystem, ContinuousBGConfig
from core.world_model import SNNWorldModel
from core.neuromodulator import NeuromodulatorSystem
from core.replay_buffer import ReplayBuffer, Experience


# =====================================================================
# Helpers
# =====================================================================

def _make_binary_pattern(dim: int, active_indices: list[int]) -> np.ndarray:
    """Create a binary spike pattern with specific neurons active."""
    p = np.zeros(dim, dtype=np.float32)
    for idx in active_indices:
        p[idx] = 1.0
    return p


# =====================================================================
# Test 1: Concept Emergence via Temporal Hebbian Clusters
# =====================================================================

class TestConceptEmergence(unittest.TestCase):
    """
    Expose a SequenceMemory to repeating patterns.
    Verify that temporal Hebbian learning creates:
      a) Bidirectional clusters when transitions are mutual (A↔B).
      b) Decreasing temporal prediction error as sequences become familiar.
    """

    def setUp(self):
        self.num_neurons = 16
        self.seq_cfg = SequenceMemoryConfig(
            learning_rate=0.05,
            decay=0.999,
            max_weight=1.0,
        )
        self.seq_mem = SequenceMemory(self.num_neurons, self.seq_cfg)

        # Two distinct sparse patterns for bidirectional association
        self.pattern_A = _make_binary_pattern(self.num_neurons, [0, 1, 2, 3])
        self.pattern_B = _make_binary_pattern(self.num_neurons, [4, 5, 6, 7])

        # Third pattern for the cycle test
        self.pattern_C = _make_binary_pattern(self.num_neurons, [8, 9, 10, 11])

        # Disjoint noise pattern — should NOT cluster with A/B
        self.pattern_noise = _make_binary_pattern(self.num_neurons, [12, 13, 14, 15])

    def test_temporal_clusters_emerge_from_mutual_transitions(self):
        """
        Alternating A→B→A→B... creates bidirectional associations.
        get_temporal_clusters() requires BOTH transition_w[i,j] > thresh AND
        transition_w[j,i] > thresh, so mutual transitions are needed.
        """
        # 80 repetitions of A→B→A→B...
        for _ in range(80):
            self.seq_mem.observe(self.pattern_A)
            self.seq_mem.observe(self.pattern_B)

        clusters = self.seq_mem.get_temporal_clusters(threshold=0.05)
        clustered_neurons = set()
        for c in clusters:
            clustered_neurons |= c

        # Neurons from A and B should form a cluster together
        expected = {0, 1, 2, 3, 4, 5, 6, 7}
        overlap = expected & clustered_neurons
        self.assertGreaterEqual(
            len(overlap), 6,
            f"Za mało neuronów z A↔B w klastrach: {overlap}"
        )

        # Noise neurons must NOT appear
        noise = {12, 13, 14, 15}
        self.assertEqual(
            len(noise & clustered_neurons), 0,
            "Neurony szumowe nie powinny wchodzić do klastra."
        )

    def test_unidirectional_transitions_build_strong_weights(self):
        """
        A→B→C→A cycle builds strong unidirectional transition weights.
        Verifies the Hebbian rule creates correct directional associations.
        """
        sequence = [self.pattern_A, self.pattern_B, self.pattern_C]
        for _ in range(50):
            for pat in sequence:
                self.seq_mem.observe(pat)

        tw = self.seq_mem.transition_w

        # A→B: neuron 0 (A) active at t → neuron 4 (B) predicted at t+1
        # transition_w[post, pre], so tw[4, 0] should be strong
        ab_strength = np.mean([tw[b, a] for a in [0, 1, 2, 3] for b in [4, 5, 6, 7]])
        # B→C
        bc_strength = np.mean([tw[c, b] for b in [4, 5, 6, 7] for c in [8, 9, 10, 11]])
        # C→A
        ca_strength = np.mean([tw[a, c] for c in [8, 9, 10, 11] for a in [0, 1, 2, 3]])

        # Non-existent transitions (A→C, B→A) should be much weaker
        ac_strength = np.mean([tw[c, a] for a in [0, 1, 2, 3] for c in [8, 9, 10, 11]])

        print(f"\n[ConceptEmergence] Transition strengths: "
              f"A→B={ab_strength:.3f}, B→C={bc_strength:.3f}, "
              f"C→A={ca_strength:.3f}, A→C(noise)={ac_strength:.3f}")

        self.assertGreater(ab_strength, 0.1, "Tranzycja A→B za słaba.")
        self.assertGreater(bc_strength, 0.1, "Tranzycja B→C za słaba.")
        self.assertGreater(ca_strength, 0.1, "Tranzycja C→A za słaba.")
        self.assertGreater(ab_strength, ac_strength * 2,
                           "Silna tranzycja A→B powinna być dużo silniejsza niż A→C.")

    def test_prediction_error_decreases_with_familiarity(self):
        """As the sequence becomes familiar, temporal prediction error drops."""
        sequence = [self.pattern_A, self.pattern_B, self.pattern_C]

        early_errors = []
        for _ in range(5):
            for pat in sequence:
                err = self.seq_mem.observe(pat)
                early_errors.append(float(np.mean(np.abs(err))))
        mean_early = np.mean(early_errors)

        for _ in range(45):
            for pat in sequence:
                self.seq_mem.observe(pat)

        late_errors = []
        for _ in range(5):
            for pat in sequence:
                err = self.seq_mem.observe(pat)
                late_errors.append(float(np.mean(np.abs(err))))
        mean_late = np.mean(late_errors)

        print(f"\n[ConceptEmergence] Średni błąd temporalny: "
              f"początek={mean_early:.4f}, koniec={mean_late:.4f}")
        self.assertLess(
            mean_late, mean_early,
            "Błąd predykcji temporalnej nie spadł po powtórzeniach sekwencji."
        )


# =====================================================================
# Test 2: Predictive Coding Free Energy Minimisation
# =====================================================================

class TestHierarchicalPredictiveCoding(unittest.TestCase):
    """
    Verify the core predictive coding dynamics:

    a) A single PC layer reduces its own prediction error over time
       (feedback_w learns to model the input → free energy drops).

    b) A two-layer hierarchy with square dimensions (N→N, N→N) lets
       the higher layer generate non-zero top-down predictions that
       modulate the lower layer's processing.

    Note: PredictiveCodingLayer.forward() returns error_spikes of dim
    num_inputs. For feedforward connections to work, the next layer's
    num_inputs must equal the previous layer's num_inputs (error space).
    Using square layers (num_inputs == num_neurons) avoids this issue.
    """

    def setUp(self):
        np.random.seed(42)
        self.dim = 8  # Square: num_inputs == num_neurons

        self.pc_config = PredictiveCodingConfig(
            k_winners=4,
            feedback_learning_rate=0.01,
            feedback_strength=0.5,
            relaxation_steps=5,
            relaxation_rate=0.15,
        )

        self.nm = NeuromodulatorSystem()

    def test_single_layer_prediction_error_drops(self):
        """
        A single PC layer exposed to constant input: prediction_error
        should decrease as feedback_w learns to model the input.
        This is the fundamental free energy minimisation mechanism.
        """
        layer = PredictiveCodingLayer(self.dim, self.dim, self.pc_config)
        layer.w = np.random.uniform(0.8, 1.5, (self.dim, self.dim)).astype(np.float32)

        stimulus = np.array([1, 1, 0.5, 0, 0, 0, 0, 0], dtype=np.float32)

        # Phase 1: early prediction errors
        early_errors = []
        for _ in range(30):
            layer.forward(stimulus)
            early_errors.append(float(np.mean(np.abs(layer.prediction_error))))
            layer.update_weights(m_t=0.5, pred_error=layer.prediction_error)
        mean_early = np.mean(early_errors[-10:])

        # Phase 2: extended training
        for _ in range(200):
            layer.forward(stimulus)
            layer.update_weights(m_t=0.5, pred_error=layer.prediction_error)

        # Phase 3: late prediction errors
        late_errors = []
        for _ in range(30):
            layer.forward(stimulus)
            late_errors.append(float(np.mean(np.abs(layer.prediction_error))))
            layer.update_weights(m_t=0.5, pred_error=layer.prediction_error)
        mean_late = np.mean(late_errors[-10:])

        print(f"\n[PC FreeEnergy] Błąd predykcji: "
              f"początek={mean_early:.4f}, koniec={mean_late:.4f}")
        self.assertLess(
            mean_late, mean_early,
            "Warstwa PC nie zminimalizowała wolnej energii (prediction_error nie spadł)."
        )

    def test_two_layer_feedback_generates_predictions(self):
        """
        Two square PC layers in a manual hierarchy: L1 feeds has_spiked to L2,
        L2 feeds predictions back to L1.

        After training, L2's feedback_w should encode a non-trivial generative
        model of its input. We verify this by checking that feedback_w energy
        grows significantly — proving that the Hebbian feedback learning rule
        has built an internal model of the stimulus.

        Note: PredictiveCodingLayer.forward() returns Poisson-encoded error_spikes
        which are too sparse for reliable feedforward excitation of a downstream
        LIF layer. We bypass NetworkGraph and directly feed L1.has_spiked
        to L2 to isolate the feedback_w learning mechanism.

        Biological analogy: feedback_w IS the generative model. The momentary
        spiking pattern is ephemeral; the learned weights are the engram.
        """
        # Stronger relaxation so L2 accumulates enough voltage to fire
        strong_pc = PredictiveCodingConfig(
            k_winners=4,
            feedback_learning_rate=0.01,
            feedback_strength=0.5,
            relaxation_steps=10,
            relaxation_rate=0.3,
        )

        l1 = PredictiveCodingLayer(self.dim, self.dim, strong_pc)
        l2 = PredictiveCodingLayer(self.dim, self.dim, strong_pc)

        # Weights within [0,1] to avoid immediate clipping by update_weights
        l1.w = np.random.uniform(0.5, 1.0, (self.dim, self.dim)).astype(np.float32)
        l2.w = np.random.uniform(0.5, 1.0, (self.dim, self.dim)).astype(np.float32)

        # All-ones stimulus ensures L1 fires consistently
        stimulus = np.ones(self.dim, dtype=np.float32)

        initial_fb_energy = float(np.sum(np.abs(l2.feedback_w)))
        l2_total_spikes = 0

        for _ in range(300):
            l1.forward(stimulus)
            l1_spikes = l1.has_spiked.astype(np.float32)

            l2.forward(l1_spikes)
            l2_total_spikes += int(np.sum(l2.has_spiked))

            prediction = l2.generate_prediction()
            l1.receive_prediction(prediction)

            l1.update_weights(m_t=0.5, pred_error=l1.prediction_error)
            l2.update_weights(m_t=0.5, pred_error=l2.prediction_error)

        final_fb_energy = float(np.sum(np.abs(l2.feedback_w)))

        print(f"\n[HierarchicalPC] L2 total spikes: {l2_total_spikes}, "
              f"feedback_w energy: {initial_fb_energy:.2f} → {final_fb_energy:.2f}")

        # The generative model (feedback_w) should have grown substantially
        self.assertGreater(
            final_fb_energy, initial_fb_energy * 2.0,
            "L2 feedback_w nie wyuczyła modelu generatywnego (energia nie wzrosła 2×)."
        )
        # L2 should have fired at least once (even if sporadically)
        self.assertGreater(
            l2_total_spikes, 0,
            "L2 nigdy nie wyemitowała impulsu w ciągu 300 kroków."
        )

    def test_hierarchical_error_reduction(self):
        """
        In a two-layer PC network, L1's prediction error should decrease
        as L2 learns to predict L1's activity via feedback.
        """
        l1 = PredictiveCodingLayer(self.dim, self.dim, self.pc_config)
        l2 = PredictiveCodingLayer(self.dim, self.dim, self.pc_config)

        l1.w = np.random.uniform(0.8, 1.5, (self.dim, self.dim)).astype(np.float32)
        l2.w = np.random.uniform(0.8, 1.5, (self.dim, self.dim)).astype(np.float32)

        net = NetworkGraph()
        net.add_layer("L1", l1)
        net.add_layer("L2", l2)
        net.connect("L1", "L2", connection_type="feedforward")
        net.connect("L2", "L1", connection_type="feedback")

        stimulus = np.array([1, 1, 0.5, 0, 0, 0, 0, 0], dtype=np.float32)

        # Phase 1: early errors
        early_errors = []
        for _ in range(30):
            net.step({"L1": stimulus}, neuromodulator=self.nm)
            early_errors.append(float(np.mean(np.abs(l1.prediction_error))))
            net.update_weights(self.nm)
            self.nm.update(l1.prediction_error, td_error=0.0)
        mean_early = np.mean(early_errors[-10:])

        # Phase 2: extended training
        for _ in range(300):
            net.step({"L1": stimulus}, neuromodulator=self.nm)
            net.update_weights(self.nm)
            self.nm.update(l1.prediction_error, td_error=0.0)

        # Phase 3: late errors
        late_errors = []
        for _ in range(30):
            net.step({"L1": stimulus}, neuromodulator=self.nm)
            late_errors.append(float(np.mean(np.abs(l1.prediction_error))))
            net.update_weights(self.nm)
            self.nm.update(l1.prediction_error, td_error=0.0)
        mean_late = np.mean(late_errors[-10:])

        print(f"\n[HierarchicalPC] L1 prediction error: "
              f"early={mean_early:.4f}, late={mean_late:.4f}")
        self.assertLess(
            mean_late, mean_early,
            "Hierarchiczna predykcja nie zredukowała błędu L1."
        )


# =====================================================================
# Test 3: Reward-Gated Multi-Layer Scalability
# =====================================================================

class TestRewardGatedScalability(unittest.TestCase):
    """
    Full-stack integration with the BG, WorldModel, Neuromodulation,
    and ReplayBuffer. Uses a square-dimension PC layer as the sensory
    front-end so the critic always receives a meaningful representation.

    Verifies:
      a) BG critic value improves across multiple episodes.
      b) World model prediction error decreases with repeated transitions.
      c) No NaN or explosion after extended training + consolidation.
    """

    def setUp(self):
        np.random.seed(42)
        self.input_dim = 8
        self.hidden_dim = 8  # Square for compatibility
        self.action_dim = 2

        pc_cfg = PredictiveCodingConfig(
            k_winners=4,
            relaxation_steps=5,
            relaxation_rate=0.15,
            feedback_learning_rate=0.005,
        )

        self.layer = PredictiveCodingLayer(self.input_dim, self.hidden_dim, pc_cfg)
        self.layer.w = np.random.uniform(0.8, 1.5,
                                          (self.input_dim, self.hidden_dim)).astype(np.float32)

        self.net = NetworkGraph()
        self.net.add_layer("L1", self.layer)

        bg_cfg = ContinuousBGConfig(critic_lr=0.05, hidden_size=64)
        self.bg = BasalGangliaAGISystem(
            state_size=self.hidden_dim, motor_dim=self.action_dim, config=bg_cfg,
        )
        self.bg.critic.w_h = np.random.uniform(
            0.1, 0.5, (self.hidden_dim, bg_cfg.hidden_size)
        ).astype(np.float32)

        self.wm = SNNWorldModel(state_size=self.hidden_dim, action_size=self.action_dim)
        self.nm = NeuromodulatorSystem()
        self.buffer = ReplayBuffer(capacity=1000)

    def _warmup_and_get_repr(self, stimulus, steps=20):
        for _ in range(steps):
            self.net.step({"L1": stimulus}, neuromodulator=self.nm)
        return self.layer.has_spiked.astype(np.float32)

    def _eval_critic_value(self, repr_vec, warmup_steps=10):
        """Evaluate V(s) cleanly using peek (no membrane accumulation bias)."""
        # Reset ensures we start from clean state
        self.bg.critic.reset_state()
        # Warmup the membrane so the hidden representation is meaningful
        for _ in range(warmup_steps):
            self.bg.critic.forward(repr_vec)
        # Use peek for the actual measurement — side-effect-free
        return self.bg.critic.peek(repr_vec)

    def _run_episode(self, stimulus, episode_len=30):
        self.net.reset_state()
        self.bg.reset_state()

        for step_i in range(episode_len):
            self.net.step({"L1": stimulus}, neuromodulator=self.nm)
            l1_repr = self.layer.has_spiked.astype(np.float32)

            is_terminal = (step_i == episode_len - 1)
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

    # ------------------------------------------------------------------

    def test_critic_value_improves_over_episodes(self):
        """
        BG critic learns to predict reward across multi-step episodes.

        Uses bg.step() with a fixed representation — the same code path
        as the real agent.  Over many episodes of (many non-terminal steps
        + terminal reward), the critic's V(s) should move closer to the
        discounted terminal reward.

        Why multi-step episodes: the SNN critic relies on eligibility trace
        accumulation across timesteps within an episode. Single-step resets
        create artificially large traces that cause oscillatory divergence.
        This matches the biological reality: synaptic eligibility tags
        need temporal integration to provide stable credit assignment.
        """
        # Fixed representation — used for BOTH training and evaluation
        fixed_repr = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)

        # Baseline: untrained critic's value estimate
        self.bg.critic.reset_state()
        initial_v = self.bg.critic.forward(fixed_repr)

        # Full reset before training
        self.bg.reset_state()

        # Training: multi-step episodes using bg.step() (same as real agent).
        # Terminal reward at end of each episode; non-terminal steps get 0.
        episode_len = 20
        for _ in range(15):
            self.bg.reset_state()
            for step_i in range(episode_len):
                is_terminal = (step_i == episode_len - 1)
                reward = 1.0 if is_terminal else 0.0
                self.bg.step(fixed_repr, reward=reward, is_terminal=is_terminal)

        # Evaluate: same representation, trained critic
        self.bg.critic.reset_state()
        final_v = self.bg.critic.forward(fixed_repr)

        # The true value depends on gamma and episode length.
        # With default gamma=0.99 and episode_len=20: V* = gamma^19 ≈ 0.826.
        # We don't assert exact convergence — just that the critic moved closer.
        initial_err = abs(initial_v)   # Before training, target is >0; initial ~0 is far
        final_err_sign = final_v       # After training, V should be positive (reward at terminal)

        print(f"\n[Scalability] Critic V(s): initial={initial_v:.4f}, final={final_v:.4f}")
        self.assertGreater(
            final_v, initial_v,
            "Krytyk nie nauczył się dodatniej wartości stanu po wielu epizodach z terminalną nagrodą."
        )

    def test_world_model_curiosity_decreases(self):
        """
        World model prediction error decreases as the same transition is repeated.

        The WM's internal encoder (PredictiveCodingLayer) needs sufficient
        synaptic drive to fire. With default hidden_size=64 and sparse input,
        excitation is spread too thinly across 64 neurons. A smaller
        hidden_size=16 concentrates excitation and ensures reliable firing,
        enabling the Hebbian decoder to actually learn.

        This tests the core curiosity principle: familiar transitions produce
        smaller prediction errors (lower surprise → lower exploration drive).
        """
        from core.config import SNNWorldModelConfig

        # Smaller hidden layer = more concentrated excitation = reliable spikes
        wm_cfg = SNNWorldModelConfig(hidden_size=16, decode_lr=0.02)
        wm = SNNWorldModel(
            state_size=self.hidden_dim, action_size=self.action_dim, config=wm_cfg,
        )

        # All-ones state ensures maximum encoder excitation
        l1_repr = np.ones(self.hidden_dim, dtype=np.float32)
        dummy_action = np.zeros(self.action_dim, dtype=np.float32)

        # Warmup: let encoder membranes stabilize
        for _ in range(50):
            combined = wm._build_input(l1_repr, dummy_action)
            wm.encoder.forward(combined)

        # Record early MSE
        early_mse = []
        for _ in range(30):
            err = wm.update(l1_repr, dummy_action, l1_repr, m_t=0.3)
            early_mse.append(float(np.mean(err ** 2)))
        mean_early_mse = np.mean(early_mse)

        # Extended training
        for _ in range(300):
            wm.update(l1_repr, dummy_action, l1_repr, m_t=0.3)

        # Record late MSE
        late_mse = []
        for _ in range(30):
            err = wm.update(l1_repr, dummy_action, l1_repr, m_t=0.3)
            late_mse.append(float(np.mean(err ** 2)))
        mean_late_mse = np.mean(late_mse)

        print(f"\n[Scalability] World Model MSE: "
              f"początek={mean_early_mse:.6f}, koniec={mean_late_mse:.6f}")
        self.assertLess(
            mean_late_mse, mean_early_mse,
            "Model świata nie zmniejszył błędu predykcji po powtórzeniach."
        )

    def test_no_nan_or_explosion_at_scale(self):
        """Extended run with all subsystems: no NaN or Inf in any component."""
        stimulus = np.array([0.5, 1, 0, 1, 0, 0.5, 0, 0], dtype=np.float32)

        for _ in range(5):
            self._run_episode(stimulus, episode_len=50)

        # Consolidation
        self.buffer.sleep_phase(self.wm, self.nm, self.bg)

        # Check all weight matrices
        weight_checks = [
            ("L1.w", self.layer.w),
            ("L1.feedback_w", self.layer.feedback_w),
            ("BG.critic.w_v", self.bg.critic.w_v),
            ("BG.critic.w_h", self.bg.critic.w_h),
            ("BG.actor.w_mu", self.bg.actor.w_mu),
            ("WM.w_decode", self.wm.w_decode),
        ]
        for name, arr in weight_checks:
            self.assertFalse(np.any(np.isnan(arr)), f"NaN w wagach {name}.")
            self.assertFalse(np.any(np.isinf(arr)), f"Inf w wagach {name}.")

        # Neuromodulator levels should be finite and in [0, 1]
        for label, level in [
            ("DA", self.nm.dopamine), ("ACh", self.nm.acetylcholine),
            ("NE", self.nm.noradrenaline), ("5-HT", self.nm.serotonin),
        ]:
            self.assertTrue(0.0 <= level <= 1.0,
                            f"Neuromodulator {label} poza [0,1]: {level}")

        print("\n[Scalability] Brak NaN/Inf po 5 epizodach × 50 kroków + konsolidacji.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
