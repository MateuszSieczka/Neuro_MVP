"""
Iterative integration tests — does the SNN actually learn?

Each test level builds on the previous one, using the arena framework.
Tests verify BOTH final performance AND learning dynamics (early vs late).
"""

import unittest
import numpy as np

from arena.core import Trainer, TrainResult, RandomAgent
from arena.environments import (
    SingleButtonEnv,
    StochasticButtonEnv,
    TwoButtonEnv,
    CorridorEnv,
    ShiftingBanditEnv,
    RiskRewardEnv,
    TMazeEnv,
    PunishmentAvoidanceEnv,
)
from arena.snn_agent import SNNAgent
from core.basal_ganglia import ContinuousBGConfig


# =====================================================================
# Helpers
# =====================================================================

class TrainResultSlice:
    """Compute action distribution over a slice of episodes."""

    def __init__(self, result: TrainResult, start: int, end: int | None):
        self._logs = result.episode_logs[start:end]

    def action_dist(self) -> dict[int, float]:
        all_actions = []
        for ep in self._logs:
            all_actions.extend(ep.actions)
        if not all_actions:
            return {}
        counts: dict[int, int] = {}
        for a in all_actions:
            counts[a] = counts.get(a, 0) + 1
        total = len(all_actions)
        return {a: c / total for a, c in sorted(counts.items())}


def print_curve_summary(name: str, result: TrainResult, window: int = 50) -> None:
    """Print a compact learning curve summary."""
    curve = result.learning_curve(window)
    if len(curve) >= 3:
        n = len(curve)
        print(f"  curve: start={curve[0]:.2f}  mid={curve[n // 2]:.2f}  end={curve[-1]:.2f}")


# =====================================================================
# Level 1: Single deterministic button
# =====================================================================

class TestLevel1_SingleButton(unittest.TestCase):
    """Press → +1.  Don't press → 0.  Must learn to always press."""

    def test_learns_to_press_button(self):
        np.random.seed(42)
        env = SingleButtonEnv()
        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.0,
                critic_lr=0.02,
                actor_lr=0.01,
                exploration_noise=0.5,
                hidden_size=32,
            ),
            use_world_model=False,
        )
        trainer = Trainer(env, agent)
        result = trainer.train(n_episodes=1000, max_steps=1)

        early = TrainResultSlice(result, 0, 100).action_dist()
        late = TrainResultSlice(result, -100, None).action_dist()
        late_press = late.get(1, 0.0)

        print(f"\n[Level 1] Early press: {early.get(1, 0):.2f}  Late press: {late_press:.2f}")
        print(f"  Mean reward last 100: {result.mean_reward(last_n=100):.2f}")
        print_curve_summary("L1", result)

        self.assertGreater(late_press, 0.7, f"Late press rate too low: {late_press:.2f}")
        self.assertTrue(result.is_improving(), "No learning improvement detected")

    def test_convergence_speed(self):
        """Should learn within 300 episodes (not 1000)."""
        np.random.seed(123)
        env = SingleButtonEnv()
        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.0, critic_lr=0.02, actor_lr=0.01,
                exploration_noise=0.5, hidden_size=32,
            ),
            use_world_model=False,
        )
        result = Trainer(env, agent).train(n_episodes=300, max_steps=1)
        late_press = result.action_distribution(last_n=50).get(1, 0.0)
        print(f"\n[Level 1b] Press rate after 300 eps: {late_press:.2f}")
        self.assertGreater(late_press, 0.6, f"Too slow to converge: {late_press:.2f}")


# =====================================================================
# Level 2: Stochastic button (lottery)
# =====================================================================

class TestLevel2_StochasticButton(unittest.TestCase):
    """EV(press)=3.9, EV(skip)=0.  Must learn despite occasional punishment."""

    def test_learns_positive_ev_button(self):
        np.random.seed(42)
        env = StochasticButtonEnv()
        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.0, critic_lr=0.02, actor_lr=0.01,
                exploration_noise=0.4, hidden_size=32,
            ),
            use_world_model=False,
        )
        result = Trainer(env, agent).train(n_episodes=1500, max_steps=1)

        late_press = result.action_distribution(last_n=200).get(1, 0.0)
        mean_rew = result.mean_reward(last_n=200)

        print(f"\n[Level 2] Late press: {late_press:.2f}  Mean rew: {mean_rew:.2f}")
        print_curve_summary("L2", result)

        self.assertGreater(late_press, 0.7, f"Should press >70%: {late_press:.2f}")
        self.assertGreater(mean_rew, 2.0, f"Mean reward too low: {mean_rew:.2f}")


# =====================================================================
# Level 3: Context-dependent two-button task
# =====================================================================

class TestLevel3_TwoButtons(unittest.TestCase):
    """Context A→btn0, Context B→btn1.  Must learn state-conditional policy."""

    def test_learns_context_dependent_policy(self):
        np.random.seed(42)
        env = TwoButtonEnv()
        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.0, critic_lr=0.02, actor_lr=0.008,
                exploration_noise=0.4, hidden_size=64,
            ),
            use_world_model=False,
        )
        result = Trainer(env, agent).train(n_episodes=3000, max_steps=1)

        mean_rew = result.mean_reward(last_n=300)
        print(f"\n[Level 3] Mean reward last 300: {mean_rew:.2f}  (random=0.0, perfect=1.0)")
        print_curve_summary("L3", result)

        self.assertGreater(mean_rew, 0.3, f"Context policy too weak: {mean_rew:.2f}")
        self.assertTrue(result.is_improving(), "No learning improvement detected")


# =====================================================================
# Level 4: 5-cell corridor — delayed reward
# =====================================================================

class TestLevel4_Corridor(unittest.TestCase):
    """Must chain 4 correct actions for +10.  Tests temporal credit assignment."""

    def test_learns_to_navigate_corridor(self):
        np.random.seed(42)
        env = CorridorEnv(corridor_length=5)
        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.95, critic_lr=0.02, actor_lr=0.008,
                exploration_noise=0.5, hidden_size=64,
            ),
            use_world_model=False,
        )
        result = Trainer(env, agent).train(n_episodes=2000, max_steps=20)

        mean_rew = result.mean_reward(last_n=200)
        print(f"\n[Level 4] Mean reward last 200: {mean_rew:.2f}  (random≈-1.0, perfect=9.6)")
        print_curve_summary("L4", result, window=100)

        self.assertGreater(mean_rew, 3.0, f"Corridor too weak: {mean_rew:.2f}")

    def test_learns_longer_corridor(self):
        """8-cell corridor — harder temporal credit assignment."""
        np.random.seed(42)
        env = CorridorEnv(corridor_length=8)
        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.97, critic_lr=0.02, actor_lr=0.008,
                exploration_noise=0.5, hidden_size=64,
            ),
            use_world_model=False,
        )
        result = Trainer(env, agent).train(n_episodes=3000, max_steps=30)

        mean_rew = result.mean_reward(last_n=300)
        print(f"\n[Level 4b] 8-cell corridor, mean reward last 300: {mean_rew:.2f}")
        print_curve_summary("L4b", result, window=100)

        # Perfect = 10 - 7*0.1 = 9.3, random ≈ -3.0
        self.assertGreater(mean_rew, 2.0, f"Long corridor too weak: {mean_rew:.2f}")


# =====================================================================
# Level 5: Shifting 3-armed bandit
# =====================================================================

class TestLevel5_ShiftingBandit(unittest.TestCase):
    """Payoffs shift every 300 episodes.  Must detect and adapt."""

    def test_adapts_to_payoff_shift(self):
        np.random.seed(42)
        env = ShiftingBanditEnv(shift_interval=300, hide_phase=False)
        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.0, critic_lr=0.03, actor_lr=0.015,
                exploration_noise=0.6, hidden_size=64,
            ),
            use_world_model=False,
        )
        result = Trainer(env, agent).train(n_episodes=1500, max_steps=1)

        mean_rew = result.mean_reward(last_n=300)
        print(f"\n[Level 5] Mean reward last 300: {mean_rew:.2f}  (random=0.50, optimal=0.80)")
        print_curve_summary("L5", result, window=100)

        self.assertGreater(mean_rew, 0.55, f"Shifting bandit too weak: {mean_rew:.2f}")

    def test_phase_adaptation(self):
        """Check that agent performs well in BOTH phases (not just one)."""
        np.random.seed(42)
        env = ShiftingBanditEnv(shift_interval=300, hide_phase=False)
        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.0, critic_lr=0.03, actor_lr=0.015,
                exploration_noise=0.6, hidden_size=64,
            ),
            use_world_model=False,
        )
        result = Trainer(env, agent).train(n_episodes=1500, max_steps=1)

        # Phase A performance (last occurrence: episodes 600-900)
        phase_a_rews = [r.total_reward for r in result.episode_logs[700:900]]
        # Phase B performance (last occurrence: episodes 900-1200)
        phase_b_rews = [r.total_reward for r in result.episode_logs[1000:1200]]

        mean_a = float(np.mean(phase_a_rews))
        mean_b = float(np.mean(phase_b_rews))

        print(f"\n[Level 5b] Phase A mean: {mean_a:.2f}  Phase B mean: {mean_b:.2f}")

        # Both phases should be above random (0.50)
        self.assertGreater(mean_a, 0.50, f"Phase A too weak: {mean_a:.2f}")
        self.assertGreater(mean_b, 0.50, f"Phase B too weak: {mean_b:.2f}")


# =====================================================================
# Level 6: Risk vs safety — asymmetric payoffs
# =====================================================================

class TestLevel6_RiskReward(unittest.TestCase):
    """Agent must learn to avoid trap (EV=-0.6) and pick context-optimal action."""

    def test_avoids_trap(self):
        """Trap action should be chosen rarely after training."""
        np.random.seed(42)
        env = RiskRewardEnv()
        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.0, critic_lr=0.03, actor_lr=0.015,
                exploration_noise=0.5, hidden_size=64,
            ),
            use_world_model=False,
        )
        result = Trainer(env, agent).train(n_episodes=3000, max_steps=1)

        dist = result.action_distribution(last_n=500)
        trap_rate = dist.get(2, 0.0)
        mean_rew = result.mean_reward(last_n=500)

        print(f"\n[Level 6] Trap rate: {trap_rate:.2f}  Mean rew: {mean_rew:.2f}")
        print(f"  Action dist: {dist}")
        print_curve_summary("L6", result, window=200)

        # Trap (action 2) should be rare; mean reward should be well above trap EV
        self.assertLess(trap_rate, 0.3, f"Trap chosen too often: {trap_rate:.2f}")
        self.assertGreater(mean_rew, 0.5, f"Mean reward too low: {mean_rew:.2f}")

    def test_context_sensitivity(self):
        """Agent should prefer risky in context 1 (EV=2.2) and safe in context 2 (EV_risky=-0.2)."""
        np.random.seed(42)
        env = RiskRewardEnv()
        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.0, critic_lr=0.03, actor_lr=0.015,
                exploration_noise=0.5, hidden_size=64,
            ),
            use_world_model=False,
        )
        result = Trainer(env, agent).train(n_episodes=5000, max_steps=1)

        mean_rew = result.mean_reward(last_n=500)
        # Random EV = (1/3)*(1.0 + 1.0 + (-0.6)) ≈ 0.47 (averaged across contexts)
        # Optimal = (1/3)*(1.0 + 2.2 + 1.0) ≈ 1.4
        print(f"\n[Level 6b] Mean reward last 500: {mean_rew:.2f}  (random≈0.47, optimal≈1.40)")
        print_curve_summary("L6b", result, window=200)

        self.assertGreater(mean_rew, 0.8, f"Not exploiting context: {mean_rew:.2f}")


# =====================================================================
# Level 7: T-maze — memory + delayed reward
# =====================================================================

class TestLevel7_TMaze(unittest.TestCase):
    """Agent must remember cue from start to choose correct arm at junction."""

    def test_learns_tmaze(self):
        np.random.seed(42)
        env = TMazeEnv()
        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.95, critic_lr=0.02, actor_lr=0.01,
                exploration_noise=0.5, hidden_size=64,
            ),
            use_world_model=False,
            trace_decay=0.8,  # Working memory: decaying trace of past states
        )
        result = Trainer(env, agent).train(n_episodes=3000, max_steps=10)

        mean_rew = result.mean_reward(last_n=300)
        # Perfect = 10 - 3*0.1 = 9.7 (3 corridor steps + correct arm)
        # Random at junction = 50% correct: 0.5*10 + 0.5*(-1) - 0.3 = 4.2
        print(f"\n[Level 7] Mean reward last 300: {mean_rew:.2f}  (random≈4.2, perfect=9.7)")
        print_curve_summary("L7", result, window=200)

        # With trace_decay=0.8, the cue signal persists: 0.8^3 = 0.51 at junction.
        # The agent should leverage this to pick the correct arm >50% of the time.
        self.assertGreater(mean_rew, 5.0, f"T-maze too weak: {mean_rew:.2f}")


# =====================================================================
# Level 8: Punishment avoidance — inhibitory learning
# =====================================================================

class TestLevel8_PunishmentAvoidance(unittest.TestCase):
    """Agent must suppress action 1 in context A (punishment) but use it in context B (reward)."""

    def test_learns_to_avoid_punishment(self):
        np.random.seed(42)
        env = PunishmentAvoidanceEnv()
        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.0, critic_lr=0.02, actor_lr=0.01,
                exploration_noise=0.4, hidden_size=64,
            ),
            use_world_model=False,
        )
        result = Trainer(env, agent).train(n_episodes=2000, max_steps=1)

        mean_rew = result.mean_reward(last_n=300)
        # Optimal: context A → action 0 (0), context B → action 1 (+2) → avg = 1.0
        # Random: context A → 50%*0 + 50%*(-3) = -1.5, B → 50%*0 + 50%*2 = 1.0 → avg = -0.25
        print(f"\n[Level 8] Mean reward last 300: {mean_rew:.2f}  (random=-0.25, optimal=1.0)")
        print_curve_summary("L8", result, window=100)

        self.assertGreater(mean_rew, 0.3, f"Punishment avoidance too weak: {mean_rew:.2f}")
        self.assertTrue(result.is_improving(), "No learning improvement detected")


# =====================================================================
# Robustness: multi-seed tests
# =====================================================================

class TestRobustness_MultiSeed(unittest.TestCase):
    """Verify learning is not seed-dependent (works across multiple seeds)."""

    def test_single_button_across_seeds(self):
        """Level 1 should pass with at least 4/5 random seeds."""
        successes = 0
        for seed in [1, 17, 42, 99, 256]:
            np.random.seed(seed)
            env = SingleButtonEnv()
            agent = SNNAgent(
                state_size=env.state_size, n_actions=env.n_actions,
                bg_config=ContinuousBGConfig(
                    gamma=0.0, critic_lr=0.02, actor_lr=0.01,
                    exploration_noise=0.5, hidden_size=32,
                ),
                use_world_model=False,
            )
            result = Trainer(env, agent).train(n_episodes=500, max_steps=1)
            press = result.action_distribution(last_n=100).get(1, 0.0)
            if press > 0.6:
                successes += 1
        print(f"\n[Robustness] Single button: {successes}/5 seeds passed")
        self.assertGreaterEqual(successes, 4, f"Only {successes}/5 seeds passed")

    def test_two_buttons_across_seeds(self):
        """Level 3 should pass with at least 4/5 random seeds."""
        successes = 0
        for seed in [1, 17, 42, 99, 256]:
            np.random.seed(seed)
            env = TwoButtonEnv()
            agent = SNNAgent(
                state_size=env.state_size, n_actions=env.n_actions,
                bg_config=ContinuousBGConfig(
                    gamma=0.0, critic_lr=0.02, actor_lr=0.008,
                    exploration_noise=0.4, hidden_size=64,
                ),
                use_world_model=False,
            )
            result = Trainer(env, agent).train(n_episodes=2000, max_steps=1)
            mean_rew = result.mean_reward(last_n=200)
            if mean_rew > 0.2:
                successes += 1
        print(f"\n[Robustness] Two buttons: {successes}/5 seeds passed")
        self.assertGreaterEqual(successes, 4, f"Only {successes}/5 seeds passed")

    def test_corridor_across_seeds(self):
        """Level 4 should pass with at least 4/5 random seeds."""
        successes = 0
        for seed in [1, 17, 42, 99, 256]:
            np.random.seed(seed)
            env = CorridorEnv(corridor_length=5)
            agent = SNNAgent(
                state_size=env.state_size, n_actions=env.n_actions,
                bg_config=ContinuousBGConfig(
                    gamma=0.95, critic_lr=0.02, actor_lr=0.008,
                    exploration_noise=0.5, hidden_size=64,
                ),
                use_world_model=False,
            )
            result = Trainer(env, agent).train(n_episodes=1500, max_steps=20)
            mean_rew = result.mean_reward(last_n=200)
            if mean_rew > 3.0:
                successes += 1
        print(f"\n[Robustness] Corridor: {successes}/5 seeds passed")
        self.assertGreaterEqual(successes, 4, f"Only {successes}/5 seeds passed")


# =====================================================================
# Baseline comparison: SNN vs Random
# =====================================================================

class TestBaseline_SNNvsRandom(unittest.TestCase):
    """SNN agent must significantly outperform random baseline."""

    def _run_comparison(self, env, agent, n_episodes, max_steps):
        """Run both SNN and Random, return (snn_reward, random_reward)."""
        snn_result = Trainer(env, agent).train(n_episodes=n_episodes, max_steps=max_steps)
        random_agent = RandomAgent(env.n_actions)
        rnd_result = Trainer(env, random_agent).train(n_episodes=n_episodes, max_steps=max_steps)
        last_n = min(200, n_episodes // 3)
        return snn_result.mean_reward(last_n=last_n), rnd_result.mean_reward(last_n=last_n)

    def test_button_beats_random(self):
        np.random.seed(42)
        env = SingleButtonEnv()
        agent = SNNAgent(
            state_size=env.state_size, n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.0, critic_lr=0.02, actor_lr=0.01,
                exploration_noise=0.5, hidden_size=32,
            ),
            use_world_model=False,
        )
        snn, rnd = self._run_comparison(env, agent, 500, 1)
        print(f"\n[vs Random] Button:   SNN={snn:.2f}  Random={rnd:.2f}  Δ={snn - rnd:+.2f}")
        self.assertGreater(snn, rnd + 0.1, f"SNN not clearly better: {snn:.2f} vs {rnd:.2f}")

    def test_corridor_beats_random(self):
        np.random.seed(42)
        env = CorridorEnv(corridor_length=8)  # Longer corridor → random struggles more
        agent = SNNAgent(
            state_size=env.state_size, n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.97, critic_lr=0.02, actor_lr=0.008,
                exploration_noise=0.5, hidden_size=64,
            ),
            use_world_model=False,
        )
        snn, rnd = self._run_comparison(env, agent, 2000, 30)
        print(f"\n[vs Random] Corridor-8: SNN={snn:.2f}  Random={rnd:.2f}  Δ={snn - rnd:+.2f}")
        self.assertGreater(snn, rnd + 0.3, f"SNN not clearly better: {snn:.2f} vs {rnd:.2f}")

    def test_risk_reward_beats_random(self):
        np.random.seed(42)
        env = RiskRewardEnv()
        agent = SNNAgent(
            state_size=env.state_size, n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.0, critic_lr=0.03, actor_lr=0.015,
                exploration_noise=0.5, hidden_size=64,
            ),
            use_world_model=False,
        )
        snn, rnd = self._run_comparison(env, agent, 2000, 1)
        print(f"\n[vs Random] Risk:     SNN={snn:.2f}  Random={rnd:.2f}  Δ={snn - rnd:+.2f}")
        self.assertGreater(snn, rnd + 0.2, f"SNN not clearly better: {snn:.2f} vs {rnd:.2f}")


# =====================================================================
# Full pipeline: BG + WorldModel + Neuromodulator
# =====================================================================

class TestFullPipeline(unittest.TestCase):
    """Verify learning works with the full SNN pipeline enabled."""

    def test_corridor_with_world_model(self):
        """Corridor should still be solved with WM + NM active."""
        np.random.seed(42)
        env = CorridorEnv(corridor_length=5)
        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.95, critic_lr=0.02, actor_lr=0.008,
                exploration_noise=0.5, hidden_size=64,
            ),
            use_world_model=True,  # Full pipeline!
        )
        result = Trainer(env, agent).train(n_episodes=2000, max_steps=20)

        mean_rew = result.mean_reward(last_n=200)
        print(f"\n[Full Pipeline] Corridor with WM: {mean_rew:.2f}")
        print_curve_summary("FP-corridor", result, window=100)

        self.assertGreater(mean_rew, 3.0, f"Full pipeline corridor too weak: {mean_rew:.2f}")

    def test_two_buttons_with_world_model(self):
        """Context task should work with WM + NM active."""
        np.random.seed(42)
        env = TwoButtonEnv()
        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=ContinuousBGConfig(
                gamma=0.0, critic_lr=0.02, actor_lr=0.008,
                exploration_noise=0.4, hidden_size=64,
            ),
            use_world_model=True,
        )
        result = Trainer(env, agent).train(n_episodes=3000, max_steps=1)

        mean_rew = result.mean_reward(last_n=300)
        print(f"\n[Full Pipeline] Two buttons with WM: {mean_rew:.2f}")
        print_curve_summary("FP-2btn", result, window=100)

        self.assertGreater(mean_rew, 0.2, f"Full pipeline two-buttons too weak: {mean_rew:.2f}")


if __name__ == "__main__":
    unittest.main()
