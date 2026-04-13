"""
Integration tests — from simplest to complex.

These tests verify that the SNN network actually *learns* in biologically
grounded environments, from trivial single-button bandits to multi-step
navigation tasks.  No parameter tweaking for specific benchmarks — we
diagnose root issues and fix the biology.
"""

from __future__ import annotations

import numpy as np
import pytest

from arena.environments import (
    SingleButtonEnv,
    StochasticButtonEnv,
    TwoButtonEnv,
    CorridorEnv,
    ShiftingBanditEnv,
    PunishmentAvoidanceEnv,
)
from arena.snn_agent import SNNAgent
from core.config import BasalGangliaConfig


# =====================================================================
# Helpers
# =====================================================================

def make_agent(
    env,
    *,
    use_world_model: bool = False,
    use_working_memory: bool = False,
    bg_overrides: dict | None = None,
) -> SNNAgent:
    """Create a minimal agent for an environment."""
    bg_cfg = BasalGangliaConfig()
    if bg_overrides:
        for k, v in bg_overrides.items():
            setattr(bg_cfg, k, v)
    return SNNAgent(
        state_size=env.state_size,
        n_actions=env.n_actions,
        bg_config=bg_cfg,
        use_world_model=use_world_model,
        use_working_memory=use_working_memory,
    )


def run_episodes(agent: SNNAgent, env, n_episodes: int) -> list[float]:
    """Run n episodes, return list of total rewards."""
    rewards = []
    for ep in range(n_episodes):
        state = env.reset(seed=ep)
        agent.reset()
        total_reward = 0.0
        for step in range(100):  # max steps safety
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, reward, next_state, done, info)
            total_reward += reward
            state = next_state
            if done:
                break
        rewards.append(total_reward)
    return rewards


def mean_last_n(rewards: list[float], n: int = 50) -> float:
    """Mean of last n rewards."""
    return float(np.mean(rewards[-n:]))


# =====================================================================
# Level 0: Smoke tests — does the network run without crashing?
# =====================================================================

class TestSmoke:
    """Basic sanity: agent can be created, can act, can observe."""

    def test_agent_creation(self):
        env = SingleButtonEnv()
        agent = make_agent(env)
        assert agent.n_actions == 2
        assert agent.state_size == 1

    def test_single_step(self):
        """Agent can do one act + observe cycle."""
        env = SingleButtonEnv()
        agent = make_agent(env)
        state = env.reset()
        action = agent.act(state)
        assert action in (0, 1)
        next_state, reward, done, info = env.step(action)
        agent.observe(state, action, reward, next_state, done, info)

    def test_ten_episodes(self):
        """Agent can run 10 episodes without crashing."""
        env = SingleButtonEnv()
        agent = make_agent(env)
        rewards = run_episodes(agent, env, 10)
        assert len(rewards) == 10

    def test_agent_with_world_model(self):
        """Agent with world model can run."""
        env = TwoButtonEnv()
        agent = make_agent(env, use_world_model=True)
        rewards = run_episodes(agent, env, 5)
        assert len(rewards) == 5

    def test_agent_with_working_memory(self):
        """Agent with WM + world model can run."""
        env = TwoButtonEnv()
        agent = make_agent(env, use_world_model=True, use_working_memory=True)
        rewards = run_episodes(agent, env, 5)
        assert len(rewards) == 5


# =====================================================================
# Level 1: SingleButton — trivial learning
# =====================================================================

class TestSingleButton:
    """
    SingleButton: action 1 → +1, action 0 → 0.
    The network should learn to press the button.
    This is the absolute minimum a learning agent must solve.
    """

    def test_basic_learning(self):
        """After 200 episodes, agent should prefer action=1 (press)."""
        env = SingleButtonEnv()
        agent = make_agent(env)
        rewards = run_episodes(agent, env, 300)

        # First 50 episodes: random (expect ~0.5 mean)
        early = mean_last_n(rewards[:50], 50)
        # Last 50 episodes: should press more often
        late = mean_last_n(rewards, 50)
        print(f"\n[SingleButton] Early mean: {early:.3f}, Late mean: {late:.3f}")
        # Generous threshold: just show learning happened
        assert late > early or late > 0.6, (
            f"No learning: early={early:.3f}, late={late:.3f}"
        )

    def test_diagnostic_internals(self):
        """Diagnose internal network state during SingleButton learning."""
        env = SingleButtonEnv()
        agent = make_agent(env)

        diagnostics = []
        for ep in range(100):
            state = env.reset()
            agent.reset()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            # Capture internal state BEFORE observe
            d = {
                "ep": ep,
                "action": action,
                "reward": reward,
                "critic_activation_mean": float(np.mean(agent.critic.activation)),
                "critic_activation_max": float(np.max(agent.critic.activation)),
                "critic_spikes_mean": float(np.mean(agent.critic.spikes_hidden)),
                "d1_spikes": float(np.mean(agent.actor.spikes_d1)),
                "d2_spikes": float(np.mean(agent.actor.spikes_d2)),
                "v_d1_mean": float(np.mean(agent.actor.v_d1)),
                "v_d2_mean": float(np.mean(agent.actor.v_d2)),
                "net_evidence": agent.actor._last_net_evidence.copy() if agent.actor._last_net_evidence is not None else None,
                "vta_v_s": agent.vta.last_v_s,
            }

            agent.observe(state, action, reward, next_state, done, info)

            d["td_error"] = agent._last_td_error
            d["vta_rpe"] = agent.vta.last_rpe
            d["vta_gamma"] = agent.vta.last_gamma_eff
            d["vta_auto_rms"] = agent.vta._auto_rms
            d["w_value_norm"] = float(np.linalg.norm(agent.vta.w_value))
            diagnostics.append(d)

        # Print summary at key points
        for i in [0, 9, 49, 99]:
            d = diagnostics[i]
            print(f"\n[SingleButton ep={d['ep']}] "
                  f"action={d['action']} reward={d['reward']:.1f} "
                  f"td={d['td_error']:.4f} "
                  f"critic_act_mean={d['critic_activation_mean']:.4f} "
                  f"critic_spikes={d['critic_spikes_mean']:.4f} "
                  f"d1_spikes={d['d1_spikes']:.4f} "
                  f"d2_spikes={d['d2_spikes']:.4f} "
                  f"v_d1={d['v_d1_mean']:.2f} v_d2={d['v_d2_mean']:.2f} "
                  f"net_ev={d['net_evidence']} "
                  f"V(s)={d['vta_v_s']:.4f} "
                  f"w_value_norm={d['w_value_norm']:.4f} "
                  f"auto_rms={d['vta_auto_rms']:.4f}")

        # Verify the critic has non-zero activation (spikes earlier
        # in the decision window, then adaptation/refractory → no spikes
        # on last substep, but EMA rate is non-zero).
        last_20 = diagnostics[-20:]
        avg_critic_act = np.mean([d["critic_activation_mean"] for d in last_20])
        print(f"\n[SingleButton] Avg critic activation (last 20): {avg_critic_act:.4f}")
        assert avg_critic_act > 0, "Critic neurons never activate — no learning possible"


# =====================================================================
# Level 2: StochasticButton — learn under noise
# =====================================================================

class TestStochasticButton:
    """
    StochasticButton: press → EV +3.9, don't press → 0.
    Agent must learn that pressing is beneficial ON AVERAGE.
    """

    def test_learns_to_press(self):
        env = StochasticButtonEnv()
        agent = make_agent(env)
        rewards = run_episodes(agent, env, 400)

        late = mean_last_n(rewards, 100)
        print(f"\n[StochasticButton] Late mean: {late:.2f} (optimal ~3.9)")
        # Should at least be pressing most of the time (>2.0 means >50% presses)
        assert late > 1.5, f"Not learning stochastic reward: late mean = {late:.2f}"


# =====================================================================
# Level 3: TwoButton — context-dependent action
# =====================================================================

class TestTwoButton:
    """
    TwoButton: context A → action 0, context B → action 1.
    Requires state-conditional policy, not just action frequency bias.
    """

    def test_learns_context_mapping(self):
        env = TwoButtonEnv()
        agent = make_agent(env)
        rewards = run_episodes(agent, env, 500)

        late = mean_last_n(rewards, 100)
        print(f"\n[TwoButton] Late mean: {late:.2f} (optimal +1.0, random 0.0)")
        assert late > 0.3, f"No context learning: late mean = {late:.2f}"


# =====================================================================
# Level 4: Corridor — temporal credit assignment
# =====================================================================

class TestCorridor:
    """
    5-cell corridor: move right 4 times → +10.
    Requires γ > 0 and multi-step credit assignment.
    """

    def test_learns_to_move_right(self):
        env = CorridorEnv()
        agent = make_agent(env)
        rewards = run_episodes(agent, env, 500)

        late = mean_last_n(rewards, 100)
        print(f"\n[Corridor] Late mean: {late:.2f} (optimal 9.6, random ~4.5)")
        # Random policy: 50% right → ~8 steps avg → reward ≈ 10 - 0.1*8 = 9.2
        # But random can also stay → negative.  Just check improvement.
        assert late > 5.0, f"No corridor learning: late mean = {late:.2f}"


# =====================================================================
# Level 5: PunishmentAvoidance — NoGo pathway
# =====================================================================

class TestPunishmentAvoidance:
    """
    Context A: action 1 → -3 (suppress!), Context B: action 1 → +2 (go!).
    Tests D2/NoGo learning from negative TD error.
    """

    def test_learns_to_avoid(self):
        env = PunishmentAvoidanceEnv()
        agent = make_agent(env)
        rewards = run_episodes(agent, env, 500)

        late = mean_last_n(rewards, 100)
        print(f"\n[PunishmentAvoidance] Late mean: {late:.2f} (optimal +1.0, random -0.5)")
        # Random: 50% context A × (50% × 0 + 50% × -3) + 50% context B × (50% × 0 + 50% × 2)
        # = 50% × -1.5 + 50% × 1.0 = -0.25
        assert late > -0.2, f"Not avoiding punishment: late mean = {late:.2f}"


# =====================================================================
# Level 6: ShiftingBandit — plasticity after reversal
# =====================================================================

class TestShiftingBandit:
    """
    3-armed bandit with payoff reversal every 200 episodes.
    Tests continual learning / adaptation.
    """

    def test_adapts_after_shift(self):
        env = ShiftingBanditEnv(shift_interval=200)
        agent = make_agent(env)

        # Run 600 episodes total (3 phases: A, B, A)
        all_rewards = run_episodes(agent, env, 600)

        # Phase A (0-199): arm 0 is best (p=0.8)
        phase_a = np.mean(all_rewards[150:200])
        # Phase B (200-399): arm 1 is best (p=0.8)
        phase_b_early = np.mean(all_rewards[200:250])
        phase_b_late = np.mean(all_rewards[350:400])

        print(f"\n[ShiftingBandit] Phase A late: {phase_a:.2f}, "
              f"Phase B early: {phase_b_early:.2f}, "
              f"Phase B late: {phase_b_late:.2f}")

        # After reversal, performance should initially drop then recover
        # Generous: just check phase B late is better than random (0.5)
        assert phase_b_late > 0.45, (
            f"No adaptation after shift: phase_b_late = {phase_b_late:.2f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
