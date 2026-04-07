"""
Diagnostic test for CartPole-v1 — traces learning dynamics step by step.
"""
import unittest
import numpy as np

from arena.benchmark import Benchmark
from arena.agent_factory import make_agent
from arena.core import Trainer, TrainResult
from arena.gym_env import GymEnv
from arena.task_config import get as get_task


class TestCartPoleDiagnostic(unittest.TestCase):

    def test_cartpole_single_seed_detailed(self):
        """Run CartPole with seed=42 and print detailed diagnostics every 20 episodes."""
        seed = 42
        np.random.seed(seed)
        task = get_task("CartPole-v1")
        env = GymEnv(
            task.env_id,
            normalize=True,
            fixed_bounds=task.obs_bounds,
            reward_scale=task.reward_scale,
        )
        env.reset(seed=seed)
        agent = make_agent(task, env)

        n_episodes = 300
        max_steps = 500
        all_rewards = []

        for ep in range(n_episodes):
            state = env.reset()
            agent.reset()
            ep_reward = 0.0
            ep_steps = 0
            actions = []

            for step in range(max_steps):
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.observe(state, action, reward, next_state, done, info)

                ep_reward += reward
                ep_steps += 1
                actions.append(action)
                state = next_state
                if done:
                    break

            all_rewards.append(ep_reward)

            if ep % 20 == 0 or ep == n_episodes - 1:
                last20 = all_rewards[-20:] if len(all_rewards) >= 20 else all_rewards
                mean20 = np.mean(last20)

                # Actor diagnostics
                w_mu = agent.bg.actor.w_mu
                w_mu_norm = np.linalg.norm(w_mu)
                w_mu_col_norms = [np.linalg.norm(w_mu[:, j]) for j in range(w_mu.shape[1])]
                logit_diff = float(np.mean(np.abs(w_mu[:, 0] - w_mu[:, 1])))
                noise = agent.bg.actor.noise_scale
                temperature = max(agent.bg.config.exploration_noise * noise, 1e-4)

                # Critic diagnostics
                w_v_norm = np.linalg.norm(agent.bg.critic.w_v)
                w_h_norm = np.linalg.norm(agent.bg.critic.w_h)

                # Neuromod
                nm = agent.neuromod
                td_rms = agent.bg._td_rms

                # Action distribution
                act_counts = [actions.count(a) for a in range(agent.n_actions)]
                act_pct = [c / max(len(actions), 1) * 100 for c in act_counts]

                print(f"\n--- Episode {ep:3d} | reward={ep_reward:6.1f} | mean20={mean20:6.1f} ---")
                print(f"  Actor:  w_mu_norm={w_mu_norm:.3f}  col_norms={[f'{n:.3f}' for n in w_mu_col_norms]}  noise_scale={noise:.4f}  temperature={temperature:.4f}")
                print(f"  Critic: w_v_norm={w_v_norm:.3f}  w_h_norm={w_h_norm:.3f}  td_rms={td_rms:.4f}")
                print(f"  Neuromod: DA={nm.dopamine:.3f}  tDA={nm.tonic_da:.3f}  5HT={nm.serotonin:.3f}  NE={nm.noradrenaline:.3f}")
                print(f"  Actions: {act_pct[0]:.0f}%/{act_pct[1]:.0f}%  (steps={ep_steps})")
                print(f"  consolidation_gate={nm.consolidation_gate:.3f}  plasticity_scale={max(0.1, 1.0 - nm.consolidation_gate):.3f}")

                # Eligibility trace norms
                e_actor_norm = np.linalg.norm(agent.bg.actor.e_actor)
                e_h_norm = np.linalg.norm(agent.bg.critic.e_h)
                e_v_norm = np.linalg.norm(agent.bg.critic.e_v)
                print(f"  Traces: e_actor={e_actor_norm:.3f}  e_h={e_h_norm:.3f}  e_v={e_v_norm:.3f}")

        env.close()

        final_mean = np.mean(all_rewards[-20:])
        print(f"\n=== FINAL: mean(last20)={final_mean:.1f}  threshold={task.solved_threshold} ===")
        self.assertGreaterEqual(final_mean, task.solved_threshold,
                                f"CartPole not solved: {final_mean:.1f} < {task.solved_threshold}")


    def test_cartpole_seed17_detailed(self):
        """Run CartPole with seed=17 (problematic seed) and trace dynamics."""
        seed = 17
        np.random.seed(seed)
        task = get_task("CartPole-v1")
        env = GymEnv(
            task.env_id, normalize=True,
            fixed_bounds=task.obs_bounds,
            reward_scale=task.reward_scale,
        )
        env.reset(seed=seed)
        agent = make_agent(task, env)

        n_episodes = 300
        max_steps = 500
        all_rewards = []

        for ep in range(n_episodes):
            state = env.reset()
            agent.reset()
            ep_reward = 0.0
            ep_steps = 0
            actions = []

            # Trace first few steps of early episodes in detail
            for step in range(max_steps):
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.observe(state, action, reward, next_state, done, info)

                if ep < 3 and step < 5:
                    # Print detailed per-step info for first episodes
                    td = agent._last_td_error
                    v = agent.bg.last_v
                    print(f"  ep={ep} step={step}: state={state[:2]}.. action={action} "
                          f"reward={reward:.1f} td={td:.4f} V={v:.4f}")

                ep_reward += reward
                ep_steps += 1
                actions.append(action)
                state = next_state
                if done:
                    break

            all_rewards.append(ep_reward)

            if ep % 10 == 0 or ep == n_episodes - 1:
                last20 = all_rewards[-20:] if len(all_rewards) >= 20 else all_rewards
                mean20 = np.mean(last20)
                nm = agent.neuromod

                w_mu = agent.bg.actor.w_mu
                # Check if actor weights differentiate between actions
                w_diff = w_mu[:, 0] - w_mu[:, 1]  # difference between action 0 and 1 weights
                logit_bias = float(np.mean(w_diff))

                act_counts = [actions.count(a) for a in range(agent.n_actions)]
                act_pct = [c / max(len(actions), 1) * 100 for c in act_counts]

                noise = agent.bg.actor.noise_scale
                temperature = max(agent.bg.config.exploration_noise * noise, 1e-4)
                td_rms = agent.bg._td_rms

                # Compute what logits look like for a zero-state
                test_logits = np.dot(np.zeros(4, dtype=np.float32), w_mu)
                test_probs = np.exp(test_logits[:2] / temperature)
                test_probs = test_probs / test_probs.sum()

                print(f"ep={ep:3d} rew={ep_reward:6.1f} mean20={mean20:6.1f} | "
                      f"noise={noise:.4f} temp={temperature:.4f} td_rms={td_rms:.3f} | "
                      f"DA={nm.dopamine:.3f} tDA={nm.tonic_da:.3f} 5HT={nm.serotonin:.3f} NE={nm.noradrenaline:.3f} | "
                      f"acts={act_pct[0]:.0f}%/{act_pct[1]:.0f}% steps={ep_steps} | "
                      f"logit_bias={logit_bias:.4f}")

        env.close()

        final_mean = np.mean(all_rewards[-20:])
        print(f"\n=== SEED 17 FINAL: mean(last20)={final_mean:.1f} ===")

    def test_cartpole_benchmark_7seeds(self):
        """Run full benchmark with 7 seeds."""
        result = Benchmark.run("CartPole-v1", seeds=[1, 17, 42, 99, 145, 256, 500], verbose=True)
        self.assertGreaterEqual(result.solve_rate, 0.85,
                                f"Solve rate too low: {result.solve_rate:.0%}")
