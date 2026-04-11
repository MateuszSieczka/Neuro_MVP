"""Diagnose D1/D2 pathway balance: firing rates, weight evolution, DA level.

Verifies that:
  1. D1 and D2 spike rates are comparable (not 7.5x asymmetry)
  2. DA level is in a useful range (not saturated at 1.0 from da_offset=0.5)
  3. Weight means and stds evolve, not stuck
  4. Action selection is not degenerate (both actions chosen)
"""
import numpy as np
np.random.seed(42)

from arena.gym_env import GymEnv
from arena.agent_factory import make_agent
from arena.task_config import get as get_task

task = get_task("CartPole-v1")
env = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds,
             reward_scale=task.reward_scale)
agent = make_agent(task, env)
actor = agent.actor
critic = agent.critic

print("=" * 72)
print("  D1/D2 Balance Diagnostics (15 episodes)")
print("=" * 72)

for ep in range(15):
    state = env.reset()
    agent.reset()
    ep_reward = 0.0
    d1_spikes_total = 0
    d2_spikes_total = 0
    da_levels = []
    action_counts = [0, 0]
    ne_levels = []

    for step in range(500):
        action = agent.act(state)
        action_counts[action] += 1
        ns, r, done, info = env.step(action)
        agent.observe(state, action, r, ns, done, info)

        d1_spikes_total += int(np.sum(actor.spikes_d1))
        d2_spikes_total += int(np.sum(actor.spikes_d2))
        da_levels.append(actor._da_level)
        if hasattr(actor, '_ne_level'):
            ne_levels.append(actor._ne_level)

        ep_reward += r
        state = ns
        if done:
            break

    steps = step + 1
    d1_rate = d1_spikes_total / max(1, steps)
    d2_rate = d2_spikes_total / max(1, steps)
    ratio = d1_rate / max(d2_rate, 1e-6)

    print(f"\nEp {ep:2d} | R={ep_reward:5.0f} steps={steps}")
    print(f"  D1 spikes/step={d1_rate:.2f}  D2 spikes/step={d2_rate:.2f}  "
          f"ratio={ratio:.2f}")
    print(f"  DA level: mean={np.mean(da_levels):.3f}  min={np.min(da_levels):.3f}  "
          f"max={np.max(da_levels):.3f}")
    if ne_levels:
        print(f"  NE level: mean={np.mean(ne_levels):.3f}  min={np.min(ne_levels):.3f}  "
              f"max={np.max(ne_levels):.3f}")
    print(f"  Actions: {action_counts}  (bias={action_counts[0]/(sum(action_counts)+1e-9):.2f})")
    print(f"  w_d1: mean={np.mean(actor.w_d1):.5f}  std={np.std(actor.w_d1):.5f}  "
          f"max={np.max(np.abs(actor.w_d1)):.4f}")
    print(f"  w_d2: mean={np.mean(actor.w_d2):.5f}  std={np.std(actor.w_d2):.5f}  "
          f"max={np.max(np.abs(actor.w_d2)):.4f}")
    print(f"  w_v:  mean={np.mean(critic.w_v):.5f}  std={np.std(critic.w_v):.5f}")

    # Warnings
    if ratio > 3.0:
        print(f"  [WARNING] D1/D2 ratio={ratio:.1f} — severe pathway asymmetry!")
    if np.mean(da_levels) > 0.9:
        print(f"  [WARNING] DA saturated at {np.mean(da_levels):.2f} — da_offset may be too high")

env.close()
print("\n" + "=" * 72)
print("  Done. D1/D2 ratio near 1.0-2.0 and DA below 0.8 = healthy.")
print("=" * 72)
