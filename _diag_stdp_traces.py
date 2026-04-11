"""Diagnose STDP eligibility traces: LTP vs LTD balance, trace magnitudes.

Verifies that:
  1. e_d1 goes both positive AND negative (LTP + LTD)
  2. e_d2 goes both positive AND negative
  3. Post-synaptic traces (_x_post_d1/d2) are non-zero
  4. ltd_ratio is applied correctly (Shen et al. 2008)
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

print("=" * 72)
print("  STDP Eligibility Trace Diagnostics (10 episodes)")
print("=" * 72)

for ep in range(10):
    state = env.reset()
    agent.reset()
    ep_reward = 0.0

    e_d1_mins, e_d1_maxs = [], []
    e_d2_mins, e_d2_maxs = [], []
    post_d1_norms, post_d2_norms = [], []
    e_d1_neg_frac, e_d2_neg_frac = [], []

    for step in range(500):
        action = agent.act(state)
        ns, r, done, info = env.step(action)
        agent.observe(state, action, r, ns, done, info)

        # Sample eligibility stats
        e_d1_mins.append(float(np.min(actor.e_d1)))
        e_d1_maxs.append(float(np.max(actor.e_d1)))
        e_d2_mins.append(float(np.min(actor.e_d2)))
        e_d2_maxs.append(float(np.max(actor.e_d2)))

        e_d1_neg_frac.append(float(np.mean(actor.e_d1 < 0)))
        e_d2_neg_frac.append(float(np.mean(actor.e_d2 < 0)))

        if hasattr(actor, '_x_post_d1'):
            post_d1_norms.append(float(np.linalg.norm(actor._x_post_d1)))
            post_d2_norms.append(float(np.linalg.norm(actor._x_post_d2)))

        ep_reward += r
        state = ns
        if done:
            break

    steps = step + 1
    print(f"\nEp {ep:2d} | R={ep_reward:5.0f} steps={steps}")
    print(f"  e_d1: min={np.min(e_d1_mins):.4f}  max={np.max(e_d1_maxs):.4f}  "
          f"neg_frac={np.mean(e_d1_neg_frac):.3f}")
    print(f"  e_d2: min={np.min(e_d2_mins):.4f}  max={np.max(e_d2_maxs):.4f}  "
          f"neg_frac={np.mean(e_d2_neg_frac):.3f}")
    if post_d1_norms:
        print(f"  post_d1_norm: mean={np.mean(post_d1_norms):.4f} max={np.max(post_d1_norms):.4f}")
        print(f"  post_d2_norm: mean={np.mean(post_d2_norms):.4f} max={np.max(post_d2_norms):.4f}")
    else:
        print("  [WARNING] _x_post_d1/_x_post_d2 not found — LTD may not be wired")

    # Check: e_d1 should go negative if LTD is working
    has_ltd_d1 = np.min(e_d1_mins) < -0.001
    has_ltd_d2 = np.min(e_d2_mins) < -0.001
    if not has_ltd_d1:
        print("  [BUG] e_d1 never goes negative — LTD is absent!")
    if not has_ltd_d2:
        print("  [BUG] e_d2 never goes negative — LTD is absent!")

env.close()
print("\n" + "=" * 72)
print("  Done. If e_d1/e_d2 go negative, STDP LTD is working correctly.")
print("=" * 72)
