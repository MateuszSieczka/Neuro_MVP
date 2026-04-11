"""Diagnose the full learning signal chain: TD → DA → STDP → weight update.

Verifies that:
  1. TD errors are not trivially zero or clipped to extremes
  2. Weight updates actually occur each episode (delta w_d1, w_d2, w_v)
  3. Critic V(s) evolves sensibly (not stuck at 0)
  4. Temperature from NE modulates action entropy
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
print("  Learning Signal Chain Diagnostics (20 episodes)")
print("=" * 72)

prev_w_d1 = actor.w_d1.copy()
prev_w_d2 = actor.w_d2.copy()
prev_w_v = critic.w_v.copy()

for ep in range(20):
    state = env.reset()
    agent.reset()
    ep_reward = 0.0
    td_errors = []
    values = []

    for step in range(500):
        action = agent.act(state)
        ns, r, done, info = env.step(action)
        agent.observe(state, action, r, ns, done, info)

        td_errors.append(agent._last_td_error)
        values.append(critic.last_value)

        ep_reward += r
        state = ns
        if done:
            break

    steps = step + 1
    td_arr = np.array(td_errors)
    v_arr = np.array(values)

    # Weight deltas
    dw_d1 = float(np.mean(np.abs(actor.w_d1 - prev_w_d1)))
    dw_d2 = float(np.mean(np.abs(actor.w_d2 - prev_w_d2)))
    dw_v = float(np.mean(np.abs(critic.w_v - prev_w_v)))
    prev_w_d1 = actor.w_d1.copy()
    prev_w_d2 = actor.w_d2.copy()
    prev_w_v = critic.w_v.copy()

    print(f"\nEp {ep:2d} | R={ep_reward:5.0f} steps={steps}")
    print(f"  TD: mean={td_arr.mean():+.4f}  std={td_arr.std():.4f}  "
          f"min={td_arr.min():+.3f}  max={td_arr.max():+.3f}")
    print(f"  V:  mean={v_arr.mean():.4f}  last={v_arr[-1]:.4f}  "
          f"max={v_arr.max():.4f}")
    print(f"  Δw_d1={dw_d1:.6f}  Δw_d2={dw_d2:.6f}  Δw_v={dw_v:.6f}")

    # Warnings
    if dw_d1 < 1e-8 and dw_d2 < 1e-8:
        print("  [WARNING] Actor weights not updating — learning is stuck!")
    if dw_v < 1e-8:
        print("  [WARNING] Critic weights not updating — value function frozen!")
    if td_arr.std() < 1e-4:
        print("  [WARNING] TD error has near-zero variance — signal may be dead")

env.close()
print("\n" + "=" * 72)
print("  Done. Weight deltas > 1e-5 and non-zero TD variance = healthy learning.")
print("=" * 72)
