"""Trace critic V(s), TD errors, and action selection over multiple episodes."""
import numpy as np
np.random.seed(42)

from arena.gym_env import GymEnv
from arena.agent_factory import make_agent
from arena.task_config import get as get_task

task = get_task("CartPole-v1")
env = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds,
             reward_scale=task.reward_scale)
state = env.reset(seed=42)
agent = make_agent(task, env)
critic = agent.critic
actor = agent.actor

print("=" * 72)
print("  Per-episode learning dynamics (15 episodes)")
print("=" * 72)

for ep in range(15):
    state = env.reset()
    agent.reset()
    ep_reward = 0.0
    td_errors = []
    values = []
    critic_spikes = 0
    d1_spikes = 0
    d2_spikes = 0
    random_actions = 0  # when total_rate < 1e-6
    action_counts = [0, 0]

    for step in range(500):
        # Check action selection mode BEFORE act
        total_rate = (np.sum(np.abs(actor.rate_d1[:actor.motor_dim])) +
                      np.sum(np.abs(actor.rate_d2[:actor.motor_dim])))
        if total_rate < 1e-6:
            random_actions += 1

        action = agent.act(state)
        action_counts[action] += 1
        ns, r, done, info = env.step(action)
        agent.observe(state, action, r, ns, done, info)

        td_errors.append(agent._last_td_error)
        values.append(critic.last_value)
        critic_spikes += int(np.sum(critic.spikes_hidden))
        d1_spikes += int(np.sum(actor.spikes_d1))
        d2_spikes += int(np.sum(actor.spikes_d2))
        ep_reward += r
        state = ns
        if done:
            break

    steps = step + 1
    td_arr = np.array(td_errors)
    v_arr = np.array(values)
    w_v_mean = float(np.mean(critic.w_v))
    w_v_max = float(np.max(np.abs(critic.w_v)))
    w_d1_mean = float(np.mean(actor.w_d1))

    print(f"\nEp {ep:2d} | R={ep_reward:5.0f} steps={steps:3d} "
          f"| crit_spk={critic_spikes:4d} d1_spk={d1_spikes:2d} d2_spk={d2_spikes:2d}")
    print(f"       | TD: mean={td_arr.mean():.3f} min={td_arr.min():.3f} max={td_arr.max():.3f}")
    print(f"       | V:  mean={v_arr.mean():.3f} last={v_arr[-1]:.3f} max={v_arr.max():.3f}")
    print(f"       | w_v: mean={w_v_mean:.4f} max_abs={w_v_max:.4f} | w_d1_mean={w_d1_mean:.4f}")
    print(f"       | random_act={random_actions}/{steps} ({random_actions/steps*100:.0f}%) "
          f"| actions={action_counts}")

env.close()
