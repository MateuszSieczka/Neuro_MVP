"""
Diagnostic for MountainCar plateau fix validation.

Runs 1000 episodes on seed 23 with extended telemetry showing:
- Episode return, steps, noise, tonic DA, serotonin
- Stagnation factor (ACC learning monitor)
- Consolidation gate (with ACC attenuation)
- Best-episode replay status
- Plasticity scale
- Rolling 50-episode average return
"""
import numpy as np
from collections import deque
from arena.gym_env import GymEnv
from arena.agent_factory import make_agent
from arena import task_config

SEED = 23
N_EPISODES = 1000

task = task_config.get("MountainCar-v0")
np.random.seed(SEED)
env = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
env.reset(seed=SEED)
agent = make_agent(task, env)

returns_window = deque(maxlen=50)
successes_window = deque(maxlen=50)
best_return = -np.inf

for ep in range(N_EPISODES):
    state = env.reset()
    agent.reset()
    ep_reward = 0.0
    v_values = []

    for step in range(task.max_steps):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.observe(state, action, reward, next_state, done, info)
        v_values.append(agent.bg.last_v)
        ep_reward += reward
        state = next_state
        if done:
            break

    returns_window.append(ep_reward)
    successes_window.append(1 if ep_reward > -200 else 0)
    if ep_reward > best_return:
        best_return = ep_reward

    avg_50 = np.mean(returns_window)
    succ_rate = np.mean(successes_window)

    if ep % 10 == 0 or ep_reward > -110 or ep_reward <= -200:
        avg_v = np.mean(v_values)
        noise = agent.bg.actor.noise_scale
        tda = agent.neuromod.tonic_da
        sero = agent.neuromod.serotonin
        gate = agent.neuromod.consolidation_gate
        stag = agent.neuromod._stagnation_factor
        plast = max(0.05, 1.0 - gate)
        has_golden = len(agent._best_episode_buffer) > 0
        golden_r = agent._best_episode_return if agent._best_episode_return > -np.inf else float('nan')
        smooth_noise = agent._smooth_noise

        # Compute sleep_gain for diagnostics (mirror snn_agent.py logic)
        _sg = 1.0
        if len(agent.neuromod._reward_history) >= 5:
            _r_arr = np.array(agent.neuromod._reward_history)
            _r_mean = float(np.mean(_r_arr))
            _r_std = float(np.std(_r_arr)) + 1e-8
            _quality = (ep_reward - _r_mean) / _r_std
            _sg = max(1.0, float(np.clip(1.0 + 0.5 * _quality, 1.0, 2.5)))

        print(f"Ep {ep:4d} | R={ep_reward:7.1f} | avg50={avg_50:7.1f} | "
              f"succ={succ_rate:.0%} | steps={step+1:3d} | "
              f"V={avg_v:+.2f} | noise={smooth_noise:.3f} | "
              f"tDA={tda:.3f} sero={sero:.3f} | "
              f"gate={gate:.3f} stag={stag:.2f} plast={plast:.2f} | "
              f"sgain={_sg:.2f} best={best_return:.0f}")

# Final summary
print("\n" + "=" * 80)
print(f"FINAL: avg_last50={np.mean(returns_window):.1f}  "
      f"success_rate={np.mean(successes_window):.0%}  "
      f"best_ever={best_return:.0f}")
print(f"Stagnation factor: {agent.neuromod._stagnation_factor:.3f}")
print(f"Consolidation gate: {agent.neuromod.consolidation_gate:.3f}")
print(f"Best episode buffer size: {len(agent._best_episode_buffer) if hasattr(agent, '_best_episode_buffer') else 'N/A'}")
print("=" * 80)

env.close()
