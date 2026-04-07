"""Quick diagnostic for MountainCar: trace curiosity, V-values, exploration."""
import numpy as np
from arena.gym_env import GymEnv
from arena.agent_factory import make_agent
from arena import task_config

task = task_config.get("MountainCar-v0")
np.random.seed(23)
env = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
env.reset(seed=23)
agent = make_agent(task, env)

for ep in range(500):
    state = env.reset()
    agent.reset()
    ep_reward = 0.0
    curiosities = []
    v_values = []

    for step in range(task.max_steps):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.observe(state, action, reward, next_state, done, info)

        if agent._use_wm:
            curiosities.append(agent.world_model.curiosity_signal())
        v_values.append(agent.bg.last_v)
        ep_reward += reward
        state = next_state
        if done:
            break

    if ep % 5 == 0 or ep_reward > -190:
        avg_cur = np.mean(curiosities) if curiosities else 0
        min_cur = np.min(curiosities) if curiosities else 0
        max_cur = np.max(curiosities) if curiosities else 0
        avg_v = np.mean(v_values)
        noise = agent.bg.actor.noise_scale
        tda = agent.neuromod.tonic_da
        sero = agent.neuromod.serotonin
        ne = agent.neuromod.noradrenaline
        print(f"Ep {ep:3d} | R={ep_reward:7.1f} | steps={step+1:3d} | "
              f"curiosity={avg_cur:.3f}[{min_cur:.3f}-{max_cur:.3f}] | "
              f"V={avg_v:.3f} | noise={noise:.3f} | "
              f"tDA={tda:.3f} sero={sero:.3f} NE={ne:.3f}")

env.close()
