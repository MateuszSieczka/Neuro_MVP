"""Multi-seed diagnostic for MountainCar."""
import sys
import numpy as np
from arena.gym_env import GymEnv
from arena.agent_factory import make_agent
from arena import task_config

SEEDS = [42, 7, 100]
N_EPISODES = 300

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")
    
    task = task_config.get("MountainCar-v0")
    np.random.seed(seed)
    env = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
    env.reset(seed=seed)
    agent = make_agent(task, env)

    rewards = []
    for ep in range(N_EPISODES):
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

        rewards.append(ep_reward)
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
    
    # Summary statistics
    rewards = np.array(rewards)
    successes = rewards > -200
    first_success = np.where(successes)[0][0] if successes.any() else -1
    
    # Count successes in last 100 episodes
    last100 = rewards[-100:]
    success_rate_last100 = np.mean(last100 > -200) * 100
    avg_reward_last100 = np.mean(last100)
    
    # Find sustained learning start (5 successes in 10 episodes)
    sustained_start = -1
    for i in range(len(rewards) - 10):
        window = rewards[i:i+10]
        if np.sum(window > -200) >= 5:
            sustained_start = i
            break
    
    print(f"\n--- SEED {seed} SUMMARY ---")
    print(f"Total successes: {successes.sum()}/{N_EPISODES}")
    print(f"First success: ep {first_success}")
    print(f"Sustained learning from: ep {sustained_start}")
    print(f"Last 100 ep: success rate={success_rate_last100:.0f}%, avg R={avg_reward_last100:.1f}")
    print(f"Best reward: {rewards.min():.0f} (most negative = worst, least negative = best)")
    sys.stdout.flush()
