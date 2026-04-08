"""500-episode convergence test for MountainCar-v0."""
import numpy as np
from arena.gym_env import GymEnv
from arena.agent_factory import make_agent
from arena import task_config

EPISODES = 500
SEED = 42

task = task_config.get("MountainCar-v0")
np.random.seed(SEED)
env = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
env.reset(seed=SEED)
agent = make_agent(task, env)

print("=" * 100)
print(f"500-EPISODE CONVERGENCE TEST  (seed={SEED})")
print("=" * 100)

all_rewards = []
successes = 0

for ep in range(EPISODES):
    state = env.reset()
    agent.reset()
    ep_reward = 0.0
    max_pos = -1.2

    for step in range(task.max_steps):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.observe(state, action, reward, next_state, done, info)
        raw_pos = state[0] * 0.9 + (-0.3)
        if raw_pos > max_pos:
            max_pos = raw_pos
        ep_reward += reward
        state = next_state
        if done:
            break

    all_rewards.append(ep_reward)
    if ep_reward > -200:
        successes += 1

    if ep % 25 == 0 or ep == EPISODES - 1:
        window = all_rewards[-25:] if len(all_rewards) >= 25 else all_rewards
        avg25 = np.mean(window)
        cur = agent.world_model.curiosity_signal() if agent._use_wm else 0
        tda = agent.neuromod.tonic_da
        sero = agent.neuromod.serotonin
        succ_rate = successes / (ep + 1)
        print(
            f"Ep {ep:3d} | R={ep_reward:7.1f} | avg25={avg25:7.1f} | "
            f"succ={succ_rate:.0%} ({successes}/{ep+1}) | "
            f"maxP={max_pos:.3f} | tDA={tda:.3f} sero={sero:.3f}"
        )

print("\n" + "=" * 100)
last100 = all_rewards[-100:]
print(f"  Avg last 100:  {np.mean(last100):.1f}")
print(f"  Best ever:     {max(all_rewards):.1f}")
print(f"  Success rate:  {successes}/{EPISODES} ({successes/EPISODES:.0%})")
print(f"  Last 100 succ: {sum(1 for r in last100 if r > -200)}/100")
env.close()
