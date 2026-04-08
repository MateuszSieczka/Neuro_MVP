"""Post-fix diagnostic: verify bootstrap deadlock is broken.

Checks vs pre-fix baselines (30 episodes):
  - V(s) = -1.00 flat        → should show variation
  - 0 action switches        → should be > 50
  - TD error → 0.0000        → should stay > 0.01
  - curiosity = 0.001         → should be > 0.05
  - sero=0.100 frozen         → should vary
  - max_pos ≈ -0.52 (stuck)   → should increase
"""
import numpy as np
from arena.gym_env import GymEnv
from arena.agent_factory import make_agent
from arena import task_config

EPISODES = 30
SEED = 42

task = task_config.get("MountainCar-v0")
np.random.seed(SEED)
env = GymEnv(
    task.env_id,
    normalize=True,
    fixed_bounds=task.obs_bounds,
    reward_scale=task.reward_scale,
)
env.reset(seed=SEED)
agent = make_agent(task, env)

print("=" * 90)
print("POST-FIX DIAGNOSTIC  (30 episodes, seed=42)")
print("=" * 90)

all_rewards = []
all_max_pos = []

for ep in range(EPISODES):
    state = env.reset()
    agent.reset()
    ep_reward = 0.0
    curiosities = []
    v_values = []
    td_errors = []
    actions = []
    max_pos = -1.2

    for step in range(task.max_steps):
        action = agent.act(state)
        actions.append(action)
        next_state, reward, done, info = env.step(action)
        agent.observe(state, action, reward, next_state, done, info)

        raw_pos = state[0] * 0.9 + (-0.3)
        if raw_pos > max_pos:
            max_pos = raw_pos

        if agent._use_wm:
            curiosities.append(agent.world_model.curiosity_signal())
        v_values.append(agent.bg.last_v)
        ep_reward += reward
        state = next_state
        if done:
            break

    switches = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])
    avg_cur = np.mean(curiosities) if curiosities else 0.0
    avg_v = np.mean(v_values)
    v_std = np.std(v_values)
    tda = agent.neuromod.tonic_da
    sero = agent.neuromod.serotonin
    ne = agent.neuromod.noradrenaline
    all_rewards.append(ep_reward)
    all_max_pos.append(max_pos)

    if ep % 5 == 0 or ep == EPISODES - 1 or ep_reward > -190:
        print(
            f"Ep {ep:3d} | R={ep_reward:7.1f} | steps={step+1:3d} | "
            f"switches={switches:3d} | cur={avg_cur:.3f} | "
            f"V={avg_v:.3f} (std={v_std:.3f}) | "
            f"tDA={tda:.3f} sero={sero:.3f} NE={ne:.3f} | "
            f"maxP={max_pos:.3f}"
        )

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"  Avg Reward (last 10): {np.mean(all_rewards[-10:]):.1f}")
print(f"  Best Reward:          {max(all_rewards):.1f}")
print(f"  Max Position reached: {max(all_max_pos):.3f}")
print(f"  Final sero:           {agent.neuromod.serotonin:.3f}")
print(f"  Final tonic_da:       {agent.neuromod.tonic_da:.3f}")

# Pass/fail checks
ok = True
if np.mean(all_rewards[-10:]) <= -200.0:
    print("  [FAIL] Avg reward still at -200 (no learning)")
    ok = False
if max(all_max_pos) < -0.45:
    print("  [FAIL] Never explored past -0.45 position")
    ok = False
if avg_cur < 0.01:
    print("  [FAIL] Curiosity collapsed to near-zero")
    ok = False
if switches < 10:
    print("  [FAIL] Almost no action switching (still stuck?)")
    ok = False
if ok:
    print("  [PASS] Bootstrap deadlock appears broken")

env.close()
