"""Post-fix diagnostic v2: verify bootstrap deadlock broken.

Tracks ALL key signals after fixes:
  A) Intrinsic reward gate removed (no more (1-tDA) suppression)
  B) Adaptive curiosity baseline (prevents collapse)
  C) 4-step minimum commitment (temporal coherence)
  D) Lower AI temperature (more decisive selection)
"""
import numpy as np
from arena.gym_env import GymEnv
from arena.agent_factory import make_agent
from arena import task_config

EPISODES = 100
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

print("=" * 100)
print(f"POST-FIX DIAGNOSTIC v2  ({EPISODES} episodes, seed={SEED})")
print("=" * 100)

all_rewards = []
all_max_pos = []

for ep in range(EPISODES):
    state = env.reset()
    agent.reset()
    ep_reward = 0.0
    curiosities = []
    v_values = []
    actions = []
    max_pos = -1.2
    decisions = 0  # count actual BG decisions (not committed repeats)

    prev_action = -1
    for step in range(task.max_steps):
        action = agent.act(state)
        if action != prev_action or step == 0:
            decisions += 1
        prev_action = action
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
    min_cur = np.min(curiosities) if curiosities else 0.0
    max_cur = np.max(curiosities) if curiosities else 0.0
    avg_v = np.mean(v_values)
    v_std = np.std(v_values)
    tda = agent.neuromod.tonic_da
    sero = agent.neuromod.serotonin
    ne = agent.neuromod.noradrenaline

    # Action distribution
    act_counts = [actions.count(i) for i in range(agent.n_actions)]
    act_str = f"L={act_counts[0]:3d} N={act_counts[1]:3d} R={act_counts[2]:3d}"

    all_rewards.append(ep_reward)
    all_max_pos.append(max_pos)

    if ep % 5 == 0 or ep == EPISODES - 1 or ep_reward > -190:
        print(
            f"Ep {ep:3d} | R={ep_reward:7.1f} | steps={step+1:3d} | "
            f"dec={decisions:3d} sw={switches:3d} | "
            f"cur={avg_cur:.3f}[{min_cur:.3f}-{max_cur:.3f}] | "
            f"V={avg_v:.1f}(std={v_std:.1f}) | "
            f"tDA={tda:.3f} sero={sero:.3f} | "
            f"{act_str} | maxP={max_pos:.3f}"
        )

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
last10_avg = np.mean(all_rewards[-10:])
print(f"  Avg Reward (last 10): {last10_avg:.1f}")
print(f"  Best Reward:          {max(all_rewards):.1f}")
print(f"  Max Position reached: {max(all_max_pos):.3f}")
print(f"  Final curiosity avg:  {avg_cur:.3f}")
print(f"  Final sero:           {agent.neuromod.serotonin:.3f}")
print(f"  Final tonic_da:       {agent.neuromod.tonic_da:.3f}")

# Pass/fail checks
ok = True
if last10_avg <= -200.0:
    print("  [FAIL] Avg reward still at -200 (no learning)")
    ok = False
else:
    print(f"  [OK]   Avg reward improved to {last10_avg:.1f}")
if max(all_max_pos) < -0.3:
    print("  [FAIL] Never explored past -0.3 position")
    ok = False
else:
    print(f"  [OK]   Explored to position {max(all_max_pos):.3f}")
if avg_cur < 0.1:
    print("  [WARN] Curiosity low at end — may be collapsing")
elif avg_cur > 0.5:
    print(f"  [OK]   Curiosity healthy at {avg_cur:.3f}")
if ok:
    print("  [PASS] Bootstrap deadlock appears broken!")

env.close()
