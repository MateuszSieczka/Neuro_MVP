"""Quick TMaze validation: 5 seeds × 600 episodes."""
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from arena.environments import TMazeEnv
from arena.snn_agent import SNNAgent

results = []
for seed in range(5):
    np.random.seed(seed * 17 + 5)
    env = TMazeEnv()
    agent = SNNAgent(
        state_size=env.state_size,
        n_actions=env.n_actions,
        use_working_memory=True,
    )

    t0 = time.time()
    rewards = []
    for ep in range(600):
        state = env.reset(seed=ep)
        agent.reset()
        total = 0.0
        for step in range(100):
            action = agent.act(state)
            ns, r, done, info = env.step(action)
            agent.observe(state, action, r, ns, done, info)
            total += r
            state = ns
            if done:
                break
        rewards.append(total)
        if (ep + 1) % 200 == 0:
            m = np.mean(rewards[-100:])
            print(f"  Seed {seed} ep {ep+1}: last100={m:.2f}")

    late = float(np.mean(rewards[-100:]))
    elapsed = time.time() - t0
    print(f"Seed {seed}: late_mean={late:.2f} elapsed={elapsed:.0f}s")
    results.append(late)

mean_late = float(np.mean(results))
print(f"\nOverall mean: {mean_late:.2f} (random ~4.2, 70%correct ~6.4)")
print(f"Per-seed: {[f'{x:.2f}' for x in results]}")
print(f"PASS: {mean_late > 4.2}")
