"""Diagnose weight drift in TMaze training."""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from arena.environments import TMazeEnv
from arena.snn_agent import SNNAgent

np.random.seed(5)
env = TMazeEnv()
agent = SNNAgent(state_size=env.state_size, n_actions=env.n_actions, use_working_memory=True)

actor = agent.bg.actor
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
    if (ep + 1) % 100 == 0:
        m = np.mean(rewards[-100:])
        w1_range = f"[{actor.w_d1.min():.2e}, {actor.w_d1.max():.2e}]"
        w2_range = f"[{actor.w_d2.min():.2e}, {actor.w_d2.max():.2e}]"
        w1_mean = f"{actor.w_d1.mean():.4f}"
        w2_mean = f"{actor.w_d2.mean():.4f}"
        # Check WM-related weights (first 8 input neurons)
        wm_d1 = actor.w_d1[:8, :]
        wm_d2 = actor.w_d2[:8, :]
        print(f"Ep {ep+1}: last100={m:.2f}  w_d1 {w1_range} mean={w1_mean}  w_d2 {w2_range} mean={w2_mean}")
        print(f"  WM→D1 mean={wm_d1.mean():.4f} max={wm_d1.max():.4f}  WM→D2 mean={wm_d2.mean():.4f} max={wm_d2.max():.4f}")
        # Check adaptation state
        print(f"  w_adapt_d1 range=[{actor.w_adapt_d1.min():.1f}, {actor.w_adapt_d1.max():.1f}]  d2=[{actor.w_adapt_d2.min():.1f}, {actor.w_adapt_d2.max():.1f}]")
