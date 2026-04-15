"""Check weight clip value and initial weight stats."""
import numpy as np
np.random.seed(5)
from arena.environments import TMazeEnv
from arena.snn_agent import SNNAgent

env = TMazeEnv()
agent = SNNAgent(state_size=env.state_size, n_actions=env.n_actions, use_working_memory=True)
actor = agent.bg.actor

print(f"_w_clip_nS = {actor._w_clip_nS:.4f}")
print(f"Init w_d1: shape={actor.w_d1.shape} mean={actor.w_d1.mean():.4f} max={actor.w_d1.max():.4f}")
print(f"Init w_d2: shape={actor.w_d2.shape} mean={actor.w_d2.mean():.4f} max={actor.w_d2.max():.4f}")

col_norms = []
for j in range(actor.action_dim):
    n1 = np.linalg.norm(actor.w_d1[:, j])
    n2 = np.linalg.norm(actor.w_d2[:, j])
    col_norms.extend([n1, n2])
    print(f"  action {j}: d1_norm={n1:.4f} d2_norm={n2:.4f}")

print(f"state_size(input)={actor.w_d1.shape[0]}, action_dim={actor.action_dim}")
print(f"_init_mean_w={actor._init_mean_w:.6f}")
