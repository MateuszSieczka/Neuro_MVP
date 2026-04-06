"""Diagnostic run to track weight dynamics."""
import numpy as np
from arena.gym_env import GymEnv
from arena.snn_agent import SNNAgent
from arena.core import Trainer
from core.basal_ganglia import ContinuousBGConfig

BOUNDS = (np.array([-2.4, -3.0, -0.21, -3.0]), np.array([2.4, 3.0, 0.21, 3.0]))
np.random.seed(1)
bg_cfg = ContinuousBGConfig(
    gamma=0.95, exploration_noise=0.1, hidden_size=128,
)
env = GymEnv("CartPole-v1", normalize=True, fixed_bounds=BOUNDS)
agent = SNNAgent(
    state_size=env.state_size, n_actions=env.n_actions,
    bg_config=bg_cfg, use_world_model=False, trace_decay=0.0,
)

for ep in range(80):
    state = env.reset()
    agent.reset()
    total_r = 0.0
    td_errors = []
    for step in range(500):
        action = agent.act(state)
        ns, r, done, info = env.step(action)
        # Track td_error
        next_aug = agent._augment_state(ns)
        is_truncated = info.get("truncated", False) if info else False
        is_terminal = done and not is_truncated
        if is_terminal:
            td = r - agent.bg.last_v
        else:
            td = r + 0.95 * agent._peek_value(next_aug) - agent.bg.last_v
        td_errors.append(td)
        agent.observe(state, action, r, ns, done, info)
        total_r += r
        state = ns
        if done:
            break
    if ep % 5 == 0 or total_r >= 490:
        c = agent.bg.critic
        a = agent.bg.actor
        norm_ev = np.linalg.norm(c.e_v)
        norm_eh = np.linalg.norm(c.e_h)
        norm_ea = np.linalg.norm(a.e_actor)
        floor_v = np.sqrt(c.e_v.size)
        floor_h = np.sqrt(c.e_h.size)
        floor_a = np.sqrt(a.e_actor.size)
        avg_td = np.mean(np.abs(td_errors))
        print(
            f"Ep {ep:3d} R={total_r:5.0f} | "
            f"avg|td|={avg_td:.2f} | "
            f"||ev||={norm_ev:.1f}/flr{floor_v:.1f} ||eh||={norm_eh:.1f}/flr{floor_h:.1f} ||ea||={norm_ea:.1f}/flr{floor_a:.1f} | "
            f"w_h:[{c.w_h.min():.2f},{c.w_h.max():.2f}] w_v:[{c.w_v.min():.2f},{c.w_v.max():.2f}]"
        )
env.close()
