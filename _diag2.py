"""Diagnostic: track V(s), actor policy, and learning dynamics."""
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

print(f"Config: critic_lr={bg_cfg.critic_lr}, actor_lr={bg_cfg.actor_lr}, "
      f"tau_e={bg_cfg.tau_e}, tau_hidden={bg_cfg.tau_hidden}, "
      f"w_clip={bg_cfg.w_clip}")
print(f"trace_decay = {np.exp(-1.0 / bg_cfg.tau_e):.4f}")
print(f"mem_decay = {agent.bg.critic._mem_decay:.4f}")
print()

for ep in range(120):
    state = env.reset()
    agent.reset()
    total_r = 0.0
    vs = []
    td_errors = []
    actions_taken = []
    for step in range(500):
        action = agent.act(state)
        actions_taken.append(action)
        v_now = agent.bg.last_v
        vs.append(v_now)
        ns, r, done, info = env.step(action)
        next_aug = agent._augment_state(ns)
        is_truncated = info.get("truncated", False) if info else False
        is_terminal = done and not is_truncated
        if is_terminal:
            td = r - v_now
        else:
            td = r + 0.95 * agent._peek_value(next_aug) - v_now
        td_errors.append(td)
        agent.observe(state, action, r, ns, done, info)
        total_r += r
        state = ns
        if done:
            break
    if ep % 5 == 0 or total_r >= 490:
        c = agent.bg.critic
        a = agent.bg.actor
        avg_td = np.mean(np.abs(td_errors))
        avg_v = np.mean(vs) if vs else 0
        max_v = np.max(np.abs(vs)) if vs else 0
        act_ratio = np.mean(actions_taken) if actions_taken else 0.5
        e_v_mag = np.mean(np.abs(c.e_v))
        e_h_mag = np.mean(np.abs(c.e_h))
        e_a_mag = np.mean(np.abs(a.e_actor))
        sero = agent.neuromod.serotonin
        ns_val = a.noise_scale
        print(
            f"Ep {ep:3d} R={total_r:5.0f} steps={len(vs):3d} | "
            f"avg|td|={avg_td:.2f} avg_V={avg_v:6.2f} max|V|={max_v:.2f} | "
            f"w_h:[{c.w_h.min():.2f},{c.w_h.max():.2f}] "
            f"w_v:[{c.w_v.min():.2f},{c.w_v.max():.2f}] "
            f"w_mu:[{a.w_mu.min():.3f},{a.w_mu.max():.3f}] | "
            f"|e_v|={e_v_mag:.2f} |e_h|={e_h_mag:.2f} |e_a|={e_a_mag:.2f} | "
            f"sero={sero:.2f} noise={ns_val:.3f} act_bias={act_ratio:.2f}"
        )
env.close()
