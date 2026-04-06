"""Diagnostic: track V(s), actor policy, and learning dynamics."""
import numpy as np
from arena.gym_env import GymEnv
from arena.snn_agent import SNNAgent
from arena.core import Trainer
from core.basal_ganglia import ContinuousBGConfig

BOUNDS = (np.array([-2.4, -3.0, -0.21, -3.0]), np.array([2.4, 3.0, 0.21, 3.0]))
np.random.seed(145)
bg_cfg = ContinuousBGConfig(
    gamma=0.95, exploration_noise=0.2, hidden_size=128,
)
env = GymEnv("CartPole-v1", normalize=True, fixed_bounds=BOUNDS)
env.reset(seed=145)
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
    # Snapshot w_mu before episode
    w_mu_before = agent.bg.actor.w_mu.copy()
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
    # w_mu change this episode
    dw_mu = np.max(np.abs(agent.bg.actor.w_mu - w_mu_before))
    if ep < 20 or ep % 5 == 0 or total_r >= 490:
        c = agent.bg.critic
        a = agent.bg.actor
        avg_td = np.mean(np.abs(td_errors))
        avg_v = np.mean(vs) if vs else 0
        act_ratio = np.mean(actions_taken) if actions_taken else 0.5
        print(
            f"Ep {ep:3d} R={total_r:5.0f} steps={len(vs):3d} | "
            f"avg|td|={avg_td:.2f} avg_V={avg_v:6.2f} | "
            f"w_mu:[{a.w_mu.min():.3f},{a.w_mu.max():.3f}] "
            f"Δw_mu={dw_mu:.4f} | "
            f"|e_a|={np.mean(np.abs(a.e_actor)):.3f} | "
            f"noise={a.noise_scale:.3f} act_bias={act_ratio:.2f}"
        )
env.close()
