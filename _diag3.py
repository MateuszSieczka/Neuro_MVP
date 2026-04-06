"""Diagnostic: trace NE, serotonin, noise dynamics."""
import numpy as np
from arena.gym_env import GymEnv
from arena.snn_agent import SNNAgent
from core.basal_ganglia import ContinuousBGConfig

BOUNDS = (np.array([-2.4, -3.0, -0.21, -3.0]), np.array([2.4, 3.0, 0.21, 3.0]))

"""Dual-environment diagnostic: CartPole + MountainCar internal dynamics."""
import numpy as np
from arena.gym_env import GymEnv
from arena.snn_agent import SNNAgent
from arena.core import Trainer
from core.basal_ganglia import ContinuousBGConfig
from core.config import NeuromodulatorConfig

BOUNDS_CP = (np.array([-2.4, -3.0, -0.21, -3.0]), np.array([2.4, 3.0, 0.21, 3.0]))
BOUNDS_MC = (np.array([-1.2, -0.07]), np.array([0.6, 0.07]))


def diagnose_env(env_name, bounds, seed, n_ep, max_steps):
    np.random.seed(seed)
    env = GymEnv(env_name, normalize=True, fixed_bounds=bounds)
    env.reset(seed=seed)
    bg_cfg = ContinuousBGConfig(gamma=0.95, exploration_noise=0.2, hidden_size=128)
    agent = SNNAgent(
        state_size=env.state_size, n_actions=env.n_actions,
        bg_config=bg_cfg, use_world_model=False, trace_decay=0.0,
    )

    print(f"\n{'='*70}")
    print(f"  {env_name} seed={seed}  ({n_ep} episodes, max_steps={max_steps})")
    print(f"{'='*70}")

    for ep in range(n_ep):
        state = env.reset()
        agent.reset()
        total_r = 0.0
        td_sum = 0.0
        td_abs_sum = 0.0
        td_count = 0
        for step in range(max_steps):
            action = agent.act(state)
            ns, r, done, info = env.step(action)
            agent.observe(state, action, r, ns, done, info)
            total_r += r
            td_sum += agent._last_td_error
            td_abs_sum += abs(agent._last_td_error)
            td_count += 1
            state = ns
            if done:
                break

        nm = agent.neuromod
        a = agent.bg.actor
        c = agent.bg.critic
        avg_td = td_sum / max(td_count, 1)
        avg_abs_td = td_abs_sum / max(td_count, 1)
        probs = a._last_probs
        w_cols = [np.linalg.norm(a.w_mu[:, j]) for j in range(a.motor_dim)]
        w_str = ",".join(f"{n:.3f}" for n in w_cols)
        p_str = ",".join(f"{p:.3f}" for p in probs) if probs is not None else "None"
        w_cv = np.linalg.norm(c.w_v)

        if ep < 25 or ep % 25 == 0 or ep == n_ep - 1:
            print(
                f"Ep{ep:4d} R={total_r:7.1f} | "
                f"tDA={nm.tonic_da:.3f} 5HT={nm.serotonin:.3f} NE={nm.noradrenaline:.3f} | "
                f"gate={nm.consolidation_gate:.3f} td_rms={agent.bg._td_rms:.3f} | "
                f"noise={a.noise_scale:.4f} avgTD={avg_td:+.3f} |avgTD|={avg_abs_td:.3f} | "
                f"p=[{p_str}] w_a=[{w_str}] w_cv={w_cv:.2f}"
            )

    env.close()


# Diagnose CartPole — pick diverse seeds (one good, one bad, one middle)
for seed in [99, 145, 256]:
    diagnose_env("CartPole-v1", BOUNDS_CP, seed, n_ep=100, max_steps=500)

# Diagnose MountainCar — same seeds
for seed in [1, 145, 42]:
    diagnose_env("MountainCar-v0", BOUNDS_MC, seed, n_ep=200, max_steps=200)
