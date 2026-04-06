"""Benchmark: MountainCar-v0 — second environment for generality testing.

MountainCar is fundamentally different from CartPole:
- Sparse reward: -1 per step, 0 at goal → total return in [-200, 0]
- Requires exploration: random policy never solves it
- 2D state (position, velocity), 3 discrete actions (left, nothing, right)
- Solved when mean return > -110 (reaches goal in < 110 steps)
- Default max_steps: 200

This tests whether the architecture generalizes beyond dense-reward environments.
"""
import numpy as np
from arena.gym_env import GymEnv
from arena.snn_agent import SNNAgent
from arena.core import Trainer
from core.basal_ganglia import ContinuousBGConfig
from core.config import SNNWorldModelConfig, NeuromodulatorConfig

# MountainCar obs: [position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07]]
MC_BOUNDS = (np.array([-1.2, -0.07]), np.array([0.6, 0.07]))
SEEDS = [1, 17, 42, 99, 145, 256, 500]


def run_mountaincar_benchmark():
    bg_cfg = ContinuousBGConfig(
    gamma=0.99,           # ← było 0.95
    exploration_noise=0.3, # ← było 0.2
    hidden_size=128,
)
    nm_cfg = NeuromodulatorConfig()

    n_ep = 1000  # MountainCar needs more episodes — sparse reward
    scores = []

    print(f"--- MountainCar-v0 Benchmark ({n_ep} episodes, solved: mean > -110) ---")

    for seed in SEEDS:
        np.random.seed(seed)
        env = GymEnv("MountainCar-v0", normalize=True, fixed_bounds=MC_BOUNDS)
        env.reset(seed=seed)

        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=bg_cfg,
            use_world_model=True,   # ← było False (curiosity dla sparse reward)
            trace_decay=0.9,        # ← było 0.0
        )

        trainer = Trainer(env, agent)
        result = trainer.train(n_episodes=n_ep, max_steps=200)

        final_score = result.mean_reward(last_n=20)
        scores.append(final_score)

        first_solved = next(
            (i for i, log in enumerate(result.episode_logs)
             if log.total_reward > -110),
            f">{n_ep}"
        )

        print(
            f"Seed {seed:3d} | Mean (last 20): {final_score:7.1f} | "
            f"First solved ep: {first_solved}"
        )
        env.close()

    solved = sum(1 for s in scores if s > -110)
    print(
        f"\nSummary: Mean={np.mean(scores):.1f} | "
        f"Solved={solved}/{len(SEEDS)}"
    )


if __name__ == "__main__":
    run_mountaincar_benchmark()
