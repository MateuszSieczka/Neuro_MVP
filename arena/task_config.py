"""
arena.task_config — Universal task registry.

DESIGN PRINCIPLE: "No task-specific tuning."
=============================================
The agent discovers optimal parameters through:
  - Exploration noise: derived from epistemic uncertainty + NE level
  - reward_scale: adaptive Welford normalization (inside neuromodulator)
  - hidden_size: based on input dimensionality × coverage factor

Only truly task-specific information lives here:
  - env_id, n_episodes, max_steps, solved_threshold (evaluation criteria)
  - obs_bounds (normalisation, not learning parameters)

Anti-cheat guarantee
--------------------
The agent never receives the env_id string.  It receives only
normalised observation vectors and structural dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, kw_only=True)
class TaskConfig:
    """Environment-specific evaluation metadata.

    All learning hyperparameters are universal — determined by
    BasalGangliaConfig / NeuromodulatorConfig defaults, NOT per-task.
    """

    env_id: str
    n_episodes: int
    max_steps: int
    solved_threshold: float
    eval_window: int = 20
    obs_bounds: tuple[np.ndarray, np.ndarray] | None = None
    use_world_model: bool = True
    use_working_memory: bool = True
    description: str = ""


# =====================================================================
# Canonical task registry — evaluation criteria ONLY
# =====================================================================

REGISTRY: dict[str, TaskConfig] = {
    "CartPole-v1": TaskConfig(
        env_id="CartPole-v1",
        n_episodes=120,
        max_steps=500,
        solved_threshold=450.0,
        eval_window=20,
        obs_bounds=(
            np.array([-2.4, -3.0, -0.21, -3.0], dtype=np.float32),
            np.array([2.4, 3.0, 0.21, 3.0], dtype=np.float32),
        ),
        use_world_model=False,
        use_working_memory=False,
        description=(
            "Dense reward, velocity in obs. "
            "World model disabled — no curiosity benefit in dense reward."
        ),
    ),
    "MountainCar-v0": TaskConfig(
        env_id="MountainCar-v0",
        n_episodes=300,
        max_steps=200,
        solved_threshold=-110.0,
        eval_window=20,
        obs_bounds=(
            np.array([-1.2, -0.07], dtype=np.float32),
            np.array([0.6, 0.07], dtype=np.float32),
        ),
        use_world_model=True,
        use_working_memory=True,
        description=(
            "Sparse reward. World model + curiosity essential. "
            "WM helps maintain momentum planning."
        ),
    ),
}


def get(env_id: str) -> TaskConfig:
    """Retrieve a TaskConfig by env_id."""
    if env_id not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(
            f"Unknown task {env_id!r}. Available: {available}. "
            f"Add a TaskConfig entry to arena/task_config.py to register."
        )
    return REGISTRY[env_id]
