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
    reward_scale: float = 1.0
    use_world_model: bool = True
    use_working_memory: bool = True
    env_class: type | None = None     # Custom arena.Environment class (None = GymEnv)
    description: str = ""


from arena.environments import (
    ShiftingBanditEnv,
    TMazeEnv,
    PunishmentAvoidanceEnv,
)


# =====================================================================
# Canonical task registry — evaluation criteria ONLY
# =====================================================================

REGISTRY: dict[str, TaskConfig] = {
    "ShiftingBandit": TaskConfig(
        env_id="ShiftingBandit",
        env_class=ShiftingBanditEnv,
        n_episodes=600,
        max_steps=1,
        solved_threshold=0.7,
        eval_window=50,
        use_world_model=False,
        use_working_memory=False,
        description=(
            "3-armed bandit with payoff reversal every 200 episodes. "
            "Tests continual learning / plasticity after distribution shift."
        ),
    ),
    "TMaze": TaskConfig(
        env_id="TMaze",
        env_class=TMazeEnv,
        n_episodes=500,
        max_steps=10,
        solved_threshold=7.0,
        eval_window=30,
        use_world_model=True,
        use_working_memory=True,
        description=(
            "T-maze with cue at start, delayed choice. "
            "Tests working memory for cue maintenance through corridor."
        ),
    ),
    "PunishmentAvoidance": TaskConfig(
        env_id="PunishmentAvoidance",
        env_class=PunishmentAvoidanceEnv,
        n_episodes=400,
        max_steps=1,
        solved_threshold=0.8,
        eval_window=30,
        use_world_model=False,
        use_working_memory=False,
        description=(
            "Context-dependent punishment avoidance. "
            "Tests inhibitory D2/NoGo pathway learning."
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
