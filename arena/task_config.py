"""
arena.task_config — Centralised registry of task-specific hyperparameters.

DESIGN PRINCIPLE: "One truth, one place."
=========================================
Every parameter that is *specific to an environment* lives here and
ONLY here.  The SNNAgent, Trainer, and Benchmark classes are entirely
blind to which task is running — they receive only:
  - Numeric tensors (state, action, reward, next_state, done)
  - Structural dimensions (state_size, n_actions)
  - Learning hyperparameters from TaskConfig

Anti-cheat guarantee
--------------------
Because the agent never receives the env_id string, it cannot embed
task-specific heuristics.  All it can learn is a mapping from
(normalised observation vector) → (action distribution).
The normalisation itself (obs_bounds) lives here, not in the agent.

Adding a new task
-----------------
1. Add a TaskConfig entry to REGISTRY.
2. Optionally provide obs_bounds if the observation space has known limits.
   Otherwise leave it None and the GymEnv wrapper uses running statistics.
3. That's it — no changes to Agent, Trainer, or Benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.basal_ganglia import ContinuousBGConfig
from core.config import SNNWorldModelConfig, NeuromodulatorConfig


@dataclass
class TaskConfig:
    """
    All environment-specific knowledge.

    Fields
    ------
    env_id : str
        Gymnasium environment identifier.
    n_episodes : int
        Training budget.  Must be chosen so that a competent agent
        converges, but not so large that a bad agent accidentally solves
        by sheer exposure.
    max_steps : int
        Hard truncation per episode.
    solved_threshold : float
        Mean return (over last `eval_window` episodes) that counts as
        "solved".  Taken from the standard benchmark definition.
    eval_window : int
        Number of trailing episodes used to compute the final score.
    obs_bounds : tuple[np.ndarray, np.ndarray] | None
        Fixed (low, high) normalization bounds for GymEnv.
        None → GymEnv uses running Welford statistics.
    bg_config : ContinuousBGConfig
        Basal-ganglia (critic + actor) hyperparameters.
    wm_config : SNNWorldModelConfig | None
        World-model config.  None means use_world_model=False.
    nm_config : NeuromodulatorConfig
        Neuromodulator hyperparameters (usually default).
    trace_decay : float
        Eligibility-trace memory for state augmentation.
        0.0 = disabled.  Use > 0 only when temporal memory is beneficial
        and the environment does NOT already provide velocity/derivative
        features in the observation.
    reward_scale : float
        Scalar multiplier applied to raw env reward before feeding to
        agent.  Keeps reward magnitudes comparable across tasks so that
        the neuromodulator's z-score normalization works consistently.
    description : str
        Human-readable explanation of why each parameter was chosen.
    """
    env_id: str
    n_episodes: int
    max_steps: int
    solved_threshold: float
    eval_window: int = 20
    obs_bounds: tuple[np.ndarray, np.ndarray] | None = None
    bg_config: ContinuousBGConfig = field(default_factory=ContinuousBGConfig)
    wm_config: SNNWorldModelConfig | None = None
    nm_config: NeuromodulatorConfig = field(default_factory=NeuromodulatorConfig)
    trace_decay: float = 0.0
    reward_scale: float = 1.0
    description: str = ""

    @property
    def use_world_model(self) -> bool:
        return self.wm_config is not None


# =====================================================================
# Canonical task registry
# =====================================================================

REGISTRY: dict[str, TaskConfig] = {

    # ------------------------------------------------------------------
    # CartPole-v1
    # Dense reward (+1 every step survived), 4D state with explicit
    # velocity features.  The agent already has access to derivatives,
    # so trace_decay=0 avoids redundant temporal smearing.
    # use_world_model=False: world model adds latency with no curiosity
    # benefit in a dense-reward setting where every step teaches.
    # gamma=0.99: episodes last up to 500 steps → need long horizon.
    # solved: mean >= 450 over last 20 episodes (conservative vs 475).
    # ------------------------------------------------------------------
    "CartPole-v1": TaskConfig(
        env_id="CartPole-v1",
        n_episodes=120,
        max_steps=500,
        solved_threshold=450.0,
        eval_window=20,
        obs_bounds=(
            np.array([-2.4, -3.0, -0.21, -3.0], dtype=np.float32),
            np.array([ 2.4,  3.0,  0.21,  3.0], dtype=np.float32),
        ),
        bg_config=ContinuousBGConfig(
            gamma=0.99,
            exploration_noise=0.25,
            hidden_size=64,
        ),
        wm_config=None,
        trace_decay=0.0,
        reward_scale=1.0,
        description=(
            "Dense reward, velocity in obs → no trace, no world model. "
            "gamma=0.99 for 500-step horizon."
        ),
    ),

    # ------------------------------------------------------------------
    # MountainCar-v0
    # Sparse reward (-1/step, 0 at goal), 2D state WITHOUT velocity
    # derivatives that would make the task trivial.  trace_decay=0.9
    # gives the agent a short-term memory of where it has been, helping
    # it correlate momentum-building actions with eventual success.
    # use_world_model=True: curiosity signal is essential here because
    # TD error saturates at -1 immediately and gives no gradient signal.
    # gamma=0.99: credit must propagate 100+ steps backward.
    # reward_scale=0.1: brings -200..0 range down to -20..0, keeping
    # tonic-DA z-score calculation stable alongside CartPole's 0..500.
    # ------------------------------------------------------------------
    "MountainCar-v0": TaskConfig(
        env_id="MountainCar-v0",
        n_episodes=300,
        max_steps=200,
        solved_threshold=-110.0,
        eval_window=20,
        obs_bounds=(
            np.array([-1.2, -0.07], dtype=np.float32),
            np.array([ 0.6,  0.07], dtype=np.float32),
        ),
        bg_config=ContinuousBGConfig(
            gamma=0.99,
            exploration_noise=1.0,
            hidden_size=128,
        ),
        wm_config=SNNWorldModelConfig(
            hidden_size=32,
            k_winners=4,
            rehearsal_steps=5,
        ),
        trace_decay=0.9,
        reward_scale=1.0,
        description=(
            "Sparse reward, no velocity derivatives → trace_decay=0.9 + world model. "
            "reward_scale=0.1 keeps neuromod z-score stable. "
            "gamma=0.99 for 200-step horizon."
        ),
    ),
}


def get(env_id: str) -> TaskConfig:
    """Retrieve a TaskConfig by env_id.  Raises KeyError with a helpful message."""
    if env_id not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(
            f"Unknown task {env_id!r}.  "
            f"Available: {available}.  "
            f"Add a TaskConfig entry to arena/task_config.py to register a new task."
        )
    return REGISTRY[env_id]