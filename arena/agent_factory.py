"""
arena.agent_factory — Constructs SNNAgent instances from TaskConfig.

Separation of concerns
======================
The factory is the ONLY place that knows both:
  (a) the TaskConfig (environment-specific hyperparameters), and
  (b) the GymEnv (structural dimensions: state_size, n_actions).

The resulting SNNAgent knows NEITHER.  It receives:
  - Two integers: state_size, n_actions
  - Learning configs: bg_config, wm_config, nm_config, trace_decay
  - Nothing else.

This is the anti-cheat contract: the agent cannot embed task-specific
knowledge because it has no way to identify which task it is solving.
Its only signal is the normalised observation vector (float32 array)
whose origin it does not know.
"""

from __future__ import annotations

from arena.core import Environment
from arena.snn_agent import SNNAgent
from arena.task_config import TaskConfig


def make_agent(task: TaskConfig, env: Environment) -> SNNAgent:
    """
    Build an SNNAgent sized and configured for *task*, without passing
    any task-identifying information to the agent itself.

    Parameters
    ----------
    task : TaskConfig
        Source of all hyperparameters (bg_config, wm_config, etc.).
    env : Environment
        Source of structural dimensions (state_size, n_actions).
        The env is ONLY queried for its dimensions — no other
        environment-specific information leaks into the agent.

    Returns
    -------
    SNNAgent
        Freshly initialised agent ready for a new training run.
    """
    return SNNAgent(
        state_size=env.state_size,
        n_actions=env.n_actions,
        bg_config=task.bg_config,
        wm_config=task.wm_config,
        nm_config=task.nm_config,
        use_world_model=task.use_world_model,
        trace_decay=task.trace_decay,
    )