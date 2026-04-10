"""
arena.agent_factory — Constructs SNNAgent instances from TaskConfig.

The factory is the ONLY place that knows both:
  (a) the TaskConfig (evaluation metadata), and
  (b) the Environment (structural dimensions: state_size, n_actions).

The resulting SNNAgent knows NEITHER the env_id nor any task-specific
hyperparameters.  All learning parameters come from universal defaults
in BasalGangliaConfig / NeuromodulatorConfig / WorldModelConfig.
"""

from __future__ import annotations

from arena.core import Environment
from arena.snn_agent import SNNAgent
from arena.task_config import TaskConfig


def make_agent(task: TaskConfig, env: Environment) -> SNNAgent:
    """Build an SNNAgent sized for *env*, with universal hyperparameters.

    Only structural information (state_size, n_actions) and the
    world_model/working_memory feature flags pass from task to agent.
    """
    return SNNAgent(
        state_size=env.state_size,
        n_actions=env.n_actions,
        use_world_model=task.use_world_model,
        use_working_memory=task.use_working_memory,
    )
