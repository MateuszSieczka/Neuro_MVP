"""arena.agent_factory — Constructs SNNAgent instances from TaskConfig."""

from __future__ import annotations

from arena.core import Environment
from arena.snn_agent import SNNAgent
from arena.task_config import TaskConfig


def make_agent(task: TaskConfig, env: Environment) -> SNNAgent:
    """Build an SNNAgent sized for *env*, with universal hyperparameters.

    Only structural information (state_size, n_actions) and the
    world_model/working_memory feature flags pass from task to agent.
    All learning hyperparameters come from universal *Config defaults.
    """
    return SNNAgent(
        state_size=env.state_size,
        n_actions=env.n_actions,
        use_world_model=task.use_world_model,
        use_working_memory=task.use_working_memory,
    )