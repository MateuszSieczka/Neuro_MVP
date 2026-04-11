"""
arena.agent_factory — Constructs SNNAgent instances from TaskConfig.

The factory is the ONLY place that knows both:
  (a) the TaskConfig (evaluation metadata), and
  (b) the Environment (structural dimensions: state_size, n_actions).

The resulting SNNAgent knows NEITHER the env_id nor any task-specific
hyperparameters.  All learning parameters come from universal defaults
in BasalGangliaConfig / NeuromodulatorConfig / WorldModelConfig.

For high-dimensional inputs (state_size >= COLUMNAR_THRESHOLD), the
factory enables columnar mode with spatial receptive fields, k-WTA
competitive selection, and spatial attention.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena.core import Environment
from arena.snn_agent import SNNAgent
from arena.task_config import TaskConfig


@dataclass(frozen=True)
class FactoryConfig:
    """Agent construction parameters (previously hardcoded)."""
    columnar_threshold: int = 16
    default_receptive_field: int = 4


_DEFAULT_FACTORY = FactoryConfig()


def make_agent(
    task: TaskConfig,
    env: Environment,
    factory_cfg: FactoryConfig | None = None,
) -> SNNAgent:
    """Build an SNNAgent sized for *env*, with universal hyperparameters.

    Only structural information (state_size, n_actions) and the
    world_model/working_memory feature flags pass from task to agent.

    For high-dim environments (state_size >= COLUMNAR_THRESHOLD), the
    columnar architecture is activated with derived receptive field size.
    """
    fcfg = factory_cfg or _DEFAULT_FACTORY
    use_columnar = env.state_size >= fcfg.columnar_threshold
    rf_size: int | None = None

    if use_columnar:
        # Derive receptive field size: largest divisor <= default_receptive_field
        rf_size = fcfg.default_receptive_field
        while env.state_size % rf_size != 0 and rf_size > 1:
            rf_size -= 1

    return SNNAgent(
        state_size=env.state_size,
        n_actions=env.n_actions,
        use_world_model=task.use_world_model,
        use_working_memory=task.use_working_memory,
        use_columnar=use_columnar,
        receptive_field_size=rf_size,
    )
