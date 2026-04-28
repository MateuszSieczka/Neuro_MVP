"""Phase 6B — reaching task convenience factory.

Thin wrapper that builds an :class:`MjxArmBody` pre-configured for
target-reaching (extrinsic reward on), paired with a matching
``ActionBrainParams`` that has ``bypass_m1=False`` so the continuous
M1 head drives the body.
"""
from __future__ import annotations

from core.backend import DEFAULT, make_key, BackendContext, PRNGKey
from core.brain_graph import (
    ActionBrainParams, ActionBrainState,
    init_action_brain_params, init_action_brain_state,
)

from .mjx_arm_body import MjxArmBody, ArmConfig, default_arm_config


def build_reacher(
    key: PRNGKey,
    *,
    ctx: BackendContext = DEFAULT,
    arm_cfg: ArmConfig | None = None,
    cortex_n_l5: int = 32,
    m1_readout_lr: float = 5e-3,
) -> tuple[ActionBrainParams, ActionBrainState, MjxArmBody]:
    """Return (params, state, body) ready to drive target reaching."""
    import jax
    k_body, k_brain = jax.random.split(key)

    cfg = arm_cfg or default_arm_config(include_target=True)
    body = MjxArmBody.create(k_body, cfg=cfg)

    params = init_action_brain_params(
        ctx,
        sensory_size=body.sensory_size,
        n_body_actions=body.n_actions,
        cortex_n_l5=cortex_n_l5,
        n_joints=cfg.n_joints,
        n_cells_per_joint=cfg.n_cells_per_joint,
        m1_readout_lr=m1_readout_lr,
        bypass_m1=False,                         # Phase 6B: M1 drives body
        use_real_proprio=True,                   # Phase 6B: MJX qpos/qvel
    )
    state = init_action_brain_state(k_brain, params)
    return params, state, body
