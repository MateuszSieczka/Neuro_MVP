"""Goal-directed reaching — active inference drives the arm to a target.

The first embodiment milestone (LEGACY_INTEGRATION.md §1): a body adapter
driving ``pc_brain`` to a reach, validated on reach success.  Each cycle is
the babble cycle with the belief chosen by **active inference** instead of
noise: clamp the goal (target-error → zero) on the sensory node, relax with
the flat-prior motor node free, read the command that the forward model says
realises the goal (Adams, Shipp & Friston 2013), execute it, and keep the
forward model adapting on the reafference.  No policy gradient anywhere.

Build a reacher with :func:`build_reacher` (pairs an :class:`MjxArmBody`
with a :func:`core.pc_brain.init_pc_brain`), babble it
(:func:`embodiment.babbling.run_babbling`), then call :func:`run_reach`.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from core.backend import DTYPE, Array, PRNGKey, split_key
from core.pc_brain import (
    PCBrainParams, PCBrainState, init_pc_brain,
    pc_brain_act, pc_brain_learn_forward,
)

from .mjx_arm_body import MjxArmBody, ArmConfig


class ReachConfig(NamedTuple):
    """Static reaching-episode + active-inference hyper-parameters."""

    max_steps: int = 500
    #: Tip within this distance (m) of the target counts as a reach.
    success_dist: float = 0.05
    #: Sparse bonus added the step the tip is inside ``success_dist``.
    success_bonus: float = 1.0
    #: Relaxation steps for the action-inference (planning) relaxation.
    act_relax_steps: int = 80
    #: Settling steps inside each closed-loop forward-model update.
    forward_settle_steps: int = 1


def build_reacher(
    key: PRNGKey,
    *,
    arm_cfg: ArmConfig | None = None,
    **brain_kwargs,
) -> tuple[PCBrainParams, PCBrainState, MjxArmBody]:
    """Build (brain params, brain state, body) ready for babble → reach.

    The body's sensory/motor sizes drive the brain construction, so the two
    are always consistent; ``brain_kwargs`` forward to
    :func:`core.pc_brain.init_pc_brain` (learning rates, ``n_relax``, …).
    """
    k_body, k_brain = split_key(key, 2)
    body = MjxArmBody.create(k_body, cfg=arm_cfg)
    params, state = init_pc_brain(
        k_brain,
        sensory_size=body.sensory_size,
        motor_size=body.motor_dim,
        **brain_kwargs,
    )
    return params, state, body


class ReachResult(NamedTuple):
    brain_state: PCBrainState
    body: MjxArmBody
    rewards: Array             # (max_steps,) shaped reward
    dists: Array               # (max_steps,) tip→target distance
    tip_traj: Array            # (max_steps, 2)
    target_traj: Array         # (max_steps, 2)
    qpos_traj: Array           # (max_steps, n_joints) — for rendering
    success: Array             # scalar bool — tip ever within success_dist


def _one_reach_cycle(
    carry, key,
    *, params: PCBrainParams, cfg: ReachConfig,
    goal_preference: Array, goal_mask: Array,
):
    """Scan body — infer goal-directed command → execute → learn forward.

    Reward is potential-based shaping with potential Φ(s) = −dist (Ng et al.
    1999): ``r = (prev_dist − dist) + success_bonus·𝟙[dist < success_dist]``.
    Mean step reward ≈ 0, so a critic does not absorb the shaping and the
    sign tracks genuine progress.
    """
    brain_state, body, prev_dist = carry

    act = pc_brain_act(
        brain_state, params, goal_preference,
        preference_mask=goal_mask, n_relax=cfg.act_relax_steps,
    )
    body, sample = body.act(key, act.joint_command)
    brain_state = pc_brain_learn_forward(
        brain_state, params, act.motor_belief, sample.sensory,
        n_relax=cfg.forward_settle_steps,
    )

    tip, tgt = body.tip_xy(), body.target_xy
    dist = jnp.linalg.norm(tip - tgt)
    progress = (prev_dist - dist).astype(DTYPE)
    bonus = jnp.where(
        dist < jnp.asarray(cfg.success_dist, DTYPE),
        jnp.asarray(cfg.success_bonus, DTYPE),
        jnp.asarray(0.0, DTYPE),
    )
    reward = (progress + bonus).astype(DTYPE)
    return (brain_state, body, dist), (reward, dist, tip, tgt, body.qpos())


def run_reach(
    brain_state: PCBrainState,
    params: PCBrainParams,
    body: MjxArmBody,
    key: PRNGKey,
    cfg: ReachConfig = ReachConfig(),
    *,
    reset_body: bool = True,
) -> ReachResult:
    """Run one reaching episode under a single ``jax.lax.scan``.

    Compiles once per distinct ``cfg.max_steps``; subsequent calls are
    device-resident loops.  The goal ("tip on target") is target-independent,
    so it is fixed for the whole episode.
    """
    goal_preference, goal_mask = body.reach_goal()

    @eqx.filter_jit
    def _episode(brain_state, body, prev_dist, keys):
        step = lambda c, k: _one_reach_cycle(
            c, k, params=params, cfg=cfg,
            goal_preference=goal_preference, goal_mask=goal_mask,
        )
        return jax.lax.scan(step, (brain_state, body, prev_dist), keys)

    k = key
    if reset_body:
        k, k_reset = split_key(k, 2)
        body, _ = body.reset(k_reset)
    # Seed Φ with the post-reset distance → first-step progress = 0.
    init_dist = jnp.linalg.norm(body.tip_xy() - body.target_xy)

    k, k_steps = split_key(k, 2)
    keys = jax.random.split(k_steps, int(cfg.max_steps))
    (brain_state, body, _), (rewards, dists, tips, tgts, qposes) = _episode(
        brain_state, body, init_dist, keys,
    )
    success = jnp.any(dists < jnp.asarray(cfg.success_dist, DTYPE))
    return ReachResult(
        brain_state=brain_state, body=body,
        rewards=rewards, dists=dists,
        tip_traj=tips, target_traj=tgts, qpos_traj=qposes,
        success=success,
    )


def collect_render_frames(
    body_template: MjxArmBody,
    qpos_traj: Array,
    target_traj: Array,
    *,
    width: int = 320, height: int = 240,
    camera: str = "topdown",
) -> Any:
    """Render a trajectory to RGB frames via ``mujoco.Renderer`` (CPU, no JIT)."""
    import numpy as np
    import mujoco

    mj_model = body_template.mj_model
    data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    target_z = body_template.cfg.target_z
    frames = []
    for q, tgt in zip(np.asarray(qpos_traj), np.asarray(target_traj)):
        data.qpos[: q.shape[0]] = q
        data.qvel[:] = 0.0
        data.mocap_pos[0, :2] = tgt
        data.mocap_pos[0, 2] = target_z
        mujoco.mj_forward(mj_model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())
    return frames
