"""Phase 6B — reaching-task run-loop + sleep-aware episode driver.

This module plumbs ``ActionBrain`` into :class:`MjxArmBody` using the
continuous-command path (``act_continuous``).  It is the Phase 6B
analogue of :func:`embodiment.run_episode`, specialised for the
continuous motor substrate and with optional babbling pre-training.

The driver never modifies the brain internals; it simply:
 1. pulls the last ``joint_command`` from ``state.m1`` each cycle;
 2. hands it to the body's ``act_continuous``;
 3. feeds the new sensory sample back into the brain.

Video rendering is **out of band**: :func:`collect_render_frames`
extracts frames from the ``mj_model`` via a CPU-only ``mujoco.Renderer``
after the trajectory has been simulated.  This means the inner loop
stays JIT-able while the Colab notebook can still produce an mp4.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from core.backend import DTYPE, Array, PRNGKey, BackendContext, split_key
from core.brain_graph import (
    ActionBrainParams, ActionBrainState, action_brain_cognitive_step,
)

from .body_interface import SensorySample
from .mjx_arm_body import MjxArmBody
from .babbling_env import ou_babble_step


class ReachResult(NamedTuple):
    rewards: Array           # (T,) float32 — extrinsic reward per cycle
    dists: Array             # (T,) float32 — tip→target distance
    tip_traj: Array          # (T, 2) tip positions
    target_traj: Array       # (T, 2) target positions
    success: Array           # scalar bool — reached within threshold
    brain_state: ActionBrainState
    body: MjxArmBody
    qpos_traj: Array         # (T, n_joints) — empty if record_qpos=False


def run_reach_episode(
    brain_state: ActionBrainState,
    brain_params: ActionBrainParams,
    ctx: BackendContext,
    body: MjxArmBody,
    key: PRNGKey,
    *,
    max_steps: int = 500,
    success_dist: float = 0.05,
    reset_body: bool = True,
    record_qpos: bool = False,
) -> ReachResult:
    """Run one reaching episode; return per-step diagnostics.

    Set ``record_qpos=True`` to also retain the joint-position trajectory
    for offline video rendering via :func:`collect_render_frames`.
    """
    k = key
    if reset_body:
        k, k_reset = split_key(k, 2)
        body, sample = body.reset(k_reset)
    else:
        sample = SensorySample(
            sensory=jnp.zeros(body.sensory_size, DTYPE),
            reward=jnp.asarray(0.0, DTYPE),
            done=jnp.asarray(0.0, DTYPE),
            info={},
        )

    rewards, dists, tips, tgts = [], [], [], []
    qposes: list[Array] = []
    prev_reward = jnp.asarray(0.0, DTYPE)
    prev_done = jnp.asarray(0.0, DTYPE)
    reached = False

    for t in range(max_steps):
        k, k_brain, k_body = split_key(k, 3)
        out = action_brain_cognitive_step(
            brain_state, brain_params, ctx,
            sample.sensory,
            prev_reward=prev_reward, prev_done=prev_done, key=k_brain,
        )
        brain_state = out.state
        jc = brain_state.m1.last_joint_command
        body, sample = body.act_continuous(k_body, jc)

        tip = body.tip_xy()
        tgt = body.target_xy
        d = jnp.linalg.norm(tip - tgt)
        dists.append(d)
        rewards.append(sample.reward)
        tips.append(tip)
        tgts.append(tgt)
        if record_qpos:
            qposes.append(body.qpos())
        prev_reward = sample.reward
        prev_done = sample.done
        if bool(d < success_dist):
            reached = True
            break
        if bool(sample.done):
            break

    return ReachResult(
        rewards=jnp.asarray(rewards, DTYPE),
        dists=jnp.asarray(dists, DTYPE),
        tip_traj=jnp.stack(tips) if tips else jnp.zeros((0, 2), DTYPE),
        target_traj=jnp.stack(tgts) if tgts else jnp.zeros((0, 2), DTYPE),
        success=jnp.asarray(reached),
        brain_state=brain_state,
        body=body,
        qpos_traj=(
            jnp.stack(qposes) if qposes
            else jnp.zeros((0, body.cfg.n_joints), DTYPE)
        ),
    )


class BabbleResult(NamedTuple):
    tip_traj: Array          # (T, 2)
    jc_traj: Array           # (T, motor_dim)
    brain_state: ActionBrainState
    body: MjxArmBody


def run_babbling(
    brain_state: ActionBrainState,
    brain_params: ActionBrainParams,
    ctx: BackendContext,
    body: MjxArmBody,
    key: PRNGKey,
    *,
    n_cycles: int = 30_000,
    ou_tau: float = 20.0,
    ou_sigma: float = 0.4,
    target_refresh: int = 400,
) -> BabbleResult:
    """OU-process motor babbling.

    The brain is driven normally (so cerebellum still sees motor PE),
    but we **override** the M1 joint command with fresh OU noise every
    cycle so the body explores broadly.  The override is stored back
    into ``state.m1.last_joint_command`` so the next cerebellum
    prediction uses it as the action context.
    """
    import equinox as eqx
    k = key
    k, k_reset = split_key(k, 2)
    body, sample = body.reset(k_reset)
    prev_reward = jnp.asarray(0.0, DTYPE)
    prev_done = jnp.asarray(0.0, DTYPE)

    motor_dim = brain_params.m1.motor_dim
    jc = jnp.zeros(motor_dim, DTYPE)
    tips, jcs = [], []

    for t in range(n_cycles):
        k, k_brain, k_body, k_noise, k_tgt = split_key(k, 5)
        out = action_brain_cognitive_step(
            brain_state, brain_params, ctx,
            sample.sensory,
            prev_reward=prev_reward, prev_done=prev_done, key=k_brain,
        )
        brain_state = out.state
        # OU noise override.
        jc = ou_babble_step(jc, k_noise, tau=ou_tau, sigma=ou_sigma)
        brain_state = eqx.tree_at(
            lambda s: s.m1.last_joint_command, brain_state, jc,
        )
        body, sample = body.act_continuous(k_body, jc)
        # Ignore extrinsic reward during babbling — curiosity drives BG.
        prev_reward = jnp.asarray(0.0, DTYPE)
        prev_done = sample.done

        if (t + 1) % target_refresh == 0:
            from .mjx_arm_body import _sample_target
            new_tgt = _sample_target(k_tgt, body.cfg.workspace_half)
            body = body._set_target(new_tgt)

        tips.append(body.tip_xy())
        jcs.append(jc)

    return BabbleResult(
        tip_traj=jnp.stack(tips) if tips else jnp.zeros((0, 2), DTYPE),
        jc_traj=jnp.stack(jcs) if jcs else jnp.zeros((0, motor_dim), DTYPE),
        brain_state=brain_state,
        body=body,
    )


# ---------------------------------------------------------------------
# Rendering — CPU only, not JIT-ed
# ---------------------------------------------------------------------


def collect_render_frames(
    body_template: MjxArmBody,
    qpos_traj: Array,
    target_traj: Array,
    *,
    width: int = 320, height: int = 240,
    camera: str = "topdown",
) -> Any:
    """Render a trajectory to a list of RGB frames using mujoco.Renderer.

    ``qpos_traj`` is ``(T, n_joints)``; ``target_traj`` is ``(T, 2)``.
    Returns a list of ``(H, W, 3)`` uint8 numpy arrays (one per cycle),
    suitable for ``imageio.mimsave(..., format='FFMPEG', fps=30)``.
    """
    import numpy as np
    import mujoco

    mj_model = body_template.mj_model
    data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    frames = []

    qpos_np = np.asarray(qpos_traj)
    tgt_np = np.asarray(target_traj)
    for q, tgt in zip(qpos_np, tgt_np):
        data.qpos[: q.shape[0]] = q
        data.qvel[:] = 0.0
        data.mocap_pos[0, :2] = tgt
        data.mocap_pos[0, 2] = 0.05
        mujoco.mj_forward(mj_model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())
    return frames
