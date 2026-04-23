"""Phase 6B — reaching-task run-loop + sleep-aware episode driver.

**Performance critical (Phase 6B).**
The earlier version dispatched one brain+MJX cycle per Python
iteration and relied on ``@eqx.filter_jit`` caching.  Measurement on
Colab T4 showed ~16.5 s/cycle regardless of n_cycles — the static
hash of ``MjxArmBody`` (which carries ``mjx_model`` as ``static=True``,
a PyTree of hundreds of MuJoCo arrays) was not cache-stable across
the freshly-returned body objects, so every step re-traced the whole
graph.  That, combined with the Python-side dispatch cost for a very
large cognitive graph, destroyed throughput.

The fix here runs **many cycles inside a single ``jax.lax.scan``**
wrapped in one ``@eqx.filter_jit``.  One XLA compilation, one host
→ device handover, N cycles on the device — Python cost drops to
amortised zero.  Target refresh (a rare event) still happens in
Python between scan chunks, so it does not interfere with the inner
graph.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from core.backend import DTYPE, Array, PRNGKey, BackendContext, split_key
from core.brain_graph import (
    ActionBrainParams, ActionBrainState, action_brain_cognitive_step,
)

from .body_interface import SensorySample
from .mjx_arm_body import MjxArmBody, _sample_target
from .babbling_env import ou_babble_step


# ---------------------------------------------------------------------
# Scan bodies — each is a pure functional one-step transition.
# Kept at module level so filter_jit's cache keys are stable.
# ---------------------------------------------------------------------


def _one_babble_cycle(
    carry: tuple, inp: tuple,
    *, brain_params, ctx, ou_tau, ou_sigma,
) -> tuple[tuple, tuple]:
    """Single babbling cycle used as the scan body.

    Carry: ``(brain_state, body, sensory, reward, done, prev_jc)``.
    Input: per-step PRNG key.
    Output (stacked): ``(jc, tip_xy)``.
    """
    brain_state, body, sensory, prev_r, prev_d, prev_jc = carry
    key = inp

    k_brain, k_body, k_noise = jax.random.split(key, 3)
    out = action_brain_cognitive_step(
        brain_state, brain_params, ctx,
        sensory,
        prev_reward=prev_r, prev_done=prev_d, key=k_brain,
    )
    brain_state = out.state
    jc = ou_babble_step(prev_jc, k_noise, tau=ou_tau, sigma=ou_sigma)
    brain_state = eqx.tree_at(
        lambda s: s.m1.last_joint_command, brain_state, jc,
    )
    body, sample = body.act_continuous(k_body, jc)
    new_carry = (
        brain_state, body,
        sample.sensory,
        jnp.asarray(0.0, DTYPE),   # no extrinsic reward in babbling
        sample.done,
        jc,
    )
    return new_carry, (jc, body.tip_xy())


def _one_reach_cycle(
    carry: tuple, inp: tuple,
    *, brain_params, ctx,
) -> tuple[tuple, tuple]:
    """Single reaching cycle used as the scan body.

    Carry: ``(brain_state, body, sensory, reward, done)``.
    Input: per-step PRNG key.
    Output (stacked): ``(reward, dist, tip_xy, target_xy, qpos)``.
    """
    brain_state, body, sensory, prev_r, prev_d = carry
    key = inp

    k_brain, k_body = jax.random.split(key, 2)
    out = action_brain_cognitive_step(
        brain_state, brain_params, ctx,
        sensory,
        prev_reward=prev_r, prev_done=prev_d, key=k_brain,
    )
    brain_state = out.state
    jc = brain_state.m1.last_joint_command
    body, sample = body.act_continuous(k_body, jc)
    tip = body.tip_xy()
    tgt = body.target_xy
    d = jnp.linalg.norm(tip - tgt)
    new_carry = (
        brain_state, body, sample.sensory, sample.reward, sample.done,
    )
    return new_carry, (sample.reward, d, tip, tgt, body.qpos())


@eqx.filter_jit
def _babble_chunk(
    brain_state, body, sensory, prev_r, prev_d, prev_jc,
    brain_params, ctx, keys, ou_tau, ou_sigma,
):
    """Run ``keys.shape[0]`` babbling cycles under one XLA graph."""
    def step_fn(c, k):
        return _one_babble_cycle(
            c, k,
            brain_params=brain_params, ctx=ctx,
            ou_tau=ou_tau, ou_sigma=ou_sigma,
        )
    init = (brain_state, body, sensory, prev_r, prev_d, prev_jc)
    final, outputs = jax.lax.scan(step_fn, init, keys)
    return final, outputs


@eqx.filter_jit
def _reach_chunk(
    brain_state, body, sensory, prev_r, prev_d,
    brain_params, ctx, keys,
):
    """Run ``keys.shape[0]`` reach cycles under one XLA graph."""
    def step_fn(c, k):
        return _one_reach_cycle(c, k, brain_params=brain_params, ctx=ctx)
    init = (brain_state, body, sensory, prev_r, prev_d)
    final, outputs = jax.lax.scan(step_fn, init, keys)
    return final, outputs


# ---------------------------------------------------------------------
# Episode / session drivers
# ---------------------------------------------------------------------


class ReachResult(NamedTuple):
    rewards: Array
    dists: Array
    tip_traj: Array
    target_traj: Array
    success: Array
    brain_state: ActionBrainState
    body: MjxArmBody
    qpos_traj: Array


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
    record_qpos: bool = False,   # qpos is always recorded now; kwarg kept for API compat
) -> ReachResult:
    """Run one reaching episode via a single ``lax.scan``.

    The inner ``_reach_chunk`` compiles once per distinct
    ``max_steps`` value; subsequent calls are device-resident loops.
    Early termination is not supported inside scan, but the returned
    ``success`` flag indicates whether the tip ever crossed
    ``success_dist`` during the episode.
    """
    del record_qpos  # always recorded; `qpos_traj` is part of the result
    k = key
    if reset_body:
        k, k_reset = split_key(k, 2)
        body, sample = body.reset(k_reset)
        sensory = sample.sensory
        prev_r = sample.reward
        prev_d = sample.done
    else:
        sensory = jnp.zeros(body.sensory_size, DTYPE)
        prev_r = jnp.asarray(0.0, DTYPE)
        prev_d = jnp.asarray(0.0, DTYPE)

    k, k_steps = split_key(k, 2)
    keys = jax.random.split(k_steps, int(max_steps))

    (brain_state, body, _, _, _), (rewards, dists, tips, tgts, qposes) = (
        _reach_chunk(
            brain_state, body, sensory, prev_r, prev_d,
            brain_params, ctx, keys,
        )
    )
    success = jnp.any(dists < jnp.asarray(success_dist, DTYPE))
    return ReachResult(
        rewards=rewards, dists=dists,
        tip_traj=tips, target_traj=tgts,
        success=success,
        brain_state=brain_state, body=body,
        qpos_traj=qposes,
    )


class BabbleResult(NamedTuple):
    tip_traj: Array
    jc_traj: Array
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
    """OU-process motor babbling driven by one ``lax.scan`` per chunk.

    The run is split into chunks of ``target_refresh`` cycles.  Each
    chunk is a single XLA kernel launch; the mocap target is rotated
    between chunks in Python (negligible cost).  If ``n_cycles`` is
    not an exact multiple of ``target_refresh`` the last chunk is
    shorter and triggers a *one-time* extra compilation.
    """
    k = key
    k, k_reset = split_key(k, 2)
    body, sample = body.reset(k_reset)

    sensory = sample.sensory
    prev_r = jnp.asarray(0.0, DTYPE)
    prev_d = sample.done
    motor_dim = brain_params.m1.motor_dim
    prev_jc = jnp.zeros(motor_dim, DTYPE)

    tau_a = jnp.asarray(ou_tau, DTYPE)
    sigma_a = jnp.asarray(ou_sigma, DTYPE)

    tips_chunks: list[Array] = []
    jcs_chunks: list[Array] = []

    remaining = int(n_cycles)
    chunk_size = int(target_refresh)

    while remaining > 0:
        this_chunk = min(chunk_size, remaining)
        k, k_chunk, k_tgt = split_key(k, 3)
        keys = jax.random.split(k_chunk, this_chunk)

        (brain_state, body, sensory, prev_r, prev_d, prev_jc), (jcs, tips) = (
            _babble_chunk(
                brain_state, body, sensory, prev_r, prev_d, prev_jc,
                brain_params, ctx, keys, tau_a, sigma_a,
            )
        )
        tips_chunks.append(tips)
        jcs_chunks.append(jcs)

        # Refresh target for the next chunk.
        new_tgt = _sample_target(k_tgt, body.cfg.workspace_half)
        body = body._set_target(new_tgt)
        remaining -= this_chunk

    tip_traj = (
        jnp.concatenate(tips_chunks, axis=0)
        if tips_chunks else jnp.zeros((0, 2), DTYPE)
    )
    jc_traj = (
        jnp.concatenate(jcs_chunks, axis=0)
        if jcs_chunks else jnp.zeros((0, motor_dim), DTYPE)
    )
    return BabbleResult(
        tip_traj=tip_traj,
        jc_traj=jc_traj,
        brain_state=brain_state,
        body=body,
    )


# ---------------------------------------------------------------------
# Rendering — CPU only, not JIT-ed.
# ---------------------------------------------------------------------


def collect_render_frames(
    body_template: MjxArmBody,
    qpos_traj: Array,
    target_traj: Array,
    *,
    width: int = 320, height: int = 240,
    camera: str = "topdown",
) -> Any:
    """Render a trajectory to RGB frames via ``mujoco.Renderer``."""
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
