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

from .body_interface import SensorySample, discretise_joint_command
from .mjx_arm_body import MjxArmBody, _sample_target


# ---------------------------------------------------------------------
# Phase 6B — keep the brain's self-model in sync with the real MJX
# physics.  The cognitive step at the start of cycle ``t`` reads
# ``state.last_body_action_id`` / ``state.last_body_action`` and
# ``state.last_joint_angles`` / ``state.last_joint_velocities`` as the
# ``action that was executed at t-1`` and ``joint state at end of
# t-1`` respectively.  Before Phase 6B the only writer of those
# fields was the synthetic pseudo-kinematics path inside
# ``action_brain_cognitive_step`` driven by the M1-committed discrete
# action.  During motor babbling the body is actually driven by an OU
# command *different from* what the brain committed, so the world
# model, cerebellum forward model, and efference-copy mossy input all
# learned (imagined-action, physical-sensory) pairs instead of the
# real (executed-action, physical-sensory) pairs.  During reaching
# with ``bypass_m1=False`` the executed action equals M1's output so
# the action fields are already consistent; the joint-angle fields
# however were still the ±0.1 rad synthetic delta, never the real
# qpos.  ``_sync_brain_to_body`` closes both loops.
# ---------------------------------------------------------------------


def _sync_brain_to_body(
    brain_state: ActionBrainState,
    body: MjxArmBody,
    jc_executed: Array,
    n_body_actions: int,
) -> ActionBrainState:
    """Overwrite the brain's ``last_*`` fields with ground-truth values.

    * ``m1.last_joint_command`` ← the jc that was actually sent to
      the physics (== M1's output in reach, == OU sample in babble).
    * ``last_body_action_id`` / ``last_body_action`` ← discretisation
      of ``jc_executed`` (same sign-split mapping the env uses in the
      fallback ``act`` path), so cortex L5 → efference-copy mossy and
      world-model action features reflect the real action.
    * ``last_joint_angles`` / ``last_joint_velocities`` ← real MJX
      ``qpos`` / ``qvel`` normalised into the brain's proprio encoder
      range (params.proprio default ±1.0 rad; body runs in radians so
      we divide by ``cfg.joint_range``).  Velocity uses the same
      factor 4 ratio the body's own proprio encoder uses so that
      ``|qvel| ≈ joint_range * 4`` saturates the encoding peak.

    Called *after* ``body.act_continuous`` (or ``body.reset``) so the
    body already holds the post-step physical state.
    """
    a_id = discretise_joint_command(jc_executed, n_body_actions)
    a_oh = (jnp.arange(n_body_actions) == a_id).astype(DTYPE)
    jr = jnp.asarray(body.cfg.joint_range, DTYPE)
    qpos = body.qpos() / jr
    qvel = body.qvel() / (jr * jnp.asarray(4.0, DTYPE))
    qpos = jnp.clip(qpos, -1.0, 1.0)
    qvel = jnp.clip(qvel, -1.0, 1.0)
    return eqx.tree_at(
        lambda s: (
            s.m1.last_joint_command,
            s.last_body_action_id,
            s.last_body_action,
            s.last_joint_angles,
            s.last_joint_velocities,
        ),
        brain_state,
        (
            jnp.asarray(jc_executed, DTYPE),
            a_id,
            a_oh,
            qpos.astype(DTYPE),
            qvel.astype(DTYPE),
        ),
    )


# ---------------------------------------------------------------------
# Scan bodies — each is a pure functional one-step transition.
# Kept at module level so filter_jit's cache keys are stable.
# ---------------------------------------------------------------------


def _one_babble_cycle(
    carry: tuple, inp: tuple,
    *, brain_params, ctx,
) -> tuple[tuple, tuple]:
    """Single babbling cycle used as the scan body.

    Phase 6B fix: babbling now drives the body with **M1's own
    noisy joint command** rather than an external Ornstein-Uhlenbeck
    process, and feeds back ``world_model.curiosity`` as the
    intrinsic reward.  This means:

    * Exploration comes from M1's NE-coupled exploration noise
      (Aston-Jones & Cohen 2005; Tumer & Brainard 2007), so the
      same node-perturbation REINFORCE rule that drives reach
      learning is also active during babbling — weights are no
      longer frozen at zero RPE for 30 k cycles.
    * The cerebellum forward model and world model see the
      (executed-action, real-sensory) pairs that *will* be used
      during reach — distribution match instead of OU mismatch.
    * Pre-fix bug: ``prev_reward = 0`` ⇒ RPE = 0 ⇒ ``m1_learn_readout``
      dw = 0 for the entire 30 k babble; M1 entered reach with a
      random PCA-synergy readout.

    Carry: ``(brain_state, body, sensory, reward, done)``.
    Input: per-step PRNG key.
    Output (stacked): ``(jc, tip_xy)``.
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
    # Phase 6B: write the executed action + real qpos/qvel into the
    # brain so next cycle's world-model, motor-PE, and efference-copy
    # see the action that actually moved the arm.
    brain_state = _sync_brain_to_body(
        brain_state, body, jc, brain_params.n_body_actions,
    )
    # Intrinsic reward: world-model curiosity (transition surprise).
    # Friston 2017 EFE epistemic value; bounded ∈ [0, 1].  This is
    # the same signal the cognitive step already feeds into
    # body-actor RPE as ``body_bonus``; using it as the extrinsic
    # reward during babbling lets the VTA/critic drive M1 learning
    # toward states the world model is still uncertain about.
    intrinsic_r = out.curiosity
    new_carry = (
        brain_state, body,
        sample.sensory,
        intrinsic_r.astype(DTYPE),
        sample.done,
    )
    return new_carry, (jc, body.tip_xy())


def _one_reach_cycle(
    carry: tuple, inp: tuple,
    *, brain_params, ctx,
    success_dist: float,
    success_bonus: float,
) -> tuple[tuple, tuple]:
    """Single reaching cycle used as the scan body.

    Reward shaping (Phase 6B fix):
        r_t = (prev_dist − dist) + success_bonus · 𝟙[dist < success_dist]

    This is potential-based shaping with potential Φ(s) = −dist
    (Ng et al. 1999 §3), which preserves the optimal policy of the
    underlying sparse-reward problem while giving a dense per-step
    progress signal.  Average step reward is ≈ 0 (no DC bias), so
    V(s) does not absorb the shaping and the RPE sign tracks actual
    *progress* events — which is what the BG D1 LTP / D2 LTD
    asymmetry was calibrated for (Collins & Frank 2014).

    Pre-fix bug: body returned ``r = -dist ∈ [-0.45, -0.15]``, a
    permanently-negative value.  V(s) tracked the long-run mean
    (≈ −0.4); RPE ≈ −0.4 − (−0.4) ≈ noise around 0 with no clear
    sign on improvements → D1 LTP rarely triggered.

    Carry: ``(brain_state, body, sensory, reward, done, prev_dist)``.
    Input: per-step PRNG key.
    Output (stacked): ``(reward, dist, tip_xy, target_xy, qpos)``.
    """
    brain_state, body, sensory, prev_r, prev_d, prev_dist = carry
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
    # Phase 6B: sync brain state with real MJX proprio.  In reach the
    # action one-hot is already consistent (M1 jc == executed jc) so
    # the sync is effectively a no-op for the action fields; its
    # purpose here is to inject the real qpos/qvel into the brain so
    # next cycle's motor-PE measures actual vs predicted kinematics.
    brain_state = _sync_brain_to_body(
        brain_state, body, jc, brain_params.n_body_actions,
    )
    tip = body.tip_xy()
    tgt = body.target_xy
    d = jnp.linalg.norm(tip - tgt)
    # Potential-based shaping + sparse success bonus.
    progress = (prev_dist - d).astype(DTYPE)
    bonus = jnp.where(
        d < jnp.asarray(success_dist, DTYPE),
        jnp.asarray(success_bonus, DTYPE),
        jnp.asarray(0.0, DTYPE),
    )
    r_shaped = (progress + bonus).astype(DTYPE)
    new_carry = (
        brain_state, body, sample.sensory, r_shaped, sample.done, d,
    )
    return new_carry, (r_shaped, d, tip, tgt, body.qpos())


@eqx.filter_jit
def _babble_chunk(
    brain_state, body, sensory, prev_r, prev_d,
    brain_params, ctx, keys,
):
    """Run ``keys.shape[0]`` babbling cycles under one XLA graph."""
    def step_fn(c, k):
        return _one_babble_cycle(
            c, k,
            brain_params=brain_params, ctx=ctx,
        )
    init = (brain_state, body, sensory, prev_r, prev_d)
    final, outputs = jax.lax.scan(step_fn, init, keys)
    return final, outputs


@eqx.filter_jit
def _reach_chunk(
    brain_state, body, sensory, prev_r, prev_d, prev_dist,
    brain_params, ctx, keys,
    success_dist, success_bonus,
):
    """Run ``keys.shape[0]`` reach cycles under one XLA graph."""
    sd = float(success_dist)
    sb = float(success_bonus)
    def step_fn(c, k):
        return _one_reach_cycle(
            c, k, brain_params=brain_params, ctx=ctx,
            success_dist=sd, success_bonus=sb,
        )
    init = (brain_state, body, sensory, prev_r, prev_d, prev_dist)
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
    success_bonus: float = 1.0,
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

    # Initial distance for potential-based shaping (Φ = −dist).  At
    # the very first step there is no "previous" tip, so seed with
    # the post-reset tip→target distance → first-step progress = 0.
    init_dist = jnp.linalg.norm(body.tip_xy() - body.target_xy)

    k, k_steps = split_key(k, 2)
    keys = jax.random.split(k_steps, int(max_steps))

    (brain_state, body, _, _, _, _), (rewards, dists, tips, tgts, qposes) = (
        _reach_chunk(
            brain_state, body, sensory, prev_r, prev_d, init_dist,
            brain_params, ctx, keys,
            float(success_dist), float(success_bonus),
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
    target_refresh: int = 400,
    ou_tau: float | None = None,    # deprecated; ignored (OU dropped Phase 6B)
    ou_sigma: float | None = None,  # deprecated; ignored
) -> BabbleResult:
    """M1-driven motor babbling under one ``lax.scan`` per chunk.

    Phase 6B fix: babbling now drives the body with M1's own noisy
    joint command (NE-coupled exploration noise; Aston-Jones &
    Cohen 2005) instead of an external Ornstein-Uhlenbeck process.
    The intrinsic reward signal is the world-model curiosity
    (Friston 2017 EFE epistemic), so the same node-perturbation
    REINFORCE rule that drives reach learning is also active during
    babbling — weights are no longer frozen at zero RPE.

    The ``ou_tau`` / ``ou_sigma`` kwargs are accepted for backward
    compatibility and ignored.

    The run is split into chunks of ``target_refresh`` cycles.  Each
    chunk is a single XLA kernel launch; the mocap target is rotated
    between chunks in Python.
    """
    del ou_tau, ou_sigma
    k = key
    k, k_reset = split_key(k, 2)
    body, sample = body.reset(k_reset)

    sensory = sample.sensory
    prev_r = jnp.asarray(0.0, DTYPE)
    prev_d = sample.done
    motor_dim = brain_params.m1.motor_dim

    tips_chunks: list[Array] = []
    jcs_chunks: list[Array] = []

    remaining = int(n_cycles)
    chunk_size = int(target_refresh)

    while remaining > 0:
        this_chunk = min(chunk_size, remaining)
        k, k_chunk, k_tgt = split_key(k, 3)
        keys = jax.random.split(k_chunk, this_chunk)

        (brain_state, body, sensory, prev_r, prev_d), (jcs, tips) = (
            _babble_chunk(
                brain_state, body, sensory, prev_r, prev_d,
                brain_params, ctx, keys,
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
