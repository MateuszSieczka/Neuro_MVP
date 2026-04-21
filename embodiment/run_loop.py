"""Stateless embodied-loop driver for ActionBrain + BodyInterface.

This is the analogue of a traditional RL agent's ``train_loop``, but it
is only a *driver*: it does not own any learning logic, hyperparameters,
or reward shaping. The brain does the learning, the body owns the
physics; the loop just connects them one decision cycle at a time.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from core.backend import DTYPE, Array, PRNGKey, split_key, BackendContext
from core.brain_graph import (
    ActionBrainParams, ActionBrainState, action_brain_step,
)

from .body_interface import BodyInterface, SensorySample


class EpisodeResult(NamedTuple):
    """Per-step history of one episode."""

    body_actions: Array       # (T,) int32
    saccade_actions: Array    # (T,) int32
    rewards: Array            # (T,) float32 — extrinsic
    total_rewards: Array      # (T,) float32 — incl. intrinsic drives
    rpes: Array               # (T,) float32
    curiosities: Array        # (T,) float32
    dones: Array              # (T,) float32
    steps_taken: int          # actual number of body transitions
    # Final carry
    brain_state: ActionBrainState
    body: BodyInterface


def run_episode(
    brain_state: ActionBrainState,
    brain_params: ActionBrainParams,
    ctx: BackendContext,
    body: BodyInterface,
    key: PRNGKey,
    *,
    max_steps: int = 200,
    reset_body: bool = True,
) -> EpisodeResult:
    """Run one episode; return trajectories and the final brain state.

    The episode stops early on the body's ``done`` flag. No JIT here —
    ``action_brain_step`` is heavy enough that per-cycle Python overhead
    is negligible, and bodies are ordinary Python objects; this keeps
    the loop trivially debuggable.
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

    body_actions = []
    saccade_actions = []
    rewards = []
    total_rewards = []
    rpes = []
    curiosities = []
    dones = []

    prev_reward = jnp.asarray(0.0, DTYPE)
    prev_done = jnp.asarray(0.0, DTYPE)

    for t in range(max_steps):
        k, k_sel, k_act = split_key(k, 3)
        out = action_brain_step(
            brain_state, brain_params, ctx,
            sample.info["image"],
            sample.info["fixation_xy"],
            prev_reward, prev_done, k_sel,
        )
        brain_state = out.state

        body, sample = body.act(k_act, out.body_action, out.saccade_action)

        body_actions.append(out.body_action)
        saccade_actions.append(out.saccade_action)
        rewards.append(sample.reward)
        total_rewards.append(out.total_reward)
        rpes.append(out.rpe)
        curiosities.append(out.curiosity)
        dones.append(sample.done)

        prev_reward = sample.reward
        prev_done = sample.done

        if bool(sample.done):
            break

    T = len(body_actions)
    return EpisodeResult(
        body_actions=jnp.asarray(body_actions, jnp.int32),
        saccade_actions=jnp.asarray(saccade_actions, jnp.int32),
        rewards=jnp.asarray(rewards, DTYPE),
        total_rewards=jnp.asarray(total_rewards, DTYPE),
        rpes=jnp.asarray(rpes, DTYPE),
        curiosities=jnp.asarray(curiosities, DTYPE),
        dones=jnp.asarray(dones, DTYPE),
        steps_taken=T,
        brain_state=brain_state,
        body=body,
    )
