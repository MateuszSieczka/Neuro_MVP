"""Motor babbling — self-supervised acquisition of the forward model.

The developmental stage that *precedes* goal-directed reaching (Oller 1980
canonical babbling; von Hofsten 2004): random motor exploration so the
motor→sensory forward model is learnt before active inference must invert it
(an untrained model makes the inferred command explode — babble first, reach
second).

Babbling and reaching share one cycle — *choose a motor belief → execute its
bounded command → learn the forward model from the reafference* — and differ
only in how the belief is chosen.  Here the belief is an Ornstein-Uhlenbeck
process in **belief space** (pre-``tanh``); ``tanh`` bounds it to the
actuator range at the body boundary, and the same pre-``tanh`` belief is the
forward-model input (:func:`core.pc_brain.pc_brain_learn_forward`).

Performance: many cycles run inside one ``jax.lax.scan`` under a single
``eqx.filter_jit`` (one XLA compile, one host→device handover); the rare
target refresh happens in Python between scan chunks.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from core.backend import DTYPE, Array, PRNGKey, split_key
from core.pc_brain import (
    PCBrainParams, PCBrainState, pc_brain_learn_forward,
)

from .mjx_arm_body import MjxArmBody, _sample_target


class BabbleConfig(NamedTuple):
    """Static babbling schedule + exploration hyper-parameters."""

    n_cycles: int = 30_000
    #: Target re-sampled every this many cycles (also the scan chunk size) so
    #: the brain sees a drifting reference without episodic resets.
    target_refresh: int = 400
    #: OU correlation time in brain cycles (≈ motor-primitive timescale,
    #: Schaal & Sternad 2001).
    tau: float = 20.0
    #: Steady-state std of the OU process in *belief* (pre-tanh) units; large
    #: enough that ``tanh`` spans most of the actuator range.
    sigma: float = 1.5
    #: Settling steps inside each forward-model update.  ``None`` ⇒ the graph's
    #: ``n_relax`` — the motor→cerebellum→sensory forward model has a hidden
    #: cerebellar cause that must settle like any latent (a 1-step settle
    #: leaves it unsettled and the model mis-fit).
    forward_settle_steps: int | None = None


def ou_babble_step(
    prev_belief: Array, key: PRNGKey, cfg: BabbleConfig,
) -> Array:
    """One Ornstein-Uhlenbeck step on the motor belief (unbounded).

    Exact-sampling update ``x_{t+1} = α·x_t + σ·√(1−α²)·ε`` with
    ``α = exp(−1/τ)`` — steady-state variance exactly ``σ²``.  Operates in
    belief space; the caller applies ``tanh`` to obtain the bounded command.
    """
    alpha = jnp.exp(-1.0 / jnp.asarray(cfg.tau, DTYPE))
    gain = jnp.asarray(cfg.sigma, DTYPE) * jnp.sqrt(1.0 - alpha * alpha)
    noise = jax.random.normal(key, prev_belief.shape, DTYPE) * gain
    return alpha * prev_belief + noise


class BabbleResult(NamedTuple):
    brain_state: PCBrainState
    body: MjxArmBody
    tip_traj: Array            # (n_cycles, 2) tip path
    command_traj: Array        # (n_cycles, motor_dim) executed commands


def _one_babble_cycle(carry, key, *, params: PCBrainParams, cfg: BabbleConfig):
    """Scan body — choose belief (OU) → execute tanh → learn forward model."""
    brain_state, body, prev_belief = carry
    k_body = key                         # body dynamics are deterministic; key unused

    belief = ou_babble_step(prev_belief, key, cfg)
    command = jnp.tanh(belief)
    body, sample = body.act(k_body, command)
    brain_state = pc_brain_learn_forward(
        brain_state, params, belief, sample.sensory,
        n_relax=cfg.forward_settle_steps,
    )
    return (brain_state, body, belief), (body.tip_xy(), command)


def run_babbling(
    brain_state: PCBrainState,
    params: PCBrainParams,
    body: MjxArmBody,
    key: PRNGKey,
    cfg: BabbleConfig = BabbleConfig(),
) -> BabbleResult:
    """Run ``cfg.n_cycles`` of motor babbling, chunked by target refresh."""

    @eqx.filter_jit
    def _chunk(brain_state, body, prev_belief, keys):
        step = lambda c, k: _one_babble_cycle(c, k, params=params, cfg=cfg)
        return jax.lax.scan(step, (brain_state, body, prev_belief), keys)

    k, k_reset = split_key(key, 2)
    body, _ = body.reset(k_reset)
    prev_belief = jnp.zeros(body.motor_dim, DTYPE)

    tip_chunks: list[Array] = []
    cmd_chunks: list[Array] = []
    remaining = int(cfg.n_cycles)
    while remaining > 0:
        this_chunk = min(int(cfg.target_refresh), remaining)
        k, k_chunk, k_tgt = split_key(k, 3)
        keys = jax.random.split(k_chunk, this_chunk)
        (brain_state, body, prev_belief), (tips, cmds) = _chunk(
            brain_state, body, prev_belief, keys,
        )
        tip_chunks.append(tips)
        cmd_chunks.append(cmds)
        body = body._set_target(_sample_target(k_tgt, body.cfg))
        remaining -= this_chunk

    return BabbleResult(
        brain_state=brain_state,
        body=body,
        tip_traj=jnp.concatenate(tip_chunks, axis=0),
        command_traj=jnp.concatenate(cmd_chunks, axis=0),
    )
