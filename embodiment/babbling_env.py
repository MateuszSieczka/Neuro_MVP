"""Phase 6B — motor-babbling wrapper around :class:`MjxArmBody`.

During babbling, we ignore extrinsic reward and feed the M1 head a
structured OU-process noise signal.  The brain's cerebellum learns a
forward model (proprio_{t+1} ≈ f(proprio_t, jc_t)) purely from its own
motor-PE climbing fibre (Phase 6A 2a.2 block).  This is the
developmental analogue of infant canonical babbling (Oller 1980,
Schaal & Sternad 2001 motor primitives).

Implementation note: the babbling env does **not** modify the brain;
it only supplies a noise input to M1 and swallows the extrinsic reward
so the BG actor learns only via curiosity (world-model PE).  The
caller of :func:`babbling_run` is responsible for zeroing the reward
signal before handing the sensory sample back to the brain.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import DTYPE, Array, PRNGKey
from .mjx_arm_body import MjxArmBody


def ou_babble_step(
    prev: Array, key: PRNGKey, *,
    tau: float = 20.0, sigma: float = 0.4,
) -> Array:
    """One step of an OU process bounded to ``[-1, 1]``.

    ``tau`` is in brain cycles (≈ 200 ms at 10 ms timestep; matches
    Schaal & Sternad 2001 primitive timescale).
    """
    alpha = jnp.exp(-1.0 / jnp.asarray(tau, DTYPE))
    noise = jax.random.normal(key, prev.shape, DTYPE) * sigma
    return jnp.clip(alpha * prev + (1.0 - alpha) * noise, -1.0, 1.0)


def reset_target_every(
    body: MjxArmBody, step: int, key: PRNGKey, *, every: int = 200,
) -> MjxArmBody:
    """Re-sample the mocap target periodically during babbling so the
    brain sees a drifting visual reference without episodic resets."""
    if (step % every) != 0 or step == 0:
        return body
    from .mjx_arm_body import _sample_target
    return body._set_target(_sample_target(key, body.cfg.workspace_half))
