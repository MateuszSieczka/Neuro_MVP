"""Proprioception encoder — joint angles + velocities → cortical afferent.

Phase 6A design
---------------
A minimal population-coded kinematic encoder.  For each of the
``n_joints`` DoFs we lay out ``n_cells_per_joint`` Gaussian tuning
curves uniformly over ``angle_range`` (angle channels) and
``velocity_range`` (velocity channels).  The final vector is the
concatenation ``[angle_pop, velocity_pop]`` per joint, flattened.

This is the standard Pouget & Sejnowski (1997) / Georgopoulos (1986)
muscle-spindle + tendon-organ readout used throughout systems motor
neuroscience.  The encoder is **body-agnostic**: Phase 6A feeds it
synthetic pseudo-joint signals derived from (last body action, position
delta); Phase 6B swaps in real MJX joint states without touching the
consumer wiring.

References
----------
- Georgopoulos (1986) *Science* 233: 1416-1419 — directional motor
  population code.
- Pouget & Sejnowski (1997) *Cereb. Cortex* 7: 222-237 — spatial
  transformations via population coding.
- Scott (2004) *Nat. Rev. Neurosci.* 5: 532-546 — sensorimotor
  integration.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from core.backend import DTYPE, Array


class ProprioceptionParams(eqx.Module):
    """Static proprioception encoder configuration."""

    angle_centers: Array     # (n_joints, n_cells_per_joint) preferred angles
    velocity_centers: Array  # (n_joints, n_cells_per_joint) preferred velocities
    angle_sigma: Array       # scalar
    velocity_sigma: Array    # scalar

    n_joints: int = eqx.field(static=True)
    n_cells_per_joint: int = eqx.field(static=True)


class ProprioState(NamedTuple):
    """Last-cycle joint kinematics, for forward-model PE estimation."""

    angles: Array      # (n_joints,)
    velocities: Array  # (n_joints,)


def init_proprioception_params(
    *,
    n_joints: int,
    n_cells_per_joint: int = 16,
    angle_range: tuple[float, float] = (-1.0, 1.0),
    velocity_range: tuple[float, float] = (-1.0, 1.0),
    angle_sigma: float | None = None,
    velocity_sigma: float | None = None,
) -> ProprioceptionParams:
    """Build a population-coding encoder.

    ``angle_range`` defaults to [-1, 1] (canonical normalised joint
    units; Phase 6B's MJX wrapper will divide by joint limits).
    Default ``sigma`` = half the inter-centre spacing (Pouget-Dayan-Zemel
    2000 half-overlap convention).
    """
    a_lo, a_hi = angle_range
    v_lo, v_hi = velocity_range
    centers_angle = jnp.linspace(a_lo, a_hi, n_cells_per_joint, dtype=DTYPE)
    centers_vel = jnp.linspace(v_lo, v_hi, n_cells_per_joint, dtype=DTYPE)

    def _default_sigma(lo: float, hi: float) -> float:
        return (hi - lo) / max(1, n_cells_per_joint - 1)

    a_sigma = _default_sigma(a_lo, a_hi) if angle_sigma is None else float(angle_sigma)
    v_sigma = _default_sigma(v_lo, v_hi) if velocity_sigma is None else float(velocity_sigma)

    # Broadcast to (n_joints, n_cells): same template on all joints
    angle_c = jnp.broadcast_to(centers_angle[None, :], (n_joints, n_cells_per_joint))
    velocity_c = jnp.broadcast_to(centers_vel[None, :], (n_joints, n_cells_per_joint))
    return ProprioceptionParams(
        angle_centers=angle_c.astype(DTYPE),
        velocity_centers=velocity_c.astype(DTYPE),
        angle_sigma=jnp.asarray(a_sigma, DTYPE),
        velocity_sigma=jnp.asarray(v_sigma, DTYPE),
        n_joints=int(n_joints),
        n_cells_per_joint=int(n_cells_per_joint),
    )


def init_proprio_state(n_joints: int) -> ProprioState:
    z = jnp.zeros((int(n_joints),), DTYPE)
    return ProprioState(angles=z, velocities=z)


def proprio_encode(
    params: ProprioceptionParams,
    angles: Array,
    velocities: Array,
) -> Array:
    """Gaussian population code.

    Parameters
    ----------
    angles : (n_joints,)
    velocities : (n_joints,)

    Returns
    -------
    Array of shape ``(n_joints * 2 * n_cells_per_joint,)`` — per-joint
    ``[angle_pop, velocity_pop]`` concatenated, then flattened across
    joints.  Values live in ``[0, 1]`` (Gaussian peak 1.0).
    """
    a = angles.astype(DTYPE)[:, None]             # (n_joints, 1)
    v = velocities.astype(DTYPE)[:, None]
    za = (a - params.angle_centers) / params.angle_sigma
    zv = (v - params.velocity_centers) / params.velocity_sigma
    angle_pop = jnp.exp(-0.5 * za * za)           # (n_joints, n_cells)
    velocity_pop = jnp.exp(-0.5 * zv * zv)
    per_joint = jnp.concatenate([angle_pop, velocity_pop], axis=1)
    return per_joint.reshape(-1).astype(DTYPE)


def proprio_output_dim(params: ProprioceptionParams) -> int:
    """Flattened encoding size."""
    return int(params.n_joints * 2 * params.n_cells_per_joint)
