"""Proprioception encoder — joint angles + velocities → a rate vector.

A population-coded kinematic encoder: for each of ``n_joints`` DoFs we lay
``n_cells_per_joint`` Gaussian tuning curves uniformly over ``angle_range``
(angle channels) and ``velocity_range`` (velocity channels).  The output is
the per-joint ``[angle_pop, velocity_pop]`` concatenation, flattened — the
flat ``[0, 1]`` rate vector that clamps the substrate's sensory node.

This is the standard Pouget & Sejnowski (1997) / Georgopoulos (1986)
muscle-spindle + tendon-organ readout used throughout systems motor
neuroscience.  The encoder is **body-agnostic**: any plant supplying joint
angles + velocities feeds it, the consumer never sees raw kinematics.

References
----------
- Georgopoulos (1986) *Science* 233: 1416-1419 — directional motor
  population code.
- Pouget & Sejnowski (1997) *Cereb. Cortex* 7: 222-237 — spatial
  transformations via population coding.
- Pouget, Dayan & Zemel (2000) — half-overlap population-code convention.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from core.backend import DTYPE, Array

#: Velocity tuning spans this multiple of the angle range by default — joint
#: speeds reach a few times the positional range over a control cycle, so the
#: velocity channels need a proportionally wider span to stay unsaturated.
DEFAULT_VELOCITY_RANGE_FACTOR = 4.0


class ProprioceptionParams(eqx.Module):
    """Static proprioception encoder configuration."""

    angle_centers: Array     # (n_joints, n_cells_per_joint) preferred angles
    velocity_centers: Array  # (n_joints, n_cells_per_joint) preferred velocities
    angle_sigma: Array       # scalar
    velocity_sigma: Array    # scalar

    n_joints: int = eqx.field(static=True)
    n_cells_per_joint: int = eqx.field(static=True)


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

    ``angle_range`` defaults to the canonical normalised joint units
    ``[-1, 1]`` (a body in physical units divides by its joint limits
    first).  Default ``sigma`` is half the inter-centre spacing
    (Pouget-Dayan-Zemel 2000 half-overlap convention).
    """
    a_lo, a_hi = angle_range
    v_lo, v_hi = velocity_range
    centers_angle = jnp.linspace(a_lo, a_hi, n_cells_per_joint, dtype=DTYPE)
    centers_vel = jnp.linspace(v_lo, v_hi, n_cells_per_joint, dtype=DTYPE)

    def _default_sigma(lo: float, hi: float) -> float:
        return (hi - lo) / max(1, n_cells_per_joint - 1)

    a_sigma = _default_sigma(a_lo, a_hi) if angle_sigma is None else float(angle_sigma)
    v_sigma = _default_sigma(v_lo, v_hi) if velocity_sigma is None else float(velocity_sigma)

    # Same tuning template on every joint → broadcast to (n_joints, n_cells).
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


def proprio_encode(
    params: ProprioceptionParams,
    angles: Array,
    velocities: Array,
) -> Array:
    """Gaussian population code of ``(angles, velocities)``.

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
    a = jnp.asarray(angles, DTYPE)[:, None]             # (n_joints, 1)
    v = jnp.asarray(velocities, DTYPE)[:, None]
    za = (a - params.angle_centers) / params.angle_sigma
    zv = (v - params.velocity_centers) / params.velocity_sigma
    angle_pop = jnp.exp(-0.5 * za * za)                 # (n_joints, n_cells)
    velocity_pop = jnp.exp(-0.5 * zv * zv)
    per_joint = jnp.concatenate([angle_pop, velocity_pop], axis=1)
    return per_joint.reshape(-1).astype(DTYPE)


def proprio_output_dim(params: ProprioceptionParams) -> int:
    """Flattened encoding size."""
    return int(params.n_joints * 2 * params.n_cells_per_joint)
