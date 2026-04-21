"""Phase 6A — proprioceptive Gaussian population code tracks joint angle.

For each joint, the argmax of the per-joint block of ``proprio_encode``
should move monotonically as the true angle sweeps its range
(Pouget & Sejnowski 1997).
"""
from __future__ import annotations

import jax.numpy as jnp

from sensory.proprioception import (
    init_proprioception_params, proprio_encode, proprio_output_dim,
)


def test_phase6a_proprio_encoding() -> None:
    n_joints, n_cells = 2, 16
    p = init_proprioception_params(
        n_joints=n_joints, n_cells_per_joint=n_cells,
        angle_range=(-1.0, 1.0), velocity_range=(-1.0, 1.0),
    )
    assert proprio_output_dim(p) == n_joints * 2 * n_cells

    # Sweep angle of joint 0, hold joint 1 fixed, zero velocities.
    sweep = jnp.linspace(-0.9, 0.9, 9)
    vels = jnp.zeros(n_joints)
    argmaxes = []
    for a in sweep:
        angles = jnp.asarray([float(a), 0.0], jnp.float32)
        enc = proprio_encode(p, angles, vels)
        assert enc.shape == (proprio_output_dim(p),)
        # Per-joint angle block: first n_cells entries of joint 0.
        block0 = enc[:n_cells]
        argmaxes.append(int(jnp.argmax(block0)))

    # Argmax is monotonically non-decreasing in the true angle.
    for a, b in zip(argmaxes[:-1], argmaxes[1:]):
        assert b >= a, f"non-monotonic population code: {argmaxes}"
    # And it actually moves.
    assert argmaxes[-1] > argmaxes[0]

    # Output finite and non-negative (gaussian bumps).
    enc = proprio_encode(
        p, jnp.asarray([0.2, -0.3]), jnp.asarray([0.1, -0.1])
    )
    assert jnp.all(jnp.isfinite(enc))
    assert float(jnp.min(enc)) >= 0.0
    assert float(jnp.max(enc)) > 0.0
