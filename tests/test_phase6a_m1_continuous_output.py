"""Phase 6A — M1 continuous head emits bounded, shape-correct commands."""
from __future__ import annotations

import jax.numpy as jnp

from core.backend import make_key
from core.m1 import init_m1_params, init_m1_state, m1_step


def test_phase6a_m1_continuous_output() -> None:
    n_l5, motor_dim = 32, 3
    p = init_m1_params(n_l5=n_l5, motor_dim=motor_dim)
    s = init_m1_state(make_key(0), p)
    assert s.motor_readout.shape == (n_l5, motor_dim)

    l5 = jnp.ones(n_l5) * 0.8
    ne = jnp.asarray(0.5)
    out = m1_step(s, p, l5, key=make_key(1), ne_level=ne)
    jc = out.joint_command
    assert jc.shape == (motor_dim,)
    assert jnp.all(jnp.isfinite(jc))
    # tanh-bounded
    assert float(jnp.max(jc)) <= 1.0
    assert float(jnp.min(jc)) >= -1.0

    # Responds to different L5 patterns (same key → only L5 differs).
    l5b = jnp.zeros(n_l5).at[0].set(1.0)
    out_b = m1_step(s, p, l5b, key=make_key(1), ne_level=ne)
    assert float(jnp.linalg.norm(jc - out_b.joint_command)) > 1e-6

    # σ is cached for the Gaussian-policy score (Williams 1992); higher
    # NE → wider exploration, and σ is always strictly positive.
    out_lo = m1_step(s, p, l5, key=make_key(2), ne_level=jnp.asarray(0.0))
    out_hi = m1_step(s, p, l5, key=make_key(2), ne_level=jnp.asarray(1.0))
    assert float(out_lo.state.last_sigma) > 0.0
    assert float(out_hi.state.last_sigma) > float(out_lo.state.last_sigma)
