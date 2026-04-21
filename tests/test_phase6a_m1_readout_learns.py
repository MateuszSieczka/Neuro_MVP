"""Phase 6A — M1 motor_readout updates under RPE (three-factor Hebbian)."""
from __future__ import annotations

import jax.numpy as jnp

from core.backend import DEFAULT, make_key
from core.m1 import init_m1_params, init_m1_state, m1_step, m1_learn_readout


def test_phase6a_m1_readout_learns() -> None:
    n_l5, motor_dim = 32, 2
    p = init_m1_params(n_l5=n_l5, motor_dim=motor_dim, readout_lr=1e-2)
    s = init_m1_state(make_key(0), p)
    init_readout = s.motor_readout

    l5 = jnp.ones(n_l5) * 0.5

    # Apply a batch of positive RPEs — readout should drift in the
    # direction of the three-factor Hebbian update outer(l5_norm, jc).
    last_out = None
    for _ in range(50):
        out = m1_step(s, p, l5)
        last_out = out
        s = m1_learn_readout(
            out.state, p,
            rpe=jnp.asarray(1.0, jnp.float32),
            l5_rate_normalised=out.l5_rate_normalised,
            joint_command=out.joint_command,
        )

    drift = s.motor_readout - init_readout
    assert jnp.all(jnp.isfinite(s.motor_readout))
    assert float(jnp.linalg.norm(drift)) > 1e-3, (
        "motor_readout did not drift under sustained RPE"
    )
    # Three-factor Hebbian law: with +RPE the drift should align with
    # outer(l5_rate_normalised, joint_command).
    hebb = jnp.outer(last_out.l5_rate_normalised, last_out.joint_command)
    cos_pos = float(
        (drift * hebb).sum()
        / (jnp.linalg.norm(drift) * jnp.linalg.norm(hebb) + 1e-8)
    )
    assert cos_pos > 0.5, f"+RPE drift not aligned with Hebbian (cos={cos_pos})"

    # Negative RPE reverses the drift sign (sanity).
    s2 = init_m1_state(make_key(0), p)
    last_out2 = None
    for _ in range(50):
        out = m1_step(s2, p, l5)
        last_out2 = out
        s2 = m1_learn_readout(
            out.state, p,
            rpe=jnp.asarray(-1.0, jnp.float32),
            l5_rate_normalised=out.l5_rate_normalised,
            joint_command=out.joint_command,
        )
    drift_neg = s2.motor_readout - init_readout
    hebb_neg = jnp.outer(last_out2.l5_rate_normalised, last_out2.joint_command)
    cos_neg = float(
        (drift_neg * hebb_neg).sum()
        / (jnp.linalg.norm(drift_neg) * jnp.linalg.norm(hebb_neg) + 1e-8)
    )
    assert cos_neg < -0.5, f"-RPE drift not anti-aligned (cos={cos_neg})"
