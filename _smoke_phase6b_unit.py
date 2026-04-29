"""Phase 6B fix smoke test (no mujoco required).

Verifies:

1. M1 ``m1_step`` injects non-zero exploration noise.
2. M1 ``m1_learn_readout`` produces non-zero dw given non-zero RPE.
3. ``m1_learn_readout`` produces *zero* dw when rpe = 0 (sanity check
   that weights are not drifting from numerical noise).
4. With persistent positive RPE on a synthetic L5 + noise pattern,
   the readout drifts in the direction predicted by REINFORCE
   (cov(\u03be, jc gain) > 0 along that channel).

Run:
    python _smoke_phase6b_unit.py
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from core.m1 import (
    M1State, init_m1_params, init_m1_state, m1_step, m1_learn_readout,
)


def main() -> None:
    n_l5, motor_dim = 32, 2
    params = init_m1_params(
        n_l5=n_l5, motor_dim=motor_dim,
        readout_lr=1e-2,           # large for visible movement
        sigma_base=0.2,
    )
    state = init_m1_state(jax.random.PRNGKey(0), params)

    print("[1] Step injects non-zero exploration noise")
    l5 = jnp.ones((n_l5,), jnp.float32) * 0.5
    out = m1_step(
        state, params, l5,
        key=jax.random.PRNGKey(42),
        ne_level=jnp.asarray(1.0, jnp.float32),
    )
    xi = np.asarray(out.exploration_noise)
    print(f"      \u03be = {xi}  |\u03be|_inf = {np.max(np.abs(xi)):.4f}")
    assert np.max(np.abs(xi)) > 1e-3, "noise is zero"

    print("[2] Learning rule produces non-zero dw under non-zero RPE")
    s_after_step = out.state
    s_after_learn = m1_learn_readout(
        s_after_step, params,
        rpe=jnp.asarray(1.0, jnp.float32),
    )
    dw = np.asarray(s_after_learn.motor_readout - s_after_step.motor_readout)
    dw_norm = float(np.linalg.norm(dw))
    print(f"      |\u0394w|_F = {dw_norm:.6f}")
    assert dw_norm > 1e-6, "dw is zero with rpe=1"

    print("[3] Learning rule is no-op when RPE = 0")
    s_zero = m1_learn_readout(
        s_after_step, params,
        rpe=jnp.asarray(0.0, jnp.float32),
    )
    dw0 = np.asarray(s_zero.motor_readout - s_after_step.motor_readout)
    dw0_norm = float(np.linalg.norm(dw0))
    print(f"      |\u0394w|_F = {dw0_norm:.2e}  (should be exactly 0)")
    assert dw0_norm == 0.0, "dw nonzero under rpe=0"

    print("[4] REINFORCE drift: with persistent positive RPE, readout "
          "follows noise direction")
    # Strategy: reset state; iteratively step + learn with RPE that is
    # POSITIVELY correlated with channel-0 of \u03be.  After many cycles,
    # mean(motor_readout column 0) should grow.
    state = init_m1_state(jax.random.PRNGKey(0), params)
    rng = np.random.default_rng(1)
    drift_keys = jax.random.split(jax.random.PRNGKey(7), 500)
    w_norms = []
    for k in drift_keys:
        l5_v = jnp.asarray(rng.uniform(0.0, 1.0, n_l5).astype(np.float32))
        out = m1_step(
            state, params, l5_v,
            key=k, ne_level=jnp.asarray(1.0, jnp.float32),
        )
        # Reward: +1 when ξ[0] > 0, -1 when ξ[0] < 0 → channel-0
        # weights should grow (REINFORCE ∇ = E[ξ · R | s] > 0).
        rpe = jnp.where(out.exploration_noise[0] > 0, 1.0, -1.0)
        state = m1_learn_readout(out.state, params, rpe=rpe)
        w_norms.append(float(np.linalg.norm(state.motor_readout)))

    w = np.asarray(state.motor_readout)
    col0_mean = float(np.mean(w[:, 0]))
    col1_mean = float(np.mean(w[:, 1]))
    print(f"      col0 mean (driven)   = {col0_mean:+.4f}  "
          "(expect > col1)")
    print(f"      col1 mean (control)  = {col1_mean:+.4f}")
    print(f"      |w| start \u2192 end       = {w_norms[0]:.4f} \u2192 "
          f"{w_norms[-1]:.4f}")
    assert col0_mean > col1_mean + 0.01, (
        f"REINFORCE failed to differentiate channels: "
        f"col0={col0_mean:+.4f} vs col1={col1_mean:+.4f}"
    )

    print()
    print("ALL UNIT-LEVEL SMOKE CHECKS PASSED \u2713")


if __name__ == "__main__":
    main()
