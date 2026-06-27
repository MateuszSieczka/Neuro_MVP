"""Phase 6A/6B — M1 motor_readout learns by node-perturbation REINFORCE.

The readout update is the Gaussian-policy score correlated with reward
(Williams 1992; Fiete & Seung 2006):

    Δw_{ij} = lr · (R − b) · l5_i · ξ_j / σ²

A *constant* reward therefore produces NO net drift (E[ξ] = 0) — that is
the whole point of node perturbation versus the old ``outer(l5, jc)``
self-reinforcing rule.  To see directed learning the reward must depend
on the perturbation.  Here we reward exploration that pushed motor
channel 0 positive (``R = ξ₀``); the readout for channel 0 must then
grow so the deterministic command μ₀ = wᵀ·l5 drifts positive, while the
unrewarded channel 1 stays put.  The sign of the reward flips the drift.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import make_key
from core.m1 import init_m1_params, init_m1_state, m1_step, m1_learn_readout


def _run(sign: float):
    n_l5, motor_dim = 16, 2
    p = init_m1_params(n_l5=n_l5, motor_dim=motor_dim, readout_lr=5e-2)
    s = init_m1_state(make_key(0), p)
    l5 = jnp.ones(n_l5) * 0.5
    ne = jnp.asarray(0.5)
    w0 = s.motor_readout
    key = make_key(1)
    for _ in range(400):
        key, k = jax.random.split(key)
        out = m1_step(s, p, l5, key=k, ne_level=ne)
        # Reward = how much this step explored channel 0 (signed).
        rpe = sign * out.exploration_noise[0]
        s = m1_learn_readout(out.state, p, rpe=rpe)
    return w0, s.motor_readout, l5


def test_phase6a_m1_readout_learns() -> None:
    w0, w_pos, l5 = _run(+1.0)
    assert jnp.all(jnp.isfinite(w_pos))

    # Deterministic command on channel 0 must rise (we rewarded +ξ₀);
    # channel 1 was never rewarded, so it should barely move.
    mu0_before = float(w0[:, 0] @ l5)
    mu0_after = float(w_pos[:, 0] @ l5)
    d0 = mu0_after - mu0_before
    d1 = abs(float(w_pos[:, 1] @ l5) - float(w0[:, 1] @ l5))
    assert d0 > 1e-2, f"channel-0 command did not rise under +ξ₀ reward (Δ={d0})"
    assert d0 > d1, "rewarded channel did not move more than the unrewarded one"

    # Flipping the reward sign reverses the drift.
    _, w_neg, _ = _run(-1.0)
    d0_neg = float(w_neg[:, 0] @ l5) - mu0_before
    assert d0_neg < -1e-2, f"channel-0 command did not fall under −ξ₀ reward (Δ={d0_neg})"
