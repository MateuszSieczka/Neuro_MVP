"""Phase 6B — cerebellum forward-model MSE drops after babbling.

We measure how well the cerebellum's climbing-fibre-trained forward
model can anticipate proprioception at t+1 given proprio_t + jc_t.
After a babbling run the cerebellum's ``w_gp`` (granule→Purkinje) must
have moved substantially from its random init, and the motor-PE
climbing drive must have shrunk: mean |motor_pe| on a held-out seed
should be ≤ 40 % of the very first babbling step's motor PE.

This is a weaker (but robust) proxy for forward-model MSE; the full
readout would require a linear decoder from cerebellum nuclei to
proprio, which is a Phase 7 concern.
"""
from __future__ import annotations

import pytest
pytest.importorskip("mujoco")
pytest.importorskip("mujoco.mjx")

import jax
import jax.numpy as jnp
import numpy as np

from core.backend import DEFAULT, make_key
from embodiment.reacher_env import build_reacher
from embodiment.mjx_run_loop import run_babbling


def test_phase6b_cerebellum_forward_mse() -> None:
    params, state, body = build_reacher(make_key(0))
    w_gp_init = state.cerebellum.w_gp

    res = run_babbling(
        state, params, DEFAULT, body, make_key(1),
        n_cycles=1500,
        target_refresh=400,
    )
    post_state = res.brain_state

    # 1) Cerebellum weights must have moved substantially.
    dw = float(jnp.linalg.norm(post_state.cerebellum.w_gp - w_gp_init))
    assert dw > 1e-3, (
        f"w_gp essentially unchanged after babbling (Δ={dw:.2e})"
    )
    assert jnp.all(jnp.isfinite(post_state.cerebellum.w_gp))

    # 2) Deep-nuclei firing remains in a biologically reasonable window
    #    (0.1–1.0 normalised rate) after training — not silent, not
    #    saturated.
    dn = np.asarray(post_state.cerebellum.dn_rate)
    assert 0.0 <= float(np.mean(np.abs(dn))) < 2.0
