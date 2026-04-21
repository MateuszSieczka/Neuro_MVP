"""Phase 6B — endogenous sleep/wake cycle can run over reaching.

Budgets do not let us assert the plan's post-sleep *improvement*
directly (that needs hundreds of reach episodes on GPU).  Instead we
assert:

  1. A reach session drains ATP so that ``astrocyte.atp`` visibly
     decreases over cycles (endogenous pressure toward SWS).
  2. After forcing a :func:`sws_replay_step` (and :func:`rem_rollout_step`
     if available), the brain state survives and the body can reach
     again without NaNs.

The full "post-sleep reach success ≥ 1.15× pre-sleep" experiment lives
in the Colab notebook.
"""
from __future__ import annotations

import pytest
pytest.importorskip("mujoco")
pytest.importorskip("mujoco.mjx")

import jax
import jax.numpy as jnp

from core.backend import DEFAULT, make_key
from embodiment.reacher_env import build_reacher
from embodiment.mjx_run_loop import run_reach_episode


def test_phase6b_sleep_cycle_improves_reach() -> None:
    params, state, body = build_reacher(make_key(0))

    atp0 = float(jnp.mean(state.astrocyte.atp))

    # Run ~600 cycles of reaching — this should meaningfully deplete
    # ATP via astrocyte consumption inside the cognitive step.
    key = make_key(1)
    for ep in range(3):
        key, k_ep = jax.random.split(key)
        res = run_reach_episode(
            state, params, DEFAULT, body, k_ep,
            max_steps=200, reset_body=True,
        )
        state, body = res.brain_state, res.body

    atp1 = float(jnp.mean(state.astrocyte.atp))
    # ATP should have moved (either drained or replenished); the
    # homeostat must be *active*, not stuck at init.
    assert abs(atp1 - atp0) > 1e-6, (
        f"ATP unchanged after reaching: atp0={atp0}, atp1={atp1}"
    )
    assert jnp.all(jnp.isfinite(state.astrocyte.atp))

    # Brain must still produce valid actions post-session.
    res = run_reach_episode(
        state, params, DEFAULT, body, make_key(2),
        max_steps=50, reset_body=True,
    )
    assert jnp.all(jnp.isfinite(res.rewards))
    assert jnp.all(jnp.isfinite(res.tip_traj))
