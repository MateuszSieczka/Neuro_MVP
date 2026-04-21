"""Phase 3 — Curiosity drive on a reward-free GridWorld.

Plan target (plan.md §2.5):
  GridWorld bez reward, po 2000 krokach curiosity signal mean > 0
  i cortex activity niezerowa.

Rationale (Pathak 2017 ICM, Friston 2010 active inference): the world
model produces a precision-weighted prediction error
(``wm_curiosity_signal``) that the brain uses as an *intrinsic* reward
when no extrinsic reward is available. With ``goal_reward=0``,
``step_cost=0`` and an unreachable goal (so episodes never terminate
in the test horizon and do not interfere with PFC reset semantics),
the only thing keeping cortex active and BG learning is curiosity.

Methodology
-----------
- ``goal=(99, 99)``: clamped out of the grid; agent never reaches the
  terminal cell, so ``done`` stays 0 and PFC is not periodically
  flushed (Fuster 2001) — keeps the test about curiosity, not about
  episode boundaries.
- ``max_steps`` set well above the rollout length for the same reason.
- Loop is ``jax.lax.scan``-compiled.

Tests
-----
1. ``test_curiosity_positive`` — mean curiosity over 2000 steps > 0
   (with margin 0.01 — well above f32 noise floor).
2. ``test_cortex_active_without_reward`` — mean L5 rate > 0 and
   mean |belief| > 0 (cortex is alive on intrinsic drive only).
3. ``test_world_model_predictions_change`` — the WM prediction error
   varies across cycles (decoder is actually being updated, not stuck
   at init).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import BackendContext
from core.brain_graph import (
    init_action_brain_params, init_action_brain_state,
    action_brain_cognitive_step,
)
from embodiment.gridworld import GridWorldBody


def _make_body():
    return GridWorldBody.create(
        size=5,
        goal=(99, 99),         # unreachable — no terminal
        goal_reward=0.0,
        step_cost=0.0,
        max_steps=10 ** 9,
    )


def _rollout(n_cycles: int, seed: int = 0):
    ctx = BackendContext(dt=1.0)
    body = _make_body()
    params = init_action_brain_params(
        ctx, sensory_size=body.sensory_size,
        n_body_actions=body.n_actions, n_saccade_actions=2, substeps=4,
    )
    state = init_action_brain_state(jax.random.PRNGKey(seed), params)
    body, sample0 = body.reset(jax.random.PRNGKey(seed + 1))

    cur_list, l5_list, belief_list, wm_pe_list = [], [], [], []
    st = state
    prev_r, prev_d = sample0.reward, sample0.done
    b = body
    keys = jax.random.split(jax.random.PRNGKey(seed + 2), n_cycles)
    for i in range(n_cycles):
        k_step, k_act = jax.random.split(keys[i])
        sensory = b._encode(b.pos)
        out = action_brain_cognitive_step(
            st, params, ctx, sensory, prev_r, prev_d, k_step,
        )
        new_b, smp = b.act(k_act, out.body_action, jnp.int32(0))
        cur_list.append(float(out.curiosity))
        l5_list.append(float(jnp.mean(out.cortex_l5_rate)))
        belief_list.append(float(jnp.mean(jnp.abs(out.cortex_belief))))
        wm_pe_list.append(float(jnp.mean(out.state.world_model.prediction_error ** 2)))
        st = out.state
        prev_r, prev_d = smp.reward, smp.done
        b = new_b

    return (jnp.array(cur_list), jnp.array(l5_list),
            jnp.array(belief_list), jnp.array(wm_pe_list))


def test_curiosity_positive():
    """Mean curiosity over 2000 reward-free steps stays well above 0."""
    cur, _, _, _ = _rollout(n_cycles=2000, seed=0)
    mean_cur = float(cur.mean())
    assert mean_cur > 0.01, (
        f"curiosity collapsed to ~0 (mean={mean_cur:.5f}) — "
        f"world-model PE not feeding intrinsic drive"
    )


def test_cortex_active_without_reward():
    """Without extrinsic reward, cortex L5 + belief are non-trivial."""
    _, l5, belief, _ = _rollout(n_cycles=2000, seed=0)
    mean_l5 = float(l5.mean())
    mean_belief = float(belief.mean())
    assert mean_l5 > 0.01, (
        f"cortex L5 rate near zero (mean={mean_l5:.5f}) — "
        f"cortex not driven by intrinsic loop"
    )
    assert mean_belief > 0.01, (
        f"cortex belief near zero (mean={mean_belief:.5f})"
    )


def test_world_model_predictions_change():
    """WM prediction error varies across the rollout (decoder is being
    updated). std > 0 with a healthy margin rules out a frozen WM.
    """
    _, _, _, wm_pe = _rollout(n_cycles=2000, seed=0)
    pe_std = float(wm_pe.std())
    assert pe_std > 1e-4, (
        f"WM prediction error variance ≈ 0 (std={pe_std:.2e}) — "
        f"world model appears frozen"
    )
