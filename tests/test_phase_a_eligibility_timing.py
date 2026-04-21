"""Phase A — Actor/critic eligibility-trace COMMIT semantics (TD(0)).

Locks in the architectural invariant introduced by Phase A:
``e_d1_committed`` / ``e_d2_committed`` / ``action_mask_committed``
represent the correlation ``pre(s_t) × post(s_t)`` and the one-hot
of ``a_t`` SELECTED in the same cycle; they are only used by the
NEXT cycle's ``actor_update`` with the RPE δ_{t+1} = r + γV(s_{t+1}) − V(s_t),
which is standard TD(0) actor-critic credit assignment
(Sutton & Barto 2018 §13.5; Gertler 2008; Shen 2008).

Regression targets:
- ``actor_commit_eligibility`` must freeze the LIVE trace exactly
  (bit-identical copy of ``state.e_d1`` / ``state.e_d2``).
- The committed ``action_mask`` must be a hard one-hot over the
  motor slice (no soft ``down_state_factor`` leak) — MSN down-state
  has ZERO plasticity per Gertler 2008 / Shen 2008.
- ``critic_commit_eligibility`` must freeze ``e_h`` identically.
- ``actor_update`` with ``td=0`` must leave ``w_d1`` / ``w_d2``
  pointwise unchanged regardless of the live trace magnitude.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import BackendContext
from core.basal_ganglia import (
    init_actor_params, init_actor_state, actor_step,
    actor_commit_eligibility, actor_update, ActorInputs,
    init_critic_params, init_critic_state, critic_step,
    critic_commit_eligibility, critic_update,
)


# --- Fixture helpers ----------------------------------------------


def _build_actor(motor_dim=4, state_size=8, n_per_action=3, seed=0):
    ctx = BackendContext(dt=1.0)
    params = init_actor_params(
        ctx, state_size=state_size, motor_dim=motor_dim,
        n_per_action=n_per_action,
    )
    state = init_actor_state(jax.random.PRNGKey(seed), params)
    return ctx, params, state


def _drive_actor(ctx, params, state, n_substeps=6):
    """Run the actor under a constant sensory input so the live trace
    builds up to a nonzero, pre-post-correlated state."""
    pre = jnp.arange(params.state_size, dtype=jnp.float32) % 2.0
    for _ in range(n_substeps):
        out = actor_step(state, params, ctx, pre, ActorInputs())
        state = out.state
    return state


# =====================================================================
# Actor commit semantics
# =====================================================================


def test_actor_commit_freezes_live_trace_bit_identical():
    """``actor_commit_eligibility`` must snapshot ``e_d1`` / ``e_d2``
    exactly — one cycle of ``actor_step`` must not have drifted
    the committed copy."""
    ctx, params, state = _build_actor()
    state = _drive_actor(ctx, params, state)
    # At this point live e_d1/e_d2 are nonzero, committed still zero.
    assert jnp.linalg.norm(state.e_d1) > 0.0
    assert jnp.linalg.norm(state.e_d1_committed) == 0.0

    committed = actor_commit_eligibility(state, params, action=jnp.int32(1))

    # Bit-identical snapshot of the live trace.
    assert jnp.array_equal(committed.e_d1_committed, state.e_d1)
    assert jnp.array_equal(committed.e_d2_committed, state.e_d2)


def test_actor_commit_mask_is_hard_one_hot_over_motor_slice():
    """The committed mask must be 1.0 on the chosen action's MSN block,
    0.0 on all other motor blocks (MSN down-state gating, Gertler
    2008) — no soft floor."""
    ctx, params, _ = _build_actor(motor_dim=4, n_per_action=3)
    # Use a fresh state so the only thing we're testing is the mask.
    state = init_actor_state(jax.random.PRNGKey(0), params)
    chosen = jnp.int32(2)
    committed = actor_commit_eligibility(state, params, action=chosen)

    mask = committed.action_mask_committed
    assert mask.shape == (params.action_dim,)
    # Motor slice: rows belonging to action 2 are 1.0, others 0.0.
    npa, tm = params.n_per_action, params.total_motor
    motor_slice = mask[:tm].reshape(params.motor_dim, npa)
    expected = jnp.zeros_like(motor_slice).at[int(chosen), :].set(1.0)
    assert jnp.array_equal(motor_slice, expected)
    # Non-chosen motor rows have EXACTLY zero (no soft factor).
    for a in range(params.motor_dim):
        if a == int(chosen):
            continue
        assert jnp.all(motor_slice[a] == 0.0)
    # Internal slots (non-motor) are always enabled.
    assert jnp.all(mask[tm:] == 1.0)


def test_actor_update_gates_to_committed_action_only():
    """``actor_update`` must only change weight rows for the action
    channel committed at the previous cycle; all other motor rows
    remain pointwise frozen."""
    ctx, params, state = _build_actor(motor_dim=4, n_per_action=3)
    state = _drive_actor(ctx, params, state)
    chosen = jnp.int32(1)
    state = actor_commit_eligibility(state, params, action=chosen)
    # DA-dependent LTP of D1 on positive TD.
    updated = actor_update(state, params, td_error=jnp.float32(+0.5))

    dw_d1 = updated.w_d1 - state.w_d1
    npa = params.n_per_action
    tm = params.total_motor
    dw_motor = dw_d1[:, :tm].reshape(params.state_size, params.motor_dim, npa)
    # Chosen action's block CAN have nonzero updates.
    # All other motor blocks MUST be zero.
    for a in range(params.motor_dim):
        if a == int(chosen):
            continue
        assert jnp.all(dw_motor[:, a, :] == 0.0), (
            f"non-chosen motor block {a} got nonzero D1 update"
        )


def test_actor_update_with_zero_td_is_noop():
    """TD(0) with δ=0 must leave weights pointwise unchanged even
    when the live trace is large — committed-trace × 0 = 0."""
    ctx, params, state = _build_actor()
    state = _drive_actor(ctx, params, state, n_substeps=10)
    state = actor_commit_eligibility(state, params, action=jnp.int32(0))
    updated = actor_update(state, params, td_error=jnp.float32(0.0))

    assert jnp.array_equal(updated.w_d1, state.w_d1)
    assert jnp.array_equal(updated.w_d2, state.w_d2)


def test_actor_commit_is_idempotent_without_new_activity():
    """Running ``actor_commit_eligibility`` twice in a row (no new
    perception between) must yield the same committed snapshot —
    the live trace didn't change."""
    ctx, params, state = _build_actor()
    state = _drive_actor(ctx, params, state)
    first = actor_commit_eligibility(state, params, action=jnp.int32(3))
    second = actor_commit_eligibility(first, params, action=jnp.int32(3))
    assert jnp.array_equal(first.e_d1_committed, second.e_d1_committed)
    assert jnp.array_equal(first.e_d2_committed, second.e_d2_committed)
    assert jnp.array_equal(
        first.action_mask_committed, second.action_mask_committed
    )


# =====================================================================
# Critic commit semantics
# =====================================================================


def _build_critic(state_size=8, hidden_size=16, seed=0):
    ctx = BackendContext(dt=1.0)
    params = init_critic_params(ctx, state_size=state_size, hidden_size=hidden_size)
    state = init_critic_state(jax.random.PRNGKey(seed), params)
    return ctx, params, state


def _drive_critic(ctx, params, state, n_substeps=6):
    pre = jnp.arange(params.state_size, dtype=jnp.float32) % 2.0
    for _ in range(n_substeps):
        out = critic_step(state, params, ctx, pre)
        state = out.state
    return state


def test_critic_commit_freezes_live_trace_bit_identical():
    ctx, params, state = _build_critic()
    state = _drive_critic(ctx, params, state)
    assert jnp.linalg.norm(state.e_h) > 0.0
    assert jnp.linalg.norm(state.e_h_committed) == 0.0

    committed = critic_commit_eligibility(state)
    assert jnp.array_equal(committed.e_h_committed, state.e_h)


def test_critic_update_uses_committed_not_live_trace():
    """If we commit at time t, then drive the critic (which updates
    the LIVE trace), ``critic_update`` must still use the FROZEN
    trace — otherwise TD(0) semantics are violated (would credit
    s_{t+1} instead of s_t)."""
    ctx, params, state = _build_critic()
    state = _drive_critic(ctx, params, state)
    committed = critic_commit_eligibility(state)
    e_h_snapshot = committed.e_h_committed
    # Drive the live trace further — e_h changes, e_h_committed must not.
    driven = _drive_critic(ctx, params, committed, n_substeps=4)
    assert not jnp.array_equal(driven.e_h, e_h_snapshot)
    assert jnp.array_equal(driven.e_h_committed, e_h_snapshot)

    updated = critic_update(driven, params, td_error=jnp.float32(0.3))
    # Δw = lr · td · e_h_committed (not e_h live)
    expected_dw = params.critic_lr * jnp.float32(0.3) * e_h_snapshot
    dw_actual = updated.w_h - driven.w_h
    assert jnp.allclose(dw_actual, expected_dw, atol=1e-6)


def test_critic_update_with_zero_td_is_noop():
    ctx, params, state = _build_critic()
    state = _drive_critic(ctx, params, state)
    state = critic_commit_eligibility(state)
    updated = critic_update(state, params, td_error=jnp.float32(0.0))
    assert jnp.array_equal(updated.w_h, state.w_h)
