"""Phase 5B — SWS reverse-replay consolidation (Wilson & McNaughton 1994).

Build a tiny replay buffer with one consistent transition and then
run SWS replay repeatedly; the world-model's prediction error on
that transition must drop meaningfully.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from core import (
    DEFAULT, make_key, split_key,
    init_action_brain_params, init_action_brain_state,
    wm_predict, sws_replay_step,
)
from core.replay_buffer import replay_store, Experience


def _mse(a, b):
    return float(jnp.mean(jnp.square(a - b)))


def test_sws_replay_reduces_wm_prediction_error():
    ctx = DEFAULT
    k = make_key(0)
    k_params, k_state, k_replay, k_sws = split_key(k, 4)

    params = init_action_brain_params(
        ctx, sensory_size=16, n_body_actions=3, n_saccade_actions=5,
        seed=0, replay_capacity=64,
    )
    state = init_action_brain_state(k_state, params)

    action_size = params.n_body_actions + params.n_saccade_actions
    state_size = params.world_model.state_size

    # --- Synthesise one consistent (s, a, s') transition in WM's
    #     state-spike space.
    s = jnp.asarray(jr.bernoulli(split_key(k_replay, 2)[0],
                                 p=0.2, shape=(state_size,)), jnp.float32)
    s_next = jnp.asarray(jr.bernoulli(split_key(k_replay, 2)[1],
                                      p=0.2, shape=(state_size,)), jnp.float32)
    a_id = jnp.asarray(0, jnp.int32)
    a_oh = (jnp.arange(action_size) == a_id).astype(jnp.float32)

    # Push 16 copies of this transition into the replay buffer with
    # high salience so SWS sees exactly this pattern.
    replay = state.replay
    zero = jnp.asarray(0.0, jnp.float32)
    for _ in range(16):
        exp = Experience(
            state=s, action=a_id, reward=zero,
            next_state=s_next,
            prediction_error=zero,
            done=zero,
            salience=jnp.asarray(1.0, jnp.float32),
            recorded_da=zero,
        )
        replay = replay_store(replay, params.replay, exp)

    import equinox as eqx
    state = init_action_brain_state(k_state, params)
    state = eqx.tree_at(lambda s: s.replay, state, replay)

    # --- Baseline WM prediction error.
    pred0 = wm_predict(
        state.world_model, params.world_model, ctx, s, a_oh, ach=1.0,
    ).predicted_state
    mse0 = _mse(pred0, s_next)

    # --- Run several SWS passes (each pass iterates over replay).
    wm = state.world_model
    hc = state.hippocampus
    replay_cur = state.replay
    keys = jr.split(k_sws, 6)
    for i in range(6):
        wm, replay_cur, hc = sws_replay_step(
            wm, params.world_model, ctx,
            replay_cur, params.replay,
            hc, params.hippocampus,
            keys[i],
            n_replay=16,
            n_body_actions=params.n_body_actions,
            n_saccade_actions=params.n_saccade_actions,
            ach=1.0,
        )

    pred1 = wm_predict(
        wm, params.world_model, ctx, s, a_oh, ach=1.0,
    ).predicted_state
    mse1 = _mse(pred1, s_next)

    assert mse1 < mse0, (
        f"SWS replay did not reduce WM PE: mse0={mse0:.4f}, mse1={mse1:.4f}"
    )
    # Meaningful improvement (≥ 10% reduction).
    assert mse1 <= 0.9 * mse0, (
        f"SWS replay reduction too small: mse0={mse0:.4f}, mse1={mse1:.4f}"
    )
