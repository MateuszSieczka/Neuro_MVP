"""Phase 5B — REM forward rollout (Hobson & McCarley 1977;
Hasselmo 2006 ACh-high/DA-low regime).

A REM rollout must not crash on an empty-ish replay buffer and must
preserve shape invariants of WM + HC state.  Stronger convergence
guarantees would require a proper generative test-bed; here we
check correctness of the offline pipeline.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from core import (
    DEFAULT, make_key, split_key,
    init_action_brain_params, init_action_brain_state,
    rem_rollout_step,
)
from core.replay_buffer import replay_store, Experience


def test_rem_rollout_preserves_shapes_and_advances_state():
    ctx = DEFAULT
    k = make_key(0)
    k_state, k_seed, k_rem = split_key(k, 3)

    params = init_action_brain_params(
        ctx, sensory_size=12, n_body_actions=2, n_saccade_actions=3,
        seed=0, replay_capacity=32,
    )
    state = init_action_brain_state(k_state, params)

    action_size = params.n_body_actions + params.n_saccade_actions
    state_size = params.world_model.state_size

    # Seed one experience so the REM sampler has something to pull.
    s = jnp.asarray(jr.bernoulli(split_key(k_seed, 2)[0],
                                 p=0.2, shape=(state_size,)), jnp.float32)
    s_next = jnp.asarray(jr.bernoulli(split_key(k_seed, 2)[1],
                                      p=0.2, shape=(state_size,)), jnp.float32)
    zero = jnp.asarray(0.0, jnp.float32)
    exp = Experience(
        state=s, action=jnp.asarray(0, jnp.int32), reward=zero,
        next_state=s_next,
        prediction_error=zero,
        done=zero, salience=jnp.asarray(1.0, jnp.float32),
        recorded_da=zero,
    )
    replay = replay_store(state.replay, params.replay, exp)

    wm_new, hc_new = rem_rollout_step(
        state.world_model, params.world_model, ctx,
        replay, params.replay,
        state.hippocampus, params.hippocampus,
        k_rem,
        k_steps=8,
        n_body_actions=params.n_body_actions,
        n_saccade_actions=params.n_saccade_actions,
        ach=1.0,
    )

    # Shapes preserved.
    assert wm_new.w_decode.shape == state.world_model.w_decode.shape
    assert hc_new.ca3.transition_w.shape == state.hippocampus.ca3.transition_w.shape

    # Finite.
    assert bool(jnp.all(jnp.isfinite(wm_new.w_decode)))
    assert bool(jnp.all(jnp.isfinite(hc_new.ca3.transition_w)))

    # WM state actually advanced — encoder rate trace mutates every
    # step the rollout drives.
    r0 = state.world_model.encoder.state_rate
    r1 = wm_new.encoder.state_rate
    assert float(jnp.abs(r1 - r0).sum()) > 0.0
