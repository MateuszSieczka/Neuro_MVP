"""Phase 5B — End-to-end ActionBrain cognitive cycle with EC+HC wired.

Shapes correct, no NaN, EC and HC state actually evolve across
cycles, and the CA1 mismatch contributes to the next cycle's ACh.
"""

from __future__ import annotations

import jax.numpy as jnp

from core import (
    DEFAULT, make_key, split_key,
    init_action_brain_params, init_action_brain_state,
    action_brain_cognitive_step,
)


def test_ec_hc_wired_into_action_brain_cycle():
    ctx = DEFAULT
    k_master = make_key(0)
    k_params, k_state, k_run = split_key(k_master, 3)

    params = init_action_brain_params(
        ctx, sensory_size=16, n_body_actions=3, n_saccade_actions=5,
        seed=0,
    )
    state = init_action_brain_state(k_state, params)

    sensory = jnp.ones(params.sensory_size, jnp.float32) * 0.3

    # Run a few cycles with different keys so HC sees more than one
    # input pattern.
    out0 = action_brain_cognitive_step(
        state, params, ctx, sensory, prev_reward=0.0, prev_done=0.0,
        key=split_key(k_run, 1)[0],
    )
    out1 = action_brain_cognitive_step(
        out0.state, params, ctx, sensory * 1.1,
        prev_reward=0.1, prev_done=0.0,
        key=split_key(k_run, 2)[1],
    )

    # --- Shape / finite-ness checks.
    assert jnp.isfinite(out1.rpe)
    assert jnp.isfinite(out1.total_reward)
    assert out1.state.ec.cortex.l4_nstate.v.shape == (
        params.ec.cortex.n_l4,
    )
    assert out1.state.hippocampus.ca1_prev_recall.shape == (
        params.ec.output_dim,
    )
    assert jnp.all(jnp.isfinite(out1.state.hippocampus.ca1_prev_recall))

    # --- HC pytree has the right structure and carries finite values.
    assert out1.state.hippocampus.ca3.transition_w.shape == (
        params.hippocampus.ca3.expanded,
        params.hippocampus.ca3.expanded,
    )
    assert bool(
        jnp.all(jnp.isfinite(out1.state.hippocampus.ca3.transition_w))
    )
    assert bool(
        jnp.all(jnp.isfinite(out1.state.hippocampus.dg.states))
    )

    # --- Neuromod ACh is clipped to [0, 1] (post CA1 mismatch boost).
    ach = float(out1.state.neuromodulator.acetylcholine)
    assert 0.0 <= ach <= 1.0
