"""Phase 5A — learning_pipeline API wiring.

The :mod:`core.learning_pipeline` module extracts the four wake
plasticity primitives (critic, body+saccade actors, cortex, attention)
into a single interface so that Phase 5B's replay learner can call
*exactly the same* entrypoints the wake cycle uses — there is no
duplicated plasticity logic that could drift apart.

This test locks the external API down:

* every ``*_learn_step`` function is importable from
  :mod:`core.learning_pipeline`;
* each function returns a pytree of the same shape as its input
  (no silent field drops);
* all output leaves are finite regardless of RPE sign — zero
  eligibility ⇒ zero weight delta is the correct behaviour and must
  not be perturbed by NaN leakage from the wrappers.

The *semantic* correctness of the wrappers (that wake learning still
converges on the bandit and gridworld tasks) is established by the
existing Phase 3/4 regression suite which now routes through the
pipeline.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from core.backend import BackendContext
from core.basal_ganglia import (
    init_actor_params, init_actor_state,
    init_critic_params, init_critic_state,
)
from core.cortex import (
    init_cortical_area_params, init_cortical_area_state,
)
from core.attention import init_attention_params, init_attention_state
from core.learning_pipeline import (
    critic_learn_step, actors_learn_step,
    cortex_learn_step, attention_learn_step,
)


def _ctx() -> BackendContext:
    return BackendContext(dt=1.0)


def _leaves_finite(pytree) -> bool:
    leaves = jax.tree_util.tree_leaves(pytree)
    return all(
        bool(jnp.isfinite(jnp.asarray(lf)).all())
        for lf in leaves
        if isinstance(lf, jnp.ndarray) or hasattr(lf, "dtype")
    )


def test_critic_learn_step_preserves_structure():
    ctx = _ctx()
    params = init_critic_params(ctx, state_size=8, hidden_size=16)
    state = init_critic_state(jax.random.PRNGKey(0), params)

    for rpe in (0.0, 0.5, -0.5):
        out = critic_learn_step(state, params, jnp.float32(rpe))
        assert eqx.tree_equal(
            jax.tree_util.tree_structure(out),
            jax.tree_util.tree_structure(state),
        ) is True
        assert _leaves_finite(out)


def test_actors_learn_step_returns_two_independent_states():
    ctx = _ctx()
    actor_params = init_actor_params(
        ctx, state_size=8, motor_dim=3, n_per_action=2,
    )
    body = init_actor_state(jax.random.PRNGKey(0), actor_params)
    saccade = init_actor_state(jax.random.PRNGKey(1), actor_params)

    new_body, new_saccade = actors_learn_step(
        body, actor_params, saccade, actor_params,
        rpe=jnp.float32(0.25),
        body_bonus=jnp.float32(0.1),
        saccade_bonus=jnp.float32(-0.1),
    )

    # Both actors have the same pytree shape as their inputs.
    assert eqx.tree_equal(
        jax.tree_util.tree_structure(new_body),
        jax.tree_util.tree_structure(body),
    ) is True
    assert eqx.tree_equal(
        jax.tree_util.tree_structure(new_saccade),
        jax.tree_util.tree_structure(saccade),
    ) is True
    assert _leaves_finite(new_body)
    assert _leaves_finite(new_saccade)


def test_cortex_learn_step_preserves_structure():
    ctx = _ctx()
    params = init_cortical_area_params(
        ctx, input_size=8,
        n_l4=8, n_l23_state=8, n_l23_error=8, n_l5=4,
    )
    state = init_cortical_area_state(jax.random.PRNGKey(0), params)

    out = cortex_learn_step(state, params, jnp.float32(0.1))
    assert eqx.tree_equal(
        jax.tree_util.tree_structure(out),
        jax.tree_util.tree_structure(state),
    ) is True
    assert _leaves_finite(out)


def test_attention_learn_step_preserves_structure():
    ctx = _ctx()
    params = init_attention_params(ctx)
    n_assoc, n_columns = 4, 6
    state = init_attention_state(
        jax.random.PRNGKey(0), n_assoc=n_assoc, n_columns=n_columns,
    )

    out = attention_learn_step(
        state, params,
        assoc_activity=jnp.ones((n_assoc,), jnp.float32),
        column_mean_rates=jnp.ones((n_columns,), jnp.float32) * 0.1,
        gains=jnp.ones((n_columns,), jnp.float32),
    )
    assert eqx.tree_equal(
        jax.tree_util.tree_structure(out),
        jax.tree_util.tree_structure(state),
    ) is True
    assert _leaves_finite(out)
