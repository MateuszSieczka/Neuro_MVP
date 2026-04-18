"""P0.10 \u2014 Cerebellum efference copy (Wolpert 1998).

Test that motor commands (body \u2295 saccade, one-hot) modulate the mossy
drive into the cerebellum, so the forward model sees what the brain
*commanded* rather than inferring it post-hoc from cortex L5.

Strategy
--------
Run action_brain_step for several cycles; compare the delay_mossy buffer
between two runs that differ only in the identity of the last body
action (via `state.last_body_action` being different one-hots). If the
efference-copy wiring is effective, delay_mossy should diverge.

We bypass stochasticity by seeding everything identically and overriding
the brain state with explicit one-hots before the step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import equinox as eqx

from core.backend import BackendContext
from core.brain_graph import (
    init_action_brain_params, init_action_brain_state, action_brain_step,
)
from embodiment.bandit import GaussianBanditBody


def _run_one_step(body_onehot, saccade_onehot):
    ctx = BackendContext(dt=1.0)
    body = GaussianBanditBody.create(jax.random.PRNGKey(0), n_actions=3)
    params = init_action_brain_params(
        ctx, sensory_size=body.sensory_size,
        n_body_actions=3, n_saccade_actions=2,
    )
    state = init_action_brain_state(jax.random.PRNGKey(1), params)

    # Override the last-action one-hots on the state pytree.
    state = eqx.tree_at(
        lambda s: (s.last_body_action, s.last_saccade_action),
        state,
        (jnp.asarray(body_onehot, jnp.float32),
         jnp.asarray(saccade_onehot, jnp.float32)),
    )
    _, sample = body.reset(jax.random.PRNGKey(2))
    out = action_brain_step(
        state, params, ctx, sample.sensory,
        jnp.asarray(0.0, jnp.float32),
        jnp.asarray(0.0, jnp.float32),
        jax.random.PRNGKey(3),
    )
    return out.state


def test_efference_copy_affects_mossy_buffer():
    """Different body actions -> different mossy buffer contents."""
    s_a = _run_one_step([1.0, 0.0, 0.0], [1.0, 0.0])
    s_b = _run_one_step([0.0, 1.0, 0.0], [1.0, 0.0])
    diff = float(jnp.abs(s_a.delay_mossy.buf - s_b.delay_mossy.buf).sum())
    assert diff > 1e-4, (
        f"efference copy did not reach mossy buffer (diff={diff:.2e})"
    )


def test_saccade_efference_also_routed():
    s_a = _run_one_step([1.0, 0.0, 0.0], [1.0, 0.0])
    s_b = _run_one_step([1.0, 0.0, 0.0], [0.0, 1.0])
    diff = float(jnp.abs(s_a.delay_mossy.buf - s_b.delay_mossy.buf).sum())
    assert diff > 1e-4, (
        f"saccade efference did not reach mossy (diff={diff:.2e})"
    )


def test_zero_efference_does_not_blow_up():
    s = _run_one_step([0.0, 0.0, 0.0], [0.0, 0.0])
    assert bool(jnp.isfinite(s.delay_mossy.buf).all())
    assert bool(jnp.isfinite(s.cerebellum.dn_rate).all())
