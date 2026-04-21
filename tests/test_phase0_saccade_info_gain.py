"""P0.8 — Saccade info-gain reward routing (Itti & Baldi 2009).

Weryfikuje że:
  1. `action_brain_step` akceptuje kwarg `info_gain` i ustawia state.last_info_gain.
  2. Z info_gain=0 → stare zachowanie; z info_gain>0 → saccade actor
     uczy się inaczej niż body.
  3. Body actor nie jest modyfikowany przez info_gain (routing).

Info-gain wchodzi do saccade RPE **addytywnie** na jednostkowej
precyzji (Friston 2017 active inference, Eq. 2.9) — żadnych
tuning-scalarów typu ``beta_saccade``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import BackendContext
from core.brain_graph import (
    init_action_brain_params, init_action_brain_state,
    action_brain_cognitive_step,
)
from embodiment.bandit import GaussianBanditBody


def _run_brain(info_gain: float, steps: int):
    ctx = BackendContext(dt=1.0)
    body = GaussianBanditBody.create(jax.random.PRNGKey(0), n_actions=3)
    params = init_action_brain_params(
        ctx, sensory_size=body.sensory_size,
        n_body_actions=3, n_saccade_actions=2,
    )
    state = init_action_brain_state(jax.random.PRNGKey(1), params)
    key = jax.random.PRNGKey(2)
    body, sample = body.reset(key)
    prev_r = jnp.asarray(0.0, jnp.float32)
    prev_d = jnp.asarray(0.0, jnp.float32)
    for _ in range(steps):
        key, k_step = jax.random.split(key)
        out = action_brain_cognitive_step(
            state, params, ctx, sample.sensory, prev_r, prev_d, k_step,
            info_gain=jnp.asarray(info_gain, jnp.float32),
        )
        state = out.state
        key, k_act = jax.random.split(key)
        body, sample = body.act(k_act, int(out.body_action), 0)
        prev_r = sample.reward
        prev_d = sample.done
    return state


def test_info_gain_stored_in_state():
    state = _run_brain(info_gain=0.3, steps=3)
    assert abs(float(state.last_info_gain) - 0.3) < 1e-5


def test_info_gain_zero_baseline():
    state = _run_brain(info_gain=0.0, steps=3)
    assert float(state.last_info_gain) == 0.0


def test_info_gain_changes_saccade_weights_not_body():
    """info_gain > 0 → saccade actor diverges; body actor invariant."""
    s0 = _run_brain(info_gain=0.0, steps=10)
    s1 = _run_brain(info_gain=0.5, steps=10)

    body_diff = float(jnp.abs(s0.actor_body.w_d1 - s1.actor_body.w_d1).sum())
    saccade_diff = float(
        jnp.abs(s0.actor_saccade.w_d1 - s1.actor_saccade.w_d1).sum()
    )

    # Body actor should be BIT-IDENTICAL across the two runs (same
    # seeds, same extrinsic reward path, info_gain is not routed to it).
    assert body_diff < 1e-5, f"body actor influenced by info_gain: {body_diff:.2e}"
    # Saccade actor should diverge.
    assert saccade_diff > 1e-4, (
        f"saccade actor should diverge: {saccade_diff:.2e}"
    )
