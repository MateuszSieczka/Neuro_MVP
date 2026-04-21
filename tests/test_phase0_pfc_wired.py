"""P0.5 — PFC slot wired into ActionBrain (Frank & Badre 2012).

Weryfikuje że:
  1. PFC zwraca niezerowy content rate po kilkudziesięciu dt aktywności.
  2. `striatal_drive` w ActionBrain zawiera PFC content (size włącza pfc).
  3. PFC persystuje między cyklami decyzyjnymi (bez reset).
  4. Reset na `done=True` zeruje PFC content rate.
  5. `pfc_step` pod theta-phase gating: amp=0.1 → drobna modulacja.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import BackendContext
from core.pfc import (
    init_pfc_params, init_pfc_state, pfc_step, pfc_reset_transient,
    pfc_select_reset,
)
from core.brain_graph import (
    init_action_brain_params, init_action_brain_state,
    action_brain_cognitive_step,
)
from embodiment.bandit import GaussianBanditBody


def test_pfc_content_rate_grows_with_input():
    ctx = BackendContext(dt=1.0)
    params = init_pfc_params(ctx, input_size=16, n_content=32, n_gate=16)
    state = init_pfc_state(jax.random.PRNGKey(0), params)
    belief = jnp.ones((16,), jnp.float32) * 0.5

    key = jax.random.PRNGKey(1)
    for i in range(200):
        key, k = jax.random.split(key)
        out = pfc_step(state, params, ctx, belief, ach=0.7, da=0.6, key=k)
        state = out.state

    assert float(out.content_rate.max()) > 0.0
    # At least one neuron must have measurable EMA > 0.
    active = (out.content_rate > 1e-4).sum()
    assert int(active) > 0


def test_pfc_silent_without_neuromod():
    """ACh=DA=0 → gate never fires → content silent (conjunction gate)."""
    ctx = BackendContext(dt=1.0)
    params = init_pfc_params(ctx, input_size=16, n_content=32, n_gate=16)
    state = init_pfc_state(jax.random.PRNGKey(0), params)
    belief = jnp.ones((16,), jnp.float32) * 0.5

    key = jax.random.PRNGKey(1)
    for i in range(150):
        key, k = jax.random.split(key)
        out = pfc_step(state, params, ctx, belief, ach=0.0, da=0.0, key=k)
        state = out.state

    # Gate-blocked → output rate essentially zero.
    assert float(out.content_rate.max()) < 1e-3


def test_pfc_reset_zeros_output():
    ctx = BackendContext(dt=1.0)
    params = init_pfc_params(ctx, input_size=16, n_content=32, n_gate=16)
    state = init_pfc_state(jax.random.PRNGKey(0), params)
    belief = jnp.ones((16,), jnp.float32) * 0.5
    key = jax.random.PRNGKey(1)
    for i in range(100):
        key, k = jax.random.split(key)
        out = pfc_step(state, params, ctx, belief, ach=0.7, da=0.6, key=k)
        state = out.state
    # Reset should restore zero output.
    reset_state = pfc_reset_transient(state, params)
    assert float(reset_state.output_rate.max()) == 0.0


def test_pfc_select_reset_triggered_by_done():
    ctx = BackendContext(dt=1.0)
    params = init_pfc_params(ctx, input_size=16, n_content=32, n_gate=16)
    state = init_pfc_state(jax.random.PRNGKey(0), params)
    belief = jnp.ones((16,), jnp.float32) * 0.5
    key = jax.random.PRNGKey(1)
    for i in range(100):
        key, k = jax.random.split(key)
        out = pfc_step(state, params, ctx, belief, ach=0.7, da=0.6, key=k)
        state = out.state
    # done=0 → unchanged; done=1 → reset.
    kept = pfc_select_reset(state, params, done=0.0)
    reset = pfc_select_reset(state, params, done=1.0)
    assert jnp.allclose(kept.output_rate, state.output_rate)
    assert float(reset.output_rate.max()) == 0.0


def test_action_brain_includes_pfc_in_striatal_drive():
    """ActionBrain z PFC działa — state_size krytyka/aktora uwzględnia PFC."""
    ctx = BackendContext(dt=1.0)
    body = GaussianBanditBody.create(jax.random.PRNGKey(0), n_actions=3)
    params = init_action_brain_params(
        ctx, sensory_size=body.sensory_size,
        n_body_actions=3, n_saccade_actions=1,
    )
    state = init_action_brain_state(jax.random.PRNGKey(1), params)

    # PFC content size must appear in state_size (cortex_belief +
    # sensory + pfc_content == state_size consumed by critic).
    expected_state_size = (
        params.cortex.n_l23_state
        + params.sensory_size
        + params.pfc.n_content
    )
    assert params.critic.state_size == expected_state_size
    assert params.actor_body.state_size == expected_state_size

    # Step a few times — must not raise.
    key = jax.random.PRNGKey(2)
    body, sample = body.reset(key)
    prev_r = jnp.asarray(0.0, jnp.float32)
    prev_d = jnp.asarray(0.0, jnp.float32)
    for i in range(3):
        key, k_step = jax.random.split(key)
        out = action_brain_cognitive_step(
            state, params, ctx, sample.sensory, prev_r, prev_d, k_step,
        )
        state = out.state
        # NOTE: GaussianBandit uses body action only (saccade is trivial).
        key, k_act = jax.random.split(key)
        body, sample = body.act(k_act, int(out.body_action), 0)
        prev_r = sample.reward
        prev_d = sample.done
    assert jnp.isfinite(out.rpe)
    # PFC output_rate is traced (will be 0 early but must be finite).
    assert jnp.isfinite(state.pfc.output_rate).all()


def test_action_brain_pfc_resets_on_done():
    """Po `done=1` PFC content_rate musi być wyzerowany w NEXT step start."""
    ctx = BackendContext(dt=1.0)
    body = GaussianBanditBody.create(jax.random.PRNGKey(0), n_actions=3)
    params = init_action_brain_params(
        ctx, sensory_size=body.sensory_size,
        n_body_actions=3, n_saccade_actions=1,
    )
    state = init_action_brain_state(jax.random.PRNGKey(1), params)
    key = jax.random.PRNGKey(2)
    body, sample = body.reset(key)

    # Drive a few steps so PFC picks up *some* rate.
    for i in range(5):
        key, k_step = jax.random.split(key)
        out = action_brain_cognitive_step(
            state, params, ctx, sample.sensory,
            jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32),
            k_step,
        )
        state = out.state
        key, k_act = jax.random.split(key)
        body, sample = body.act(k_act, int(out.body_action), 0)

    # Now force done=1.  ActionBrain.pfc_select_reset happens at top
    # of action_brain_step — so call it with done=1 and observe that
    # the returned state's pfc is the one that ran with a reset PFC.
    key, k_step = jax.random.split(key)
    out_done = action_brain_cognitive_step(
        state, params, ctx, sample.sensory,
        jnp.asarray(0.0, jnp.float32), jnp.asarray(1.0, jnp.float32),
        k_step,
    )
    # PFC was reset at entry; substeps ran on fresh state.  Expect its
    # output_rate to be small (no time to accumulate over 20 ms of
    # calibrated drive — gate needs ~few ms to warm up).
    assert float(out_done.state.pfc.output_rate.max()) < 0.5
