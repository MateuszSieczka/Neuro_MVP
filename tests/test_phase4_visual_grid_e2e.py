"""Phase 4 e2e — VisualGrid + brain-side SensoryStack integration smoke.

Verifies the Phase 0.7 wiring end-to-end:

  * :class:`VisualGridBody` exposes ``image`` in its info dict and
    feeds the brain raw pixels (the eye is CNS, not body — Rodieck
    1998).
  * :func:`init_action_brain_params` accepts ``sensory_stack_params``
    and re-derives the effective afferent size from V1 L2/3 belief
    width.
  * :func:`action_brain_step` consumes ``image`` + ``fixation_xy``,
    runs the retina → LGN → V1 chain inside the brain, threads the
    V1 belief into thalamus / striatum / world-model, and updates
    ``state.prev_pe_rate`` for the next saccade-info-gain reward.
  * No NaNs / infs propagate through 60 decision cycles of fully
    closed-loop perception–action.

These are minimal smoke assertions; quantitative behavioural metrics
live in :mod:`test_phase4_saccade_selection` and
:mod:`test_phase4_v1_emergence_full`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import BackendContext
from core.brain_graph import (
    init_action_brain_params, init_action_brain_state, action_brain_step,
)
from sensory.retina import RetinaConfig
from sensory.sensory_stack import init_sensory_stack_params
from embodiment.visual_grid import VisualGridBody


def _build(seed: int = 0, *, n_l23_state: int = 32):
    ctx = BackendContext(dt=1.0)
    cfg = RetinaConfig(fovea_size=8, n_pyramid=2, periphery_tile=4)
    body = VisualGridBody.create(
        jax.random.PRNGKey(seed),
        size=3, start=(0, 0), goal=(2, 2),
        tex_size=32, max_steps=40, retina_cfg=cfg,
    )
    ss_params = init_sensory_stack_params(
        ctx, retina_cfg=cfg,
        n_l4=64, n_l23_state=n_l23_state, n_l23_error=16, n_l5=16,
    )
    params = init_action_brain_params(
        ctx,
        sensory_size=0,                  # overridden by sensory_stack
        n_body_actions=4,
        sensory_stack_params=ss_params,
    )
    state = init_action_brain_state(jax.random.PRNGKey(seed + 100), params)
    return ctx, body, params, state


def test_action_brain_accepts_sensory_stack_params():
    ctx, body, params, state = _build()
    # Effective sensory_size must equal V1 L4 population (the
    # corticostriatal afferent emitted by SensoryStack — Smith 2004).
    assert params.sensory_size == 64
    assert params.sensory_stack is not None
    assert state.sensory_stack is not None
    # prev_pe_rate is a scalar tracked across cycles.
    assert state.prev_pe_rate.shape == ()


def test_image_path_runs_one_cycle():
    """A single action_brain_step with image+fixation completes
    cleanly and updates the sensory_stack state."""
    ctx, body, params, state = _build()
    body, sample = body.reset(jax.random.PRNGKey(7))
    image = sample.info["image"]
    fix = sample.info["fixation_xy"]

    out = action_brain_step(
        state, params, ctx,
        prev_reward=0.0, prev_done=0.0,
        key=jax.random.PRNGKey(11),
        image=image, fixation_xy=fix,
    )

    # Belief shape must match V1 n_l23_state.
    assert out.cortex_belief.shape[0] == params.cortex.n_l23_state
    assert jnp.isfinite(out.cortex_belief).all()
    assert jnp.isfinite(out.state.prev_pe_rate)
    # First-cycle PE rate is finite (V1 has just seen its first image).
    assert float(out.state.prev_pe_rate) >= 0.0
    # Sensory stack state advanced.
    new_v1 = out.state.sensory_stack.v1
    old_v1 = state.sensory_stack.v1
    # At least one V1 leaf must change after one step (membrane voltage).
    assert not jnp.array_equal(new_v1.l4_nstate.v, old_v1.l4_nstate.v)


def test_closed_loop_60_cycles_no_nans():
    """60 decision cycles of VisualGrid + ActionBrain with the brain
    owning the retina. Asserts numerical stability across the full
    perceive → act → learn loop."""
    ctx, body, params, state = _build(seed=1)
    body, sample = body.reset(jax.random.PRNGKey(3))
    prev_r = jnp.asarray(0.0, jnp.float32)
    prev_d = jnp.asarray(0.0, jnp.float32)
    key = jax.random.PRNGKey(5)

    pe_history = []
    info_gain_history = []
    for _ in range(60):
        key, k_step, k_act = jax.random.split(key, 3)
        out = action_brain_step(
            state, params, ctx,
            prev_reward=prev_r, prev_done=prev_d,
            key=k_step,
            image=sample.info["image"],
            fixation_xy=sample.info["fixation_xy"],
        )
        state = out.state
        pe_history.append(float(state.prev_pe_rate))
        info_gain_history.append(float(state.last_info_gain))
        body, sample = body.act(
            k_act, int(out.body_action), int(out.saccade_action),
        )
        prev_r = sample.reward
        prev_d = sample.done

    pe_arr = jnp.asarray(pe_history)
    ig_arr = jnp.asarray(info_gain_history)
    assert jnp.isfinite(pe_arr).all(), "V1 PE rate diverged"
    assert jnp.isfinite(ig_arr).all(), "info_gain diverged"
    # PE rate must vary across cycles (V1 is responding to changing
    # input, not stuck at zero or pinned).
    assert float(jnp.std(pe_arr)) > 1e-4, (
        f"V1 PE rate is constant across 60 cycles (std={jnp.std(pe_arr):.2e})"
    )
    # Info-gain must be non-negative everywhere (relu) and active at
    # least once (the saccade actor needs a non-trivial training
    # signal somewhere in the trace).
    assert (ig_arr >= 0.0).all()
    assert float(ig_arr.max()) > 0.0, (
        "saccade info-gain reward never fired across 60 cycles — "
        "Bayesian-surprise loop is not closed"
    )


def test_v1_state_evolves_under_closed_loop():
    """Closed-loop V1 inside :func:`action_brain_step` must integrate
    input across the fixation window without numerical blowup, and
    its dynamic state (membrane voltages, rate EMAs) must evolve.

    STDP weight convergence is a long-timescale phenomenon (Olshausen
    & Field 1996 — thousands of natural-image patches); within 30
    decision cycles we only assert the substrate is alive and stable.
    """
    ctx, body, params, state = _build(seed=2)
    w_init = state.sensory_stack.v1.w_l4_in
    rate_init = state.sensory_stack.v1.rate_l4

    body, sample = body.reset(jax.random.PRNGKey(4))
    prev_r = jnp.asarray(0.0, jnp.float32)
    prev_d = jnp.asarray(0.0, jnp.float32)
    key = jax.random.PRNGKey(6)
    for _ in range(30):
        key, k_step, k_act = jax.random.split(key, 3)
        out = action_brain_step(
            state, params, ctx,
            prev_reward=prev_r, prev_done=prev_d,
            key=k_step,
            image=sample.info["image"],
            fixation_xy=sample.info["fixation_xy"],
        )
        state = out.state
        body, sample = body.act(
            k_act, int(out.body_action), int(out.saccade_action),
        )
        prev_r, prev_d = sample.reward, sample.done

    w_final = state.sensory_stack.v1.w_l4_in
    rate_final = state.sensory_stack.v1.rate_l4
    # Integration sanity: weights and rates remain finite under STDP.
    assert jnp.isfinite(w_final).all(), "V1 weights diverged"
    assert jnp.isfinite(rate_final).all(), "V1 rate EMAs diverged"
    # Weights must remain non-negative (cortex invariant).
    assert float(w_final.min()) >= 0.0
    # Rate EMA must have moved away from the all-zero init (V1 is
    # actually firing in response to the foveated visual input).
    rate_delta = float(jnp.abs(rate_final - rate_init).sum())
    assert rate_delta > 0.0, (
        "V1 L4 rate EMA never updated \u2014 V1 is silent in closed loop"
    )
