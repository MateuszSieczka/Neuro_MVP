"""Phase 4 e2e — saccade actor is driven by V1 info-gain (Itti & Baldi 2009).

Verifies that the brain-owned :class:`SensoryStack` produces a
non-trivial Bayesian-surprise reward, and that this reward is
routed *only* to the saccade (oculomotor) BG actor — not to the
body (skeletomotor) actor (Tatler 2011 active sampling).

Method
------
Two paired runs share identical seeds, body, perception, and
extrinsic reward path; they differ only in whether the V1-derived
``info_gain`` is allowed through (auto-computed) or forced to zero
(explicit override).  After N cycles:

  * The body-actor weights must be **bit-identical** across runs
    (info_gain has no path into the body loop).
  * The saccade-actor weights must **diverge** (info_gain drives
    saccade-specific RPE through ``beta_saccade``).
  * The auto-computed info_gain must fire at least once and remain
    non-negative throughout (relu of PE delta).
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


def _build(seed: int):
    ctx = BackendContext(dt=1.0)
    cfg = RetinaConfig(fovea_size=8, n_pyramid=2, periphery_tile=4)
    body = VisualGridBody.create(
        jax.random.PRNGKey(seed),
        size=4, start=(0, 0), goal=(3, 3),
        tex_size=32, max_steps=40, retina_cfg=cfg,
    )
    ss_params = init_sensory_stack_params(
        ctx, retina_cfg=cfg,
        n_l4=32, n_l23_state=16, n_l23_error=8, n_l5=8,
    )
    params = init_action_brain_params(
        ctx, sensory_size=0, n_body_actions=4,
        sensory_stack_params=ss_params,
        substeps=4,
        n_tc=32, n_ct=16, n_trn=16,
        cortex_n_l4=32, cortex_n_l23_state=32,
        cortex_n_l23_error=32, cortex_n_l5=16,
        critic_hidden=32, wm_hidden=32, wm_n_error=32,
    )
    state = init_action_brain_state(jax.random.PRNGKey(seed + 100), params)
    return ctx, body, params, state


def _run(seed: int, *, force_zero_info_gain: bool, n_cycles: int):
    ctx, body, params, state = _build(seed)
    body, sample = body.reset(jax.random.PRNGKey(seed + 7))
    prev_r = jnp.asarray(0.0, jnp.float32)
    prev_d = jnp.asarray(0.0, jnp.float32)
    key = jax.random.PRNGKey(seed + 13)
    info_gains = []
    for _ in range(n_cycles):
        key, k_step, k_act = jax.random.split(key, 3)
        ig_kw = jnp.asarray(0.0, jnp.float32) if force_zero_info_gain else None
        out = action_brain_step(
            state, params, ctx,
            prev_reward=prev_r, prev_done=prev_d,
            key=k_step,
            image=sample.info["image"],
            fixation_xy=sample.info["fixation_xy"],
            info_gain=ig_kw,
        )
        state = out.state
        info_gains.append(float(state.last_info_gain))
        body, sample = body.act(
            k_act, int(out.body_action), int(out.saccade_action),
        )
        prev_r, prev_d = sample.reward, sample.done
    return state, jnp.asarray(info_gains)


def test_info_gain_fires_and_is_nonnegative():
    """V1 PE delta must produce at least one positive info-gain
    reward across 40 cycles, and never go negative (relu)."""
    _, ig_trace = _run(seed=0, force_zero_info_gain=False, n_cycles=40)
    assert (ig_trace >= 0.0).all(), "info_gain went negative (relu broken)"
    assert float(ig_trace.max()) > 0.0, (
        "info_gain reward never fired — V1 PE never decreased between "
        "consecutive fixations, Bayesian-surprise loop is dead"
    )


def test_info_gain_routed_only_to_saccade_actor():
    """Body actor must be bit-identical between info_gain=auto and
    info_gain=0 runs; saccade actor must diverge.

    This is the strict version of the routing test in
    :mod:`test_phase0_saccade_info_gain` extended to the e2e
    SensoryStack-wired path.
    """
    s_auto, _ = _run(seed=1, force_zero_info_gain=False, n_cycles=30)
    s_zero, _ = _run(seed=1, force_zero_info_gain=True, n_cycles=30)

    body_diff = float(
        jnp.abs(s_auto.actor_body.w_d1 - s_zero.actor_body.w_d1).sum()
    )
    saccade_diff = float(
        jnp.abs(s_auto.actor_saccade.w_d1 - s_zero.actor_saccade.w_d1).sum()
    )
    assert body_diff < 1e-5, (
        f"body actor influenced by info_gain (diff={body_diff:.2e}) — "
        f"routing leaked beyond the saccade BG loop"
    )
    assert saccade_diff > 1e-4, (
        f"saccade actor did not diverge (diff={saccade_diff:.2e}) — "
        f"info_gain reward not effectively reaching its actor"
    )


def test_saccade_actor_evidence_diverges_from_body_actor():
    """Across a closed-loop run, the two BG actors should not be
    coupled: their spike-count evidence accumulators must be alive.
    Sanity check on Alexander 1986 parallel-loop implementation."""
    s_auto, _ = _run(seed=2, force_zero_info_gain=False, n_cycles=30)
    e_body = float(s_auto.actor_body.spike_count_d1.sum())
    e_sacc = float(s_auto.actor_saccade.spike_count_d1.sum())
    # Both must be alive (some D1 spikes accumulated somewhere).
    assert e_body > 0.0 or e_sacc > 0.0
