"""Faza U — §6 multi-cycle eligibility traces on the temporal (``w_dyn``) edges.

Eligibility is the **one rule extended in time**, not a second mechanism: a
temporal edge accumulates a decaying trace of its presynaptic factor
``φ(μ_prev[src])`` and commits ``ΔW = η·ξ_dst ⊗ trace``.  The destination's
own precision-weighted error ``ξ`` is the learning signal, so a value/policy
temporal edge is modulated by the value-node ε (the RPE / DA) with no
separate global broadcast.  The trace bridges credit across a gap wider than
one cycle — exactly what the 1-cycle ``mu_prev`` carry cannot do.

These tests assert: the trace is inert when the flag is off (the §5b
temporal path is byte-identical), and a reward delivered ≥2 cycles after a
cue credits the cue→reward temporal edge only when eligibility is on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.pc_graph import (
    init_pc_graph_params, init_pc_graph_state,
    pc_graph_step, pc_graph_learn,
)


def _cycle(state, params, clamp):
    cv = {k: jnp.asarray(v, jnp.float32) for k, v in clamp.items()}
    return pc_graph_step(state, params, cv, n_steps=5, update_precision=False).state


# ---------------------------------------------------------------------
# inert when off — the §5b temporal commit is unchanged
# ---------------------------------------------------------------------


def test_eligibility_off_keeps_trace_zero():
    """``eligibility=False`` ⇒ the trace never accumulates (§5b path intact)."""
    p = init_pc_graph_params((1, 1), (), dyn_edges=((0, 1),), act="linear")
    s = init_pc_graph_state(jax.random.PRNGKey(0), p)
    assert len(s.elig) == 1 and float(jnp.sum(jnp.abs(s.elig[0]))) == 0.0
    assert p.elig_mode is False

    for _ in range(5):
        s = _cycle(s, p, {0: [1.0], 1: [1.0]})
    # The trace state exists but is left untouched at zero when off.
    assert float(jnp.sum(jnp.abs(s.elig[0]))) == 0.0


def test_eligibility_on_accumulates_a_decaying_trace():
    """With the flag on, the trace fills with the (decayed) presynaptic factor."""
    p = init_pc_graph_params(
        (1, 1), (), dyn_edges=((0, 1),), act="linear", eligibility=True,
    )
    s = init_pc_graph_state(jax.random.PRNGKey(1), p)
    # Drive the source, then learn: μ_prev lags one cycle, so the trace
    # picks the cue up on the *next* learn and decays thereafter.
    s = _cycle(s, p, {0: [1.0], 1: [0.0]})           # μ_prev[0] ← 1
    s = _cycle(s, p, {0: [0.0], 1: [0.0]})           # trace ← φ(1) = 1
    t1 = float(s.elig[0][0])
    s = _cycle(s, p, {0: [0.0], 1: [0.0]})           # trace ← λ·1 (decays)
    t2 = float(s.elig[0][0])
    assert t1 > 0.9, f"trace did not capture the cue: {t1:.3f}"
    assert 0.0 < t2 < t1, f"trace did not decay: {t1:.3f} → {t2:.3f}"
    assert abs(t2 - float(p.elig_decay) * t1) < 1e-4, "decay ≠ λ"


# ---------------------------------------------------------------------
# the headline: credit a reward delivered ≥2 cycles after the cue
# ---------------------------------------------------------------------


def _train_cue_reward(eligibility: bool) -> float:
    """A cue at t, blanks, then a reward 3 cycles later; return the learned
    cue→value temporal weight.  Node 0 = cue, node 1 = value; one temporal
    edge (cue(t−1) → value(t)).  Both nodes clamped throughout so the cue's
    carry is exactly zero at the reward step — an immediate (1-cycle) commit
    therefore cannot credit it.
    """
    p = init_pc_graph_params(
        (1, 1), (), dyn_edges=((0, 1),), act="linear",
        eta_w=0.2, eligibility=eligibility,
    )
    s = init_pc_graph_state(jax.random.PRNGKey(2), p)
    episode = (
        {0: [1.0], 1: [0.0]},     # cue
        {0: [0.0], 1: [0.0]},     # blank
        {0: [0.0], 1: [0.0]},     # blank
        {0: [0.0], 1: [1.0]},     # reward, ≥2 cycles after the cue
    )
    for _ in range(60):
        for clamp in episode:
            s = _cycle(s, p, clamp)
    return float(s.w_dyn[0][0, 0])


def test_eligibility_bridges_a_delayed_reward():
    """Only eligibility lets the cue→value edge learn the delayed reward."""
    w_on = _train_cue_reward(eligibility=True)
    w_off = _train_cue_reward(eligibility=False)
    assert w_on > 0.3, (
        f"eligibility failed to credit the cue from a delayed reward: {w_on:.3f}"
    )
    assert w_off < 0.1, (
        f"a 1-cycle edge must NOT bridge the gap, got {w_off:.3f}"
    )
    assert w_on > 3.0 * abs(w_off), "delayed credit must come from the trace"
