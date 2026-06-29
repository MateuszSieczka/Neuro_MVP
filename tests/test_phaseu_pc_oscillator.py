"""Faza U — §4 theta/gamma oscillator + the HC encode/retrieve theta gate.

Asserts the theta phase partitions each cycle into mutually exclusive
encode / retrieve windows, NE speeds and 5-HT slows theta, SWS suppresses
gamma, gamma resets at the theta trough (PAC), and — the §3-deferred piece
— that the theta gate confines hippocampal encoding and completion to
opposite phases.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from core.pc_oscillator import (
    init_oscillator_params, init_oscillator_state, oscillator_step,
)
from core.pc_graph import init_region_graph, REGION_INDEX
from core.pc_hippocampus import (
    init_hippocampus, hippocampus_encode, hippocampus_complete,
)


def _count_resets(ne=0.0, sero=0.0, n=60):
    op = init_oscillator_params()
    st = init_oscillator_state()
    n_reset = 0
    for _ in range(n):
        st, out = oscillator_step(st, op, ne_level=ne, sero_level=sero)
        n_reset += int(out.theta_reset)
    return n_reset


# ---------------------------------------------------------------------
# Theta phase structure
# ---------------------------------------------------------------------


def test_encode_retrieve_phases_partition_the_cycle():
    op = init_oscillator_params()
    st = init_oscillator_state()
    enc = ret = both = 0
    for _ in range(40):
        st, out = oscillator_step(st, op)
        enc += int(out.encoding_phase)
        ret += int(out.retrieval_phase)
        both += int(bool(out.encoding_phase) and bool(out.retrieval_phase))
    assert enc > 0 and ret > 0, "both windows occur"
    assert both == 0, "encode and retrieve are mutually exclusive"


def test_ne_speeds_and_sero_slows_theta():
    assert _count_resets(ne=1.0) > _count_resets(ne=0.0), "NE speeds theta"
    assert _count_resets(sero=1.0) < _count_resets(sero=0.0), "5-HT slows theta"


def test_sws_suppresses_gamma():
    op = init_oscillator_params()
    st = init_oscillator_state()
    st, _ = oscillator_step(st, op, sws_mode=True)
    assert float(st.gamma_amplitude) == 0.0, "gamma is silenced in SWS"


def test_pac_gamma_resets_at_theta_trough():
    """When theta crosses π (the trough) gamma phase-resets to ~0 (Lisman 2013)."""
    op = init_oscillator_params()
    st = init_oscillator_state()
    saw_trough_reset = False
    for _ in range(60):
        prev_theta = float(st.theta_phase)
        st, out = oscillator_step(st, op)
        crossed_trough = prev_theta < jnp.pi <= float(st.theta_phase)
        if crossed_trough:
            assert bool(out.gamma_reset), "gamma must reset at the theta trough"
            assert float(st.gamma_phase) < float(op.gamma_freq) * 2.0 * jnp.pi + 1e-4
            saw_trough_reset = True
    assert saw_trough_reset, "the run should cross at least one theta trough"


# ---------------------------------------------------------------------
# HC theta gate — the §3-deferred encode/retrieve gating
# ---------------------------------------------------------------------


def _hc_setup(seed=0):
    gp, gs = init_region_graph(jax.random.PRNGKey(seed))
    hc = REGION_INDEX["hippocampus"]
    hp, store = init_hippocampus(jax.random.PRNGKey(seed + 1),
                                 gp.node_sizes[hc], mismatch_gate=0.1)
    return gp, gs, hp, store, hc


def _set_hc(graph, hc, value):
    mu = list(graph.mu)
    mu[hc] = value.astype(mu[hc].dtype)
    return eqx.tree_at(lambda s: s.mu, graph, tuple(mu))


def test_theta_gate_confines_encoding_to_encoding_phase():
    gp, gs, hp, store, hc = _hc_setup()
    belief = jax.random.normal(jax.random.PRNGKey(7), (gp.node_sizes[hc],))
    g = _set_hc(gs, hc, belief)
    off = hippocampus_encode(g, gp, hp, store, gate=1.0, phase_gate=False)
    on = hippocampus_encode(g, gp, hp, store, gate=1.0, phase_gate=True)
    assert not bool(off.stored), "no encoding off the storage phase"
    assert bool(on.stored), "encoding proceeds on the storage phase"


def test_theta_gate_confines_completion_to_retrieval_phase():
    gp, gs, hp, store, hc = _hc_setup()
    belief = jax.random.normal(jax.random.PRNGKey(7), (gp.node_sizes[hc],))
    g = _set_hc(gs, hc, belief)
    stored = hippocampus_encode(g, gp, hp, store, gate=1.0)
    cue = _set_hc(g, hc, belief + 0.05 * jax.random.normal(jax.random.PRNGKey(8),
                                                           (gp.node_sizes[hc],)))
    off = hippocampus_complete(cue, hp, stored.state, phase_gate=False)
    on = hippocampus_complete(cue, hp, stored.state, phase_gate=True)
    assert not bool(off.completed), "no recall off the retrieval phase"
    assert bool(on.completed), "recall proceeds on the retrieval phase"
