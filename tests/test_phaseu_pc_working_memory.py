"""Faza U — §5 working memory as a leaky temporal self-edge.

Working memory is persistence: a node holding its belief μ across cycles
when the input is absent.  In this substrate that is the §5b ``w_dyn``
temporal self-edge ``(node→node)`` initialised to ``gain·I`` — no new rule,
no new state type, only a named persistence gain.  ``gain ≤ 1`` makes it a
*leaky* integrator (decays by ``gain`` per cycle) rather than a runaway
attractor; the global ``leak`` adds further forgetting.

These tests assert: the self-edge holds μ across cycles with the input
removed (and collapses to zero with no persistence gain), the hold leaks
geometrically, and the region graph wires + initialises a ``pfc``
persistence node behind the opt-in flag without disturbing the base graph.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from core.pc_graph import (
    init_pc_graph_params, init_pc_graph_state,
    pc_graph_relax, pc_graph_roll, pc_graph_step, pc_graph_errors,
    apply_wm_persistence_init,
    init_region_graph, REGION_NODES, REGION_INDEX,
)


def _cos(a, b):
    return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-9))


def _hold_node(gain: float):
    """A 1-node graph with a persistence self-edge; return (held μ, target)."""
    p = init_pc_graph_params(
        (4,), (), dyn_edges=((0, 0),), act="tanh", n_relax=60, leak=0.0,
    )
    s = init_pc_graph_state(jax.random.PRNGKey(0), p)
    s = apply_wm_persistence_init(s, p, 0, gain)
    target = jnp.array([0.5, -0.3, 0.2, 0.4], jnp.float32)
    s = pc_graph_roll(eqx.tree_at(lambda st: st.mu, s, (target,)))   # μ_prev ← target
    held = pc_graph_relax(s, p, clamp=()).mu[0]                       # no input
    return held, target


# ---------------------------------------------------------------------
# the self-edge holds μ across a cycle with the input removed
# ---------------------------------------------------------------------


def test_persistence_self_edge_holds_belief():
    """gain·I self-edge sustains the belief; gain=0 lets it collapse."""
    held, target = _hold_node(gain=0.9)
    held0, _ = _hold_node(gain=0.0)

    assert jnp.linalg.norm(held) > 0.5 * jnp.linalg.norm(target), "WM did not hold"
    assert _cos(held, target) > 0.9, "held pattern drifted from the stored one"
    assert jnp.linalg.norm(held0) < 0.1 * jnp.linalg.norm(held), (
        "with no persistence gain the belief must decay away"
    )


def test_persistence_is_a_leaky_integrator():
    """Repeated empty cycles decay the held belief geometrically (not frozen)."""
    gain = 0.8
    p = init_pc_graph_params(
        (4,), (), dyn_edges=((0, 0),), act="tanh", n_relax=60, leak=0.0,
    )
    s = init_pc_graph_state(jax.random.PRNGKey(1), p)
    s = apply_wm_persistence_init(s, p, 0, gain)
    target = jnp.array([0.4, 0.4, 0.4, 0.4], jnp.float32)
    s = pc_graph_roll(eqx.tree_at(lambda st: st.mu, s, (target,)))

    norms = []
    for _ in range(4):
        s = pc_graph_relax(s, p, clamp=())     # hold, no input
        norms.append(float(jnp.linalg.norm(s.mu[0])))
        s = pc_graph_roll(s)                   # carry forward
    # Monotonic decay, still alive after several cycles (leaky, not 0, not ∞).
    assert all(norms[i + 1] < norms[i] for i in range(len(norms) - 1)), norms
    assert norms[-1] > 0.0 and norms[0] < jnp.linalg.norm(target) + 1e-4


# ---------------------------------------------------------------------
# region-graph integration: a pfc persistence node behind the flag
# ---------------------------------------------------------------------


def test_region_graph_working_memory_is_additive_and_initialised():
    """The flag appends one pfc node + its persistence self-edge; base intact."""
    gp0, _ = init_region_graph(jax.random.PRNGKey(2))
    gp1, gs1 = init_region_graph(
        jax.random.PRNGKey(2), working_memory=True, wm_persistence_gain=0.9,
    )
    assert gp0.n_nodes == 11
    assert gp1.n_nodes == 12                       # + pfc
    pfc = 11
    assert gp1.node_sizes[pfc] == 32

    # Base region names/sizes still resolve unchanged.
    for name in REGION_NODES:
        i = REGION_INDEX[name]
        assert gp1.node_sizes[i] == gp0.node_sizes[i]

    # The persistence self-edge exists and is initialised to gain·I.
    assert (pfc, pfc) in gp1.dyn_edges
    e = list(gp1.dyn_edges).index((pfc, pfc))
    assert jnp.allclose(gs1.w_dyn[e], 0.9 * jnp.eye(32, dtype=jnp.float32))

    # Static wiring connects it to the deep cortical cause both ways.
    c3 = REGION_INDEX["cortex_l3"]
    assert (c3, pfc) in gp1.edges and (pfc, c3) in gp1.edges


def test_region_graph_with_working_memory_relaxes_finite():
    """A full region graph with WM runs a cycle to a finite free energy."""
    gp, gs = init_region_graph(
        jax.random.PRNGKey(3), working_memory=True, eta_w=1e-2, n_relax=20,
    )
    s_idx = REGION_INDEX["sensory"]
    obs = jax.random.normal(jax.random.PRNGKey(4), (gp.node_sizes[s_idx],))
    out = pc_graph_step(gs, gp, {s_idx: obs}, n_steps=20)
    assert jnp.isfinite(out.free_energy)
    for w in out.state.w_dyn:
        assert jnp.all(jnp.isfinite(w))
