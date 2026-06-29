"""Faza U — §3 hippocampal node group: pattern separation / completion.

The hippocampus node gains its function from a one-shot episodic store,
no second learning rule.  Asserts:

* CA1 mismatch is exactly the node's own prediction error (mean|ε|);
* a belief encoded one-shot is pattern-completed from a noisy cue and
  written back onto the node;
* encoding is gated by novelty / mismatch (a matched belief is not
  re-stored).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from core.pc_graph import (
    init_region_graph, pc_graph_errors, REGION_INDEX,
)
from core.pc_hippocampus import (
    init_hippocampus, hippocampus_mismatch,
    hippocampus_encode, hippocampus_complete,
)


def _set_belief(graph, idx, value):
    mu = list(graph.mu)
    mu[idx] = value.astype(mu[idx].dtype)
    return eqx.tree_at(lambda s: s.mu, graph, tuple(mu))


def _hc_graph(seed=0):
    gp, gs = init_region_graph(jax.random.PRNGKey(seed))
    hc_idx = REGION_INDEX["hippocampus"]
    hp, store = init_hippocampus(jax.random.PRNGKey(seed + 1),
                                 gp.node_sizes[hc_idx], mismatch_gate=0.1)
    return gp, gs, hp, store, hc_idx


# ---------------------------------------------------------------------
# CA1 comparator = the node's own prediction error
# ---------------------------------------------------------------------


def test_ca1_mismatch_is_node_error():
    gp, gs, hp, _store, hc_idx = _hc_graph()
    belief = jax.random.normal(jax.random.PRNGKey(5), (gp.node_sizes[hc_idx],))
    g = _set_belief(gs, hc_idx, belief)

    eps = pc_graph_errors(g, gp)[hc_idx]
    expected = float(jnp.mean(jnp.abs(eps)))
    got = float(hippocampus_mismatch(g, gp, hp))
    assert abs(got - expected) < 1e-6, "CA1 mismatch ≠ node ε"


# ---------------------------------------------------------------------
# one-shot encode → pattern completion
# ---------------------------------------------------------------------


def test_encode_then_complete_restores_belief():
    gp, gs, hp, store, hc_idx = _hc_graph()
    d = gp.node_sizes[hc_idx]

    v0 = jax.random.normal(jax.random.PRNGKey(7), (d,))
    g = _set_belief(gs, hc_idx, v0)
    out = hippocampus_encode(g, gp, hp, store, gate=1.0)
    assert bool(out.stored), "episode not encoded"

    # Corrupt the belief, then complete from the store.
    cue = v0 + 0.05 * jax.random.normal(jax.random.PRNGKey(8), (d,))
    g = _set_belief(g, hc_idx, cue)
    comp = hippocampus_complete(g, hp, out.state)

    assert bool(comp.completed), "confident match should complete"
    restored = comp.graph.mu[hc_idx]
    assert float(jnp.linalg.norm(restored - v0)) < 1e-5, "belief not restored"


def test_encode_gate_blocks_familiar_belief():
    gp, gs, hp, store, hc_idx = _hc_graph()
    d = gp.node_sizes[hc_idx]
    v0 = jax.random.normal(jax.random.PRNGKey(9), (d,))
    g = _set_belief(gs, hc_idx, v0)

    first = hippocampus_encode(g, gp, hp, store, gate=1.0)
    # Same belief again — no longer novel, second write is a no-op.
    second = hippocampus_encode(g, gp, hp, first.state, gate=1.0)
    assert bool(first.stored) and not bool(second.stored)


def test_default_gate_is_mismatch_driven():
    """gate=None uses CA1 mismatch: a zero-error belief is not committed."""
    gp, gs, hp, store, hc_idx = _hc_graph()
    # Fresh graph: μ_hc = 0 ⇒ ε_hc = 0 ⇒ mismatch 0 < gate ⇒ no store.
    out = hippocampus_encode(gs, gp, hp, store)        # gate defaults to mismatch
    assert not bool(out.stored), "zero-mismatch belief should not encode"
