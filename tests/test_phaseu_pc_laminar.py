"""Faza U — §5 laminar cortex: the canonical PC microcircuit per region.

Each cortical region is split into three populations wired by the **one
rule** (Bastos 2012): L4 = ε (granular error, separable Π), L2/3 = μ (the
cause — kept as the existing ``cortex_lN`` index/read-out), L5 = prediction
(the descending output).  Inter-region generative edges re-origin from L5
into the lower region's L4; the deep consumers read ``L5_cortex_l3``.  It is
sub-populations + intra-region edges on the one graph — no new rule, and
relaxation handles the extra nodes like any others (U.3 arbitrary topology).

These tests assert: the split is additive (base graph byte-identical when
off), the enriched graph relaxes to a finite free energy and trains, its
L4/L2-3/L5 populations are live and distinct, and it does no worse than the
flat-node baseline at explaining the sensory input.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.pc_graph import (
    pc_graph_step, pc_graph_clamp, pc_graph_relax, pc_graph_errors,
    init_region_graph, REGION_NODES, REGION_INDEX,
)

# Laminar nodes are appended after the 11 base nodes, in this order.
C1_L4, C1_L5 = 11, 12
C2_L4, C2_L5 = 13, 14
C3_L4, C3_L5 = 15, 16


def _obs(key, dim):
    # A structured (low-frequency) pattern, not white noise — gives the
    # hierarchy something learnable to compress.
    return jnp.sin(jnp.linspace(0.0, 3.0, dim) + jax.random.uniform(key))


# ---------------------------------------------------------------------
# additive: flat graph unchanged, laminar adds the right nodes + edges
# ---------------------------------------------------------------------


def test_laminar_is_additive_and_preserves_readouts():
    gp0, _ = init_region_graph(jax.random.PRNGKey(0))
    gp1, _ = init_region_graph(jax.random.PRNGKey(0), laminar_cortex=True)

    assert gp0.n_nodes == 11 and gp0.n_edges == 15
    # +2 nodes per cortical region (L4, L5) and +2 intra edges per region.
    assert gp1.n_nodes == 17 and gp1.n_edges == 21

    # Base region names/sizes (incl. the cortex_l* read-outs) unchanged.
    for name in REGION_NODES:
        i = REGION_INDEX[name]
        assert gp1.node_sizes[i] == gp0.node_sizes[i]

    # cortex_lN stays the L2/3 cause (the read-out pc_brain.cortex_top reads).
    c1, c2, c3 = (REGION_INDEX[f"cortex_l{n}"] for n in (1, 2, 3))
    s, wm = REGION_INDEX["sensory"], REGION_INDEX["world_model"]

    # Intra-region triad edges: cause→L4 (error) and cause→L5 (prediction).
    for cause, l4, l5 in ((c1, C1_L4, C1_L5), (c2, C2_L4, C2_L5), (c3, C3_L4, C3_L5)):
        assert (cause, l4) in gp1.edges and (cause, l5) in gp1.edges
    # Descending output re-origins from L5 into the lower region's L4 / sensory.
    assert (C1_L5, s) in gp1.edges
    assert (C2_L5, C1_L4) in gp1.edges
    assert (C3_L5, C2_L4) in gp1.edges
    # Deep consumers read L5_cortex_l3 (corticofugal output), not the cause.
    assert (C3_L5, wm) in gp1.edges
    assert (C3_L5, REGION_INDEX["value"]) in gp1.edges
    assert (C3_L5, REGION_INDEX["motor"]) in gp1.edges


# ---------------------------------------------------------------------
# the enriched graph relaxes to a finite FE and trains under the one rule
# ---------------------------------------------------------------------


def test_laminar_region_graph_relaxes_and_trains():
    gp, gs = init_region_graph(
        jax.random.PRNGKey(1), laminar_cortex=True, eta_w=1e-2, n_relax=30,
    )
    s_idx = REGION_INDEX["sensory"]
    obs = _obs(jax.random.PRNGKey(2), gp.node_sizes[s_idx])

    s = gs
    for _ in range(12):
        out = pc_graph_step(s, gp, {s_idx: obs}, n_steps=30)
        s = out.state
        assert jnp.isfinite(out.free_energy), "laminar region-graph FE not finite"
    for w in s.weights:
        assert jnp.all(jnp.isfinite(w)), "laminar weight blew up"
    moved = sum(float(jnp.sum(jnp.abs(a - b))) for a, b in zip(s.weights, gs.weights))
    assert moved > 0.0, "the one rule did not update the laminar edges"


def test_laminar_populations_are_live_and_distinct():
    """L4 carries a (finite, non-trivial) ε; L5 ≠ L2/3 after relaxation."""
    gp, gs = init_region_graph(
        jax.random.PRNGKey(3), laminar_cortex=True, eta_w=2e-2, n_relax=40,
    )
    s_idx = REGION_INDEX["sensory"]
    obs = _obs(jax.random.PRNGKey(4), gp.node_sizes[s_idx])

    s = gs
    for _ in range(10):
        s = pc_graph_step(s, gp, {s_idx: obs}, n_steps=40).state

    relaxed = pc_graph_relax(pc_graph_clamp(s, {s_idx: obs}), gp, clamp=(s_idx,), n_steps=40)
    eps = pc_graph_errors(relaxed, gp)
    # L4 error populations carry a finite, non-zero prediction error.
    for l4 in (C1_L4, C2_L4):
        assert jnp.all(jnp.isfinite(eps[l4]))
        assert float(jnp.sum(jnp.abs(eps[l4]))) > 0.0
    # L5 (prediction) and L2/3 (cause) are genuinely distinct populations.
    c1 = REGION_INDEX["cortex_l1"]
    assert float(jnp.sum(jnp.abs(relaxed.mu[C1_L5] - relaxed.mu[c1]))) > 1e-4


# ---------------------------------------------------------------------
# no regression: laminar does no worse than flat at explaining the input
# ---------------------------------------------------------------------


def _sensory_error(laminar: bool) -> tuple[float, float]:
    gp, gs = init_region_graph(
        jax.random.PRNGKey(5), laminar_cortex=laminar, eta_w=2e-2, n_relax=30,
    )
    s_idx = REGION_INDEX["sensory"]
    obs = _obs(jax.random.PRNGKey(6), gp.node_sizes[s_idx])

    def err(state):
        relaxed = pc_graph_relax(
            pc_graph_clamp(state, {s_idx: obs}), gp, clamp=(s_idx,), n_steps=30,
        )
        return float(jnp.mean(jnp.abs(pc_graph_errors(relaxed, gp)[s_idx])))

    e0 = err(gs)
    s = gs
    for _ in range(25):
        s = pc_graph_step(s, gp, {s_idx: obs}, n_steps=30).state
    return e0, err(s)


def test_laminar_does_no_worse_than_flat_baseline():
    flat0, flatN = _sensory_error(laminar=False)
    lam0, lamN = _sensory_error(laminar=True)

    # Both learn to explain the input.
    assert flatN < 0.8 * flat0, f"flat baseline did not learn: {flat0:.3f}→{flatN:.3f}"
    assert lamN < 0.8 * lam0, f"laminar did not learn: {lam0:.3f}→{lamN:.3f}"
    # Enrichment does not regress reconstruction (generous tolerance).
    assert lamN <= flatN * 1.25, (
        f"laminar regressed vs flat: flat={flatN:.4f} laminar={lamN:.4f}"
    )
