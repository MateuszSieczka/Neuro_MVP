"""Faza U — krok U.2/U.3: jeden silnik, jedna reguła, jeden graf.

``core.pc_graph`` is the unified substrate: every region is a node, every
projection a generative edge, the only plasticity is the single rule
``ΔW = η·Π·ε·φ(μ)`` and the only inference is free-energy relaxation —
on an *arbitrary* topology (Salvatori 2022).  These tests assert the
substrate's load-bearing properties:

* relaxation descends the one global objective ``graph_free_energy``;
* the one rule learns a supervised mapping through a deep node chain
  (§9.4 credit scaling) and through a multi-parent / cyclic topology
  (§9.5 arbitrary topology);
* all biological regions instantiate as nodes of one graph and run a
  full clamp→relax→learn cycle (the U.2 big-bang statement).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from core.pc_graph import (
    init_pc_graph_params, init_pc_graph_state,
    pc_graph_relax, pc_graph_clamp, graph_free_energy, pc_graph_step,
    pc_graph_predictions, init_region_graph, REGION_INDEX,
)


# ---------------------------------------------------------------------
# helpers — a chain graph: cause(top) → h2 → h1 → output(bottom)
# ---------------------------------------------------------------------


def _chain_graph(sizes, key, **kw):
    """Nodes 0..L bottom→top; edge (j+1 → j): higher predicts lower."""
    L = len(sizes) - 1
    edges = tuple((l + 1, l) for l in range(L))   # src=higher predicts dst=lower
    params = init_pc_graph_params(sizes, edges, **kw)
    state = init_pc_graph_state(key, params)
    return params, state


def _chain_feedforward_output(state, params, top_input):
    """Output (node 0) from a feedforward pass through the chain."""
    from core.pc_module import _phi
    L = params.n_nodes - 1
    a = [None] * (L + 1)
    a[L] = top_input
    # edge e predicts node e from node e+1 (see _chain_graph ordering)
    for l in range(L - 1, -1, -1):
        a[l] = state.weights[l] @ _phi(params.act, a[l + 1])
    return a[0]


# ---------------------------------------------------------------------
# relaxation descends the global free energy
# ---------------------------------------------------------------------


def test_relaxation_descends_global_free_energy():
    sizes = (4, 8, 8, 6)              # output=4 … cause(input)=6
    params, state = _chain_graph(
        sizes, jax.random.PRNGKey(0), eta_mu=0.1, n_relax=200,
    )
    k2, k3 = jax.random.split(jax.random.PRNGKey(1), 2)
    inp = jax.random.normal(k2, (sizes[-1],))
    out_ff = _chain_feedforward_output(state, params, inp)
    target = out_ff + 0.3 * jax.random.normal(k3, (sizes[0],))

    top = params.n_nodes - 1
    clamped = pc_graph_clamp(state, {top: inp, 0: target})

    energies = [
        float(graph_free_energy(
            pc_graph_relax(clamped, params, clamp=(0, top), n_steps=k), params,
        ))
        for k in range(0, 201, 25)
    ]
    assert energies[-1] < energies[0] * 0.8, (
        f"global FE barely moved: {energies[0]:.4f} → {energies[-1]:.4f}"
    )
    for i in range(len(energies) - 1):
        assert energies[i + 1] <= energies[i] + 1e-5, f"FE rose: {energies}"


# ---------------------------------------------------------------------
# one rule learns through a DEEP node chain (credit scaling, §9.4)
# ---------------------------------------------------------------------


def test_one_rule_learns_deep_chain():
    sizes = (3, 16, 16, 16, 6)        # 4 generative edges = deep hierarchy
    params, state = _chain_graph(
        sizes, jax.random.PRNGKey(2),
        eta_mu=0.1, eta_w=5e-2, n_relax=40,
    )
    k2, k3 = jax.random.split(jax.random.PRNGKey(3), 2)
    inp = jax.random.normal(k2, (sizes[-1],))
    target = jax.random.normal(k3, (sizes[0],)) * 0.5
    top = params.n_nodes - 1

    loss0 = float(0.5 * jnp.sum(
        (_chain_feedforward_output(state, params, inp) - target) ** 2))

    for _ in range(300):
        out = pc_graph_step(state, params, {top: inp, 0: target}, n_steps=40)
        state = out.state

    lossN = float(0.5 * jnp.sum(
        (_chain_feedforward_output(state, params, inp) - target) ** 2))
    assert lossN < loss0 * 0.2, (
        f"deep chain did not learn under one rule: {loss0:.4f} → {lossN:.4f}"
    )


# ---------------------------------------------------------------------
# arbitrary topology: multi-parent + cycle relaxes to finite FE (§9.5)
# ---------------------------------------------------------------------


def test_arbitrary_topology_relaxes():
    # 4 nodes; node 0 has TWO parents (1 and 2); 1↔2 form a cycle.
    sizes = (6, 8, 8, 5)
    edges = ((1, 0), (2, 0), (3, 1), (1, 2), (2, 1))   # multi-parent + cycle
    params = init_pc_graph_params(sizes, edges, eta_mu=0.05, n_relax=100)
    state = init_pc_graph_state(jax.random.PRNGKey(4), params)
    obs = jax.random.normal(jax.random.PRNGKey(5), (sizes[0],))

    clamped = pc_graph_clamp(state, {0: obs})
    f0 = float(graph_free_energy(clamped, params))
    relaxed = pc_graph_relax(clamped, params, clamp=(0,), n_steps=100)
    fN = float(graph_free_energy(relaxed, params))

    assert jnp.isfinite(fN), "FE diverged on cyclic topology"
    assert fN <= f0 + 1e-5, f"relaxation increased FE on cyclic graph: {f0}→{fN}"


# ---------------------------------------------------------------------
# the U.2 big-bang: all regions as nodes of ONE graph, ONE rule
# ---------------------------------------------------------------------


def test_region_graph_runs_one_cycle():
    params, state = init_region_graph(
        jax.random.PRNGKey(6), eta_mu=0.05, eta_w=1e-2, n_relax=30,
    )
    # Every region is present as a node.
    assert params.n_nodes == 11
    assert params.n_edges == 15

    s_idx = REGION_INDEX["sensory"]
    obs = jax.random.normal(jax.random.PRNGKey(7), (params.node_sizes[s_idx],))

    out = pc_graph_step(state, params, {s_idx: obs}, n_steps=30)
    # One full clamp→relax→learn cycle produced a finite objective and
    # finite weights everywhere (no per-region rule, no NaNs).
    assert jnp.isfinite(out.free_energy), "region-graph FE not finite"
    for w in out.state.weights:
        assert jnp.all(jnp.isfinite(w)), "region-graph weight blew up"

    # Learning changed at least some edges (the one rule is live on them).
    moved = sum(
        float(jnp.sum(jnp.abs(a - b))) for a, b in zip(out.state.weights, state.weights)
    )
    assert moved > 0.0, "one rule did not update any edge"


def test_sustained_inference_stays_bounded_under_adaptive_precision():
    """Many clamp→relax→learn cycles must not blow up (precision stability).

    Regression for the explicit-Euler instability: with a fixed ``eta_mu``
    the raw step ``μ -= eta_mu·∂F/∂μ`` diverges once adaptive precision Π
    sharpens past ``η_μ·Π ≈ 1`` — sustained babbling drove ``cortex_l1`` μ
    to ~900 and then NaN.  The curvature-preconditioned step
    (:func:`core.pc_graph._graph_relax_step`) keeps the effective step ``≈
    eta_mu`` for any Π, so beliefs stay bounded with no leak.  Here a
    persisted-belief loop with under-relaxation (``n_steps=1``) and adaptive
    precision is the stress that used to diverge.
    """
    params, state = init_region_graph(
        jax.random.PRNGKey(0), eta_mu=0.1, eta_w=1e-2, leak=0.0, n_relax=20,
    )
    s_idx = REGION_INDEX["sensory"]
    sens_dim = params.node_sizes[s_idx]

    def cycle(st, key):
        obs = jax.random.normal(key, (sens_dim,))            # fresh clamp each step
        st = pc_graph_step(st, params, {s_idx: obs}, n_steps=1).state
        max_mu = jnp.max(jnp.stack([jnp.max(jnp.abs(m)) for m in st.mu]))
        return st, max_mu

    keys = jax.random.split(jax.random.PRNGKey(1), 2000)
    state, max_mu = eqx.filter_jit(
        lambda st, ks: jax.lax.scan(cycle, st, ks)
    )(state, keys)

    assert jnp.all(jnp.isfinite(max_mu)), "beliefs went non-finite during sustained inference"
    # Pre-fix this reached the hundreds; preconditioned it sits near O(1).
    assert float(max_mu[-1]) < 50.0, f"beliefs diverged: max|mu|={float(max_mu[-1])}"
