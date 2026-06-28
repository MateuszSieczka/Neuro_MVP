"""Faza U — krok U.4b: structural plasticity (self-wiring) by free energy.

``core.pc_structural`` grows and prunes connectivity under the same
free-energy objective as inference and learning.  Asserts:

* **self-wiring beats frozen sparse**: starting sparse, the graph grows
  connections where they reduce FE and ends with far lower loss than the
  same sparse graph that only adjusts existing weights (plan §9.3 — the
  graph improves FE by changing its own connectivity);
* **pruning** removes synapses too weak to pay their wiring cost;
* **homeostasis** keeps the active density bounded (no runaway growth).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.pc_graph import (
    init_pc_graph_params, init_pc_graph_state,
    pc_graph_clamp, pc_graph_relax,
)
from core.pc_structural import (
    init_sparse_masks, pc_structural_learn, pc_structural_update,
    pc_structural_step, active_count, active_fraction,
)
from core.pc_module import _phi


def _ff_loss(state, inp, target):
    a = state.weights[1] @ _phi("tanh", inp)     # node1 from node2 (edge 2→1)
    a = state.weights[0] @ _phi("tanh", a)       # node0 from node1 (edge 1→0)
    return float(0.5 * jnp.sum((a - target) ** 2))


def _task(seed=0):
    p = init_pc_graph_params(
        (4, 12, 6), ((1, 0), (2, 1)),
        act="tanh", eta_mu=0.1, eta_w=0.05, n_relax=40,
    )
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed), 4)
    s = init_pc_graph_state(k1, p)
    s, masks = init_sparse_masks(s, p, k2, density=0.1)
    inp = jax.random.normal(k3, (6,))
    target = jax.random.normal(k4, (4,)) * 0.5
    return p, s, masks, inp, target


def test_self_wiring_beats_frozen_sparse():
    p, s0, masks0, inp, target = _task(0)
    top = 2

    # Frozen-sparse baseline: only existing synapses adapt.
    sb = s0
    for _ in range(150):
        c = pc_graph_clamp(sb, {top: inp, 0: target})
        r = pc_graph_relax(c, p, clamp=(0, top), n_steps=40)
        sb = pc_structural_learn(r, p, masks0)
    loss_base = _ff_loss(sb, inp, target)

    # Self-wiring: grow connections that reduce free energy.
    sg, mg = s0, masks0
    for _ in range(150):
        out = pc_structural_step(
            sg, p, mg, {top: inp, 0: target},
            n_steps=40, spawn_threshold=0.03, max_active_frac=0.6,
        )
        sg, mg = out.state, out.masks
    loss_grow = _ff_loss(sg, inp, target)

    assert int(active_count(mg)) > int(active_count(masks0)), "graph did not grow"
    assert loss_grow < loss_base * 0.5, (
        f"self-wiring did not beat frozen sparse: {loss_base:.4f} vs {loss_grow:.4f}"
    )


def test_pruning_removes_weak_synapses():
    p, s0, _, inp, target = _task(1)
    top = 2
    dense = tuple(jnp.ones_like(w) for w in s0.weights)

    c = pc_graph_clamp(s0, {top: inp, 0: target})
    relaxed = pc_graph_relax(c, p, clamp=(0, top), n_steps=40)

    # Prune hard, no growth (spawn_threshold huge).
    _, pruned_masks = pc_structural_update(
        relaxed, p, dense,
        prune_threshold=0.2, spawn_threshold=1e9, max_active_frac=1.0,
    )
    full = sum(float(m.size) for m in dense)
    assert int(active_count(pruned_masks)) < full, "pruning removed nothing"


def test_homeostatic_density_cap():
    p, s0, masks0, inp, target = _task(2)
    top = 2
    cap = 0.2
    sg, mg = s0, masks0
    for _ in range(120):
        out = pc_structural_step(
            sg, p, mg, {top: inp, 0: target},
            n_steps=30, spawn_threshold=0.01, max_active_frac=cap,
        )
        sg, mg = out.state, out.masks
    frac = float(active_fraction(mg))
    assert frac < cap + 0.15, f"density cap breached: frac={frac:.3f} (cap {cap})"
