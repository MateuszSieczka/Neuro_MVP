"""Structural plasticity on the PC graph — self-wiring (Faza U, U.4b).

The graph grows and prunes its own connectivity by the *same* free-energy
criterion that drives inference and learning — no separate rule (plan
§U.4b).  A synapse is grown where turning it on would reduce free energy
and pruned where it is too weak to pay its wiring cost; a homeostatic cap
keeps density bounded (the stability the plan flags as essential, §8.3).

Mechanism
---------
Connectivity is a boolean ``mask`` over each edge's weight matrix (the
JIT-safe pre-allocated-budget trick of ``core/sparse.py``, here at the
graph level): inactive synapses are held at 0 and excluded from learning;
"grow" = unmask, "prune" = mask + zero.  Shapes never change, so the same
compiled relaxation runs for the network's life.

* **Grow** where the one-rule gradient is large.  ``ΔW = η·ξ_dst⊗φ(μ_src)``
  is ``−∂F/∂W``; a large ``|ΔW|`` on an *inactive* synapse means switching
  it on descends free energy.  Growth therefore reuses the exact learning
  signal — the structure follows the same objective as everything else.
* **Prune** synapses with ``|W| < prune_threshold`` (wiring cost: a
  synapse too weak to predict is not worth maintaining; Chklovskii 2004,
  Rakic 1988 developmental pruning).
* **Homeostasis**: growth is gated off once an edge's active fraction
  reaches ``max_active_frac`` — hard density cap against runaway growth.

Because masked learning keeps inactive synapses at 0, ``φ(μ)`` and the
predictions are unaffected by dormant slots; the graph is exactly the
sparse network its mask describes.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, split_key
from .pc_module import _phi
from .pc_graph import (
    PCGraphParams, PCGraphState,
    pc_graph_clamp, pc_graph_relax, pc_graph_learn,
    pc_graph_predictions, graph_free_energy,
)


# =====================================================================
# Masks
# =====================================================================


def init_sparse_masks(
    state: PCGraphState, params: PCGraphParams, key: PRNGKey,
    *, density: float = 0.1,
) -> tuple[PCGraphState, tuple]:
    """Mask each edge to a random ``density`` fraction; zero the rest.

    Returns ``(sparsified_state, masks)`` where ``masks[e]`` is a
    float ``{0,1}`` array the shape of edge ``e``'s weight matrix.  Start
    sparse so growth has room (and so the self-wiring test has something
    to wire).
    """
    E = params.n_edges
    keys = split_key(key, max(1, E))
    masks = []
    weights = list(state.weights)
    for e in range(E):
        m = (jax.random.uniform(keys[e], weights[e].shape, DTYPE) < density).astype(DTYPE)
        masks.append(m)
        weights[e] = weights[e] * m
    new_state = eqx.tree_at(lambda s: s.weights, state, tuple(weights))
    return new_state, tuple(masks)


def active_fraction(masks: tuple) -> Array:
    """Overall fraction of active synapses across all edges."""
    tot = sum(float(m.size) for m in masks)
    act = sum(jnp.sum(m) for m in masks)
    return act / jnp.asarray(tot, DTYPE)


def active_count(masks: tuple) -> Array:
    """Total active synapse count across all edges."""
    return sum(jnp.sum(m) for m in masks)


# =====================================================================
# Masked learning
# =====================================================================


def pc_structural_learn(
    state: PCGraphState, params: PCGraphParams, masks: tuple,
    *, update_precision: bool = True,
) -> PCGraphState:
    """One-rule weight update restricted to active synapses.

    Runs :func:`core.pc_graph.pc_graph_learn` then re-applies the mask so
    inactive slots stay at 0 (the dense update would otherwise resurrect
    them, dissolving the structure).
    """
    learned = pc_graph_learn(state, params, update_precision=update_precision)
    weights = tuple(learned.weights[e] * masks[e] for e in range(params.n_edges))
    return eqx.tree_at(lambda s: s.weights, learned, weights)


# =====================================================================
# Grow / prune  (the structural step)
# =====================================================================


def pc_structural_update(
    state: PCGraphState, params: PCGraphParams, masks: tuple,
    *,
    prune_threshold: float = 1e-3,
    spawn_threshold: float = 0.05,
    max_active_frac: float = 0.5,
    init_weight: float = 0.02,
) -> tuple[PCGraphState, tuple]:
    """Prune weak synapses, grow free-energy-reducing ones (homeostatic).

    Operates at the inference equilibrium (call after relaxation), so the
    growth gradient ``ξ_dst⊗φ(μ_src)`` reflects the settled beliefs.
    Returns ``(new_state, new_masks)``.
    """
    act = params.act
    preds = pc_graph_predictions(state, params)
    new_masks = list(masks)
    new_weights = list(state.weights)

    for e, (src, dst) in enumerate(params.edges):
        W = state.weights[e]
        m = masks[e]
        size = int(W.size)                                  # static
        budget = int(max_active_frac * size)                # static, hard cap
        xi = state.pi[dst] * (state.mu[dst] - preds[dst])
        grad = jnp.outer(xi, _phi(act, state.mu[src]))      # −∂F/∂W

        # Prune: active synapses strong enough to pay their wiring cost.
        keep = m * (jnp.abs(W) >= jnp.asarray(prune_threshold, DTYPE))
        # Grow candidates: inactive synapses whose activation reduces FE.
        growable = (1.0 - m) * (
            jnp.abs(grad) > jnp.asarray(spawn_threshold, DTYPE)
        ).astype(DTYPE)

        # Priority score: kept synapses (score ≥ 1) outrank any grow
        # candidate (score ∈ [0, 0.5]) — existing structure is preserved
        # first, growth only fills spare capacity.  Among grow candidates,
        # larger |gradient| (bigger FE reduction) wins.
        gmax = jnp.max(jnp.abs(grad)) + jnp.asarray(1e-6, DTYPE)
        score = keep * (jnp.abs(W) + 1.0) + growable * (0.5 * jnp.abs(grad) / gmax)
        cand = jnp.clip(keep + growable, 0.0, 1.0)

        # Hard homeostatic cap: keep only the top ``budget`` by score.
        flat = jnp.where(cand > 0, score, -jnp.inf).ravel()
        if budget <= 0:
            new_m = jnp.zeros_like(m)
        elif budget >= size:
            new_m = cand
        else:
            cutoff = jnp.sort(flat)[size - budget]
            new_m = (jnp.where(cand > 0, score, -jnp.inf) >= cutoff).astype(DTYPE) * cand

        was_active = m * new_m                               # kept-and-selected
        grown = (1.0 - m) * new_m                            # newly grown
        Wn = W * was_active + jnp.sign(grad) * jnp.asarray(init_weight, DTYPE) * grown
        new_masks[e] = new_m.astype(DTYPE)
        new_weights[e] = Wn.astype(DTYPE)

    new_state = eqx.tree_at(lambda s: s.weights, state, tuple(new_weights))
    return new_state, tuple(new_masks)


class StructuralStepOutput(NamedTuple):
    state: PCGraphState
    masks: tuple
    free_energy: Array
    active: Array          # total active synapse count after the step


def pc_structural_step(
    state: PCGraphState, params: PCGraphParams, masks: tuple,
    clamp_values: dict,
    *,
    n_steps: int | None = None,
    prune_threshold: float = 1e-3,
    spawn_threshold: float = 0.05,
    max_active_frac: float = 0.5,
    init_weight: float = 0.02,
    update_precision: bool = True,
) -> StructuralStepOutput:
    """clamp → relax → masked-learn → grow/prune: one self-wiring cycle."""
    clamped = pc_graph_clamp(state, clamp_values)
    relaxed = pc_graph_relax(
        clamped, params, clamp=tuple(clamp_values.keys()), n_steps=n_steps,
    )
    fe = graph_free_energy(relaxed, params)
    learned = pc_structural_learn(
        relaxed, params, masks, update_precision=update_precision,
    )
    grown, new_masks = pc_structural_update(
        learned, params, masks,
        prune_threshold=prune_threshold, spawn_threshold=spawn_threshold,
        max_active_frac=max_active_frac, init_weight=init_weight,
    )
    return StructuralStepOutput(
        state=grown, masks=new_masks, free_energy=fe, active=active_count(new_masks),
    )
