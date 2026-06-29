"""Hippocampal node group — pattern separation / completion (Faza U, §3).

The ``hippocampus`` node already exists in the region graph
(:data:`core.pc_graph.REGION_INDEX`); the entorhinal hub (``entorhinal``)
is its multi-parent input.  This module gives that node its hippocampal
*function* without adding a second learning rule — it couples the node's
belief μ to a one-shot :class:`~core.pc_memory.EpisodicStore`:

  * **DG pattern separation + one-shot encoding** — the node belief is
    sparsified into a DG code and written to the episodic store, gated by
    the CA1 mismatch (novel input ⇒ high mismatch ⇒ store; Sara 2009,
    McGaughy 2008 mismatch-driven encoding).  One-shot, high-salience:
    the fast hippocampal weights of complementary learning systems
    (McClelland, McNaughton & O'Reilly 1995).

  * **CA3 auto-associative completion** — a partial / noisy belief cue is
    matched in DG space and the stored belief is read back onto the node,
    completing the episode (Treves & Rolls 1994).

  * **CA1 comparator = the node's own prediction error** — the substrate
    already computes ε at every node; the hippocampal mismatch is just
    ``mean|ε|`` at the ``hippocampus`` node, no extra machinery.  It is
    both the encoding gate and a novelty read-out.

The store is auto-associative (key width = value width = the node size),
so completion operates directly in the node's belief space.  Temporal
sequence prediction (legacy CA3 transition matrix) is deferred to the
§6 temporal-credit milestone, where it becomes a one-rule dynamic edge
rather than a second plasticity rule.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey
from .free_energy import variational_free_energy
from .pc_graph import PCGraphParams, PCGraphState, pc_graph_errors, REGION_INDEX
from .pc_memory import (
    EpisodicParams, EpisodicState,
    init_episodic_params, init_episodic_state,
    episodic_store, episodic_recall, StoreOutput, RecallOutput,
)


class HippocampusParams(eqx.Module):
    """Episodic-store params + the graph index of the hippocampus node."""

    episodic: EpisodicParams
    hc_idx: int = eqx.field(static=True)


def init_hippocampus(
    key: PRNGKey,
    hc_size: int,
    *,
    hc_idx: int | None = None,
    capacity: int = 256,
    dg_expansion_factor: int = 5,
    dg_sparsity: float = 0.05,
    similarity_thresh: float = 0.85,
    mismatch_gate: float = 0.3,
) -> tuple[HippocampusParams, EpisodicState]:
    """Build the hippocampal group bound to the ``hippocampus`` graph node.

    ``hc_size`` must equal the node's width (auto-associative store).
    ``mismatch_gate`` is the CA1-mismatch threshold above which a belief
    is committed one-shot (the novelty gate).
    """
    idx = REGION_INDEX["hippocampus"] if hc_idx is None else int(hc_idx)
    ep = init_episodic_params(
        key_dim=int(hc_size), value_dim=int(hc_size),
        capacity=int(capacity),
        dg_expansion_factor=int(dg_expansion_factor),
        dg_sparsity=float(dg_sparsity),
        similarity_thresh=float(similarity_thresh),
        gate_thresh=float(mismatch_gate),
    )
    store = init_episodic_state(key, ep)
    return HippocampusParams(episodic=ep, hc_idx=idx), store


def hippocampus_mismatch(
    graph: PCGraphState, gparams: PCGraphParams, params: HippocampusParams,
) -> Array:
    """CA1 comparator: ``mean|ε|`` at the hippocampus node.

    The substrate's prediction error at the node *is* the comparison
    between the top-down episodic expectation and the entorhinal input —
    a novelty signal (McGaughy 2008) and the one-shot encoding gate.
    """
    eps = pc_graph_errors(graph, gparams)
    return jnp.mean(jnp.abs(eps[params.hc_idx]))


def hippocampus_surprise(
    graph: PCGraphState, gparams: PCGraphParams, params: HippocampusParams,
) -> Array:
    """Free energy carried at the hippocampus node (``½ Π·ε²``)."""
    eps = pc_graph_errors(graph, gparams)
    j = params.hc_idx
    return variational_free_energy(graph.pi[j], eps[j])


def hippocampus_encode(
    graph: PCGraphState, gparams: PCGraphParams,
    params: HippocampusParams, store: EpisodicState,
    *,
    gate: Array | float | None = None,
    phase_gate: Array | bool | None = None,
) -> StoreOutput:
    """One-shot encode the current hippocampal belief into the store.

    Stored auto-associatively (key = value = the node belief).  ``gate``
    defaults to the CA1 mismatch so only novel episodes are committed;
    pass an explicit surprise / neuromodulatory scalar to override.

    ``phase_gate`` is the theta encoding window
    (:attr:`core.pc_oscillator.OscillatorOutput.encoding_phase`): when
    given and ``False`` the write is suppressed (the gate is forced below
    threshold), so encoding only happens on the storage phase of theta
    (Hasselmo 2002).  ``None`` ⇒ always eligible (timing-agnostic, the §3
    behaviour).
    """
    belief = graph.mu[params.hc_idx]
    g = hippocampus_mismatch(graph, gparams, params) if gate is None else gate
    if phase_gate is not None:
        # Off-phase ⇒ gate forced to 0 < gate_thresh ⇒ no store (branchless).
        g = jnp.asarray(g, DTYPE) * jnp.asarray(phase_gate, DTYPE)
    return episodic_store(store, params.episodic, belief, belief, g)


class CompletionOutput(NamedTuple):
    graph: PCGraphState          # graph with the hippocampus belief completed
    recall: RecallOutput         # the matched value + its cosine similarity
    completed: Array             # bool scalar — whether the belief was written


def hippocampus_complete(
    graph: PCGraphState, params: HippocampusParams, store: EpisodicState,
    *,
    phase_gate: Array | bool | None = None,
) -> CompletionOutput:
    """Pattern-complete the hippocampus belief from the episodic store.

    Uses the current node belief as the cue, recalls the best DG match,
    and writes it back onto the node when the match clears
    ``similarity_thresh`` (a confident completion); otherwise the belief
    is left untouched.  Completion is a belief edit, not a learning step —
    the one rule still owns all weight changes.

    ``phase_gate`` is the theta retrieval window
    (:attr:`core.pc_oscillator.OscillatorOutput.retrieval_phase`): when
    given and ``False`` no completion is applied, so recall only happens
    on the retrieval phase of theta — opposite to encoding (Hasselmo
    2002).  ``None`` ⇒ always eligible (the §3 behaviour).
    """
    cue = graph.mu[params.hc_idx]
    out = episodic_recall(store, params.episodic, cue)
    completed = out.similarity >= params.episodic.similarity_thresh
    if phase_gate is not None:
        completed = completed & jnp.asarray(phase_gate, bool)
    new_belief = jnp.where(completed, out.value.astype(DTYPE), cue)
    mu = list(graph.mu)
    mu[params.hc_idx] = new_belief
    new_graph = eqx.tree_at(lambda s: s.mu, graph, tuple(mu))
    return CompletionOutput(graph=new_graph, recall=out, completed=completed)
