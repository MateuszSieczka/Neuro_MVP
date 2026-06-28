"""Predictive-coding graph — the unified substrate (Faza U, kroki U.2 + U.3).

This is the big-bang of Faza U: **one rule on a shared graph**.  Every
biological region becomes a node (a :class:`~core.pc_module` value
population μ with its own precision Π); every projection becomes a
generative edge with weights ``W``; the *only* plasticity rule anywhere
is the local, precision-weighted Hebbian descent on free energy

    ΔW_(i→j) = η_w · ξ_j ⊗ φ(μ_i)            ξ_j = Π_j ⊙ ε_j

and the *only* inference is relaxation of the value nodes on the same
free energy

    ε_j   = μ_j − Σ_{i→j} W_(i→j) φ(μ_i)      (sum of incoming predictions)
    ∂F/∂μ_j = ξ_j − φ'(μ_j) ⊙ Σ_{j→c} W_(j→c)ᵀ ξ_c
    μ_j ← μ_j − η_μ ∂F/∂μ_j                   (Incremental PC; Salvatori 2024)

The graph topology is **arbitrary** (Salvatori et al. 2022, "Learning on
Arbitrary Graph Topologies via Predictive Coding"): multi-parent nodes,
skip edges and cycles are all permitted — relaxation does not need a
feedforward order.  Adding a region = adding nodes + edges; the rule does
not change.  This replaces, in one substrate:

  * cortical STDP / anti-Hebb,          * cerebellar Marr-Albus LTD,
  * VTA TD(0),                          * BG actor-critic three-factor,
  * M1 node-perturbation REINFORCE,     * HC one-shot,
  * the hand-coded ``action_brain_cognitive_step`` sequence.

The global objective is one number, ``graph_free_energy`` (= Σ over nodes
of :func:`core.free_energy.variational_free_energy`), which inference and
learning both minimise and which §9.7 diagnoses.

Scope of this module
--------------------
Pure rate-mode graph engine + a biologically-structured region assembly
(:func:`init_region_graph`) that instantiates all the regions above as
nodes of one graph under the one rule.  Wiring this graph to the live
MJX sensory/motor loop (replacing ``action_brain_cognitive_step`` end to
end) is U.3's closure (plan §10 step 8); the legacy ``brain_graph`` stays
on disk until the graph reproduces its capabilities C1–C6.

Single canonical node math lives in :mod:`core.pc_module`; the chain
there is the linear special case of this graph and the two share the
activation helpers and the free-energy definition.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, split_key
from .pc_module import _phi, _phi_prime
from .free_energy import variational_free_energy


# =====================================================================
# Params / state
# =====================================================================


class PCGraphParams(eqx.Module):
    """Static topology + hyper-params of a predictive-coding graph.

    ``node_sizes[j]`` is the dimensionality of node ``j``'s value
    population μ_j.  ``edges[e] = (src, dst)`` means node ``src``
    generates a top-down prediction of node ``dst`` through weight
    ``W_e`` of shape ``(node_sizes[dst], node_sizes[src])``.
    """

    eta_mu: Array
    eta_w: Array
    leak: Array
    pi_alpha: Array
    var_floor: Array

    node_sizes: tuple = eqx.field(static=True)
    edges: tuple = eqx.field(static=True)        # tuple[(src:int, dst:int)]
    act: str = eqx.field(static=True)
    n_relax: int = eqx.field(static=True)

    @property
    def n_nodes(self) -> int:
        return len(self.node_sizes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def _incoming(edges: tuple, n_nodes: int) -> tuple:
    """Per-node list of edge indices that *predict* this node (dst == j)."""
    inc = [[] for _ in range(n_nodes)]
    for e, (_src, dst) in enumerate(edges):
        inc[dst].append(e)
    return tuple(tuple(x) for x in inc)


def _outgoing(edges: tuple, n_nodes: int) -> tuple:
    """Per-node list of edge indices where this node is the source (src == j)."""
    out = [[] for _ in range(n_nodes)]
    for e, (src, _dst) in enumerate(edges):
        out[src].append(e)
    return tuple(tuple(x) for x in out)


def init_pc_graph_params(
    node_sizes: tuple[int, ...],
    edges: tuple[tuple[int, int], ...],
    *,
    act: str = "tanh",
    eta_mu: float = 0.1,
    eta_w: float = 1e-2,
    leak: float = 0.0,
    tau_pi_steps: float = 1000.0,
    var_floor: float = 1e-4,
    n_relax: int = 20,
) -> PCGraphParams:
    sizes = tuple(int(s) for s in node_sizes)
    edges = tuple((int(a), int(b)) for (a, b) in edges)
    for (a, b) in edges:
        if not (0 <= a < len(sizes) and 0 <= b < len(sizes)):
            raise ValueError(f"edge ({a},{b}) out of range for {len(sizes)} nodes")
    f = lambda x: jnp.asarray(x, DTYPE)
    pi_alpha = 1.0 - jnp.exp(-1.0 / jnp.asarray(tau_pi_steps, DTYPE))
    return PCGraphParams(
        eta_mu=f(eta_mu), eta_w=f(eta_w), leak=f(leak),
        pi_alpha=f(pi_alpha), var_floor=f(var_floor),
        node_sizes=sizes, edges=edges, act=act, n_relax=int(n_relax),
    )


class PCGraphState(eqx.Module):
    """Dynamic graph state: node beliefs μ, edge weights, node precision."""

    mu: tuple              # (n_nodes) arrays, mu[j] shape (node_sizes[j],)
    weights: tuple         # (n_edges) arrays, W[e] shape (size[dst], size[src])
    pi: tuple              # (n_nodes) arrays, pi[j] shape (node_sizes[j],)
    pe_var: tuple          # (n_nodes) arrays


def init_pc_graph_state(
    key: PRNGKey, params: PCGraphParams, *, dtype=DTYPE,
) -> PCGraphState:
    """LeCun-scaled generative edges; μ = 0; Π = 1 (uniform prior)."""
    sizes = params.node_sizes
    E = params.n_edges
    keys = split_key(key, max(1, E))
    weights = []
    for e, (src, dst) in enumerate(params.edges):
        n_out, n_in = sizes[dst], sizes[src]
        scale = 1.0 / jnp.sqrt(jnp.asarray(n_in, dtype))
        weights.append(jax.random.normal(keys[e], (n_out, n_in), dtype) * scale)
    mu = tuple(jnp.zeros(s, dtype) for s in sizes)
    pi = tuple(jnp.ones(s, dtype) for s in sizes)
    pe_var = tuple(jnp.ones(s, dtype) for s in sizes)
    return PCGraphState(
        mu=mu, weights=tuple(weights), pi=pi, pe_var=pe_var,
    )


# =====================================================================
# Predictions / errors / free energy
# =====================================================================


def pc_graph_predictions(state: PCGraphState, params: PCGraphParams) -> tuple:
    """Top-down prediction of each node = Σ over incoming edges."""
    act = params.act
    preds = [jnp.zeros(n, DTYPE) for n in params.node_sizes]
    for e, (src, dst) in enumerate(params.edges):
        preds[dst] = preds[dst] + state.weights[e] @ _phi(act, state.mu[src])
    return tuple(preds)


def pc_graph_errors(state: PCGraphState, params: PCGraphParams) -> tuple:
    """``ε_j = μ_j − Σ_{i→j} W φ(μ_i)`` for every node."""
    preds = pc_graph_predictions(state, params)
    return tuple(state.mu[j] - preds[j] for j in range(params.n_nodes))


def graph_free_energy(state: PCGraphState, params: PCGraphParams) -> Array:
    """Global free energy ``F = Σ_j ½ Π_j · ε_j²`` — the one objective."""
    eps = pc_graph_errors(state, params)
    total = jnp.asarray(0.0, DTYPE)
    for j in range(params.n_nodes):
        total = total + variational_free_energy(state.pi[j], eps[j])
    return total


# =====================================================================
# Relaxation (inference)
# =====================================================================


def _graph_relax_step(
    mu: tuple, weights: tuple, pi: tuple, params: PCGraphParams,
    outgoing: tuple, clamp_mask: tuple,
) -> tuple:
    """One inference sweep: μ_j ← μ_j − η_μ ∂F/∂μ_j on the free nodes."""
    act = params.act
    N = params.n_nodes
    # Predictions + precision-weighted errors.
    preds = [jnp.zeros(n, DTYPE) for n in params.node_sizes]
    for e, (src, dst) in enumerate(params.edges):
        preds[dst] = preds[dst] + weights[e] @ _phi(act, mu[src])
    eps = [mu[j] - preds[j] for j in range(N)]
    xi = [pi[j] * eps[j] for j in range(N)]

    new_mu = list(mu)
    for j in range(N):
        if clamp_mask[j]:
            continue
        # value term: this node carries its own error ε_j.
        g = xi[j]
        # source term: this node predicts each of its children.
        if outgoing[j]:
            phip = _phi_prime(act, mu[j])
            acc = jnp.zeros_like(mu[j])
            for e in outgoing[j]:
                dst = params.edges[e][1]
                acc = acc + weights[e].T @ xi[dst]
            g = g - phip * acc
        g = g + params.leak * mu[j]
        new_mu[j] = mu[j] - params.eta_mu * g
    return tuple(new_mu)


def pc_graph_relax(
    state: PCGraphState, params: PCGraphParams,
    clamp: tuple[int, ...] = (),
    *,
    n_steps: int | None = None,
) -> PCGraphState:
    """Relax free value nodes; ``clamp`` lists node indices held fixed.

    Clamped nodes are observations (sensory) and, during supervised
    learning, target nodes.  All others are inferred.  Arbitrary
    topology — multi-parent, skip edges and cycles all relax fine.
    """
    steps = params.n_relax if n_steps is None else int(n_steps)
    clamp_set = set(int(c) for c in clamp)
    clamp_mask = tuple(j in clamp_set for j in range(params.n_nodes))
    outgoing = _outgoing(params.edges, params.n_nodes)
    weights, pi = state.weights, state.pi

    def body(_, mu):
        return _graph_relax_step(mu, weights, pi, params, outgoing, clamp_mask)

    mu = jax.lax.fori_loop(0, steps, body, state.mu)
    return eqx.tree_at(lambda s: s.mu, state, mu)


def pc_graph_clamp(state: PCGraphState, values: dict[int, Array]) -> PCGraphState:
    """Set μ of the given node indices (e.g. sensory obs, supervised target)."""
    mu = list(state.mu)
    for idx, val in values.items():
        mu[idx] = val.astype(DTYPE)
    return eqx.tree_at(lambda s: s.mu, state, tuple(mu))


# =====================================================================
# Learning — the single rule, on every edge
# =====================================================================


def pc_graph_learn(
    state: PCGraphState, params: PCGraphParams,
    *,
    update_precision: bool = True,
) -> PCGraphState:
    """One Hebbian step ``ΔW_(i→j) = η_w ξ_j ⊗ φ(μ_i)`` on every edge.

    The single plasticity rule of the whole brain.  Precision per node
    tracks the EMA of ε² (Friston 2010 inverse-variance weighting).
    Call after :func:`pc_graph_relax` so μ sits at the inference
    equilibrium.
    """
    act = params.act
    N, E = params.n_nodes, params.n_edges
    preds = pc_graph_predictions(state, params)
    eps = [state.mu[j] - preds[j] for j in range(N)]
    xi = [state.pi[j] * eps[j] for j in range(N)]

    new_weights = list(state.weights)
    for e, (src, dst) in enumerate(params.edges):
        new_weights[e] = state.weights[e] + params.eta_w * jnp.outer(
            xi[dst], _phi(act, state.mu[src]),
        )

    new_pi = list(state.pi)
    new_pe_var = list(state.pe_var)
    if update_precision:
        a = params.pi_alpha
        for j in range(N):
            pe_var_j = (1.0 - a) * state.pe_var[j] + a * eps[j] ** 2
            new_pe_var[j] = pe_var_j
            new_pi[j] = 1.0 / (pe_var_j + params.var_floor)

    return PCGraphState(
        mu=state.mu, weights=tuple(new_weights),
        pi=tuple(new_pi), pe_var=tuple(new_pe_var),
    )


class PCGraphStepOutput(NamedTuple):
    state: PCGraphState
    free_energy: Array


def pc_graph_step(
    state: PCGraphState, params: PCGraphParams,
    clamp_values: dict[int, Array],
    *,
    n_steps: int | None = None,
    update_precision: bool = True,
) -> PCGraphStepOutput:
    """Clamp → relax → learn: one cognitive cycle on the graph.

    The relaxation order *emerges* from error flow over the edges — there
    is no hand-coded region sequence (the thing U.3 removes).
    """
    clamped = pc_graph_clamp(state, clamp_values)
    relaxed = pc_graph_relax(
        clamped, params, tuple(clamp_values.keys()), n_steps=n_steps,
    )
    fe = graph_free_energy(relaxed, params)
    learned = pc_graph_learn(relaxed, params, update_precision=update_precision)
    return PCGraphStepOutput(state=learned, free_energy=fe)


# =====================================================================
# Region assembly — all biological regions as nodes of ONE graph
# =====================================================================


# Node indices for the canonical region graph.  The macro-architecture
# (which regions, their roles) is the genetic blueprint Faza U keeps
# (plan §4); PC runs ON it, the regions are no longer separate rules.
REGION_NODES = (
    "sensory",      # 0  afferent observation (clamped)
    "cortex_l1",    # 1  early cortex
    "cortex_l2",    # 2  mid cortex
    "cortex_l3",    # 3  deep cortex (abstract cause)
    "world_model",  # 4  generative model of sensory transition
    "value",        # 5  VTA/critic value (temporal-PC node)
    "policy",       # 6  BG policy precision
    "motor",        # 7  M1 desired-proprioception prediction
    "cerebellum",   # 8  forward-model node
    "hippocampus",  # 9  episodic cause
)
REGION_INDEX = {name: i for i, name in enumerate(REGION_NODES)}


def init_region_graph(
    key: PRNGKey,
    *,
    sensory_size: int = 12,
    cortex_size: int = 32,
    wm_size: int = 32,
    motor_size: int = 8,
    value_size: int = 1,
    policy_size: int = 4,
    cb_size: int = 8,
    hc_size: int = 16,
    **graph_kwargs,
) -> tuple[PCGraphParams, PCGraphState]:
    """Instantiate every region as a node of a single PC graph (U.2).

    Topology (src predicts dst; generative / top-down):
      cortical hierarchy generates sensory : c1→sensory, c2→c1, c3→c2
      world model is a cortical cause of sensory : c3→wm, wm→sensory
      value & policy read the deep cortical cause : c3→value, c3→policy
      motor predicted by cortex + cerebellum : c3→motor, cb→motor
      efference copy closes a forward-model loop : motor→cb   (cycle!)
      motor predicts its proprioceptive reafference : motor→sensory
        (the forward model used by active inference, U.5 — preferring a
         sensory outcome infers the motor command, Adams 2013)
      hippocampal loop : c3→hc, hc→c1

    The motor↔cerebellum cycle exercises the arbitrary-topology claim
    (Salvatori 2022) — relaxation handles it without a feedforward order.
    Every edge learns by the one rule; there are no per-region rules.
    """
    R = REGION_INDEX
    sizes = [0] * len(REGION_NODES)
    sizes[R["sensory"]] = sensory_size
    sizes[R["cortex_l1"]] = cortex_size
    sizes[R["cortex_l2"]] = cortex_size
    sizes[R["cortex_l3"]] = cortex_size
    sizes[R["world_model"]] = wm_size
    sizes[R["value"]] = value_size
    sizes[R["policy"]] = policy_size
    sizes[R["motor"]] = motor_size
    sizes[R["cerebellum"]] = cb_size
    sizes[R["hippocampus"]] = hc_size

    edges = (
        (R["cortex_l1"], R["sensory"]),
        (R["cortex_l2"], R["cortex_l1"]),
        (R["cortex_l3"], R["cortex_l2"]),
        (R["cortex_l3"], R["world_model"]),
        (R["world_model"], R["sensory"]),
        (R["cortex_l3"], R["value"]),
        (R["cortex_l3"], R["policy"]),
        (R["cortex_l3"], R["motor"]),
        (R["cerebellum"], R["motor"]),
        (R["motor"], R["cerebellum"]),
        (R["motor"], R["sensory"]),
        (R["cortex_l3"], R["hippocampus"]),
        (R["hippocampus"], R["cortex_l1"]),
    )

    params = init_pc_graph_params(tuple(sizes), edges, **graph_kwargs)
    state = init_pc_graph_state(key, params)
    return params, state
