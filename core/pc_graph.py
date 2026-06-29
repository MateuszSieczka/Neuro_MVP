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
not change.  This single substrate subsumes what used to be separate
per-region rules:

  * cortical STDP / anti-Hebb,          * cerebellar Marr-Albus LTD,
  * VTA TD(0),                          * BG actor-critic three-factor,
  * M1 node-perturbation REINFORCE,     * HC one-shot,
  * a hand-coded region-call sequence.

The global objective is one number, ``graph_free_energy`` (= Σ over nodes
of :func:`core.free_energy.variational_free_energy`), which inference and
learning both minimise and which §9.7 diagnoses.

Scope of this module
--------------------
Pure rate-mode graph engine + a biologically-structured region assembly
(:func:`init_region_graph`) that instantiates all the regions above as
nodes of one graph under the one rule.  :mod:`core.pc_brain` runs it as a
cognitive cycle; a body adapter driving that cycle is the next, external
build (plan §12).

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
from .pc_precision import welford_precision_update


# Node-precision tracking modes for :func:`pc_graph_learn`.
PRECISION_EMA = "ema"            # zero-centred ε² EMA (default, legacy)
PRECISION_WELFORD = "welford"    # mean-centred Welford EMA (§4, richer)


# =====================================================================
# Params / state
# =====================================================================


class PCGraphParams(eqx.Module):
    """Static topology + hyper-params of a predictive-coding graph.

    ``node_sizes[j]`` is the dimensionality of node ``j``'s value
    population μ_j.  ``edges[e] = (src, dst)`` means node ``src``
    generates a top-down prediction of node ``dst`` through weight
    ``W_e`` of shape ``(node_sizes[dst], node_sizes[src])``.

    ``dyn_edges[e] = (src, dst)`` is a **temporal** generative edge
    (§6): identical to a spatial edge except its source is the
    *previous* cycle's belief ``μ_src(t−1)``, so it predicts node ``dst``
    from the past, not the present (dynamic / generalized predictive
    coding, Friston 2008).  It is the substrate of temporal credit —
    a value(t−1)→value(t) edge realises the TD bootstrap and a
    world_model self-edge realises the sensory-transition model — and it
    learns by the *same* one rule, with ``φ(μ_src(t−1))`` as the
    presynaptic factor.  Default empty: a static-only graph is unchanged.
    """

    eta_mu: Array
    eta_w: Array
    leak: Array
    pi_alpha: Array
    var_floor: Array
    elig_decay: Array          # eligibility-trace decay λ (per step); inert if off

    node_sizes: tuple = eqx.field(static=True)
    edges: tuple = eqx.field(static=True)        # tuple[(src:int, dst:int)]
    dyn_edges: tuple = eqx.field(static=True)    # tuple[(src:int, dst:int)] temporal
    act: str = eqx.field(static=True)
    n_relax: int = eqx.field(static=True)
    precision_mode: str = eqx.field(static=True)  # PRECISION_EMA | PRECISION_WELFORD
    elig_mode: bool = eqx.field(static=True)      # multi-cycle eligibility on w_dyn edges

    @property
    def n_nodes(self) -> int:
        return len(self.node_sizes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_dyn_edges(self) -> int:
        return len(self.dyn_edges)


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
    dyn_edges: tuple[tuple[int, int], ...] = (),
    act: str = "tanh",
    eta_mu: float = 0.1,
    eta_w: float = 1e-2,
    leak: float = 0.0,
    tau_pi_steps: float = 1000.0,
    var_floor: float = 1e-4,
    n_relax: int = 20,
    precision_mode: str = PRECISION_EMA,
    eligibility: bool = False,
    tau_elig_steps: float = 4.0,
) -> PCGraphParams:
    sizes = tuple(int(s) for s in node_sizes)
    edges = tuple((int(a), int(b)) for (a, b) in edges)
    dyn_edges = tuple((int(a), int(b)) for (a, b) in dyn_edges)
    for (a, b) in edges + dyn_edges:
        if not (0 <= a < len(sizes) and 0 <= b < len(sizes)):
            raise ValueError(f"edge ({a},{b}) out of range for {len(sizes)} nodes")
    if precision_mode not in (PRECISION_EMA, PRECISION_WELFORD):
        raise ValueError(
            f"precision_mode must be {PRECISION_EMA!r} or {PRECISION_WELFORD!r}, "
            f"got {precision_mode!r}"
        )
    if tau_elig_steps <= 0.0:
        raise ValueError(f"tau_elig_steps must be > 0, got {tau_elig_steps}")
    f = lambda x: jnp.asarray(x, DTYPE)
    pi_alpha = 1.0 - jnp.exp(-1.0 / jnp.asarray(tau_pi_steps, DTYPE))
    elig_decay = jnp.exp(-1.0 / jnp.asarray(tau_elig_steps, DTYPE))
    return PCGraphParams(
        eta_mu=f(eta_mu), eta_w=f(eta_w), leak=f(leak),
        pi_alpha=f(pi_alpha), var_floor=f(var_floor), elig_decay=f(elig_decay),
        node_sizes=sizes, edges=edges, dyn_edges=dyn_edges,
        act=act, n_relax=int(n_relax),
        precision_mode=precision_mode, elig_mode=bool(eligibility),
    )


class PCGraphState(eqx.Module):
    """Dynamic graph state: node beliefs μ, edge weights, node precision.

    ``w_dyn`` holds one weight matrix per temporal edge
    (:attr:`PCGraphParams.dyn_edges`); ``mu_prev`` is the 1-cycle delayed
    carry — the previous cycle's relaxed belief that every temporal edge
    reads as its source.  Both are empty / inert when the graph declares
    no temporal edges, so a static-only graph is byte-identical to before.

    ``elig`` holds one eligibility trace per temporal edge — a decaying
    accumulation of the edge's presynaptic factor ``φ(μ_prev[src])`` (§6
    multi-cycle credit, the rate analogue of e-prop; Bellec 2020).  It is
    the *time-extension of the one rule*, not a second mechanism: when
    :attr:`PCGraphParams.elig_mode` is on, a temporal edge commits
    ``ΔW = η·ξ_dst ⊗ elig`` instead of the 1-cycle ``ΔW = η·ξ_dst ⊗
    φ(μ_prev[src])`` — so a reward/error arriving several cycles after the
    presynaptic event still credits the edge (a 1-cycle carry cannot).
    Zeroed and untouched when ``elig_mode`` is off, so the §5b temporal
    path is byte-identical.
    """

    mu: tuple              # (n_nodes) arrays, mu[j] shape (node_sizes[j],)
    weights: tuple         # (n_edges) arrays, W[e] shape (size[dst], size[src])
    pi: tuple              # (n_nodes) arrays, pi[j] shape (node_sizes[j],)
    pe_var: tuple          # (n_nodes) arrays — EMA of ε² (Π = 1/(pe_var+floor))
    pe_mean: tuple         # (n_nodes) arrays — EMA of ε (Welford mode only; else 0)
    w_dyn: tuple           # (n_dyn_edges) arrays, W[e] shape (size[dst], size[src])
    mu_prev: tuple         # (n_nodes) arrays — previous cycle's μ (temporal source)
    elig: tuple            # (n_dyn_edges) arrays, elig[e] shape (size[src],)


def init_pc_graph_state(
    key: PRNGKey, params: PCGraphParams, *, dtype=DTYPE,
) -> PCGraphState:
    """LeCun-scaled generative edges; μ = 0; Π = 1 (uniform prior)."""
    sizes = params.node_sizes
    E, D = params.n_edges, params.n_dyn_edges
    keys = split_key(key, max(1, E + D))

    def _lecun(k, src, dst):
        n_out, n_in = sizes[dst], sizes[src]
        scale = 1.0 / jnp.sqrt(jnp.asarray(n_in, dtype))
        return jax.random.normal(k, (n_out, n_in), dtype) * scale

    weights = [_lecun(keys[e], src, dst) for e, (src, dst) in enumerate(params.edges)]
    w_dyn = [_lecun(keys[E + e], src, dst) for e, (src, dst) in enumerate(params.dyn_edges)]
    mu = tuple(jnp.zeros(s, dtype) for s in sizes)
    pi = tuple(jnp.ones(s, dtype) for s in sizes)
    pe_var = tuple(jnp.ones(s, dtype) for s in sizes)
    pe_mean = tuple(jnp.zeros(s, dtype) for s in sizes)
    mu_prev = tuple(jnp.zeros(s, dtype) for s in sizes)
    elig = tuple(jnp.zeros(sizes[src], dtype) for (src, _dst) in params.dyn_edges)
    return PCGraphState(
        mu=mu, weights=tuple(weights), pi=pi, pe_var=pe_var, pe_mean=pe_mean,
        w_dyn=tuple(w_dyn), mu_prev=mu_prev, elig=elig,
    )


# =====================================================================
# Foveal Gabor prior — opt-in init of a cortex→sensory generative edge
# =====================================================================


class FovealGaborInit(NamedTuple):
    """Geometry for seeding a generative edge with a foveal Gabor prior.

    Door 2 of the integration contract (LEGACY_INTEGRATION.md §2): the
    first cortical edge ``cortex_l1→sensory`` is given the classical V1
    simple-cell prior — orientation- and spatial-frequency-tuned Gabors
    (Hubel & Wiesel 1962; Jones & Palmer 1987) — instead of a flat random
    init.  In the *generative* direction each **column** of the weight is
    one cortical unit's projective field onto the sensory vector; placing a
    Gabor in the foveal ON/OFF sub-blocks makes a unit predict an oriented
    edge at the centre of gaze, which natural-image statistics then refine
    under the one rule (Olshausen & Field 1996 — a prior, not a fixed
    feature).  Carries plain geometry only, so :mod:`core` never imports
    :mod:`sensory`; the vision adapter builds it from its ``RetinaConfig``.

    Attributes
    ----------
    patch_size:
        Side ``P`` of the square foveal patch (``fovea_size``); the ON and
        OFF blocks are each ``P·P`` rows of the sensory vector.
    on_offset, off_offset:
        Start indices of the fovea-ON / fovea-OFF blocks in the flat
        sensory vector (the layout of
        :meth:`sensory.retina.RetinalSample.as_afferent`).
    n_orientations, n_sf, sf_min, sf_max:
        Gabor bank tiling of orientation × spatial-frequency space.
    mix:
        Blend in ``[0, 1]`` of the Gabor prior over the existing
        LeCun-random weights on the foveal rows (``1`` = pure Gabor, ``0``
        = unchanged); the default keeps some random variation so adjacent
        units are not exactly tied (Ringach 2002).
    """

    patch_size: int
    on_offset: int
    off_offset: int
    n_orientations: int = 8
    n_sf: int = 4
    sf_min: float = 1.0
    sf_max: float = 6.0
    mix: float = 0.7


def _gabor_patch(
    size: int, theta: float, sf: float, phase: float, sigma: float,
) -> Array:
    """One zero-mean, unit-L2 ``(size, size)`` Gabor (Jones & Palmer 1987).

    ``theta`` orientation (rad), ``sf`` spatial frequency (cycles/patch),
    ``phase`` carrier phase, ``sigma`` envelope width (px).
    """
    yy, xx = jnp.mgrid[0:size, 0:size].astype(DTYPE)
    c = (size - 1) / 2.0
    x, y = xx - c, yy - c
    x_rot = jnp.cos(theta) * x + jnp.sin(theta) * y
    y_rot = -jnp.sin(theta) * x + jnp.cos(theta) * y
    envelope = jnp.exp(-(x_rot ** 2 + y_rot ** 2) / (2.0 * sigma ** 2))
    carrier = jnp.cos(2.0 * jnp.pi * sf * x_rot / size + phase)
    g = envelope * carrier
    g = g - g.mean()                                   # zero-mean
    norm = jnp.sqrt((g * g).sum()) + jnp.asarray(1e-6, DTYPE)
    return (g / norm).astype(DTYPE)


def _foveal_gabor_bank(n_filters: int, cfg: FovealGaborInit) -> Array:
    """``(n_filters, P, P)`` Gabor bank tiling orientation × SF (cycled).

    The first ``2·n_orientations·n_sf`` filters tile orientation × SF ×
    {even, odd} phase; any extra filters cycle the same bank (Ringach 2002
    replication-with-jitter), one filter per cortical unit (column).
    """
    P = cfg.patch_size
    sigma = P / 4.0                                    # envelope ~ half patch
    phases = (0.0, float(jnp.pi) / 2.0)                # even / odd Gabors
    sfs = jnp.linspace(cfg.sf_min, cfg.sf_max, cfg.n_sf)
    thetas = jnp.linspace(0.0, float(jnp.pi), cfg.n_orientations, endpoint=False)
    base = [
        _gabor_patch(P, float(theta), float(sf), phase, float(sigma))
        for theta in thetas for sf in sfs for phase in phases
    ]
    base_bank = jnp.stack(base, axis=0)                # (B, P, P)
    idx = jnp.arange(n_filters) % base_bank.shape[0]
    return base_bank[idx]


def _edge_index(edges: tuple, src: int, dst: int) -> int:
    for e, (s, d) in enumerate(edges):
        if s == src and d == dst:
            return e
    raise ValueError(f"no edge ({src}→{dst}) in graph to seed with a Gabor prior")


def apply_foveal_gabor_init(
    state: PCGraphState, params: PCGraphParams,
    src_idx: int, dst_idx: int, cfg: FovealGaborInit,
) -> PCGraphState:
    """Seed the ``src→dst`` generative edge's foveal rows with a Gabor prior.

    The edge weight ``W`` has shape ``(dst_dim, src_dim)``; its columns are
    the source units' projective fields onto the destination (sensory)
    node.  The foveal ON block gets the half-rectified ``+`` part of each
    column's Gabor, the OFF block the ``−`` part — so a centre-of-gaze edge
    drives a unit through both ON and OFF afferents.  Gabor magnitudes are
    matched to the LeCun column scale so the rheobase-free drive of the
    default init is preserved (LeCun init stays the baseline; this only
    *shapes* the foveal rows).  Peripheral / motion rows are untouched.
    """
    e = _edge_index(params.edges, int(src_idx), int(dst_idx))
    W = state.weights[e]
    n_out, n_in = W.shape
    P2 = cfg.patch_size * cfg.patch_size
    on0, off0 = int(cfg.on_offset), int(cfg.off_offset)
    if max(on0, off0) + P2 > n_out:
        raise ValueError(
            f"foveal blocks [{on0}:{on0 + P2}], [{off0}:{off0 + P2}] exceed "
            f"the {n_out}-d destination node"
        )

    bank = _foveal_gabor_bank(n_in, cfg)               # (n_in, P, P)
    g = bank.reshape(n_in, P2)                          # one filter per column
    on_w = jnp.clip(g, 0.0, None).T                    # (P², n_in)
    off_w = jnp.clip(-g, 0.0, None).T
    # Match the half-rectified Gabor mean to the LeCun column init mean
    # (half-normal mean = scale·√(2/π)), preserving drive strength.
    lecun = 1.0 / jnp.sqrt(jnp.asarray(n_in, DTYPE))
    target_mean = lecun * jnp.sqrt(jnp.asarray(2.0 / jnp.pi, DTYPE))
    eps = jnp.asarray(1e-8, DTYPE)
    on_w = on_w * (target_mean / (on_w.mean() + eps))
    off_w = off_w * (target_mean / (off_w.mean() + eps))

    mix = jnp.asarray(cfg.mix, DTYPE)
    W = W.at[on0:on0 + P2].set(mix * on_w + (1.0 - mix) * W[on0:on0 + P2])
    W = W.at[off0:off0 + P2].set(mix * off_w + (1.0 - mix) * W[off0:off0 + P2])

    new_weights = list(state.weights)
    new_weights[e] = W.astype(DTYPE)
    return eqx.tree_at(lambda s: s.weights, state, tuple(new_weights))


# =====================================================================
# Working-memory persistence — opt-in init of a temporal self-edge
# =====================================================================


def _dyn_edge_index(dyn_edges: tuple, src: int, dst: int) -> int:
    for e, (s, d) in enumerate(dyn_edges):
        if s == src and d == dst:
            return e
    raise ValueError(f"no temporal edge ({src}→{dst}) in graph to seed for persistence")


def apply_wm_persistence_init(
    state: PCGraphState, params: PCGraphParams, node_idx: int, gain: float,
) -> PCGraphState:
    """Seed a node's temporal **self**-edge with ``gain·I`` (working memory).

    Working memory is persistence: a node holding its belief μ across
    cycles when the input is absent (Goldman-Rakic 1995; Compte 2000 bump
    attractor → leaky integrator).  In this substrate that is exactly the
    §5b temporal self-edge ``(node→node)`` (door 2, the ``w_dyn`` primitive)
    initialised to a near-identity transition: with ``gain·I`` the edge
    predicts ``μ(t) ≈ gain·φ(μ(t−1))``, so μ persists (decaying by ``gain``
    per cycle — a leaky integrator, not a runaway attractor; keep
    ``gain ≤ 1``).  No new rule and no new state type — the self-edge still
    learns by the one rule and the only new thing is this named persistence
    gain.  The global ``leak`` supplies the additional forgetting term.
    Off by default; the random LeCun init of :func:`init_pc_graph_state`
    stays the baseline for every other temporal edge.
    """
    e = _dyn_edge_index(params.dyn_edges, int(node_idx), int(node_idx))
    n = params.node_sizes[node_idx]
    W = jnp.asarray(gain, DTYPE) * jnp.eye(n, dtype=DTYPE)
    new_w_dyn = list(state.w_dyn)
    new_w_dyn[e] = W
    return eqx.tree_at(lambda s: s.w_dyn, state, tuple(new_w_dyn))


# =====================================================================
# Predictions / errors / free energy
# =====================================================================


def _temporal_predictions(
    mu_prev: tuple, w_dyn: tuple, params: PCGraphParams,
) -> tuple:
    """Per-node Σ of temporal-edge predictions ``W_dyn @ φ(μ_prev[src])``.

    ``μ_prev`` is frozen for the whole cycle, so each temporal edge is a
    constant additive drive on its destination — a prior from the past
    (Friston 2008 generalized coordinates) computed once.  Returns a
    zeros tuple when the graph declares no temporal edges, so it adds
    nothing to a static-only graph.
    """
    act = params.act
    temporal = [jnp.zeros(n, DTYPE) for n in params.node_sizes]
    for e, (src, dst) in enumerate(params.dyn_edges):
        temporal[dst] = temporal[dst] + w_dyn[e] @ _phi(act, mu_prev[src])
    return tuple(temporal)


def pc_graph_predictions(state: PCGraphState, params: PCGraphParams) -> tuple:
    """Top-down prediction of each node = Σ over incoming spatial+temporal edges."""
    act = params.act
    preds = [jnp.zeros(n, DTYPE) for n in params.node_sizes]
    for e, (src, dst) in enumerate(params.edges):
        preds[dst] = preds[dst] + state.weights[e] @ _phi(act, state.mu[src])
    temporal = _temporal_predictions(state.mu_prev, state.w_dyn, params)
    return tuple(preds[j] + temporal[j] for j in range(params.n_nodes))


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
    outgoing: tuple, hold: tuple, temporal: tuple,
) -> tuple:
    """One inference sweep: ``μ_j ← μ_j − η_μ ∂F/∂μ_j / L_j`` on free dims.

    The raw gradient descent ``μ_j ← μ_j − η_μ ∂F/∂μ_j`` is **explicit
    Euler**, stable only while ``η_μ·Π ≲ 1``.  But precision ``Π`` is
    adaptive and sharpens as the model fits (up to ``1/var_floor``), so a
    fixed ``η_μ`` eventually violates that bound and the relaxation
    diverges — the high-precision error ``ξ = Π·ε`` of one node kicks its
    parents through the cross term faster than the step can integrate.

    The step is therefore **preconditioned by the diagonal of the
    free-energy curvature**

        L_j = ∂²F/∂μ_j² = Π_j + φ'(μ_j)² · Σ_{j→c} (W_(j→c)²)ᵀ Π_c

    (own-error precision + the precision each child contributes back through
    its edge).  Dividing the gradient by ``L_j`` is diagonal-Newton /
    natural-gradient inference: the effective per-dimension step is ``≈ η_μ``
    regardless of how large any ``Π`` grows, so the sweep is
    **unconditionally stable** — when a child precision climbs, the same
    precision appears in ``L_j`` and shrinks the step in step.  It is a
    *preconditioner*, not a new term: it rescales the search direction and
    leaves the stationary point ``∂F/∂μ = 0`` (hence the learned fixed point
    and the PC≡backprop equilibrium) unchanged.  No magic constant — only
    the existing ``η_μ`` and a ``finfo.tiny`` floor guarding the divide.

    ``hold[j]`` is a per-dimension boolean mask of node ``j``: ``True``
    dimensions are observations held fixed (a clamp), ``False`` ones are
    inferred.  Whole-node clamping is the all-``True`` special case; a
    partial mask pins a *subset* of a node's dimensions — the primitive a
    partial preference (goal on some channels) or an occluded observation
    needs.  Held dimensions still contribute their error ε to children, so
    a pinned channel drives the nodes that predict it.

    ``temporal[j]`` is node ``j``'s constant temporal-edge prediction from
    the previous cycle (frozen across the sweep): it enters ``ε_j`` like
    any incoming prediction but, since its source is the carry and not a
    live node, contributes no up-term and no curvature — only the spatial
    ``outgoing`` edges propagate error and curvature to their sources.
    """
    act = params.act
    N = params.n_nodes
    tiny = jnp.asarray(jnp.finfo(DTYPE).tiny, DTYPE)
    # Predictions + precision-weighted errors (spatial now + temporal past).
    preds = [jnp.zeros(n, DTYPE) for n in params.node_sizes]
    for e, (src, dst) in enumerate(params.edges):
        preds[dst] = preds[dst] + weights[e] @ _phi(act, mu[src])
    eps = [mu[j] - preds[j] - temporal[j] for j in range(N)]
    xi = [pi[j] * eps[j] for j in range(N)]

    new_mu = list(mu)
    for j in range(N):
        # value term: this node carries its own error ε_j (curvature Π_j).
        g = xi[j]
        L = pi[j]
        # source term: this node predicts each of its children — the error
        # propagates up (g) and the child precision adds to the curvature (L).
        if outgoing[j]:
            phip = _phi_prime(act, mu[j])
            acc = jnp.zeros_like(mu[j])
            curv = jnp.zeros_like(mu[j])
            for e in outgoing[j]:
                dst = params.edges[e][1]
                acc = acc + weights[e].T @ xi[dst]
                curv = curv + (weights[e] ** 2).T @ pi[dst]
            g = g - phip * acc
            L = L + phip ** 2 * curv
        g = g + params.leak * mu[j]
        L = L + params.leak
        # Diagonal-Newton step: divide by the curvature ⇒ stable for any Π.
        updated = mu[j] - params.eta_mu * g / (L + tiny)
        # Free dimensions descend the gradient; held ones keep their value.
        new_mu[j] = jnp.where(hold[j], mu[j], updated)
    return tuple(new_mu)


def _build_hold(
    params: PCGraphParams,
    clamp: tuple[int, ...],
    clamp_masks: dict[int, Array] | None,
) -> tuple:
    """Per-node boolean hold masks from whole-node + partial clamps.

    ``clamp`` lists nodes pinned in full; ``clamp_masks`` maps a node
    index to a per-dimension boolean mask for partial pinning.  A node may
    appear in only one of the two.
    """
    masks = clamp_masks or {}
    clamp_set = set(int(c) for c in clamp)
    overlap = clamp_set.intersection(int(k) for k in masks)
    if overlap:
        raise ValueError(
            f"node(s) {sorted(overlap)} given both a whole-node clamp and a "
            f"partial clamp mask — use one or the other"
        )
    hold = []
    for j in range(params.n_nodes):
        size = params.node_sizes[j]
        if j in clamp_set:
            hold.append(jnp.ones(size, dtype=bool))
        elif j in masks:
            hold.append(jnp.asarray(masks[j], dtype=bool))
        else:
            hold.append(jnp.zeros(size, dtype=bool))
    return tuple(hold)


def pc_graph_relax(
    state: PCGraphState, params: PCGraphParams,
    clamp: tuple[int, ...] = (),
    *,
    clamp_masks: dict[int, Array] | None = None,
    n_steps: int | None = None,
) -> PCGraphState:
    """Relax free value nodes; pinned dimensions are held fixed.

    ``clamp`` lists node indices held fixed in full (observations and,
    during supervised learning, target nodes).  ``clamp_masks`` pins a
    *subset* of a node's dimensions (a partial preference or partial
    observation): each value is a per-dimension boolean mask, ``True`` =
    held.  All unpinned dimensions are inferred.  Arbitrary topology —
    multi-parent, skip edges and cycles all relax fine.
    """
    steps = params.n_relax if n_steps is None else int(n_steps)
    hold = _build_hold(params, clamp, clamp_masks)
    outgoing = _outgoing(params.edges, params.n_nodes)
    weights, pi = state.weights, state.pi
    # Temporal-edge drive is constant across the sweep (μ_prev is frozen).
    temporal = _temporal_predictions(state.mu_prev, state.w_dyn, params)

    def body(_, mu):
        return _graph_relax_step(mu, weights, pi, params, outgoing, hold, temporal)

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

    # Temporal edges learn by the *same* rule — the only difference is the
    # presynaptic factor is the previous cycle's belief φ(μ_prev[src]).
    # With ``elig_mode`` that presynaptic factor becomes a *decaying trace*
    # of φ(μ_prev[src]) (§6 multi-cycle credit, the rate e-prop analogue):
    # the commit ``ΔW = η·ξ_dst ⊗ elig`` then credits the edge for a
    # presynaptic event several cycles before the error ξ_dst arrived — the
    # destination's own precision-weighted error is the learning signal, so
    # for the value / policy temporal edges that ξ *is* the RPE (DA), with
    # no separate global modulator.  The trace is the one rule extended in
    # time, not a second mechanism.
    new_w_dyn = list(state.w_dyn)
    new_elig = list(state.elig)
    for e, (src, dst) in enumerate(params.dyn_edges):
        phi_prev = _phi(act, state.mu_prev[src])
        if params.elig_mode:
            trace = params.elig_decay * state.elig[e] + phi_prev
            new_elig[e] = trace
            presyn = trace
        else:
            presyn = phi_prev
        new_w_dyn[e] = state.w_dyn[e] + params.eta_w * jnp.outer(xi[dst], presyn)

    new_pi = list(state.pi)
    new_pe_var = list(state.pe_var)
    new_pe_mean = list(state.pe_mean)
    if update_precision:
        a = params.pi_alpha
        welford = params.precision_mode == PRECISION_WELFORD
        for j in range(N):
            if welford:
                # Mean-centred Welford EMA: tracks ε's running mean so a
                # biased error stream is not permanently low-precision.
                new_pe_mean[j], new_pe_var[j], new_pi[j] = welford_precision_update(
                    state.pe_mean[j], state.pe_var[j], eps[j],
                    a, params.var_floor,
                )
            else:
                # Zero-centred ε² EMA (default, unchanged): pe_mean is
                # carried untouched so the two modes share one state shape.
                pe_var_j = (1.0 - a) * state.pe_var[j] + a * eps[j] ** 2
                new_pe_var[j] = pe_var_j
                new_pi[j] = 1.0 / (pe_var_j + params.var_floor)

    return PCGraphState(
        mu=state.mu, weights=tuple(new_weights),
        pi=tuple(new_pi), pe_var=tuple(new_pe_var), pe_mean=tuple(new_pe_mean),
        w_dyn=tuple(new_w_dyn), mu_prev=state.mu_prev, elig=tuple(new_elig),
    )


def pc_graph_roll(state: PCGraphState) -> PCGraphState:
    """Advance the temporal carry: ``μ_prev ← μ`` (call at the cycle boundary).

    Next cycle's temporal edges read this cycle's relaxed belief as their
    source.  Run once per cognitive cycle, *after* relax+learn.  Inert
    (shape-only) when the graph has no temporal edges.
    """
    return eqx.tree_at(lambda s: s.mu_prev, state, state.mu)


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
    is no hand-coded region sequence (the thing U.3 removes).  The temporal
    carry is rolled at the end (``μ_prev ← μ``) so the next cycle's
    temporal edges see this cycle's belief.
    """
    clamped = pc_graph_clamp(state, clamp_values)
    relaxed = pc_graph_relax(
        clamped, params, tuple(clamp_values.keys()), n_steps=n_steps,
    )
    fe = graph_free_energy(relaxed, params)
    learned = pc_graph_learn(relaxed, params, update_precision=update_precision)
    return PCGraphStepOutput(state=pc_graph_roll(learned), free_energy=fe)


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
    "entorhinal",   # 10 EC convergence hub feeding the hippocampus
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
    ec_size: int = 32,
    pfc_size: int = 32,
    gabor_foveal_init: FovealGaborInit | None = None,
    temporal_edges: bool = False,
    laminar_cortex: bool = False,
    working_memory: bool = False,
    wm_persistence_gain: float = 0.9,
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
      entorhinal convergence feeds the hippocampus : c3→ec, motor→ec, ec→hc
        (EC is a multi-parent hub integrating the deep cortical cause and
         the motor efference, Witter 2007; the hippocampus reads it and
         completes back to cortex)
      hippocampal completion feedback : hc→c1

    The motor↔cerebellum cycle and the multi-parent EC node both exercise
    the arbitrary-topology claim (Salvatori 2022) — relaxation handles
    them without a feedforward order.  Every edge learns by the one rule;
    there are no per-region rules.

    ``gabor_foveal_init`` (opt-in, default ``None`` ⇒ unchanged LeCun init)
    seeds the ``cortex_l1→sensory`` generative edge with a foveal Gabor
    prior (:class:`FovealGaborInit`) — the V1 simple-cell prior for vision
    (§2).  Backward-compatible: omitting it reproduces the original graph
    exactly.

    ``temporal_edges`` (opt-in, default ``False`` ⇒ no temporal edges,
    region graph byte-identical) adds the two **temporal** self-edges that
    give the substrate temporal credit (§6):

      * ``value(t−1)→value(t)`` — the TD bootstrap.  The value node's own
        prediction error against this edge *is* the temporal-difference
        error (no separate critic / TD update); its dopaminergic read-out
        feeds :mod:`core.pc_neuromod` (closing the §4 DA-proxy deferral).
      * ``world_model(t−1)→world_model(t)`` — the sensory-transition cause.
        Composed with the existing static ``world_model→sensory`` edge it
        predicts the *next* sensory state, the one-rule replacement for the
        deferred sequence-memory transition rule.

    Both learn by the single rule with ``φ(μ_prev)`` as the presynaptic
    factor; there is no second plasticity mechanism.

    ``laminar_cortex`` (opt-in, default ``False`` ⇒ flat single nodes,
    region graph byte-identical) splits each cortical region into the
    canonical PC microcircuit (Bastos 2012) — three populations wired by
    the one rule:

      * **L2/3 = μ** (the cause) — *kept as the existing* ``cortex_lN``
        index, so every read-out (``cortex_top_idx``, vision, memory) and
        the ``hc→cortex_l1`` / consumer edges still resolve.
      * **L4 = ε** (granular error) — a new appended node with its own Π,
        making error precision separable from the cause's (Feldman &
        Friston 2010 superficial/deep precision).  Predicted by its cause
        ``(L2/3→L4)`` and by the region above's deep output, so its ε is
        the feedforward/feedback mismatch (a comparator).
      * **L5 = prediction** — a new appended node carrying the region's
        descending output; inter-region generative edges re-origin here
        ``(L5_N→L4_{N−1})`` and the deep consumers read ``L5_cortex_l3``.

    Appended in order ``[c1_l4, c1_l5, c2_l4, c2_l5, c3_l4, c3_l5]`` after
    the 11 base nodes, so base indices are unchanged.  Every intra-region
    edge learns by the one rule; relaxation handles the extra populations
    like any other nodes (U.3 arbitrary topology).

    ``working_memory`` (opt-in, default ``False``) appends a ``pfc``
    persistence node and gives it a **leaky temporal self-edge** — the same
    §5b ``w_dyn`` primitive, initialised to ``wm_persistence_gain·I`` so it
    holds its belief μ across cycles when input is absent (bump attractor →
    leaky integrator; Goldman-Rakic 1995).  ``(cortex_l3→pfc)`` feeds it the
    deep cortical cause, ``(pfc→cortex_l3)`` returns the held context.  No
    new rule, no new state type — only the named persistence gain.

    Multi-cycle **eligibility traces** (§6) are enabled with
    ``eligibility=True`` (flows through to :func:`init_pc_graph_params`):
    every temporal edge then commits against a decaying trace of its
    presynaptic factor instead of the 1-cycle value, bridging credit across
    >1 cycle.  The destination node's own precision-weighted error is the
    learning signal, so the value/policy temporal edges are modulated by the
    value-node ε (the RPE / DA) with no separate broadcast.
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
    sizes[R["entorhinal"]] = ec_size

    # Laminar split (§5): append L4/L5 for each cortical region; L2/3 stays
    # the base ``cortex_lN`` index.  Off ⇒ ``lam`` empty and every cortical
    # endpoint below collapses to the single base node (flat, byte-identical).
    CORTICAL = ("cortex_l1", "cortex_l2", "cortex_l3")
    lam: dict[str, tuple[int, int]] = {}
    if laminar_cortex:
        for name in CORTICAL:
            l4 = len(sizes); sizes.append(cortex_size)
            l5 = len(sizes); sizes.append(cortex_size)
            lam[name] = (l4, l5)

    def cause(name: str) -> int:        # L2/3 (μ) — the read-out, always base
        return R[name]

    def gran(name: str) -> int:         # L4 (ε) — granular target of descent
        return lam[name][0] if laminar_cortex else R[name]

    def out(name: str) -> int:          # L5 (prediction) — descending source
        return lam[name][1] if laminar_cortex else R[name]

    # Inter-region generative edges: descend from a region's L5 output into
    # the lower region's L4 granular layer; deep consumers read L5_cortex_l3.
    # With ``laminar_cortex=False`` every out()/gran() is the base node, so
    # this reproduces the original 15-edge graph in the original order.
    edges = [
        (out("cortex_l1"), R["sensory"]),
        (out("cortex_l2"), gran("cortex_l1")),
        (out("cortex_l3"), gran("cortex_l2")),
        (out("cortex_l3"), R["world_model"]),
        (R["world_model"], R["sensory"]),
        (out("cortex_l3"), R["value"]),
        (out("cortex_l3"), R["policy"]),
        (out("cortex_l3"), R["motor"]),
        (R["cerebellum"], R["motor"]),
        (R["motor"], R["cerebellum"]),
        (R["motor"], R["sensory"]),
        (out("cortex_l3"), R["entorhinal"]),
        (R["motor"], R["entorhinal"]),
        (R["entorhinal"], R["hippocampus"]),
        (R["hippocampus"], gran("cortex_l1")),
    ]

    # Intra-region laminar edges (one rule): the cause predicts its granular
    # error population (L2/3→L4, giving L4 a separable Π) and its deep output
    # (L2/3→L5, the descending message).  None when flat.
    if laminar_cortex:
        for name in CORTICAL:
            edges.append((cause(name), gran(name)))
            edges.append((cause(name), out(name)))

    # Temporal self-edges (§6): a node predicting its own next state from the
    # previous cycle — the value TD bootstrap and the world-model
    # sensory-transition cause.  Empty by default (region graph unchanged).
    dyn_edges: list[tuple[int, int]] = []
    if temporal_edges:
        dyn_edges.append((R["value"], R["value"]))
        dyn_edges.append((R["world_model"], R["world_model"]))

    # Working memory (§5): a persistence node with a leaky temporal self-edge
    # (the §5b w_dyn primitive) holding μ across cycles; fed by and feeding
    # back the deep cortical cause.  Appended after any laminar nodes.
    pfc_idx: int | None = None
    if working_memory:
        pfc_idx = len(sizes)
        sizes.append(pfc_size)
        edges.append((out("cortex_l3"), pfc_idx))
        edges.append((pfc_idx, gran("cortex_l3")))
        dyn_edges.append((pfc_idx, pfc_idx))

    params = init_pc_graph_params(
        tuple(sizes), tuple(edges), dyn_edges=tuple(dyn_edges), **graph_kwargs,
    )
    state = init_pc_graph_state(key, params)
    if gabor_foveal_init is not None:
        state = apply_foveal_gabor_init(
            state, params, out("cortex_l1"), R["sensory"], gabor_foveal_init,
        )
    if working_memory:
        state = apply_wm_persistence_init(state, params, pfc_idx, wm_persistence_gain)
    return params, state
