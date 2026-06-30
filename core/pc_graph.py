"""Predictive-coding graph — the unified substrate (Faza U, kroki U.2 + U.3).

This is the big-bang of Faza U: **one rule on a shared graph**.  Every
biological region becomes a node (a :class:`~core.pc_module` value
population μ with its own precision Π); every projection becomes a
generative edge with weights ``W``; the *only* plasticity rule anywhere
is the local, curvature-preconditioned Hebbian descent on free energy

    ΔW_(i→j) = η_w · ε_j ⊗ φ(μ_i)

(raw gradient ``−ξ_j ⊗ φ(μ_i)``, ``ξ_j = Π_j ⊙ ε_j``; dividing by the edge
curvature ``∝ Π_j`` cancels the precision so the step is stable for any ``Π``
— the learning twin of the preconditioned μ-step below, same fixed point
``∂F/∂W = 0``; precision still sets the inference metric)

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
    #: Nodes whose precision Π is held fixed (not tracked by the ε² EMA) — the
    #: flat-prior *action* nodes of active inference (:func:`set_action_prior`).
    #: An action variable carries no prior preference and is inferred purely to
    #: satisfy a goal, so its precision must stay flat; without this the first
    #: learning step overwrites the flat prior with the ε² EMA and the action
    #: node becomes a stiff perceptual node that the goal can no longer move.
    fixed_pi_nodes: tuple = eqx.field(static=True)
    #: Nodes whose generative prediction carries a learnable DC ``bias`` term.
    #: Empty by default ⇒ ``bias`` stays zero everywhere and every prediction
    #: is byte-identical to the bias-free model.  Enabled only where the
    #: generative target is non-zero-mean and a node must reconstruct it — the
    #: forward-model output (``sensory``) — *not* abstract/value nodes, where an
    #: always-on baseline would absorb a reward and steal credit from the
    #: intermittent edges that should learn it (the temporal-credit edges), and
    #: *not* a frozen granule layer (``cerebellum``) whose ``bias`` is its fixed
    #: random threshold, not a learnable DC term.
    bias_nodes: tuple = eqx.field(static=True)
    #: Edges whose ``W`` is **never** updated by the learning rule — a frozen
    #: generative projection.  The canonical use is the cerebellar granule
    #: layer ``motor→cerebellum``: in Marr–Albus the mossy→granule expansion is
    #: a *fixed* high-dimensional random non-linearity and only the
    #: granule→Purkinje (``cerebellum→sensory``) synapse is plastic
    #: (:func:`apply_granule_expansion_init`).  Making the deep edge plastic
    #: *and* feeding a free hidden layer is exactly what collapses the forward
    #: model to predict-the-mean (the local rule drives ``ΔW=−η·W·φφᵀ`` →
    #: ``W→0``); freezing the expansion removes the edge that would decay.
    #: Empty by default ⇒ every edge learns, byte-identical to before.
    frozen_edges: tuple = eqx.field(static=True)
    #: Nodes whose belief μ is a **deterministic function of its top-down
    #: prediction** each relaxation step (``μ_j ← Σ_{i→j} W φ(μ_i) + bias_j``)
    #: instead of being freely inferred — an *amortised / feedforward
    #: recognition* node, not an iterative latent.  Error and curvature still
    #: backprop **through** it to its parents by the exact chain rule (it is the
    #: zero-relaxation-time limit of a free node: its effective upstream message
    #: is ``φ'(μ_j)·Σ_{j→c} W_{j→c}ᵀ ξ_c``), so a goal on a descendant still
    #: drives the node's ancestors.  This is the load-bearing half of the
    #: Marr–Albus fix: a frozen ``motor→cerebellum`` edge is *not enough* if the
    #: cerebellum stays a free latent — the over-complete latent reconstructs
    #: the clamped target off the motor manifold and the readout trains at a
    #: different operating point than inference uses.  Making the granule layer
    #: feedforward pins it to ``g(motor)`` during both learning and inversion.
    #: Empty by default ⇒ every node is a free latent, byte-identical to before.
    feedforward_nodes: tuple = eqx.field(static=True)
    #: Edges whose weight step is **normalised by presynaptic feature energy**
    #: (NLMS: ``ΔW = η_w·ε ⊗ φ / (‖φ‖²+δ)``) — the curvature-complete
    #: (natural-gradient) form of the one rule, the input-side twin of the
    #: existing output-side ``÷Π`` preconditioning.  Plain LMS on a fixed rich
    #: basis is stable only for ``η < 2/‖φ‖²``; a dense ``cb_size``-wide granule
    #: readout (``‖φ‖² ~ cb_size``) at the substrate ``η_w`` diverges, so the
    #: granule→Purkinje readout uses NLMS, which is unconditionally stable for
    #: ``η_w < 2`` regardless of feature energy.  The normaliser rescales the
    #: whole edge update by one positive scalar, so the gradient *direction*
    #: (and the ``∂F/∂W=0`` fixed point) is unchanged; only the rate adapts.
    #: Empty by default ⇒ plain LMS everywhere, byte-identical to before.
    nlms_edges: tuple = eqx.field(static=True)

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


def _feedforward_topo_order(feedforward_nodes: tuple, edges: tuple) -> tuple:
    """Topological order (parents before children) over the feedforward nodes.

    A feedforward node is a deterministic function of its top-down prediction,
    so when one feedforward node feeds another the upstream one must be
    evaluated first.  Only edges *between* feedforward nodes constrain the
    order; edges from/to free nodes do not (a free parent's belief is read
    as-is, a free child consumes the result downstream).  Returns the order as
    a tuple of node indices.  Raises if the feedforward sub-graph has a cycle —
    a deterministic node cannot depend (even transitively) on itself.
    """
    ff = set(int(j) for j in feedforward_nodes)
    if not ff:
        return ()
    # Adjacency restricted to the feedforward sub-graph.
    children = {j: set() for j in ff}
    indeg = {j: 0 for j in ff}
    for (src, dst) in edges:
        if src in ff and dst in ff and dst not in children[src]:
            children[src].add(dst)
            indeg[dst] += 1
    # Kahn's algorithm (deterministic: process in ascending index order).
    ready = sorted(j for j in ff if indeg[j] == 0)
    order: list[int] = []
    while ready:
        j = ready.pop(0)
        order.append(j)
        for c in sorted(children[j]):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
        ready.sort()
    if len(order) != len(ff):
        raise ValueError(
            "feedforward nodes form a cycle — a deterministic feedforward node "
            "cannot depend on itself (directly or transitively)"
        )
    return tuple(order)


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
    fixed_pi_nodes: tuple[int, ...] = (),
    bias_nodes: tuple[int, ...] = (),
    frozen_edges: tuple[int, ...] = (),
    feedforward_nodes: tuple[int, ...] = (),
    nlms_edges: tuple[int, ...] = (),
) -> PCGraphParams:
    sizes = tuple(int(s) for s in node_sizes)
    edges = tuple((int(a), int(b)) for (a, b) in edges)
    dyn_edges = tuple((int(a), int(b)) for (a, b) in dyn_edges)
    fixed_pi_nodes = tuple(sorted(int(j) for j in fixed_pi_nodes))
    bias_nodes = tuple(sorted(int(j) for j in bias_nodes))
    frozen_edges = tuple(sorted(int(e) for e in frozen_edges))
    feedforward_nodes = tuple(sorted(int(j) for j in feedforward_nodes))
    nlms_edges = tuple(sorted(int(e) for e in nlms_edges))
    for (a, b) in edges + dyn_edges:
        if not (0 <= a < len(sizes) and 0 <= b < len(sizes)):
            raise ValueError(f"edge ({a},{b}) out of range for {len(sizes)} nodes")
    for e in frozen_edges + nlms_edges:
        if not (0 <= e < len(edges)):
            raise ValueError(f"edge index {e} out of range for {len(edges)} edges")
    for j in feedforward_nodes:
        if not (0 <= j < len(sizes)):
            raise ValueError(f"feedforward node {j} out of range for {len(sizes)} nodes")
    # A feedforward node is set to its top-down prediction, so it must have a
    # source: a feedforward node with no incoming spatial edge would be pinned
    # to its bias alone (a constant), which is never the intent.
    _ff = set(feedforward_nodes)
    _have_parent = {dst for (_s, dst) in edges}
    orphan_ff = sorted(_ff - _have_parent)
    if orphan_ff:
        raise ValueError(
            f"feedforward node(s) {orphan_ff} have no incoming spatial edge to "
            f"predict them — a feedforward node needs a generative parent"
        )
    # Compute a topological order over the feedforward sub-graph (parents
    # before children) up front so the relaxation never re-derives it; a cycle
    # among feedforward nodes makes the deterministic recursion ill-defined.
    _feedforward_topo_order(feedforward_nodes, edges)
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
        fixed_pi_nodes=fixed_pi_nodes, bias_nodes=bias_nodes,
        frozen_edges=frozen_edges, feedforward_nodes=feedforward_nodes,
        nlms_edges=nlms_edges,
    )


class PCGraphState(eqx.Module):
    """Dynamic graph state: node beliefs μ, edge weights, node precision.

    ``bias`` holds one vector per node — the generative prediction's DC term:
    a node's top-down prediction is ``Σ_{i→j} W φ(μ_i) + bias_j``.  It is the
    weight of an implicit always-on unit (``φ ≡ 1``) and learns by the same
    one rule (``Δbias_j = η ε_j``); without it the generative model is forced
    through the origin and cannot fit a non-zero-mean target (e.g. a
    population code, whose activations are non-negative).  Initialised to 0,
    so a freshly built graph is byte-identical to the bias-free model until it
    learns.

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
    bias: tuple            # (n_nodes) arrays, bias[j] shape (node_sizes[j],)
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
    bias = tuple(jnp.zeros(s, dtype) for s in sizes)
    pi = tuple(jnp.ones(s, dtype) for s in sizes)
    pe_var = tuple(jnp.ones(s, dtype) for s in sizes)
    pe_mean = tuple(jnp.zeros(s, dtype) for s in sizes)
    mu_prev = tuple(jnp.zeros(s, dtype) for s in sizes)
    elig = tuple(jnp.zeros(sizes[src], dtype) for (src, _dst) in params.dyn_edges)
    return PCGraphState(
        mu=mu, weights=tuple(weights), bias=bias, pi=pi, pe_var=pe_var, pe_mean=pe_mean,
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
# Marr–Albus granule expansion — fixed random init of a frozen edge
# =====================================================================


def apply_granule_expansion_init(
    state: PCGraphState, params: PCGraphParams,
    src_idx: int, dst_idx: int, key: PRNGKey, *, gain: float = 1.3,
) -> PCGraphState:
    """Seed ``src→dst`` as a fixed random Marr–Albus granule expansion.

    Marr–Albus (Marr 1969; Albus 1971): the mossy-fibre → granule projection
    is a **fixed, high-dimensional, random non-linear expansion** of its input;
    only the downstream granule → Purkinje synapse is plastic.  This init makes
    the ``motor → cerebellum`` edge that expansion — a random ``(dst, src)``
    weight plus a random per-unit threshold (the destination node's ``bias``,
    the granule rheobase diversity).  Pair it with ``frozen_edges`` (the edge
    never learns) and ``feedforward_nodes`` (the granule layer is a
    deterministic function of its input, not a free latent); the bias must then
    *not* be in ``bias_nodes`` (it is the fixed threshold, not a learnable DC
    term).

    ``gain`` is the random-feature / ELM scale: large enough that the granule
    pre-activations span the informative range of their ``tanh`` non-linearity
    (pre-activation std ≈ 1 over the command distribution) — a *representational*
    criterion, not tuned to any downstream task.  A fixed expansion + a plastic
    readout has **no deep plastic edge to collapse**, so the one rule learns the
    forward model without decaying it to predict-the-mean.
    """
    e = _edge_index(params.edges, int(src_idx), int(dst_idx))
    n_out, n_in = state.weights[e].shape
    kw, kb = split_key(key, 2)
    g = jnp.asarray(gain, DTYPE)
    W = (jax.random.normal(kw, (n_out, n_in), DTYPE) * g).astype(DTYPE)
    b = (jax.random.normal(kb, (n_out,), DTYPE) * g).astype(DTYPE)
    new_weights = list(state.weights)
    new_weights[e] = W
    new_bias = list(state.bias)
    new_bias[int(dst_idx)] = b
    return eqx.tree_at(
        lambda s: (s.weights, s.bias), state,
        (tuple(new_weights), tuple(new_bias)),
    )


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
    """Top-down prediction of each node = Σ incoming spatial+temporal edges + bias."""
    act = params.act
    preds = [jnp.zeros(n, DTYPE) for n in params.node_sizes]
    for e, (src, dst) in enumerate(params.edges):
        preds[dst] = preds[dst] + state.weights[e] @ _phi(act, state.mu[src])
    temporal = _temporal_predictions(state.mu_prev, state.w_dyn, params)
    return tuple(preds[j] + temporal[j] + state.bias[j] for j in range(params.n_nodes))


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
    incoming: tuple, outgoing: tuple, hold: tuple, temporal: tuple,
    ff_set: frozenset, ff_fwd_order: tuple, ff_rev_order: tuple,
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

    ``temporal[j]`` is node ``j``'s constant additive drive across the sweep
    — the temporal-edge prediction from the previous cycle plus the node bias
    (an always-on unit).  It enters ``ε_j`` like any incoming prediction but,
    having a constant source, contributes no up-term and no curvature.

    **Feedforward nodes** (``ff_set``) are not freely inferred: each is set to
    its own top-down prediction ``μ_j ← Σ_{i→j} W φ(μ_i) + temporal_j``
    (deterministic recognition activity), in ``ff_fwd_order`` so a feedforward
    node sees already-updated feedforward parents.  Such a node has ``ε_j ≡ 0``
    and so contributes nothing to its own free energy, but it still relays the
    chain-rule message of its children to its parents: in ``ff_rev_order``
    (children first) its *effective* upstream error and curvature are

        ξ_eff_j   = φ'(μ_j) · Σ_{j→c} W_(j→c)ᵀ ξ_c
        curv_eff_j = φ'(μ_j)² · Σ_{j→c} (W_(j→c)²)ᵀ curv_c

    — exactly the values a *free* node would settle to (the zero-relaxation
    limit), so error and curvature backprop through it without it absorbing the
    goal as a free latent.  This is what lets a frozen, feedforward Marr–Albus
    granule layer carry a sensory goal back to the motor command: the command's
    inference curvature ``L_motor`` stays strictly positive (through the
    granule's ``curv_eff``) even with a flat action prior ``Π_motor = 0``.
    """
    act = params.act
    N = params.n_nodes
    tiny = jnp.asarray(jnp.finfo(DTYPE).tiny, DTYPE)
    mu = list(mu)

    def _spatial_pred(mu_: list, j: int) -> Array:
        acc = jnp.zeros(params.node_sizes[j], DTYPE)
        for e in incoming[j]:
            acc = acc + weights[e] @ _phi(act, mu_[params.edges[e][0]])
        return acc

    # Feedforward forward pass: pin each feedforward node to its prediction
    # (parents before children); held dims keep their clamped value.
    for j in ff_fwd_order:
        pred_j = _spatial_pred(mu, j) + temporal[j]
        mu[j] = jnp.where(hold[j], mu[j], pred_j)

    # Predictions + own errors for every node from the (feedforward-updated) μ.
    preds = [jnp.zeros(n, DTYPE) for n in params.node_sizes]
    for e, (src, dst) in enumerate(params.edges):
        preds[dst] = preds[dst] + weights[e] @ _phi(act, mu[src])
    pred_full = [preds[j] + temporal[j] for j in range(N)]
    phip = [_phi_prime(act, mu[j]) for j in range(N)]

    # Per-node upstream messages: free nodes send ξ = Π·ε and curvature Π;
    # feedforward nodes (ε ≡ 0) relay their children's chain-rule message.
    xi: list = [None] * N
    curv: list = [None] * N
    for j in range(N):
        if j in ff_set:
            continue
        xi[j] = pi[j] * (mu[j] - pred_full[j])
        curv[j] = pi[j]
    for j in ff_rev_order:                       # children first
        acc = jnp.zeros_like(mu[j])
        cacc = jnp.zeros_like(mu[j])
        for e in outgoing[j]:
            dst = params.edges[e][1]
            acc = acc + weights[e].T @ xi[dst]
            cacc = cacc + (weights[e] ** 2).T @ curv[dst]
        xi[j] = phip[j] * acc
        curv[j] = phip[j] ** 2 * cacc

    # Update free nodes by the curvature-preconditioned gradient step.
    new_mu = list(mu)
    for j in range(N):
        if j in ff_set:
            continue                             # already set to its prediction
        g = xi[j]
        L = pi[j]
        # source term: this node predicts each of its children — the error
        # propagates up (g) and the child curvature adds to L (a feedforward
        # child contributes its relayed curv_eff, a free child its Π).
        if outgoing[j]:
            acc = jnp.zeros_like(mu[j])
            cu = jnp.zeros_like(mu[j])
            for e in outgoing[j]:
                dst = params.edges[e][1]
                acc = acc + weights[e].T @ xi[dst]
                cu = cu + (weights[e] ** 2).T @ curv[dst]
            g = g - phip[j] * acc
            L = L + phip[j] ** 2 * cu
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
    incoming = _incoming(params.edges, params.n_nodes)
    outgoing = _outgoing(params.edges, params.n_nodes)
    ff_set = frozenset(params.feedforward_nodes)
    ff_fwd_order = _feedforward_topo_order(params.feedforward_nodes, params.edges)
    ff_rev_order = tuple(reversed(ff_fwd_order))
    weights, pi = state.weights, state.pi
    # Constant additive drive on each node across the sweep: the temporal-edge
    # prediction (μ_prev frozen) plus the node bias (an always-on unit).  Both
    # enter ε like an incoming prediction but, having a constant source,
    # contribute no up-term and no curvature.
    temporal = _temporal_predictions(state.mu_prev, state.w_dyn, params)
    temporal = tuple(temporal[j] + state.bias[j] for j in range(params.n_nodes))

    def body(_, mu):
        return _graph_relax_step(
            mu, weights, pi, params, incoming, outgoing, hold, temporal,
            ff_set, ff_fwd_order, ff_rev_order,
        )

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
    """One Hebbian step ``ΔW_(i→j) = η_w ε_j ⊗ φ(μ_i)`` on every edge.

    The single plasticity rule of the whole brain, **curvature-preconditioned**
    — the learning analogue of the preconditioned μ-step in
    :func:`_graph_relax_step`.  The raw free-energy gradient is
    ``∂F/∂W_(i→j) = −ξ_j ⊗ φ(μ_i)`` with ``ξ_j = Π_j ε_j``; descending it with
    a fixed ``η_w`` is **explicit Euler**, unstable once ``Π_j`` sharpens
    (``Π → 1/var_floor``), so the precision-weighted error of a well-fit node
    keeps driving its incoming weights even as ε → 0 — ``|W|`` then drifts
    without bound (worst on a recurrent loop).  Dividing the step by the
    edge's free-energy curvature ``∂²F/∂W² ∝ Π_j`` cancels that factor and
    leaves ``ΔW = η_w ε_j ⊗ φ(μ_i)``: the effective step is ``≈ η_w`` for any
    ``Π``, exactly as the μ-step's division by ``L_j`` makes the belief step
    ``≈ η_μ``.  It is the same preconditioner, not a new rule — the stationary
    point ``∂F/∂W = 0`` (hence the learned fixed point) is unchanged, and
    precision still sets the *inference* metric (which μ, hence which ε, is
    learned).  The per-node ``bias`` learns by the same rule with a constant
    ``φ ≡ 1`` presynaptic factor.  Precision per node still tracks the EMA of
    ε² for inference (Friston 2010), except for ``fixed_pi_nodes`` (flat-prior
    action nodes) whose precision is held.  Call after :func:`pc_graph_relax`
    so μ sits at the inference equilibrium.
    """
    act = params.act
    N, E = params.n_nodes, params.n_edges
    ff_set = frozenset(params.feedforward_nodes)
    frozen = frozenset(params.frozen_edges)
    nlms = frozenset(params.nlms_edges)
    # NLMS regulariser δ: one unit of feature energy, guarding the divide for a
    # (near-)silent presynaptic population.  Negligible against a dense granule
    # readout (‖φ‖² ~ cb_size) yet keeps the step finite when ‖φ‖² → 0.
    nlms_delta = jnp.asarray(1.0, DTYPE)
    preds = pc_graph_predictions(state, params)
    eps = [state.mu[j] - preds[j] for j in range(N)]

    new_weights = list(state.weights)
    for e, (src, dst) in enumerate(params.edges):
        if e in frozen:
            continue                             # frozen projection (e.g. granule)
        phi_src = _phi(act, state.mu[src])
        step = params.eta_w * jnp.outer(eps[dst], phi_src)
        if e in nlms:
            # Natural-gradient (NLMS) step: normalise by presynaptic energy so
            # a fixed η_w is stable for any feature scale.  One positive scalar
            # ⇒ the gradient direction (and ∂F/∂W=0 fixed point) is unchanged.
            step = step / (jnp.sum(phi_src ** 2) + nlms_delta)
        new_weights[e] = state.weights[e] + step

    # Per-node generative bias = weight of an always-on (φ ≡ 1) unit, same
    # rule, only on the opted-in ``bias_nodes`` (elsewhere bias stays zero).
    new_bias = [
        state.bias[j] + params.eta_w * eps[j] if j in params.bias_nodes
        else state.bias[j]
        for j in range(N)
    ]

    # Temporal edges learn by the *same* (preconditioned) rule — the only
    # difference is the presynaptic factor is the previous cycle's belief
    # φ(μ_prev[src]).  With ``elig_mode`` it becomes a *decaying trace* of
    # φ(μ_prev[src]) (§6 multi-cycle credit, the rate e-prop analogue): the
    # commit ``ΔW = η·ε_dst ⊗ elig`` then credits the edge for a presynaptic
    # event several cycles before the error arrived — the destination's own
    # error ε_dst is the learning signal, so for the value / policy temporal
    # edges that ε *is* the RPE (DA / temporal-difference error itself, no
    # longer scaled by precision), with no separate global modulator.
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
        new_w_dyn[e] = state.w_dyn[e] + params.eta_w * jnp.outer(eps[dst], presyn)

    new_pi = list(state.pi)
    new_pe_var = list(state.pe_var)
    new_pe_mean = list(state.pe_mean)
    if update_precision:
        a = params.pi_alpha
        welford = params.precision_mode == PRECISION_WELFORD
        for j in range(N):
            if j in params.fixed_pi_nodes or j in ff_set:
                # Flat-prior action node: precision held (not learned), so the
                # goal can always move it (see PCGraphParams.fixed_pi_nodes).
                # Feedforward node: ε ≡ 0 by construction, so there is no error
                # stream to track — its precision stays at init (and is unused,
                # since a feedforward node relays φ'-scaled child messages, not
                # its own Π).
                continue
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
        mu=state.mu, weights=tuple(new_weights), bias=tuple(new_bias),
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
    # Cerebellum = hidden nonlinear layer of the motor→sensory forward model
    # (Marr-Albus granule expansion): wide enough to represent the
    # population-coded reafference as a tanh basis for the linear readout.
    cb_size: int = 64,
    hc_size: int = 16,
    ec_size: int = 32,
    pfc_size: int = 32,
    gabor_foveal_init: FovealGaborInit | None = None,
    temporal_edges: bool = False,
    laminar_cortex: bool = False,
    working_memory: bool = False,
    wm_persistence_gain: float = 0.9,
    marr_albus_cerebellum: bool = True,
    granule_gain: float = 1.3,
    **graph_kwargs,
) -> tuple[PCGraphParams, PCGraphState]:
    """Instantiate every region as a node of a single PC graph (U.2).

    Topology (src predicts dst; generative / top-down):
      cortical hierarchy generates sensory : c1→sensory, c2→c1, c3→c2
      world model is a cortical cause of sensory : c3→wm, wm→sensory
      value & policy read the deep cortical cause : c3→value, c3→policy
      motor predicted by cortex : c3→motor
      forward model = motor → cerebellum → sensory : motor→cb, cb→sensory
        (the cerebellum is the hidden nonlinear layer — Marr-Albus granule
         expansion — that makes the motor→reafference map representable, so
         active inference can invert a preferred sensory outcome to a motor
         command, Adams 2013; a direct linear motor→sensory edge cannot
         generate the population-coded reafference)
      entorhinal convergence feeds the hippocampus : c3→ec, motor→ec, ec→hc
        (EC is a multi-parent hub integrating the deep cortical cause and
         the motor efference, Witter 2007; the hippocampus reads it and
         completes back to cortex)
      hippocampal completion feedback : hc→c1

    The motor→cerebellum→sensory→…→motor loop (closed through the cortical
    generative model) and the multi-parent EC node both exercise the
    arbitrary-topology claim (Salvatori 2022) — relaxation handles them
    without a feedforward order.  Every edge learns by the one rule; there
    are no per-region rules.

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

    ``marr_albus_cerebellum`` (opt-in, **default ``True``**) builds the forward
    model as a true Marr–Albus circuit: the ``motor→cerebellum`` edge is a
    *frozen* random granule expansion (:func:`apply_granule_expansion_init`,
    scaled by ``granule_gain``), the cerebellum is a *feedforward* deterministic
    layer (``feedforward_nodes``), and only ``cerebellum→sensory`` (Purkinje) is
    plastic (NLMS-stabilised).  This is the fix for the forward-model collapse:
    a plastic deep edge feeding a *free* wide cerebellum decays to predict-the-
    mean under the one rule (the over-complete latent reconstructs the clamped
    reafference off the motor manifold, and ``ΔW_mc ≈ −η·W_mc·φφᵀ`` drives the
    edge to zero).  Set ``False`` for the legacy free-latent cerebellum
    (diagnostics / ablation only).

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
    # With ``laminar_cortex=False`` every out()/gran() is the base node.
    edges = [
        (out("cortex_l1"), R["sensory"]),
        (out("cortex_l2"), gran("cortex_l1")),
        (out("cortex_l3"), gran("cortex_l2")),
        (out("cortex_l3"), R["world_model"]),
        (R["world_model"], R["sensory"]),
        (out("cortex_l3"), R["value"]),
        (out("cortex_l3"), R["policy"]),
        (out("cortex_l3"), R["motor"]),
        # Forward model = motor → cerebellum → sensory: the cerebellum is the
        # *hidden nonlinear layer* (Marr-Albus granule expansion) that makes
        # the motor→reafference map representable.  A single direct
        # motor→sensory edge is linear in φ(motor) and cannot generate the
        # population-coded (nonlinear) reafference, so active inference had
        # nothing accurate to invert; routing through the cerebellum's tanh
        # nonlinearity does.  Generative/top-down: the motor command (cause)
        # predicts the cerebellar state, which predicts the sensory effect
        # (Adams, Shipp & Friston 2013).  No cerebellum→motor edge — that made
        # a recurrent motor↔cerebellum cycle whose precision-weighted weights
        # ran away and gave the goal a second, action-free way to be explained.
        (R["motor"], R["cerebellum"]),
        (R["cerebellum"], R["sensory"]),
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

    edges_t = tuple(edges)
    mc_edge = _edge_index(edges_t, R["motor"], R["cerebellum"])
    cs_edge = _edge_index(edges_t, R["cerebellum"], R["sensory"])

    if marr_albus_cerebellum:
        # Marr–Albus forward model: the motor→cerebellum granule expansion is a
        # FROZEN random non-linearity and the cerebellum is a FEEDFORWARD
        # (deterministic) layer, so there is no deep *plastic* edge into a free
        # hidden layer to collapse to predict-the-mean.  Only the
        # cerebellum→sensory Purkinje readout is plastic, learned on a fixed
        # rich basis (NLMS-stabilised: ‖φ‖² ~ cb_size makes plain LMS diverge).
        # The cerebellum's bias is the fixed random granule threshold, so it is
        # NOT a learnable bias node — only the readout DC term (sensory) is.
        frozen_edges = (mc_edge,)
        feedforward_nodes = (R["cerebellum"],)
        nlms_edges = (cs_edge,)
        bias_nodes = (R["sensory"],)
    else:
        # Legacy free-latent cerebellum with a plastic deep edge (collapses —
        # kept only for diagnostics / ablation).
        frozen_edges = ()
        feedforward_nodes = ()
        nlms_edges = ()
        bias_nodes = (R["sensory"], R["cerebellum"])

    params = init_pc_graph_params(
        tuple(sizes), edges_t, dyn_edges=tuple(dyn_edges),
        # The motor node is the action variable of active inference: its flat
        # prior must not be overwritten by precision learning (see
        # PCGraphParams.fixed_pi_nodes / core.pc_active.set_action_prior).
        fixed_pi_nodes=(R["motor"],),
        bias_nodes=bias_nodes,
        frozen_edges=frozen_edges,
        feedforward_nodes=feedforward_nodes,
        nlms_edges=nlms_edges,
        **graph_kwargs,
    )
    k_state, k_granule = split_key(key, 2)
    state = init_pc_graph_state(k_state, params)
    if marr_albus_cerebellum:
        state = apply_granule_expansion_init(
            state, params, R["motor"], R["cerebellum"], k_granule, gain=granule_gain,
        )
    if gabor_foveal_init is not None:
        state = apply_foveal_gabor_init(
            state, params, out("cortex_l1"), R["sensory"], gabor_foveal_init,
        )
    if working_memory:
        state = apply_wm_persistence_init(state, params, pfc_idx, wm_persistence_gain)
    return params, state
