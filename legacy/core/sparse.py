"""
Sparse, JIT-compatible connectivity with a fixed connection budget.

JAX JIT requires static array shapes.  Structural plasticity that adds
or removes connections at runtime therefore cannot use true variable-NNZ
sparse matrices — every shape change would re-trace the compiled step.

The solution is a *pre-allocated* connection budget: each connection
fabric reserves ``max_connections`` slots (COO pairs).  "Creating" a
synapse means unmasking an already-allocated slot by writing a non-zero
weight; "pruning" means writing zero.  The BCOO array's shape is
invariant, so the same compiled step runs for the life of the network.

Density budgeting:
    Biological cortical connectivity: 5-15 % (Braitenberg & Schüz 1998).
    Default here: 15 % — generous enough for post-hoc growth without
    starving the density-driven pruning rule.

Distance-dependent wiring:
    p_connect(d) = p_local · exp(-d² / 2σ²)       (Hellwig 2000)
    Used when caller supplies a ``positions`` array; otherwise the
    initial mask is uniformly random over the budget.

File layout:
    ``SparseConnectivity`` — ``eqx.Module`` holding BCOO-backed weights
        plus a boolean ``active`` mask (redundant with ``weights != 0``
        but allows tracking pruning age without floating-point quirks).
    ``init_random_sparse`` / ``init_distance_dependent`` — constructors
        used at graph construction time.
    ``matvec``                — ``I = W·s`` for presynaptic spike vector.
    ``prune_below`` / ``unmask``
                              — JIT-safe plasticity primitives that
                                modify ``weights`` in place while
                                preserving shape.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsp

from .backend import DTYPE, Array, PRNGKey


# ======================================================================
# Core container
# ======================================================================


class SparseConnectivity(eqx.Module):
    """Fixed-budget sparse connection fabric.

    Stores indices + weights as a BCOO matrix.  Active synapses have
    ``weights != 0``; masked slots sit at zero awaiting reuse by
    structural plasticity (synaptogenesis = write non-zero, pruning =
    write zero).  Shape of every leaf is constant for the network's
    lifetime, making the whole object safe inside ``jax.jit`` /
    ``lax.scan``.

    Fields:
        indices:   (nnz, 2) int32 — pre-row / post-col for each slot.
        weights:   (nnz,) float32 — connection weights (0 ⇒ inactive).
        shape:     (n_pre, n_post) — dense equivalent shape for matvec.
    """

    indices: Array          # (nnz, 2) int32
    weights: Array          # (nnz,) float32
    shape: tuple[int, int] = eqx.field(static=True)

    @property
    def n_pre(self) -> int:
        return self.shape[0]

    @property
    def n_post(self) -> int:
        return self.shape[1]

    @property
    def n_slots(self) -> int:
        """Total allocated connection budget (includes masked slots)."""
        return int(self.weights.shape[0])


# ======================================================================
# Constructors (called outside JIT)
# ======================================================================


def init_random_sparse(
    n_pre: int,
    n_post: int,
    density: float,
    key: PRNGKey,
    *,
    weight_scale: float = 0.1,
    active_fraction: float = 1.0,
) -> SparseConnectivity:
    """Uniform random COO initialisation.

    Reserves ``ceil(density · n_pre · n_post)`` slots.  A fraction
    ``active_fraction`` of them receive a weight drawn from
    ``Normal(0, weight_scale)``; the rest stay at 0 — available for
    synaptogenesis to fill later.

    ``key`` is consumed and split internally; pass a fresh key.
    """
    assert 0.0 < density <= 1.0, "density must lie in (0, 1]"
    assert 0.0 <= active_fraction <= 1.0

    n_slots = max(1, int(round(density * n_pre * n_post)))
    k_idx, k_w, k_act = jax.random.split(key, 3)

    # Sample without replacement from the full (pre, post) grid by
    # drawing flat indices.  At density ≤ 15 % and realistic sizes this
    # is cheap; for very dense sparse we'd switch to a Fisher-Yates
    # shuffle.  We allow duplicates for simplicity — downstream code
    # just sums contributions, which is correct.
    flat = jax.random.randint(
        k_idx, shape=(n_slots,), minval=0, maxval=n_pre * n_post, dtype=jnp.int32,
    )
    pre = flat // n_post
    post = flat % n_post
    indices = jnp.stack([pre, post], axis=1).astype(jnp.int32)

    weights = jax.random.normal(k_w, shape=(n_slots,), dtype=DTYPE) * weight_scale

    if active_fraction < 1.0:
        active_mask = jax.random.uniform(k_act, shape=(n_slots,)) < active_fraction
        weights = jnp.where(active_mask, weights, 0.0).astype(DTYPE)

    return SparseConnectivity(
        indices=indices, weights=weights, shape=(n_pre, n_post),
    )


def init_distance_dependent(
    positions_pre: Array,
    positions_post: Array,
    sigma: float,
    p_local: float,
    max_budget: int,
    key: PRNGKey,
    *,
    weight_scale: float = 0.1,
) -> SparseConnectivity:
    """Distance-dependent connectivity (Hellwig 2000).

    For each candidate slot, sample a (pre, post) pair whose connection
    probability is ``p_local · exp(-d² / 2σ²)`` where ``d = ‖pos_pre -
    pos_post‖``.  Budget is capped at ``max_budget`` slots to satisfy
    JIT's static-shape constraint.

    Positions may be any finite-dim coordinate (``(n, d)`` shape).
    """
    n_pre = positions_pre.shape[0]
    n_post = positions_post.shape[0]

    k_pre, k_post, k_accept, k_w = jax.random.split(key, 4)
    pre_candidates = jax.random.randint(
        k_pre, shape=(max_budget,), minval=0, maxval=n_pre, dtype=jnp.int32,
    )
    post_candidates = jax.random.randint(
        k_post, shape=(max_budget,), minval=0, maxval=n_post, dtype=jnp.int32,
    )

    d2 = jnp.sum(
        (positions_pre[pre_candidates] - positions_post[post_candidates]) ** 2,
        axis=-1,
    )
    p_accept = p_local * jnp.exp(-d2 / (2.0 * sigma ** 2))
    accept = jax.random.uniform(k_accept, shape=(max_budget,)) < p_accept

    indices = jnp.stack([pre_candidates, post_candidates], axis=1).astype(jnp.int32)
    weights = jax.random.normal(k_w, shape=(max_budget,), dtype=DTYPE) * weight_scale
    weights = jnp.where(accept, weights, 0.0).astype(DTYPE)

    return SparseConnectivity(
        indices=indices, weights=weights, shape=(n_pre, n_post),
    )


# ======================================================================
# Core operations
# ======================================================================


def matvec(conn: SparseConnectivity, pre_signal: Array) -> Array:
    """Compute ``post = Wᵀ · pre_signal`` — i.e. accumulate presynaptic
    contributions into each postsynaptic neuron.

    Uses ``jax.ops.segment_sum`` over the post index — robust to
    duplicate slots (multiple synapses between the same pair) which
    the random constructor may produce.  Masked slots contribute 0
    weight, so the same routine handles pre/post-plasticity without
    branching.
    """
    pre_idx = conn.indices[:, 0]
    post_idx = conn.indices[:, 1]
    contributions = pre_signal[pre_idx] * conn.weights
    return jax.ops.segment_sum(
        contributions, post_idx, num_segments=conn.shape[1],
    ).astype(DTYPE)


def prune_below(
    conn: SparseConnectivity, threshold: float,
) -> SparseConnectivity:
    """Zero out slots whose ``|weight| < threshold`` (shape preserved).

    Structural pruning rule; safe inside jit.
    """
    mask = jnp.abs(conn.weights) >= threshold
    new_weights = jnp.where(mask, conn.weights, 0.0).astype(DTYPE)
    return eqx.tree_at(lambda c: c.weights, conn, new_weights)


def unmask(
    conn: SparseConnectivity,
    slot_mask: Array,
    new_weights: Array,
) -> SparseConnectivity:
    """Write ``new_weights`` into slots where ``slot_mask`` is true.

    ``slot_mask`` is a boolean array of shape ``(n_slots,)``; typically
    computed as ``is_masked & should_spawn``.  Shape is preserved.
    """
    updated = jnp.where(slot_mask, new_weights, conn.weights).astype(DTYPE)
    return eqx.tree_at(lambda c: c.weights, conn, updated)


def to_bcoo(conn: SparseConnectivity) -> jsp.BCOO:
    """Export as a ``jax.experimental.sparse.BCOO`` for interop.

    Useful when a consumer prefers the canonical BCOO API (e.g. for
    ``bcoo_dot_general``).  The BCOO is dense in representation since
    we keep all slots — zero-weighted slots simply contribute nothing.
    """
    return jsp.BCOO(
        (conn.weights, conn.indices),
        shape=conn.shape,
    )


# ======================================================================
# Plasticity helpers
# ======================================================================


def synaptogenesis(
    conn: SparseConnectivity,
    pre_spikes: Array,
    post_spikes: Array,
    key: PRNGKey,
    *,
    spawn_prob: float = 1e-3,
    init_weight: float = 0.05,
) -> SparseConnectivity:
    """Stochastically "create" synapses in empty slots whose endpoints
    co-fired in this step (Butz & van Ooyen 2013).

    For every slot whose weight is currently 0 AND whose ``pre`` and
    ``post`` both spiked, flip the slot active with probability
    ``spawn_prob`` and initialise its weight to ``init_weight``.
    Preserves shape.
    """
    pre_idx = conn.indices[:, 0]
    post_idx = conn.indices[:, 1]
    is_empty = conn.weights == 0.0
    co_fired = (pre_spikes[pre_idx] > 0) & (post_spikes[post_idx] > 0)
    candidate = is_empty & co_fired
    sample = jax.random.uniform(key, shape=conn.weights.shape) < spawn_prob
    spawn_mask = candidate & sample
    new_weights = jnp.where(spawn_mask, init_weight, conn.weights).astype(DTYPE)
    return eqx.tree_at(lambda c: c.weights, conn, new_weights)


def active_count(conn: SparseConnectivity) -> Array:
    """Number of currently-active synapses (non-zero weights)."""
    return jnp.sum(conn.weights != 0.0)


def density(conn: SparseConnectivity) -> Array:
    """Fraction of budget currently active — for monitoring."""
    return active_count(conn) / conn.weights.shape[0]
