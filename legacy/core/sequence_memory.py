"""Temporal sequence memory — pure JAX.

Lisman & Jensen (2013); Hawkins & Ahmad (2016); Rolls (2013);
Treves & Rolls (1994).

Single-scale implementation: DG-like pattern separation (plastic random
projection + k-WTA) followed by outer-product transition learning
``W[i, j] = P(neuron i fires at t | neuron j fired at t-1)``.

Differences from legacy:
- Multi-scale ``HierarchicalSequenceMemory`` requires dynamic-length
  Python buffers and is therefore deferred — the cortex composition
  layer (Phase 2) owns phase/episode pooling via fixed-budget ring
  buffers tied to the oscillator state.
- k-WTA threshold uses ``jnp.sort`` (differentiable, JIT-friendly)
  rather than ``np.partition`` (host-only).
- DG Hebbian update is always executed and scaled by activity mask;
  the legacy ``if np.any(sparse > 0)`` gate is a compute optimisation
  that is incompatible with JIT.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey


# =====================================================================
# Params / state
# =====================================================================


class SeqMemParams(eqx.Module):
    """Static hyperparameters + precomputed sizes."""

    learning_rate: Array
    decay: Array
    max_weight: Array
    dg_learning_rate: Array
    n_in: int = eqx.field(static=True)
    expanded: int = eqx.field(static=True)
    k: int = eqx.field(static=True)


def init_seqmem_params(
    n_in: int,
    *,
    expansion_factor: int = 4,
    sparsity_k: float = 0.1,
    learning_rate: float = 0.01,
    decay: float = 0.999,
    max_weight: float = 1.0,
    dg_learning_rate: float = 0.001,
) -> SeqMemParams:
    expanded = n_in * expansion_factor
    k = max(1, int(expanded * sparsity_k))
    f = lambda x: jnp.asarray(x, DTYPE)
    return SeqMemParams(
        learning_rate=f(learning_rate),
        decay=f(decay),
        max_weight=f(max_weight),
        dg_learning_rate=f(dg_learning_rate),
        n_in=n_in, expanded=expanded, k=k,
    )


class SeqMemState(eqx.Module):
    """DG projection + column-norm reference + transitions + traces."""

    w_dg: Array                # (n_in, expanded)
    init_col_norm: Array       # (1, expanded) reference for Oja-like rescale
    transition_w: Array        # (expanded, expanded)
    prev_pattern: Array        # (expanded,)
    predicted_next: Array      # (n_in,)
    temporal_error: Array      # (n_in,)


def init_seqmem_state(
    key: PRNGKey, params: SeqMemParams, *, dtype=DTYPE,
) -> SeqMemState:
    """DG projection ~ ``N(0, 1/√n_in)`` (Rolls 2013 sparse expander)."""
    w_dg = jax.random.normal(
        key, (params.n_in, params.expanded), dtype=dtype,
    ) / jnp.sqrt(jnp.asarray(params.n_in, dtype))
    col_norm = jnp.maximum(
        jnp.linalg.norm(w_dg, axis=0, keepdims=True),
        jnp.asarray(1e-8, dtype),
    )
    return SeqMemState(
        w_dg=w_dg,
        init_col_norm=col_norm,
        transition_w=jnp.zeros((params.expanded, params.expanded), dtype=dtype),
        prev_pattern=jnp.zeros(params.expanded, dtype=dtype),
        predicted_next=jnp.zeros(params.n_in, dtype=dtype),
        temporal_error=jnp.zeros(params.n_in, dtype=dtype),
    )


# =====================================================================
# Pattern separation + step
# =====================================================================


def _pattern_separate(
    state: SeqMemState, params: SeqMemParams, pattern: Array,
) -> tuple[Array, SeqMemState]:
    """Project, k-WTA, apply Oja-normalised competitive Hebbian update.

    Returns ``(sparse_expanded, state_with_updated_w_dg)``.
    """
    projected = jnp.maximum(pattern @ state.w_dg, 0.0)
    # k-WTA: keep top-k via jnp.sort descending; threshold = k-th value.
    sorted_desc = jnp.sort(projected)[::-1]
    threshold = sorted_desc[params.k - 1]
    # ``max(projected) < 1e-10`` branch → produce all-zero sparse code.
    nonempty = jnp.max(projected) > 1e-10
    sparse = jnp.where(
        (projected >= threshold) & nonempty,
        projected,
        jnp.asarray(0.0, DTYPE),
    )
    # Competitive Hebbian: winners strengthen inputs (Rolls 2013).
    dw = params.dg_learning_rate * (pattern[:, None] * sparse[None, :])
    w_dg_new = state.w_dg + dw
    # Oja-like column renormalisation toward initial column norm.
    col_norm = jnp.maximum(
        jnp.linalg.norm(w_dg_new, axis=0, keepdims=True),
        jnp.asarray(1e-8, DTYPE),
    )
    w_dg_new = w_dg_new * (state.init_col_norm / col_norm)
    new_state = eqx.tree_at(lambda s: s.w_dg, state, w_dg_new)
    return sparse, new_state


def _predict_from(state: SeqMemState, sparse: Array) -> Array:
    """One-step prediction: expand → transition → fold back via ``w_dg.T``.

    Matches legacy: ``sparse @ transition_w.T`` then project through
    ``w_dg.T`` and clip to ``[0, 1]``.
    """
    raw_expanded = sparse @ state.transition_w.T
    raw_clipped = jnp.clip(raw_expanded, 0.0, 1.0)
    raw = raw_clipped @ state.w_dg.T
    return jnp.clip(raw, 0.0, 1.0)


class SeqMemOutput(NamedTuple):
    state: SeqMemState
    temporal_error: Array
    sparse: Array


def seqmem_step(
    state: SeqMemState, params: SeqMemParams, pattern: Array,
) -> SeqMemOutput:
    """Observe ``pattern``, learn transition from previous, predict next.

    Returns the temporal error (per-input dimension) so upstream code
    can turn it into a novelty signal or a neuromodulatory drive.
    """
    pat = pattern.astype(DTYPE)
    sparse, state = _pattern_separate(state, params, pat)

    temporal_error = pat - state.predicted_next

    # Transition weight update: dW = lr · outer(sparse, prev). Then decay
    # and clip. Gate contributions by activity magnitude of both ends
    # (legacy ``if any(prev) and any(sparse)`` → scalar gate multiplier).
    any_prev = jnp.maximum(jnp.sum(state.prev_pattern), 0.0)
    any_cur = jnp.maximum(jnp.sum(sparse), 0.0)
    gate = jnp.minimum(any_prev, 1.0) * jnp.minimum(any_cur, 1.0)
    gate = jnp.clip(gate, 0.0, 1.0)
    dw = gate * params.learning_rate * (sparse[:, None] * state.prev_pattern[None, :])
    transition_w = jnp.clip(
        (state.transition_w + dw) * params.decay,
        0.0, params.max_weight,
    )

    predicted_next = _predict_from(
        eqx.tree_at(lambda s: s.transition_w, state, transition_w),
        sparse,
    )

    new_state = eqx.tree_at(
        lambda s: (s.transition_w, s.prev_pattern, s.predicted_next, s.temporal_error),
        state,
        (transition_w, sparse, predicted_next, temporal_error),
    )
    return SeqMemOutput(
        state=new_state, temporal_error=temporal_error, sparse=sparse,
    )


def seqmem_novelty(state: SeqMemState) -> Array:
    """Scalar novelty ∈ [0, 1] from temporal prediction error magnitude."""
    return jnp.clip(jnp.mean(jnp.abs(state.temporal_error)), 0.0, 1.0)


def seqmem_reset_transient(state: SeqMemState, params: SeqMemParams) -> SeqMemState:
    """Clear transient predictions/errors; keep w_dg + transition_w."""
    return eqx.tree_at(
        lambda s: (s.prev_pattern, s.predicted_next, s.temporal_error),
        state,
        (
            jnp.zeros(params.expanded, DTYPE),
            jnp.zeros(params.n_in, DTYPE),
            jnp.zeros(params.n_in, DTYPE),
        ),
    )
