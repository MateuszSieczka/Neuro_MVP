"""Variational free energy — pure JAX primitives (Friston 2010).

All routines are stateless, JIT-safe, and precision-weighted. Precision
vectors shorter than the target are broadcast via nearest-zone indexing so
that astrocyte-level ``(n_zones,)`` precision can drive neuron-level
``(n_neurons,)`` updates without Python-side logic.
"""

from __future__ import annotations

import jax.numpy as jnp

from .backend import DTYPE, Array


def broadcast_precision(precision: Array, target_n: int) -> Array:
    """Map ``(n_zones,)`` precision to ``(target_n,)`` via nearest-zone index.

    If shapes already match, returns the array unchanged. This mirrors the
    legacy semantics (``np.linspace(...).astype(int)``) but is fully JIT-safe
    because ``target_n`` is a Python int (static under JIT).
    """
    precision = precision.astype(DTYPE)
    n = precision.shape[0]
    if n == target_n:
        return precision
    idx = jnp.linspace(0, n - 1, target_n).astype(jnp.int32)
    return precision[idx]


def variational_free_energy(
    prediction_error: Array,
    precision: Array | float = 1.0,
) -> Array:
    """Scalar ``F = ½ εᵀ Π ε`` — precision-weighted prediction-error energy."""
    eps = prediction_error.astype(DTYPE)
    if isinstance(precision, (int, float)):
        return jnp.asarray(0.5, DTYPE) * jnp.asarray(precision, DTYPE) * jnp.sum(
            eps * eps
        )
    pi = precision.astype(DTYPE)
    if pi.shape[0] != eps.shape[0]:
        pi = broadcast_precision(pi, eps.shape[0])
    return jnp.asarray(0.5, DTYPE) * jnp.sum(pi * eps * eps)


def precision_weighted_update(
    prediction_error: Array,
    precision: Array | float,
    learning_rate: float,
) -> Array:
    """Return ``Δμ = lr · Π · ε`` for belief/weight gradient updates."""
    eps = prediction_error.astype(DTYPE)
    lr = jnp.asarray(learning_rate, DTYPE)
    if isinstance(precision, (int, float)):
        return lr * jnp.asarray(precision, DTYPE) * eps
    pi = precision.astype(DTYPE)
    if pi.shape[0] != eps.shape[0]:
        pi = broadcast_precision(pi, eps.shape[0])
    return lr * pi * eps


def expected_free_energy(
    pragmatic_value: float | Array,
    epistemic_value: float | Array,
    ambiguity: float | Array = 0.0,
    epistemic_weight: float | Array = 1.0,
) -> Array:
    """Expected free energy for action selection (active inference).

    ``G(a) = −pragmatic + ambiguity − β · epistemic``. Lower is better.
    ``epistemic_weight`` is typically modulated by NE (curiosity).
    """
    prag = jnp.asarray(pragmatic_value, DTYPE)
    epi = jnp.asarray(epistemic_value, DTYPE)
    amb = jnp.asarray(ambiguity, DTYPE)
    beta = jnp.asarray(epistemic_weight, DTYPE)
    return -prag + amb - beta * epi
