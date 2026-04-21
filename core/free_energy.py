"""Free energy / EFE primitives — pure JAX (Friston 2010, 2017).

Only :func:`broadcast_precision` is currently wired into the live
circuit (via ``error_neuron.en_update_weights``). :func:`expected_free_energy`
is retained for Phase 9 active-inference planning but is *not*
exported from ``core.__init__`` until a live call-site is added — this
keeps ``__all__`` truthful about what is live in the brain graph.

The earlier ``variational_free_energy`` and ``precision_weighted_update``
helpers were removed as redundant one-liners (the former is equivalent
to ``0.5 * jnp.sum(precision * error ** 2)``; the latter is exactly the
STDP three-factor gradient, already implemented per-module in
``plasticity.py`` and ``error_neuron.py``).
"""

from __future__ import annotations

import jax.numpy as jnp

from .backend import DTYPE, Array


__all__ = ["broadcast_precision"]


def broadcast_precision(precision: Array, target_n: int) -> Array:
    """Map ``(n_zones,)`` precision to ``(target_n,)`` via nearest-zone index.

    If shapes already match, returns the array unchanged. This mirrors
    the legacy semantics (``np.linspace(...).astype(int)``) but is
    fully JIT-safe because ``target_n`` is a Python int (static under
    JIT).
    """
    precision = precision.astype(DTYPE)
    n = precision.shape[0]
    if n == target_n:
        return precision
    idx = jnp.linspace(0, n - 1, target_n).astype(jnp.int32)
    return precision[idx]


def expected_free_energy(
    pragmatic_value: float | Array,
    epistemic_value: float | Array,
    ambiguity: float | Array = 0.0,
    epistemic_weight: float | Array = 1.0,
) -> Array:
    """Expected free energy for action selection (Friston 2017).

    ``G(a) = −pragmatic + ambiguity − β · epistemic``. Lower is better.
    ``epistemic_weight`` is typically modulated by NE (curiosity).

    NOTE: not exported from ``core.__init__`` — wire-in is scheduled
    for Phase 9 (active-inference planner). Kept here to avoid a
    module-delete / re-create churn when that phase lands.
    """
    prag = jnp.asarray(pragmatic_value, DTYPE)
    epi = jnp.asarray(epistemic_value, DTYPE)
    amb = jnp.asarray(ambiguity, DTYPE)
    beta = jnp.asarray(epistemic_weight, DTYPE)
    return -prag + amb - beta * epi
