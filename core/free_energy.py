"""Free energy primitives — pure JAX (Friston 2010).

Both functions here have live consumers:

* :func:`variational_free_energy` — the single scalar objective
  ``F = ½ Σ Π·ε²`` minimised by relaxation and learning in
  :mod:`core.pc_module` / :mod:`core.pc_graph` (Faza U).
* :func:`broadcast_precision` — maps zone precision to error-neuron
  precision in ``error_neuron.en_update_weights``.

:func:`expected_free_energy` (EFE) is the action-selection objective of
U.5 (active-inference motor head): live consumer in
:mod:`core.pc_active` (``pc_efe`` / ``efe_select``).
"""

from __future__ import annotations

import jax.numpy as jnp

from .backend import DTYPE, Array


__all__ = [
    "broadcast_precision", "variational_free_energy", "expected_free_energy",
]


def variational_free_energy(precision: Array, error: Array) -> Array:
    """Gaussian variational free energy ``F = ½ Σ Π ⊙ ε²`` (Friston 2010).

    The single scalar objective that *everything* in Faza U minimises:
    perception (state relaxation), learning (weight updates) and action
    (expected-FE). Restored from the earlier delete (it was removed as a
    "redundant one-liner" before any consumer existed) now that
    :mod:`core.pc_module` / :mod:`core.pc_graph` relax and learn on it.

    ``precision`` and ``error`` are broadcast together; pass matching
    shapes (per-error-unit precision) or a scalar precision.
    """
    precision = precision.astype(DTYPE)
    error = error.astype(DTYPE)
    return 0.5 * jnp.sum(precision * error ** 2)


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
    """Expected free energy for active-inference action selection (Friston 2017).

    ``G(a) = −pragmatic + ambiguity − β · epistemic``; lower is better.
    A policy is chosen by minimising ``G`` (``core.pc_active.efe_select``).

    * ``pragmatic_value`` — expected progress toward preferred states
      (goal / reward); the exploitation term.
    * ``epistemic_value`` — expected information gain (curiosity); the
      exploration term, weighted by ``β`` (NE-modulated, Parr & Friston
      2017).  ``core.world_model.wm_learning_progress`` is its standing
      approximation in the legacy circuit.
    * ``ambiguity`` — expected outcome uncertainty (penalised).

    Broadcasts elementwise, so passing vectors of candidate-policy
    values returns a vector of ``G`` per policy.
    """
    prag = jnp.asarray(pragmatic_value, DTYPE)
    epi = jnp.asarray(epistemic_value, DTYPE)
    amb = jnp.asarray(ambiguity, DTYPE)
    beta = jnp.asarray(epistemic_weight, DTYPE)
    return -prag + amb - beta * epi
