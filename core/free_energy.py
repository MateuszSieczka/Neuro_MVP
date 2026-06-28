"""Free energy primitives — pure JAX (Friston 2010).

The two scalar objectives of the predictive-coding substrate:

* :func:`variational_free_energy` — ``F = ½ Σ Π·ε²``, minimised by
  relaxation (perception) and learning (weights) in :mod:`core.pc_module`
  / :mod:`core.pc_graph`.
* :func:`expected_free_energy` — the action-selection objective of
  active inference (:mod:`core.pc_active` ``efe_select`` / ``pc_efe``).
"""

from __future__ import annotations

import jax.numpy as jnp

from .backend import DTYPE, Array


__all__ = ["variational_free_energy", "expected_free_energy"]


def variational_free_energy(precision: Array, error: Array) -> Array:
    """Gaussian variational free energy ``F = ½ Σ Π ⊙ ε²`` (Friston 2010).

    The single scalar objective inference and learning both minimise.
    ``precision`` and ``error`` are broadcast together; pass matching
    shapes (per-error-unit precision) or a scalar precision.
    """
    precision = precision.astype(DTYPE)
    error = error.astype(DTYPE)
    return 0.5 * jnp.sum(precision * error ** 2)


def expected_free_energy(
    pragmatic_value: float | Array,
    epistemic_value: float | Array,
    ambiguity: float | Array = 0.0,
    epistemic_weight: float | Array = 1.0,
) -> Array:
    """Expected free energy for active-inference action selection (Friston 2017).

    ``G(a) = −pragmatic + ambiguity − β · epistemic``; lower is better.
    A policy is chosen by minimising ``G`` (:func:`core.pc_active.efe_select`).

    * ``pragmatic_value`` — expected progress toward preferred states
      (goal / reward); the exploitation term.
    * ``epistemic_value`` — expected information gain (curiosity); the
      exploration term, weighted by ``β`` (NE-modulated, Parr & Friston
      2017).
    * ``ambiguity`` — expected outcome uncertainty (penalised).

    Broadcasts elementwise, so passing vectors of candidate-policy
    values returns a vector of ``G`` per policy.
    """
    prag = jnp.asarray(pragmatic_value, DTYPE)
    epi = jnp.asarray(epistemic_value, DTYPE)
    amb = jnp.asarray(ambiguity, DTYPE)
    beta = jnp.asarray(epistemic_weight, DTYPE)
    return -prag + amb - beta * epi
