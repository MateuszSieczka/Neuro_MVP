"""JAX backend — type aliases, default dtype and PRNG helpers.

The single base dependency of the predictive-coding substrate.  Pure,
jit-friendly, no side effects: all state lives in the caller's pytree.
``float32`` is the default dtype; cast once at the edges and stay there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array as _Array

    Array = _Array
    PRNGKey = _Array
else:
    Array = jax.Array
    PRNGKey = jax.Array

# Default precision for beliefs, weights, precision and errors.
DTYPE = jnp.float32


# ------------------------------------------------------------------
# PRNG helpers.
# ------------------------------------------------------------------


def make_key(seed: int) -> PRNGKey:
    """Create a PRNG key from an integer seed."""
    return jax.random.PRNGKey(seed)


def split_key(key: PRNGKey, n: int = 2) -> Array:
    """Split ``key`` into ``n`` fresh keys.  ``n=2`` by default."""
    return jax.random.split(key, n)


def fold_in_step(key: PRNGKey, step: int | Array) -> PRNGKey:
    """Deterministically derive a subkey for iteration ``step``.

    Prefer this over carrying a running key through a loop when you need
    reproducibility per step without threading state.
    """
    return jax.random.fold_in(key, jnp.asarray(step, dtype=jnp.uint32))
