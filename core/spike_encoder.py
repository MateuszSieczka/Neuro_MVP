"""Spike encoders — pure JAX.

- :func:`gaussian_population_encode`: continuous vector → distributed rates in [0,1]
  via tiled Gaussian receptive fields (Pouget, Dayan & Zemel 2000).
- :func:`poisson_spike`: rate vector in [0,1] → binary spikes at one timestep.

Both are stateless and JIT-safe. ``PopulationEncoderParams`` precomputes
receptive-field centres and ``inv_2sigma2`` for reuse across steps.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey


class PopulationEncoderParams(eqx.Module):
    """Precomputed tiled Gaussian receptive fields."""

    centers: Array
    inv_2sigma2: Array
    n_dims: int = eqx.field(static=True)
    n_neurons_per_dim: int = eqx.field(static=True)


def init_population_encoder(
    n_dims: int,
    n_neurons_per_dim: int = 15,
    value_min: float = -1.0,
    value_max: float = 1.0,
    *,
    dtype=DTYPE,
) -> PopulationEncoderParams:
    """Construct tiled Gaussian receptive fields with a 10% boundary margin."""
    margin = 0.1 * (value_max - value_min)
    centers = jnp.linspace(
        value_min - margin, value_max + margin, n_neurons_per_dim, dtype=dtype
    )
    if n_neurons_per_dim > 1:
        spacing = centers[1] - centers[0]
    else:
        spacing = jnp.asarray(1.0, dtype=dtype)
    inv_2sigma2 = jnp.asarray(1.0, dtype=dtype) / (
        jnp.asarray(2.0, dtype=dtype) * (jnp.asarray(0.5, dtype=dtype) * spacing) ** 2
    )
    return PopulationEncoderParams(
        centers=centers,
        inv_2sigma2=inv_2sigma2,
        n_dims=n_dims,
        n_neurons_per_dim=n_neurons_per_dim,
    )


def gaussian_population_encode(
    params: PopulationEncoderParams, values: Array
) -> Array:
    """Encode ``(n_dims,)`` vector into ``(n_dims × n_per_dim,)`` rates in [0,1]."""
    v = values.astype(DTYPE).reshape(-1, 1)
    rates = jnp.exp(-((v - params.centers) ** 2) * params.inv_2sigma2)
    return rates.reshape(-1)


def poisson_spike(key: PRNGKey, rates: Array) -> Array:
    """One timestep of Bernoulli spikes.

    ``rate=0 → 0``, ``rate≥1 → 1``, else spike with probability ``rate``.
    """
    rates = jnp.clip(rates, 0.0, 1.0).astype(DTYPE)
    u = jax.random.uniform(key, shape=rates.shape, dtype=DTYPE)
    return (u < rates).astype(DTYPE)


class PoissonOutput(NamedTuple):
    spikes: Array
    key: PRNGKey


def poisson_step(key: PRNGKey, rates: Array) -> PoissonOutput:
    """Split key and produce one step of spikes + next key for scan loops."""
    key, sub = jax.random.split(key)
    return PoissonOutput(spikes=poisson_spike(sub, rates), key=key)
