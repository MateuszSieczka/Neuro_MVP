"""Gaussian population coding of a scalar into a rate vector.

The canonical Pouget-Dayan-Zemel (2000) cortical population code: ``n``
units with preferred values uniformly spaced over ``[x_min, x_max]``, each
firing a Gaussian bump around its preferred value.  Turns a continuous
quantity (a target-error component, a stimulus value) into the dense ``[0,
peak]`` rate vector the substrate clamps — without revealing the raw scalar
to the brain.
"""

from __future__ import annotations

import jax.numpy as jnp

from core.backend import DTYPE, Array


def gaussian_population_encode(
    x: Array | float, n: int, *,
    x_min: float, x_max: float,
    sigma: float | None = None,
    peak: float = 1.0,
) -> Array:
    """Gaussian population code of scalar ``x`` across ``n`` tuning curves.

    Each unit ``i`` has preferred value ``c_i`` uniformly spaced on
    ``[x_min, x_max]``; activation is ``peak · exp(-½·((x − c_i)/σ)²)``.
    Default ``sigma`` is the centre spacing ``(x_max − x_min)/(n − 1)``,
    giving half-overlap between adjacent curves.

    Returns a dense ``(n,)`` float32 vector in ``[0, peak]``.
    """
    centers = jnp.linspace(x_min, x_max, n, dtype=DTYPE)
    sigma_val = (x_max - x_min) / max(1, n - 1) if sigma is None else float(sigma)
    z = (jnp.asarray(x, DTYPE) - centers) / jnp.asarray(sigma_val, DTYPE)
    return (jnp.asarray(peak, DTYPE) * jnp.exp(-0.5 * z * z)).astype(DTYPE)
