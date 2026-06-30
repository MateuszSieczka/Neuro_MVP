"""Population coding of a scalar into a rate vector — two tuning families.

* **Gaussian** (Pouget-Dayan-Zemel 2000): ``n`` units with preferred values
  uniformly spaced over ``[x_min, x_max]``, each firing a bump around its
  preferred value.  The standard cortical code for *afferent* quantities
  (stimulus value, joint angle): dense, distributed, hides the raw scalar.

* **Monotonic** ("thermometer" / cumulative): each unit's activation is the
  cumulative-logistic of the value past its threshold, so the population is
  monotone in the encoded scalar.  This is the code for a *controllable /
  goal* variable that active inference must **invert**: a goal is clamped on
  these channels and the command is read off by minimising free energy
  (= L2 on the code).  A narrow Gaussian bump makes that L2 **flat** wherever
  the predicted and target bumps do not overlap (both ≈0 off-peak) — the
  inversion gradient vanishes and the command is never driven.  A monotonic
  code keeps the L2 gradient non-zero across the whole range (its derivative
  is a bump that tiles the range), so the inversion has signal everywhere —
  without revealing the raw scalar (still ``n`` distributed units).
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


def monotonic_population_encode(
    x: Array | float, n: int, *,
    x_min: float, x_max: float,
    sigma: float | None = None,
    peak: float = 1.0,
) -> Array:
    """Monotonic ("thermometer") population code of scalar ``x`` across ``n`` units.

    Unit ``i`` has threshold ``c_i`` uniformly spaced on ``[x_min, x_max]``
    and fires ``peak·σ((x − c_i)/s)`` (logistic cumulative), so the
    population is monotone increasing in ``x``: low-threshold units saturate
    to ``peak``, high-threshold units to ``0``, the transition sitting at
    ``x``.  Unlike a Gaussian bump, the code's derivative ``dᵢ/dx`` is a
    logistic bump that *tiles the whole range*, so the squared-error (free
    energy) between two codes has a non-vanishing gradient at any
    separation — the property that lets active inference invert a clamped
    goal on these channels into a command everywhere in the workspace, not
    only when already on target.  Default ``sigma`` is the centre spacing
    (adjacent thresholds overlap).  Returns a dense ``(n,)`` float32 vector
    in ``[0, peak]``.
    """
    centers = jnp.linspace(x_min, x_max, n, dtype=DTYPE)
    sigma_val = (x_max - x_min) / max(1, n - 1) if sigma is None else float(sigma)
    z = (jnp.asarray(x, DTYPE) - centers) / jnp.asarray(sigma_val, DTYPE)
    return (jnp.asarray(peak, DTYPE) / (1.0 + jnp.exp(-z))).astype(DTYPE)
