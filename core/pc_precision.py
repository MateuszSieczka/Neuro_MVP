"""Online precision tracking — Welford EMA + multi-channel composition (§4).

Precision Π is the substrate's attention and its adaptive learning rate
(Feldman & Friston 2010; plan §U.1 pt.5): a node's error ε is weighted by
Π in both inference (``ξ = Π·ε``) and the one learning rule, so tracking Π
well *is* tracking how much each channel should be trusted.

Two complementary tools, both pure / step-unit (no wall clock — the
substrate has no ``dt``, so every timescale is in cognitive-step units,
the same convention as :mod:`core.pc_graph`'s ``pi_alpha`` and
:mod:`core.pc_sleep`'s ``pressure_alpha``):

* **Vector node precision** (:func:`welford_precision_update`) — the
  mean-centred Welford EMA of ε.  Richer than ``pc_graph_learn``'s inline
  zero-centred ``pe_var ← (1−α)·pe_var + α·ε²``: it tracks the running
  *mean* of ε as well, so the variance is about that mean rather than
  about zero.  A biased error stream (systematic offset) then reports
  high precision once its variance settles, instead of being permanently
  penalised by the offset.  This is the opt-in precision path
  ``pc_graph`` folds in (``precision_mode="welford"``); the zero-centred
  EMA stays the default.

* **Scalar channels** (:class:`PrecisionChannel`) — one ``(mean, var)``
  EMA pair per reward-like signal, with :func:`precision_compose`
  (inverse-variance-weighted mean, for fusing estimates of one latent)
  and :func:`precision_standardize` (z-score, for summing signals of
  *different* kinds — DA reward vs curiosity vs surprise — without magic
  β weights).  Used by :mod:`core.pc_neuromod` to combine its drivers.

References
----------
  Friston (2010)               — free energy; precision as inverse variance.
  Feldman & Friston (2010)     — precision = attention / gain.
  West (1979)                  — the EMA-variance recurrence.
  FitzGerald, Dolan, Friston (2015) — neuromodulatory precision control.
"""

from __future__ import annotations

from typing import Iterable

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array


def step_alpha(tau_steps: float) -> Array:
    """EMA weight for a timescale of ``tau_steps`` cognitive steps.

    ``α = 1 − exp(−1/τ)`` — the same step-unit convention as
    ``core.pc_graph.init_pc_graph_params`` (``pi_alpha``).  Larger τ ⇒
    slower (smaller α) tracking.
    """
    return 1.0 - jnp.exp(-1.0 / jnp.asarray(tau_steps, DTYPE))


# =====================================================================
# Vector node precision — mean-centred Welford EMA (the pc_graph fold)
# =====================================================================


def welford_precision_update(
    pe_mean: Array, pe_var: Array, eps: Array, alpha: Array | float,
    var_floor: Array | float,
) -> tuple[Array, Array, Array]:
    """One mean-centred EMA step on ``(mean, var)`` of a node's error ε.

    West (1979) recurrence, vectorised over the node's error units::

        δ    = ε − mean
        mean ← mean + α·δ
        var  ← (1 − α)·(var + α·δ²)
        Π    = 1 / (var + var_floor)

    Returns ``(new_mean, new_var, new_pi)``.  The pre-update ``mean`` is
    used for ``δ`` so the step is causal.  Reduces to the zero-centred
    ``pe_var`` EMA only when the error mean is already zero; tracking the
    mean is the enrichment over ``pc_graph_learn``'s inline path.
    """
    a = jnp.asarray(alpha, DTYPE)
    eps = eps.astype(DTYPE)
    delta = eps - pe_mean
    new_mean = pe_mean + a * delta
    new_var = (1.0 - a) * (pe_var + a * delta * delta)
    new_pi = 1.0 / (new_var + jnp.asarray(var_floor, DTYPE))
    return new_mean, new_var, new_pi


# =====================================================================
# Scalar channels — one (mean, var) EMA per reward-like signal
# =====================================================================


class PrecisionChannel(eqx.Module):
    """Running EMA mean + variance + decay for one scalar signal.

    ``alpha`` is the per-step EMA rate in ``(0, 1]`` (precompute with
    :func:`step_alpha`).  ``var_floor`` is the inverse-precision ceiling:
    no channel has infinite precision (all neural signals carry noise).
    Warm start ``var = 1`` ⇒ ``Π ≈ 1`` (uniform prior before any data).
    """

    mean: Array
    var: Array
    alpha: Array
    var_floor: Array


def init_precision_channel(
    *,
    tau_steps: float = 1000.0,
    init_mean: float = 0.0,
    init_var: float = 1.0,
    var_floor: float = 1e-4,
    dtype=DTYPE,
) -> PrecisionChannel:
    """Warm-start a channel with ``Π ≈ 1`` (uniform prior)."""
    return PrecisionChannel(
        mean=jnp.asarray(init_mean, dtype),
        var=jnp.asarray(init_var, dtype),
        alpha=step_alpha(tau_steps).astype(dtype),
        var_floor=jnp.asarray(var_floor, dtype),
    )


def precision_update(channel: PrecisionChannel, x: float | Array) -> PrecisionChannel:
    """One EMA step on ``(mean, var)`` given scalar observation ``x``.

    Same West (1979) recurrence as :func:`welford_precision_update`, scalar
    form.  In the stationary limit ``var → Var(x)`` regardless of ``α``.
    """
    x = jnp.asarray(x, DTYPE)
    a = channel.alpha
    delta = x - channel.mean
    new_mean = channel.mean + a * delta
    new_var = (1.0 - a) * (channel.var + a * delta * delta)
    return eqx.tree_at(lambda c: (c.mean, c.var), channel, (new_mean, new_var))


def precision_value(channel: PrecisionChannel) -> Array:
    """``Π = 1 / (var + var_floor)`` — higher ⇒ more trustworthy channel."""
    return 1.0 / (channel.var + channel.var_floor)


def precision_mean(channel: PrecisionChannel) -> Array:
    """Running mean ``E[x]`` (a Bayer & Glimcher reward baseline)."""
    return channel.mean


def precision_standardize(channel: PrecisionChannel, x: float | Array) -> Array:
    """Z-score ``(x − mean) / √(var + var_floor)``.

    The principled way to combine reward-like signals of *different* kinds
    (extrinsic reward, curiosity, surprise) into one drive: each is
    centred and scaled by its own running statistics, so the sum is
    scale-invariant and self-calibrating — no hand-tuned β weights.  Use
    this (not :func:`precision_compose`) when the signals are not noisy
    estimates of a single latent quantity.
    """
    x = jnp.asarray(x, DTYPE)
    return (x - channel.mean) / jnp.sqrt(channel.var + channel.var_floor)


def precision_compose(
    channels: Iterable[PrecisionChannel], values: Iterable[float | Array],
) -> Array:
    """Inverse-variance-weighted mean ``Σ_k Π_k·x_k / Σ_k Π_k``.

    Appropriate when the ``x_k`` are noisy estimates of the *same* latent
    quantity (multi-sensor fusion).  Channels must be up to date (producers
    write on observation, consumers read here).
    """
    channels = list(channels)
    values = list(values)
    if len(channels) != len(values):
        raise ValueError(
            "precision_compose: len(channels) != len(values) "
            f"({len(channels)} != {len(values)})"
        )
    if not channels:
        return jnp.asarray(0.0, DTYPE)
    pis = jnp.stack([precision_value(c) for c in channels])
    xs = jnp.stack([jnp.asarray(v, DTYPE) for v in values])
    return jnp.sum(pis * xs) / (jnp.sum(pis) + jnp.asarray(1e-12, DTYPE))
