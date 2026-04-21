"""Online precision tracking bus — pure JAX.

Friston (2010) Free-energy principle; Yu & Dayan (2005) uncertainty →
ACh/NE; Tobler et al. (2005) D2 RMS; Bayer & Glimcher (2005) reward
baseline; Mathys et al. (2011) volatility.

Motivation
----------
Whenever two or more reward-like signals are combined into one scalar
drive (critic TD target, actor advantage, active-inference G), their
*relative magnitudes* must be normalised. Hand-tuned β weights (as in
``r_actor = rpe + β·curiosity``) are magic constants that break the
moment a third channel is added or the environment's reward scale
changes. The principled solution is precision weighting: each channel
carries its own running-variance estimate, and the composed reward is
the variance-inverse-weighted mean

    π_k = 1 / (Var_k + ε),            μ_k = E[x_k]    (both running EMAs)
    r_composed = Σ_k π_k · x_k / Σ_k π_k

so noisy channels are auto-downweighted and well-calibrated channels
dominate. This is the standard generative-model precision update
(Friston 2010 §3.2) applied to reward signals rather than sensory
prediction errors.

Design
------
* ``PrecisionChannel``: one EMA pair ``(mean, var)`` plus a decay rate
  ``α`` (derived from a timescale τ via ``ctx.complement(τ)``). A
  channel is a pytree leaf, so it composes trivially into any host
  state.
* ``precision_update(channel, x)``: Welford-style EMA update,
  ``μ ← μ + α(x − μ)``, ``Var ← (1 − α) (Var + α (x − μ)²)`` (this is
  the standard EMA-variance recurrence, see West 1979). The
  pre-update ``μ`` is used for the squared-deviation so the update is
  causal.
* ``precision_value(channel)``: returns ``π = 1 / (Var + ε)``. Warm
  start is ``Var = 1.0`` → ``π ≈ 1.0`` (uniform prior, before any
  observation).
* ``precision_compose(channels, values)``: given a tuple of channels
  and matching scalar values, returns
  ``Σ π_k x_k / Σ π_k``. Pure function, JIT-friendly.

Timescale convention
--------------------
Default τ = 10 s (10 000 dt at dt = 1 ms) matches D2 autoreceptor
desensitisation (Benoit-Marand 2011) — long enough to smooth over
single-trial noise, short enough to track task-level drift. A per-
channel τ can be set for signals with known faster / slower dynamics
(e.g. immediate sensory PE uses shorter τ than slow appetitive
baseline).
"""

from __future__ import annotations

from typing import Iterable

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array, BackendContext


# =====================================================================
# Channel pytree
# =====================================================================


class PrecisionChannel(eqx.Module):
    """Running EMA mean + variance + decay for one scalar signal.

    ``alpha`` is the per-step EMA update rate (dimensionless, in
    ``(0, 1]``). It is pre-computed via ``ctx.complement(tau)`` so
    the update is JIT-safe and τ-independent of ``dt``.

    ``var_floor`` guards against division by zero in
    ``precision_value``. It is a *biophysical* floor: no channel can
    have infinite precision because all neural signals have shot
    noise. Default 1e-4 (in the units of the signal, typically
    normalised ~O(1)).
    """

    mean: Array           # scalar, running EMA
    var: Array            # scalar, running EMA variance
    alpha: Array          # scalar, per-step update rate
    var_floor: Array      # scalar, numerical + biophysical floor


def init_precision_channel(
    ctx: BackendContext,
    *,
    tau_ms: float = 10_000.0,
    init_mean: float = 0.0,
    init_var: float = 1.0,
    var_floor: float = 1e-4,
    dtype=DTYPE,
) -> PrecisionChannel:
    """Warm-start with ``Var = 1`` so ``π ≈ 1`` (uniform prior).

    The uniform prior is the neutral point for precision weighting:
    before any data, all channels contribute equally. As each channel
    accumulates observations its precision self-calibrates.
    """
    alpha = ctx.complement(jnp.asarray(tau_ms, dtype))
    return PrecisionChannel(
        mean=jnp.asarray(init_mean, dtype),
        var=jnp.asarray(init_var, dtype),
        alpha=alpha,
        var_floor=jnp.asarray(var_floor, dtype),
    )


# =====================================================================
# Per-channel update / read
# =====================================================================


def precision_update(
    channel: PrecisionChannel, x: float | Array,
) -> PrecisionChannel:
    """One EMA step on ``(mean, var)`` given a scalar observation ``x``.

    Standard EMA-variance recurrence (West 1979 §2):

        δ    = x − mean
        mean ← mean + α · δ
        var  ← (1 − α) · (var + α · δ²)

    The ``(1 − α)`` multiplier on ``var`` is the usual EMA decay; the
    additive ``α · δ²`` term is the new-observation contribution. In
    the stationary limit ``Var → Var(x)`` regardless of ``α``.
    """
    x = jnp.asarray(x, DTYPE)
    a = channel.alpha
    delta = x - channel.mean
    new_mean = channel.mean + a * delta
    new_var = (1.0 - a) * (channel.var + a * delta * delta)
    return eqx.tree_at(
        lambda c: (c.mean, c.var),
        channel,
        (new_mean, new_var),
    )


def precision_value(channel: PrecisionChannel) -> Array:
    """``π = 1 / (Var + var_floor)`` — the precision of this channel.

    Higher π → this channel is trustworthy (low variance) → it should
    dominate in a weighted composition.
    """
    return 1.0 / (channel.var + channel.var_floor)


def precision_standardize(
    channel: PrecisionChannel, x: float | Array,
) -> Array:
    """Z-score ``(x − μ) / √(Var + var_floor)``.

    The principled way to combine reward-like signals of *different*
    kinds (extrinsic hedonic reward, curiosity, information gain,
    surprise, …) into one actor RPE. Each signal is centered on its
    running mean and scaled by its running stdev before summing, so
    the composite ``rpe + z(curiosity) + z(info_gain)`` is
    scale-invariant and self-calibrating — no hand-tuned β weights,
    and the composition automatically adapts to each signal's natural
    scale (r_ext can be in dollars, curiosity in nats, info_gain in
    dimensionless PE units — after z-scoring they are all in units of
    "σ above mean" and add meaningfully).

    Inverse-variance weighted *averages* (``Σ π_k x_k / Σ π_k``) are
    only appropriate when the ``x_k`` are noisy estimates of the
    *same* latent quantity (e.g. multi-sensor fusion of a single
    physical state). Actor-bonus composition is not that case.
    """
    x = jnp.asarray(x, DTYPE)
    return (x - channel.mean) / jnp.sqrt(channel.var + channel.var_floor)


def precision_mean(channel: PrecisionChannel) -> Array:
    """Running mean ``E[x]``; useful as a Bayer & Glimcher baseline."""
    return channel.mean


# =====================================================================
# Multi-channel weighted composition
# =====================================================================


def precision_compose(
    channels: Iterable[PrecisionChannel], values: Iterable[float | Array],
) -> Array:
    """Variance-inverse-weighted mean of the supplied scalar values.

    ``r_composed = Σ_k π_k · x_k / Σ_k π_k``

    All channels must be up-to-date (caller responsibility — update
    happens when the *signal* is observed, not when it is composed).
    This separation keeps the API symmetric: consumers read, producers
    write.
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
    num = jnp.sum(pis * xs)
    den = jnp.sum(pis) + jnp.asarray(1e-12, DTYPE)
    return num / den


def precision_weight(
    channel: PrecisionChannel,
    total_precision: Array,
) -> Array:
    """Normalised weight ``π_k / Σ π`` for this channel in a composition.

    Useful for diagnostic output (which channel dominated a given
    decision) or when the caller wants to apply the weight
    asymmetrically (e.g. only the epistemic half of a composed reward).
    """
    return precision_value(channel) / (
        total_precision + jnp.asarray(1e-12, DTYPE)
    )
