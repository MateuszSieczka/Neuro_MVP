"""Phase 0.5 — Precision bus: Welford EMA converges to running mean/var.

The precision bus (``core.precision_bus.PrecisionChannel``) is the
scale-normalisation substrate that lets :func:`action_brain_cognitive_step`
additively compose RPE with intrinsic bonuses (curiosity, info_gain)
that live in different physical units. Each channel must:

1. Track mean and variance of its input stream online (no batch).
2. Converge to the true statistics of a stationary source.
3. Produce unit-variance, zero-mean z-scores on its own estimates.

Failure of (3) would leak raw-scale differences back into the RPE
summation — exactly the scale-mismatch bug the refactor fixes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from core.backend import BackendContext
from core.precision_bus import (
    init_precision_channel, precision_update, precision_standardize,
)


def test_precision_channel_converges_to_input_statistics():
    """After many samples of N(5, 4), μ → 5 and Var → 4."""
    ctx = BackendContext(dt=1.0)
    # τ = 1000 steps → α ≈ 1e-3, well-mixed after 10 τ.
    ch = init_precision_channel(ctx, tau_ms=1000.0)
    key = jax.random.PRNGKey(0)
    samples = jax.random.normal(key, (20_000,)) * 2.0 + 5.0
    for s in samples:
        ch = precision_update(ch, s)
    assert abs(float(ch.mean) - 5.0) < 0.2, (
        f"mean did not converge: got {float(ch.mean):.3f}, expected ~5"
    )
    assert abs(float(ch.var) - 4.0) < 0.5, (
        f"variance did not converge: got {float(ch.var):.3f}, expected ~4"
    )


def test_precision_standardize_produces_unit_variance():
    """Standardised stream has mean ≈ 0 and variance ≈ 1."""
    ctx = BackendContext(dt=1.0)
    ch = init_precision_channel(ctx, tau_ms=1000.0)
    key = jax.random.PRNGKey(1)
    samples = jax.random.normal(key, (20_000,)) * 3.0 - 2.0
    # Warm-up: first pass to let statistics converge.
    for s in samples[:5000]:
        ch = precision_update(ch, s)
    zs = []
    for s in samples[5000:]:
        ch = precision_update(ch, s)
        zs.append(float(precision_standardize(ch, s)))
    zs = np.asarray(zs)
    assert abs(zs.mean()) < 0.1, f"z-score mean off: {zs.mean():.3f}"
    assert abs(zs.std() - 1.0) < 0.15, f"z-score std off: {zs.std():.3f}"


def test_precision_scale_invariance():
    """Two streams with 1000× scale difference produce same-scale z-scores.

    This is the contract that makes additive composition ``rpe +
    curiosity_z + info_gain_z`` robust: whether the raw bonus was in
    units of 1e-1 or 1e+2, after standardise it contributes an O(1)
    additive term.

    Note: scale-invariance holds only when ``Var(x) >> var_floor``
    (the precision floor, Friston 2010 — protects against amplifying
    noise on a dead/near-constant channel). Both scales here are
    well above that floor.
    """
    ctx = BackendContext(dt=1.0)
    key = jax.random.PRNGKey(2)
    raw = jax.random.normal(key, (20_000,))

    ch_a = init_precision_channel(ctx, tau_ms=1000.0)
    ch_b = init_precision_channel(ctx, tau_ms=1000.0)
    zs_a, zs_b = [], []
    for x in raw:
        s_a = x * 1e-1
        s_b = x * 1e2
        ch_a = precision_update(ch_a, s_a)
        ch_b = precision_update(ch_b, s_b)
        zs_a.append(float(precision_standardize(ch_a, s_a)))
        zs_b.append(float(precision_standardize(ch_b, s_b)))
    zs_a = np.asarray(zs_a[5000:])
    zs_b = np.asarray(zs_b[5000:])
    # Both should be ~unit-variance regardless of raw scale.
    assert 0.8 < zs_a.std() < 1.2, f"std_a={zs_a.std():.3f}"
    assert 0.8 < zs_b.std() < 1.2, f"std_b={zs_b.std():.3f}"
    # And pointwise they should track each other (same latent x,
    # standardised copies) → high correlation.
    corr = np.corrcoef(zs_a, zs_b)[0, 1]
    assert corr > 0.95, f"scale-invariance broken: corr={corr:.3f}"


def test_precision_floor_protects_dead_channel():
    """A near-constant channel must NOT be standardised to unit variance.

    If ``Var(x) ≈ 0``, dividing by √Var would amplify micro-noise to
    O(1) and corrupt downstream summation. ``var_floor`` (Friston
    2010 precision floor) bounds the denominator; the standardised
    output is then ≪ 1 for dead channels, which is the correct
    behaviour (silent channel contributes silently).
    """
    ctx = BackendContext(dt=1.0)
    ch = init_precision_channel(ctx, tau_ms=1000.0, var_floor=1e-4)
    # Deterministic near-zero stream with tiny numerical noise.
    key = jax.random.PRNGKey(3)
    noise = jax.random.normal(key, (20_000,)) * 1e-6
    zs = []
    for s in noise:
        ch = precision_update(ch, s)
        zs.append(float(precision_standardize(ch, s)))
    zs = np.asarray(zs[5000:])
    # Floor clamps the denom to √1e-4 = 0.01, so |z| ≤ 1e-6/1e-2 = 1e-4.
    assert zs.std() < 0.05, (
        f"dead channel standardised to nonzero variance: std={zs.std():.4f}"
    )
