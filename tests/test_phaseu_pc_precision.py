"""Faza U — §4 online precision tracking (precision_bus salvage).

Asserts the mean-centred Welford EMA is a faithful variance estimator, is
*richer* than the zero-centred ε² EMA on a biased error stream, that the
scalar-channel compose/standardize helpers behave, and that the opt-in
``pc_graph`` Welford path is backward-compatible (EMA stays the default,
identical numerics; pe_mean inert unless Welford is selected).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.pc_precision import (
    step_alpha, welford_precision_update,
    init_precision_channel, precision_update, precision_value,
    precision_standardize, precision_compose,
)
from core.pc_graph import (
    init_region_graph, pc_graph_step, PRECISION_EMA, PRECISION_WELFORD,
)


# ---------------------------------------------------------------------
# Welford vector update
# ---------------------------------------------------------------------


def test_welford_tracks_variance_of_zero_mean_noise():
    """On zero-mean noise the Welford variance converges to the true variance."""
    key = jax.random.PRNGKey(0)
    alpha = step_alpha(50.0)
    floor = jnp.asarray(1e-4)
    mean = jnp.zeros(1)
    var = jnp.ones(1)
    for i in range(4000):
        eps = 2.0 * jax.random.normal(jax.random.fold_in(key, i), (1,))  # var ≈ 4
        mean, var, pi = welford_precision_update(mean, var, eps, alpha, floor)
    assert abs(float(var[0]) - 4.0) < 0.8, "Welford var should track ≈ Var(ε)"
    assert abs(float(mean[0])) < 0.3, "running mean should sit near 0"


def test_welford_beats_zero_centred_ema_on_biased_error():
    """A biased (offset) error stream: Welford reads high precision, ε² EMA low.

    The enrichment over ``pc_graph_learn``'s inline path — a systematic
    offset inflates the zero-centred second moment forever, but the
    mean-centred variance settles near 0, so Welford precision ≫ EMA.
    """
    alpha = step_alpha(50.0)
    floor = jnp.asarray(1e-4)
    bias = 3.0
    w_mean, w_var = jnp.zeros(1), jnp.ones(1)
    ema_var = jnp.ones(1)
    a = alpha
    for _ in range(4000):
        eps = jnp.asarray([bias])                    # constant biased error
        w_mean, w_var, w_pi = welford_precision_update(w_mean, w_var, eps, a, floor)
        ema_var = (1.0 - a) * ema_var + a * eps ** 2  # the legacy inline path
    ema_pi = 1.0 / (ema_var + floor)
    assert float(w_var[0]) < 0.05, "Welford variance collapses around the mean"
    assert float(w_pi[0]) > 100.0 * float(ema_pi[0]), "Welford ≫ EMA precision on bias"


# ---------------------------------------------------------------------
# Scalar channels
# ---------------------------------------------------------------------


def test_channel_precision_rises_for_calibrated_signal():
    """A low-variance channel earns higher precision than a noisy one."""
    key = jax.random.PRNGKey(1)
    calm = init_precision_channel(tau_steps=50.0)
    noisy = init_precision_channel(tau_steps=50.0)
    for i in range(2000):
        calm = precision_update(calm, 0.1 * jax.random.normal(jax.random.fold_in(key, i)))
        noisy = precision_update(noisy, 5.0 * jax.random.normal(jax.random.fold_in(key, 10_000 + i)))
    assert float(precision_value(calm)) > float(precision_value(noisy))


def test_compose_downweights_noisy_channel():
    """Inverse-variance composition trusts the calibrated estimate."""
    key = jax.random.PRNGKey(2)
    calm = init_precision_channel(tau_steps=50.0)
    noisy = init_precision_channel(tau_steps=50.0)
    for i in range(2000):
        calm = precision_update(calm, 1.0 + 0.05 * jax.random.normal(jax.random.fold_in(key, i)))
        noisy = precision_update(noisy, 3.0 * jax.random.normal(jax.random.fold_in(key, 9 + i)))
    composed = float(precision_compose([calm, noisy], [1.0, 5.0]))
    assert abs(composed - 1.0) < abs(composed - 5.0), "composition near the calm value"


def test_standardize_is_scale_invariant():
    """Z-score puts a 1σ deviation at ≈ 1 regardless of the signal's scale."""
    key = jax.random.PRNGKey(3)
    ch = init_precision_channel(tau_steps=100.0)
    for i in range(4000):
        ch = precision_update(ch, 10.0 * jax.random.normal(jax.random.fold_in(key, i)))
    z = float(precision_standardize(ch, ch.mean + jnp.sqrt(ch.var)))
    assert abs(z - 1.0) < 0.1


# ---------------------------------------------------------------------
# pc_graph opt-in Welford path — backward-compatible
# ---------------------------------------------------------------------


def test_ema_is_default_and_pe_mean_inert():
    gp, gs = init_region_graph(jax.random.PRNGKey(0))
    assert gp.precision_mode == PRECISION_EMA
    out = pc_graph_step(gs, gp, {0: jnp.ones(gp.node_sizes[0])})
    assert all(bool(jnp.all(m == 0.0)) for m in out.state.pe_mean), \
        "EMA mode must leave pe_mean untouched (zero)"


def test_welford_mode_opt_in_tracks_mean_same_pi_shapes():
    gp, gs = init_region_graph(jax.random.PRNGKey(0), precision_mode=PRECISION_WELFORD)
    out = pc_graph_step(gs, gp, {0: jnp.ones(gp.node_sizes[0])})
    # pe_mean now carries information; precision shapes unchanged (drop-in).
    assert any(bool(jnp.any(m != 0.0)) for m in out.state.pe_mean)
    for j in range(gp.n_nodes):
        assert out.state.pi[j].shape == (gp.node_sizes[j],)
