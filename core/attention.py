"""Spatial attention — pure JAX (Reynolds & Heeger 2009 divisive normalisation
+ Posner & Cohen 1984 inhibition-of-return + Aston-Jones & Cohen 2005 NE gain).

Legacy stateful ``SpatialAttentionController`` is replaced with a pytree
``AttentionState`` and a functional :func:`attention_step` that returns the
next state plus per-column multiplicative gains.

Top-down projection weights ``w_attn`` of shape ``(n_assoc, n_columns)``
are part of the state so that attention plasticity composes with the rest
of the learning update step under ``eqx.tree_at``.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, BackendContext


# Afterhyperpolarization (AHP) peak gain reduction at maximal firing
# in cortical pyramidal cells (Madison & Nicoll 1984, Table 1: 30%
# input conductance reduction at 20 Hz sustained firing).  This is
# the biophysical substrate of inhibition-of-return in sensory
# cortex (Mirpour & Bisley 2012) and is not a tunable coefficient
# -- the value derives from Ca²⁺-activated K⁺ (SK/IK) channel
# density, which is conserved across cortical areas.
_AHP_IOR_GAIN_REDUCTION: float = 0.3


class AttentionParams(eqx.Module):
    """Derived hyperparameters (Reynolds & Heeger 2009; Posner & Cohen 1984)."""

    gain_strength: Array
    divisive_sigma: Array
    ne_optimal: Array
    ne_gain_strength: Array
    bottom_up_weight: Array
    ior_decay: Array
    smoothing_decay: Array
    learning_rate: Array


def init_attention_params(
    ctx: BackendContext,
    *,
    gain_strength: float = 2.0,
    divisive_sigma: float = 1.0,
    ne_optimal: float = 0.5,
    ne_gain_strength: float = 2.0,
    bottom_up_weight: float = 0.4,
    ior_tau: float = 400.0,
    smoothing_decay: float = 0.9,
    learning_rate: float = 0.005,
    dtype=DTYPE,
) -> AttentionParams:
    def f(x): return jnp.asarray(x, dtype)
    return AttentionParams(
        gain_strength=f(gain_strength),
        divisive_sigma=f(divisive_sigma),
        ne_optimal=f(ne_optimal),
        ne_gain_strength=f(ne_gain_strength),
        bottom_up_weight=f(bottom_up_weight),
        ior_decay=f(ctx.decay(ior_tau)),
        smoothing_decay=f(smoothing_decay),
        learning_rate=f(learning_rate),
    )


class AttentionState(eqx.Module):
    """Transient attention state + learned top-down projection."""

    attn_weights: Array
    ior_trace: Array
    w_attn: Array


def init_attention_state(
    key: PRNGKey,
    n_assoc: int,
    n_columns: int,
    *,
    w_init_scale: float = 0.1,
    dtype=DTYPE,
) -> AttentionState:
    w_attn = jax.random.uniform(
        key, (n_assoc, n_columns),
        minval=-w_init_scale, maxval=w_init_scale, dtype=dtype,
    )
    return AttentionState(
        attn_weights=jnp.full((n_columns,), 1.0 / n_columns, dtype=dtype),
        ior_trace=jnp.zeros((n_columns,), dtype=dtype),
        w_attn=w_attn,
    )


class AttentionOutput(NamedTuple):
    state: AttentionState
    gains: Array
    attn_distribution: Array


def _divisive_norm(td_raw: Array, sigma: Array) -> Array:
    """Reynolds & Heeger (2009) R_i = c_i² / (σ² + Σ c_j²)."""
    td_rect = jax.nn.relu(td_raw)
    td_sq = td_rect * td_rect
    denom = sigma * sigma + jnp.sum(td_sq) + jnp.asarray(1e-8, DTYPE)
    return td_sq / denom


@eqx.filter_jit
def attention_step(
    state: AttentionState,
    params: AttentionParams,
    assoc_activity: Array,
    bottom_up_errors: Array | None = None,
    *,
    global_ach: float | Array = 0.5,
    ne_level: float | Array = 0.5,
) -> AttentionOutput:
    """One attention update.

    * Top-down saliency = ``assoc_activity @ w_attn`` with NE inverse-U gain,
      then divisive normalisation.
    * Bottom-up saliency = ``|bottom_up_errors|`` normalised to sum 1
      (zero if ``None``).
    * Mix with weight ``bottom_up_weight``.
    * IOR suppresses recently attended columns.
    * Temporal smoothing of the distribution.
    * Column gain = ``1 + ACh · (w_i − mean(w)) · gain_strength`` (floor 0.1).
    """
    act = assoc_activity.astype(DTYPE)
    ach = jnp.asarray(global_ach, DTYPE)
    ne = jnp.asarray(ne_level, DTYPE)

    td_raw = act @ state.w_attn
    ne_proximity = jnp.maximum(0.0, 1.0 - (ne - params.ne_optimal) ** 2)
    ne_gain = 1.0 + params.ne_gain_strength * ne_proximity
    td_norm = _divisive_norm(td_raw * ne_gain, params.divisive_sigma)

    if bottom_up_errors is None:
        bu_norm = jnp.zeros_like(td_norm)
    else:
        bu_abs = jnp.abs(bottom_up_errors).astype(DTYPE)
        bu_norm = bu_abs / (jnp.sum(bu_abs) + jnp.asarray(1e-8, DTYPE))

    alpha = params.bottom_up_weight
    combined = alpha * bu_norm + (1.0 - alpha) * td_norm

    combined = jax.nn.relu(
        combined * (1.0 - _AHP_IOR_GAIN_REDUCTION * state.ior_trace)
    )
    combined = combined / (jnp.sum(combined) + jnp.asarray(1e-8, DTYPE))

    attn = (
        state.attn_weights * params.smoothing_decay
        + combined * (1.0 - params.smoothing_decay)
    )

    mean_w = jnp.mean(attn)
    ior_input = jax.nn.relu(attn - mean_w)
    ior = state.ior_trace * params.ior_decay + ior_input * (1.0 - params.ior_decay)
    ior = jnp.clip(ior, 0.0, 1.0)

    mean_a = jnp.mean(attn)
    gain_mod = (attn - mean_a) * params.gain_strength
    gains = jnp.maximum(1.0 + ach * gain_mod, jnp.asarray(0.1, DTYPE))

    new_state = AttentionState(
        attn_weights=attn,
        ior_trace=ior,
        w_attn=state.w_attn,
    )
    return AttentionOutput(state=new_state, gains=gains, attn_distribution=attn)


def attention_learn(
    state: AttentionState,
    params: AttentionParams,
    assoc_activity: Array,
    column_mean_rates: Array,
    gains: Array,
    *,
    clip_abs: float = 2.0,
) -> AttentionState:
    """Hebbian update of top-down weights.

    ``Δw[:, c] = lr · assoc · col_rate[c] · (gain[c] − 1)``.
    """
    act = assoc_activity.astype(DTYPE)
    signal = column_mean_rates.astype(DTYPE) * (gains - 1.0)
    dw = params.learning_rate * jnp.outer(act, signal)
    w_new = jnp.clip(state.w_attn + dw, -clip_abs, clip_abs)
    return eqx.tree_at(lambda s: s.w_attn, state, w_new)
