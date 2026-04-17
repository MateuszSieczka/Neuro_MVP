"""
Plasticity — pure functional STDP + three-factor eligibility.

Reference:
  Bi & Poo (1998, 2001) — pair-based STDP window.
  Gerstner et al. (2018) — eligibility traces & three-factor rule.
  Izhikevich (2007) — dopamine-gated eligibility (DA-STDP).

All functions take and return pytrees; the pre-synaptic and
post-synaptic STDP traces live on the respective ``NeuronState``
objects (``x_pre`` on the pre layer, ``x_post`` on the post layer;
``spikes`` carries the last-step binary spike).

Three-factor rule (Izhikevich 2007):
    Δw = lr · M(t) · E(t)       M = modulator (DA, ACh, error signal)
                                 E = eligibility (STDP-weighted)
    dE/dt = -E/τ_e  +  stdp_pair(pre_trace, post_trace, pre_spk, post_spk)

Pair-based STDP window (Bi & Poo 1998):
    pre_spike ∧ prior post-trace > 0  ⇒  LTD of -a_minus · x_post
    post_spike ∧ prior pre-trace > 0  ⇒  LTP of +a_plus  · x_pre
(The sign convention here treats ``a_minus`` as the magnitude of LTD.)

We avoid explicit spike-time counters (the legacy code used
``t_since_*_spike`` for a ±20 ms window) because the decaying traces
already implement a smooth exponential window — Gerstner's original
form.  This is both more biological and vmap-friendly.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array, BackendContext
from .state import EligibilityState


# ======================================================================
# Parameters
# ======================================================================


class STDPParams(eqx.Module):
    """Precomputed STDP time constants as per-step decay multipliers.

    ``pre_decay``   — applied to pre-synaptic trace  (τ_plus)
    ``post_decay``  — applied to post-synaptic trace (τ_minus)
    ``elig_decay``  — applied to eligibility trace   (τ_e)
    ``a_plus / a_minus`` — LTP / LTD amplitudes (Bi & Poo 2001: 1.05× ratio).
    """

    a_plus: Array
    a_minus: Array
    pre_decay: Array
    post_decay: Array
    elig_decay: Array


def init_stdp_params(
    ctx: BackendContext,
    *,
    a_plus: float = 0.01,
    a_minus: float = 0.0105,
    tau_plus: float = 17.0,
    tau_minus: float = 34.0,
    tau_eligibility: float = 20.0,
) -> STDPParams:
    f = lambda x: jnp.asarray(x, DTYPE)
    return STDPParams(
        a_plus=f(a_plus),
        a_minus=f(a_minus),
        pre_decay=ctx.decay(tau_plus),
        post_decay=ctx.decay(tau_minus),
        elig_decay=ctx.decay(tau_eligibility),
    )


# ======================================================================
# Trace updates (population-local)
# ======================================================================


def update_pre_trace(
    x_pre: Array, pre_spikes: Array, decay: Array,
) -> Array:
    """Decay pre-synaptic STDP trace and increment on pre spikes.

    Called on the PRE layer's state each step.  ``x_pre`` has the
    pre-layer's shape ``(n_pre,)``.
    """
    return x_pre * decay + pre_spikes


def update_post_trace(
    x_post: Array, post_spikes: Array, decay: Array,
) -> Array:
    """Decay post-synaptic STDP trace and increment on post spikes."""
    return x_post * decay + post_spikes


# ======================================================================
# Eligibility update (pair-based STDP → eligibility)
# ======================================================================


def stdp_pair_update(
    elig: EligibilityState,
    params: STDPParams,
    *,
    pre_spikes: Array,       # (n_pre,)  float32 0/1
    post_spikes: Array,      # (n_post,) float32 0/1
    x_pre: Array,            # (n_pre,)  pre trace at THIS step (pre-update)
    x_post: Array,           # (n_post,) post trace at THIS step (pre-update)
) -> EligibilityState:
    """Pair-based STDP accumulation into eligibility (Gerstner 2018).

    Event-driven outer-product updates:
        LTP:  post spiked now → credit every pre that has x_pre > 0
                Δe[i,j] += +a_plus  · x_pre[i]  · post_spikes[j]
        LTD:  pre spiked now  → punish every post that has x_post > 0
                Δe[i,j] += -a_minus · pre_spikes[i] · x_post[j]
    Plus exponential decay of the eligibility trace.

    Sign convention: eligibility is the un-modulated pending weight
    change; apply the third factor (``weight_update``) to commit.
    """
    # Decay first, then accumulate new evidence
    e_decayed = elig.e * params.elig_decay

    ltp = params.a_plus * jnp.outer(x_pre, post_spikes)
    ltd = params.a_minus * jnp.outer(pre_spikes, x_post)

    return EligibilityState(e=(e_decayed + ltp - ltd).astype(DTYPE))


# ======================================================================
# Three-factor rule (commit eligibility → weight change)
# ======================================================================


def weight_update_three_factor(
    weights: Array,
    elig: EligibilityState,
    *,
    lr: Array | float,
    modulator: Array | float,
    mask: Array | None = None,
    clip_abs: float | None = None,
) -> Array:
    """Apply a three-factor weight update (Izhikevich 2007).

        Δw[i,j] = lr · modulator · e[i,j]

    ``modulator`` may be a scalar (global DA) or an array broadcastable
    to ``weights`` (per-synapse attention-gated, error-neuron precision).
    ``mask`` (optional) is a boolean array used to freeze some synapses
    (e.g. only plastic on active sparse slots).

    When ``clip_abs`` is provided, weights are clipped to
    ``[-clip_abs, clip_abs]`` after the update — cheap stability guard
    that's still far less restrictive than the legacy hard-clip.
    """
    dw = jnp.asarray(lr, DTYPE) * jnp.asarray(modulator, DTYPE) * elig.e
    if mask is not None:
        dw = jnp.where(mask, dw, 0.0)
    new_w = weights + dw
    if clip_abs is not None:
        new_w = jnp.clip(new_w, -clip_abs, clip_abs)
    return new_w.astype(DTYPE)


# ======================================================================
# Convenience: combined STDP pair + modulated weight update
# ======================================================================


class PlasticityOutput(NamedTuple):
    elig: EligibilityState
    weights: Array


def stdp_step(
    weights: Array,
    elig: EligibilityState,
    params: STDPParams,
    *,
    pre_spikes: Array,
    post_spikes: Array,
    x_pre: Array,
    x_post: Array,
    lr: Array | float = 1.0,
    modulator: Array | float = 1.0,
    mask: Array | None = None,
    clip_abs: float | None = None,
) -> PlasticityOutput:
    """Run one full plasticity step: update eligibility, apply weight delta.

    Returns the new eligibility and new weights.  Shape-preserving, so
    safe inside ``jax.jit`` and ``lax.scan`` for arbitrary connection
    fabrics (dense matrices OR sparse ``weights``-array slots).
    """
    new_elig = stdp_pair_update(
        elig, params,
        pre_spikes=pre_spikes, post_spikes=post_spikes,
        x_pre=x_pre, x_post=x_post,
    )
    new_w = weight_update_three_factor(
        weights, new_elig,
        lr=lr, modulator=modulator, mask=mask, clip_abs=clip_abs,
    )
    return PlasticityOutput(elig=new_elig, weights=new_w)
