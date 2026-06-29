"""Spatial attention as precision gain over sensory sub-fields (§4).

Attention *is* precision in this substrate (Feldman & Friston 2010): no
separate routing mechanism, just a per-dimension gain on the sensory
node's precision Π.  :func:`core.pc_active.scale_node_precision` already
multiplies a node's Π by an arbitrary array, so a ``(sensory_dim,)`` gain
vector — high on the attended channels, low elsewhere — *is* spatial
attention, fed through the very same ``precision_gains`` hook the
cognitive step already exposes.

The gain is built from two classic ingredients, both pure / step-unit:

* **Divisive normalisation** (Reynolds & Heeger 2009): a field's drive is
  ``s² / (σ² + Σ s²)`` — competition across channels, so attention is a
  limited resource that sharpens contrast.
* **Inhibition of return** (Posner & Cohen 1984): a slow trace of
  recently attended fields suppresses them, so attention keeps sampling
  new sub-fields instead of locking on.  The suppression strength is the
  cortical AHP gain reduction (Madison & Nicoll 1984), a biophysical
  constant, not a free knob.

Bottom-up saliency defaults to the sensory node's own prediction error
``|ε_sensory|`` — the substrate's "what is currently surprising" signal —
so attention is drawn to the channels the model is failing to explain.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array
from .pc_graph import PCGraphParams, PCGraphState, pc_graph_errors, REGION_INDEX
from .pc_precision import step_alpha


# Afterhyperpolarisation (AHP) peak gain reduction at sustained firing in
# cortical pyramidal cells (Madison & Nicoll 1984: ~30% conductance
# reduction at 20 Hz).  The biophysical substrate of inhibition-of-return
# (Mirpour & Bisley 2012) — a conserved Ca²⁺-activated K⁺ density, not a
# tunable coefficient.
_AHP_IOR_GAIN_REDUCTION: float = 0.3


class AttentionParams(eqx.Module):
    """Divisive-norm + IOR hyper-params (Reynolds & Heeger 2009; Posner 1984)."""

    gain_strength: Array       # how far attention pushes Π from 1
    divisive_sigma: Array      # semi-saturation constant of the normalisation
    ior_decay: Array           # per-step retention of the IOR trace
    ior_gain_reduction: Array  # AHP suppression strength (biophysical)
    gain_floor: Array          # gains never fall below this


def init_attention_params(
    *,
    gain_strength: float = 2.0,
    divisive_sigma: float = 1.0,
    tau_ior: float = 20.0,
    gain_floor: float = 0.1,
    dtype=DTYPE,
) -> AttentionParams:
    """``tau_ior`` is the IOR trace timescale in cognitive steps.

    The trace decays with retention ``exp(−1/tau_ior)`` (step-unit, no
    wall clock); larger ``tau_ior`` ⇒ longer-lasting suppression of a
    visited field.
    """
    f = lambda x: jnp.asarray(x, dtype)
    ior_decay = jnp.exp(-1.0 / jnp.asarray(tau_ior, dtype))
    return AttentionParams(
        gain_strength=f(gain_strength),
        divisive_sigma=f(divisive_sigma),
        ior_decay=ior_decay.astype(dtype),
        ior_gain_reduction=f(_AHP_IOR_GAIN_REDUCTION),
        gain_floor=f(gain_floor),
    )


class AttentionState(eqx.Module):
    """Just the inhibition-of-return trace over the sensory sub-fields."""

    ior_trace: Array           # (sensory_dim,)


def init_attention_state(sensory_dim: int, *, dtype=DTYPE) -> AttentionState:
    return AttentionState(ior_trace=jnp.zeros(int(sensory_dim), dtype))


def _divisive_norm(saliency: Array, sigma: Array) -> Array:
    """Reynolds & Heeger (2009) ``R_i = s_i² / (σ² + Σ s_j²)``."""
    s = jax.nn.relu(saliency)
    s_sq = s * s
    denom = sigma * sigma + jnp.sum(s_sq) + jnp.asarray(1e-8, DTYPE)
    return s_sq / denom


class AttentionOutput(NamedTuple):
    state: AttentionState      # IOR trace advanced one step
    gains: Array               # (sensory_dim,) per-field precision gain
    distribution: Array        # (sensory_dim,) normalised attention weights


def attention_step(
    state: AttentionState, params: AttentionParams, saliency: Array,
) -> AttentionOutput:
    """One attention update → per-field sensory precision gain.

    ``saliency`` is the per-field bottom-up drive (e.g. ``|ε_sensory|``).
    Divisive normalisation makes the fields compete; the IOR trace
    suppresses recently attended ones; the result is a gain
    ``1 + strength·(attn − mean(attn))`` floored at ``gain_floor``.
    Feed ``gains`` as the sensory entry of ``precision_gains``.
    """
    sal = saliency.astype(DTYPE)
    norm = _divisive_norm(sal, params.divisive_sigma)
    # IOR suppression, then renormalise to a distribution.
    suppressed = jax.nn.relu(norm * (1.0 - params.ior_gain_reduction * state.ior_trace))
    distribution = suppressed / (jnp.sum(suppressed) + jnp.asarray(1e-8, DTYPE))

    mean_a = jnp.mean(distribution)
    gains = jnp.maximum(
        params.gain_floor,
        1.0 + params.gain_strength * (distribution - mean_a),
    )

    # Advance the IOR trace: above-mean (attended) fields accumulate
    # suppression, decaying back over ``tau_ior``.
    ior_input = jax.nn.relu(distribution - mean_a)
    ior = state.ior_trace * params.ior_decay + ior_input * (1.0 - params.ior_decay)
    ior = jnp.clip(ior, 0.0, 1.0)

    return AttentionOutput(
        state=AttentionState(ior_trace=ior.astype(DTYPE)),
        gains=gains.astype(DTYPE),
        distribution=distribution.astype(DTYPE),
    )


def sensory_error_saliency(
    graph: PCGraphState, gparams: PCGraphParams,
    *, sensory_idx: int | None = None,
) -> Array:
    """Bottom-up saliency = ``|ε|`` at the sensory node (what is surprising)."""
    s_idx = REGION_INDEX["sensory"] if sensory_idx is None else int(sensory_idx)
    return jnp.abs(pc_graph_errors(graph, gparams)[s_idx])


def attention_precision_gains(
    state: AttentionState, params: AttentionParams,
    graph: PCGraphState, gparams: PCGraphParams,
    *, sensory_idx: int | None = None,
) -> tuple[AttentionState, dict[int, Array]]:
    """Convenience: derive sensory saliency, step, and return the gain dict.

    Returns ``(new_attention_state, {sensory_idx: gains})`` ready to pass
    as ``precision_gains`` to
    :func:`core.pc_brain.pc_brain_cognitive_step`.
    """
    s_idx = REGION_INDEX["sensory"] if sensory_idx is None else int(sensory_idx)
    sal = sensory_error_saliency(graph, gparams, sensory_idx=s_idx)
    out = attention_step(state, params, sal)
    return out.state, {s_idx: out.gains}
