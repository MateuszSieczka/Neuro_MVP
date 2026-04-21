"""PFC — thin wrapper around ``working_memory`` for embodied action loops.

Frank & Badre (2012): PFC hierarchical gating biases basal-ganglia
action selection by providing a persistent content signal that
outlasts a single decision cycle.  Hasselmo (2005): theta-phase
encoding vs retrieval — we gate the WM content input with
``1 + amp·cos(θ + φ_pfc)`` so encoding peaks at θ=0 (opposite trough
where gamma is maximal → retrieval).

This module does three things:
1. Delegates core dynamics to ``wm_step`` (content AdEx + gate AdEx).
2. Maintains a slow output-rate EMA projected to the BG as an
   additional striatal drive.
3. Provides a ``done``-gated reset so the attractor is cleared between
   episodes.

PFC content dimensions default to n=64 (Durstewitz 2000); gate
dimensions default to n_gate=32.  Both are static JAX shapes.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, BackendContext
from .working_memory import (
    WMParams, WMState, init_wm_params, init_wm_state,
    wm_step, wm_reset_transient,
)


# =====================================================================
# Params / state
# =====================================================================


class PFCParams(eqx.Module):
    """PFC = WM + slow output-rate EMA decay + theta-phase gain depth."""

    wm: WMParams
    out_rate_decay: Array
    theta_gain_depth: Array  # amplitude of theta excitability modulation

    input_size: int = eqx.field(static=True)
    n_content: int = eqx.field(static=True)


class PFCState(eqx.Module):
    wm: WMState
    # Per-neuron low-pass of content spikes (τ ~ 100 ms).  This is the
    # signal projected into basal-ganglia striatum.
    output_rate: Array  # (n_content,)


def init_pfc_params(
    ctx: BackendContext,
    input_size: int,
    *,
    n_content: int = 64,
    n_gate: int = 32,
    tau_out_ms: float = 100.0,
    theta_gain_depth: float = 0.1,
    **wm_kwargs,
) -> PFCParams:
    wm_p = init_wm_params(
        ctx, n_in=input_size, n=n_content, n_gate=n_gate, **wm_kwargs,
    )
    return PFCParams(
        wm=wm_p,
        out_rate_decay=jnp.asarray(ctx.decay(tau_out_ms), DTYPE),
        theta_gain_depth=jnp.asarray(theta_gain_depth, DTYPE),
        input_size=int(input_size),
        n_content=int(n_content),
    )


def init_pfc_state(key: PRNGKey, params: PFCParams, *, dtype=DTYPE) -> PFCState:
    wm_s = init_wm_state(key, params.wm, dtype=dtype)
    return PFCState(
        wm=wm_s,
        output_rate=jnp.zeros(params.n_content, dtype=dtype),
    )


# =====================================================================
# Step
# =====================================================================


class PFCOutput(NamedTuple):
    state: PFCState
    content_rate: Array  # (n_content,) — projected to BG striatum


@eqx.filter_jit
def pfc_step(
    state: PFCState,
    params: PFCParams,
    ctx: BackendContext,
    cortex_belief: Array,
    ach: float | Array,
    da: float | Array,
    key: PRNGKey,
    *,
    theta_phase: float | Array = 0.0,
    phase_offset: float | Array = 0.0,
) -> PFCOutput:
    """Advance PFC one ``dt``.

    ``cortex_belief`` is L2/3 state rate from sensory cortex (size
    ``params.input_size``).  ACh and DA drive the conjunction gate
    (O'Reilly & Frank 2006).  Theta phase multiplicatively modulates
    the content feedforward conductance (Hasselmo 2005 encoding).
    """
    theta = jnp.asarray(theta_phase, DTYPE)
    phi = jnp.asarray(phase_offset, DTYPE)
    rg = jnp.asarray(1.0, DTYPE) + params.theta_gain_depth * jnp.cos(theta + phi)

    wm_out = wm_step(
        state.wm, params.wm, ctx,
        external_input=cortex_belief.astype(DTYPE),
        ach=ach, da=da, key=key, receptor_gain=rg,
    )

    # Slow rate EMA projected to BG — this is the PFC "goal" signal.
    out_rate = (
        state.output_rate * params.out_rate_decay
        + wm_out.spikes * (1.0 - params.out_rate_decay)
    )
    new_state = PFCState(wm=wm_out.state, output_rate=out_rate)
    return PFCOutput(state=new_state, content_rate=out_rate)


def pfc_reset_transient(state: PFCState, params: PFCParams) -> PFCState:
    """Clear PFC dynamics on episode boundary; keep learned weights.

    Biologically: at task-reset the attractor context is abandoned
    (Fuster 2001 — PFC cells signal goal; goal finished → new slot).
    """
    wm_reset = wm_reset_transient(state.wm, params.wm)
    return PFCState(
        wm=wm_reset,
        output_rate=jnp.zeros_like(state.output_rate),
    )


def pfc_select_reset(
    state: PFCState, params: PFCParams, done: float | Array,
) -> PFCState:
    """JAX-safe conditional reset: ``lerp(done, current, reset)``.

    Used inside ``action_brain_step`` where ``done`` is a float-scalar
    transition flag.  Implemented branch-free via ``jnp.where`` over
    the reset state and the current state — both must be same pytree
    shape, which they are by construction.
    """
    reset = pfc_reset_transient(state, params)
    d = jnp.asarray(done, DTYPE)
    # Use structure-preserving where: fields are all same-shape arrays.
    import jax
    return jax.tree_util.tree_map(
        lambda cur, rst: jnp.where(d > 0.5, rst, cur),
        state, reset,
    )
