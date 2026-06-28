"""
Neuron — pure functional AdEx integrate-and-fire step.

Replaces the legacy class-based ``AdExLayer`` with stateless functions
operating on ``NeuronState`` / ``NeuronParams`` pytrees.

Membrane (Brette & Gerstner 2005):
    C_m dV/dt = -g_L(V - E_L) + g_L·Δ_T·exp((V-V_T)/Δ_T) + I_syn - w
    τ_w dw/dt = a·(V - E_L) - w
Spike:  V ≥ V_cutoff  ⇒  V ← V_reset,  w ← w + b

Integration uses exponential Euler (Rotter & Diesmann 1999) — A-stable,
handles the AdEx exp term + NMDA stiffness without substepping.

Neuromorphic note: setting ``a=0, b=0, delta_t→0`` degenerates the kernel
to pure LIF — no code branch needed.  Akida / TrueNorth export clamps
params at conversion time; DYNAPs / Loihi 2 / SpiNNaker 2 run the full
AdEx natively.

STDP + eligibility traces live in a separate module (Phase 1.5 will add
``core/plasticity.py``) because they operate on *pairs* of populations
(pre + post) and the connection fabric.  This step is population-local.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array, BackendContext
from .state import NeuronState, NeuronParams


# ======================================================================
# Optional astrocyte modulation — lightweight pytree
# ======================================================================


class AstroMod(NamedTuple):
    """Per-neuron astrocyte modulation factors (Krok 1.3).

    ``threshold_shift`` is added to V_T (mV); ``leak_gain`` multiplies
    g_L.  Both have shape ``(n,)`` matching the population.  Pass
    ``None`` when astrocyte coupling is disabled.
    """

    threshold_shift: Array
    leak_gain: Array


# ======================================================================
# Parameter factory — builds NeuronParams from raw constants
# ======================================================================


def init_neuron_params(
    ctx: BackendContext,
    *,
    v_rest: float = -70.0,
    v_thresh: float = -55.0,
    v_reset: float = -75.0,
    v_spike_cutoff: float = -30.0,
    delta_t: float = 2.0,
    C_m: float = 281.0,
    g_L: float = 30.0,
    tau_w: float = 144.0,
    a: float = 4.0,
    b: float = 80.5,
    refrac_period_ms: float = 2.0,
) -> NeuronParams:
    """Build a ``NeuronParams`` pytree.

    Defaults: Regular-Spiking pyramidal (Brette & Gerstner 2005 Tab. 1).
    Other neuron types change ``(a, b, tau_w)``:
        Fast-Spiking  (FS):  a=0,  b=0,   tau_w irrelevant
        Intr. Bursting(IB):  a=4,  b=150, tau_w=30
        Late Spiking  (LS):  a=80, b=40,  tau_w=720
    Pure LIF:  a=0, b=0, delta_t→0.01 (near-zero for numeric safety).
    """
    return NeuronParams(
        v_rest=jnp.asarray(v_rest, DTYPE),
        v_thresh=jnp.asarray(v_thresh, DTYPE),
        v_reset=jnp.asarray(v_reset, DTYPE),
        v_spike_cutoff=jnp.asarray(v_spike_cutoff, DTYPE),
        delta_t=jnp.asarray(delta_t, DTYPE),
        C_m=jnp.asarray(C_m, DTYPE),
        g_L=jnp.asarray(g_L, DTYPE),
        tau_w=jnp.asarray(tau_w, DTYPE),
        a=jnp.asarray(a, DTYPE),
        b=jnp.asarray(b, DTYPE),
        refrac_period=jnp.asarray(int(ctx.ms_to_steps(refrac_period_ms)), jnp.int32),
        w_decay=ctx.decay(tau_w),
        w_gain=ctx.complement(tau_w),
    )


# ======================================================================
# Core step
# ======================================================================


def _adex_drift(
    v: Array,
    w: Array,
    i_syn: Array,
    g_syn: Array,
    params: NeuronParams,
    eff_v_thresh: Array,
    eff_g_L: Array,
) -> tuple[Array, Array]:
    """Return ``(F_V, J_V)`` for the exp-Euler integrator.

    F_V(V) = (1/C_m) · [-g_L(V-E_L) + g_L·Δ_T·exp((V-V_T)/Δ_T) + I_syn - w]
    J_V(V) = ∂F/∂V = (1/C_m) · [-g_L + g_L·exp((V-V_T)/Δ_T) - g_syn]

    ``g_syn`` is the total synaptic conductance from ``synapse_step``
    (conductance-based synapses contribute ``-g_syn`` to the Jacobian
    because I_syn = g·(E-V)).
    """
    exp_arg = jnp.clip((v - eff_v_thresh) / params.delta_t, -20.0, 10.0)
    exp_term = jnp.exp(exp_arg)
    inv_Cm = 1.0 / params.C_m

    F_v = inv_Cm * (
        -eff_g_L * (v - params.v_rest)
        + eff_g_L * params.delta_t * exp_term
        + i_syn
        - w
    )
    J_v = inv_Cm * (-eff_g_L + eff_g_L * exp_term - g_syn)
    return F_v, J_v


def neuron_step(
    state: NeuronState,
    params: NeuronParams,
    ctx: BackendContext,
    i_syn: Array,
    g_syn: Array,
    *,
    astro: Optional[AstroMod] = None,
    v_thresh_adaptive: Optional[Array] = None,
    ne_level: Array | float = 0.0,
    ne_thresh_drop: float = 3.0,
) -> tuple[NeuronState, Array]:
    """One AdEx integration step.

    Args:
        state:  current ``NeuronState``.
        params: static (per-run) ``NeuronParams``.
        ctx:    backend context (exp_euler_step, dt).
        i_syn:  ``(n,)`` total synaptic current from ``synapse_step``.
        g_syn:  ``(n,)`` total synaptic conductance (Jacobian term).
        astro:  optional per-zone V_T shift + g_L gain (Krok 1.3).
        v_thresh_adaptive: optional homeostatic threshold ``(n,)``;
                overrides ``params.v_thresh`` when provided.
        ne_level: scalar NE neuromodulator level (reduces threshold).
        ne_thresh_drop: mV reduction in threshold per unit NE.

    Returns:
        ``(new_state, spikes)`` where ``spikes`` is a float32 ``(n,)``
        array of 0/1 values (post-reset, post-refractory).
    """
    # 1. Effective threshold + leak (homeostatic + astrocyte + NE)
    base_thresh = v_thresh_adaptive if v_thresh_adaptive is not None else params.v_thresh
    if astro is not None:
        eff_v_thresh = base_thresh + astro.threshold_shift
        eff_g_L = params.g_L * astro.leak_gain
    else:
        eff_v_thresh = base_thresh
        eff_g_L = params.g_L
    eff_v_thresh = eff_v_thresh - jnp.asarray(ne_level, DTYPE) * ne_thresh_drop

    # 2. Refractory countdown (clamped ≥ 0)
    new_refrac = jnp.maximum(state.refrac - 1, 0)
    in_refrac = new_refrac > 0

    # 3. Drift + exp-Euler on V (using effective V_T / g_L)
    F_v, J_v = _adex_drift(
        state.v, state.w_adapt, i_syn, g_syn, params, eff_v_thresh, eff_g_L,
    )
    v_integrated = ctx.exp_euler_step(state.v, F_v, J_v)
    v_integrated = jnp.minimum(v_integrated, jnp.asarray(50.0, DTYPE))
    v_held = jnp.where(in_refrac, params.v_reset, v_integrated)

    # 4. Spike detection — at or above cutoff, not in refractory
    spike_threshold = jnp.minimum(params.v_spike_cutoff, eff_v_thresh)
    spiked = (v_held >= spike_threshold) & ~in_refrac
    spikes_f = spiked.astype(DTYPE)

    # 5. Reset V, trigger adaptation jump, arm refractory, bump post trace
    v_post = jnp.where(spiked, params.v_reset, v_held)
    w_after_spike = state.w_adapt + spikes_f * params.b
    refrac_after = jnp.where(spiked, params.refrac_period, new_refrac).astype(jnp.int32)

    # 6. Subthreshold adaptation: w = w·decay + a·(V−E_L)·gain
    w_new = (
        w_after_spike * params.w_decay
        + params.a * (v_post - params.v_rest) * params.w_gain
    )

    # 7. Post-synaptic STDP trace: decay + event-based increment
    #    (pre-trace lives in the *upstream* layer's state)
    x_post_new = state.x_post + spikes_f
    # x_pre left untouched here — managed by STDP module with pre layer.

    new_state = eqx.tree_at(
        lambda s: (s.v, s.w_adapt, s.refrac, s.x_post, s.spikes),
        state,
        (v_post, w_new, refrac_after, x_post_new, spikes_f),
    )
    return new_state, spikes_f


# ======================================================================
# Convenience: STDP pre-trace update (independent of neuron_step)
# ======================================================================


def decay_pre_trace(
    state: NeuronState,
    decay: Array | float,
    incoming_spikes: Array,
) -> NeuronState:
    """Decay ``x_pre`` and increment on incoming spikes.

    Used when ``x_pre`` tracks the post-synaptic neuron's *incoming*
    spike rate (e.g. for per-target heterosynaptic rules).  Shape of
    ``incoming_spikes`` must match ``state.x_pre``.
    """
    new_xpre = state.x_pre * decay + incoming_spikes
    return eqx.tree_at(lambda s: s.x_pre, state, new_xpre)
