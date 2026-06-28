"""Fast-spiking PV+ interneuron pool — pure JAX.

Brunel & Wang (2003); Woodin, Ganguly & Poo (2003);
Isaacson & Scanziani (2011); Seamans & Yang (2004).

Replaces algorithmic k-WTA with biophysical E→I→E competition:
- FS AdEx interneurons (``a=0``, ``b=0`` — degenerate AdEx) with
  ``g_L = C_m / tau_m_inh`` so exp-Euler time constant matches
  the configured ``tau_m_inh``.
- E→I drive is conductance-based in units of nS; weights are
  pre-calibrated at init so that ``I = g · (E_exc − V_inh)`` reaches
  rheobase for the expected number of active excitatory inputs.
- I→E feedback splits into GABA-A (fast ~70%) and GABA-B (slow ~30%)
  conductances, recombined with a driving-force to ``E_inh`` that is
  self-limiting as ``V_exc → E_inh``.
- Inhibitory STDP: E→I Hebbian, I→E anti-Hebbian (Woodin 2003).
- DA D2 modulation of I→E gain exposed as a helper scalar (Hill),
  passed into ``ipool_step`` — no hidden mutable state.

Differences from legacy:
- SWS GABA-surge and DA-D2 gain are *parameters* of ``ipool_step`` so
  the caller owns any mode switching (required for JIT purity).
- Eligibility-style ``_trace_exc`` / ``_trace_inh`` kept as exp-filtered
  traces (``tau=20 ms``) — the legacy implementation did the same, only
  wrapped behind a per-step decay.
- Astrocyte ATP coupling dropped at this level; surfacing it at the
  cortex composition layer (Phase 2).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, BackendContext, split_key
from .state import NeuronParams, NeuronState, init_neuron_state
from .neuron import init_neuron_params, neuron_step


# =====================================================================
# Params / state
# =====================================================================


class IPoolParams(eqx.Module):
    """FS interneuron pool parameters."""

    inh: NeuronParams

    # Reversal potentials
    e_exc: Array           # excitatory reversal (0 mV typical)
    e_inh: Array           # inhibitory / Cl⁻ reversal (−75 mV typical)
    v_thresh_ref: Array    # reference V_thresh for driving-force normalisation

    # GABA decays + balance
    gaba_a_decay: Array
    gaba_b_decay: Array
    gaba_b_ratio: Array

    # Inhibitory STDP
    trace_decay: Array
    inh_stdp_lr: Array
    ei_balance_lr: Array

    # Static sizes
    n_exc: int = eqx.field(static=True)
    n_inh: int = eqx.field(static=True)


def init_ipool_params(
    ctx: BackendContext,
    n_exc: int,
    *,
    n_inh: int = 32,
    v_rest: float = -70.0,
    v_thresh: float = -55.0,
    v_reset: float = -75.0,
    v_spike_cutoff: float = -30.0,
    tau_m_inh: float = 8.0,
    delta_t: float = 2.0,
    C_m: float = 281.0,
    refrac_period_ms: float = 2.0,
    e_exc: float = 0.0,
    e_inh: float = -75.0,
    tau_gaba_a: float = 6.0,
    tau_gaba_b: float = 150.0,
    gaba_b_ratio: float = 0.25,
    tau_trace: float = 20.0,
    inh_stdp_lr: float = 1e-4,
    ei_balance_lr: float = 5e-5,
) -> IPoolParams:
    """Build params for an FS PV+ pool with biophysically consistent leak."""
    g_L = C_m / tau_m_inh
    inh = init_neuron_params(
        ctx,
        v_rest=v_rest, v_thresh=v_thresh, v_reset=v_reset,
        v_spike_cutoff=v_spike_cutoff, delta_t=delta_t,
        C_m=C_m, g_L=g_L, tau_w=tau_m_inh,     # tau_w irrelevant (a=b=0)
        a=0.0, b=0.0, refrac_period_ms=refrac_period_ms,
    )
    f = lambda x: jnp.asarray(x, DTYPE)
    return IPoolParams(
        inh=inh,
        e_exc=f(e_exc), e_inh=f(e_inh), v_thresh_ref=f(v_thresh),
        gaba_a_decay=f(ctx.decay(tau_gaba_a)),
        gaba_b_decay=f(ctx.decay(tau_gaba_b)),
        gaba_b_ratio=f(gaba_b_ratio),
        trace_decay=f(ctx.decay(tau_trace)),
        inh_stdp_lr=f(inh_stdp_lr),
        ei_balance_lr=f(ei_balance_lr),
        n_exc=n_exc, n_inh=n_inh,
    )


class IPoolState(eqx.Module):
    """FS interneuron pool state."""

    inh: NeuronState
    w_ei: Array          # (n_exc, n_inh) E→I weights (nS, pre-calibrated)
    w_ie: Array          # (n_inh, n_exc) I→E weights (nS)
    g_gaba_a: Array      # (n_exc,) fast GABA conductance on E pop
    g_gaba_b: Array      # (n_exc,) slow GABA conductance on E pop
    trace_exc: Array     # (n_exc,) pre-synaptic trace
    trace_inh: Array     # (n_inh,) post-synaptic trace


def _calibrate_ei_scale(
    params: IPoolParams, w_ei_mean: float, target_sparsity: float = 0.1,
) -> float:
    """Scale so that expected E→I drive reaches rheobase.

    ``I_rheo = g_L · (gap − Δ_T)`` and the raw weight product is
    ``expected_active · w_ei_mean``; we divide by the E_exc-driving
    force to convert to conductance (nS).
    """
    ncfg = params.inh
    gap = float(abs(ncfg.v_thresh - ncfg.v_rest))
    i_rheo = float(ncfg.g_L) * (gap - float(ncfg.delta_t))
    expected_active = max(1.0, min(3.0, params.n_exc * target_sparsity))
    input_gain = i_rheo / (expected_active * w_ei_mean)
    df = float(params.e_exc - ncfg.v_rest)
    return input_gain / max(df, 1.0)


def init_ipool_state(
    key: PRNGKey, params: IPoolParams,
    *,
    w_ei_mean: float = 0.5,
    w_ei_cv: float = 0.3,
    w_ie_mean: float = 0.5,
    w_ie_cv: float = 0.2,
    target_sparsity: float = 0.1,
    dtype=DTYPE,
) -> IPoolState:
    """Half-normal weights scaled by AdEx rheobase so E→I drives I to threshold."""
    k_ei, k_ie = split_key(key, 2)
    cond_scale = _calibrate_ei_scale(params, w_ei_mean, target_sparsity)

    w_ei = jnp.abs(
        jax.random.normal(k_ei, (params.n_exc, params.n_inh), dtype=dtype)
        * (w_ei_mean * w_ei_cv) + w_ei_mean
    ) * jnp.asarray(cond_scale, dtype)
    w_ie = jnp.abs(
        jax.random.normal(k_ie, (params.n_inh, params.n_exc), dtype=dtype)
        * (w_ie_mean * w_ie_cv) + w_ie_mean
    )

    inh_state = init_neuron_state(params.n_inh, v_rest=float(params.inh.v_rest))
    return IPoolState(
        inh=inh_state,
        w_ei=w_ei, w_ie=w_ie,
        g_gaba_a=jnp.zeros(params.n_exc, dtype),
        g_gaba_b=jnp.zeros(params.n_exc, dtype),
        trace_exc=jnp.zeros(params.n_exc, dtype),
        trace_inh=jnp.zeros(params.n_inh, dtype),
    )


# =====================================================================
# Step
# =====================================================================


class IPoolOutput(NamedTuple):
    state: IPoolState
    i_inh: Array          # (n_exc,) inhibitory current (pA-scale) on E pop
    inh_spikes: Array     # (n_inh,) for diagnostics


def ipool_step(
    state: IPoolState, params: IPoolParams, ctx: BackendContext,
    exc_spikes: Array,
    v_exc: Array,
    *,
    ie_gain: float | Array = 1.0,
    apply_stdp: bool = True,
) -> IPoolOutput:
    """One timestep of E→I→E inhibition with conductance-based drive.

    ``ie_gain`` multiplies the I→E feedback (carries DA D2 and
    SWS surge modulations — callers compose them as a single scalar).

    Returns an inhibitory *current* (pA-scale) the caller subtracts
    from excitatory ``I_syn``. The driving force normalises by
    ``V_thresh − E_inh`` so at the threshold the effective inhibition
    matches current-based models; below threshold it is reduced
    (self-limiting, Isaacson & Scanziani 2011).
    """
    exc = exc_spikes.astype(DTYPE)
    v_e = v_exc.astype(DTYPE)
    gain = jnp.asarray(ie_gain, DTYPE)

    # --- E→I drive (conductance-based) ---
    g_ei = exc @ state.w_ei                                  # (n_inh,) nS
    i_input = g_ei * (params.e_exc - state.inh.v)            # pA

    new_inh, inh_spikes = neuron_step(
        state.inh, params.inh, ctx, i_syn=i_input, g_syn=g_ei,
    )

    # --- I→E feedback split into GABA-A / GABA-B ---
    fb = (inh_spikes @ state.w_ie) * gain                    # (n_exc,)
    g_a = state.g_gaba_a * params.gaba_a_decay + (1.0 - params.gaba_b_ratio) * fb
    g_b = state.g_gaba_b * params.gaba_b_decay + params.gaba_b_ratio * fb

    # --- Inhibitory STDP traces ---
    tr_exc = state.trace_exc * params.trace_decay + exc
    tr_inh = state.trace_inh * params.trace_decay + inh_spikes

    # --- Weight updates (Dale-clamped) ---
    if apply_stdp:
        # E→I Hebbian: pre-trace × post-spike. Scalar gate on total
        # post-activity keeps compute cheap but stays JIT-safe.
        any_post = jnp.any(inh_spikes)
        dw_ei = params.inh_stdp_lr * (tr_exc[:, None] * inh_spikes[None, :])
        w_ei = jnp.where(any_post, state.w_ei + dw_ei, state.w_ei)
        w_ei = jnp.maximum(w_ei, 0.0)
        # I→E anti-Hebbian: homeostatic correction.
        any_pre = jnp.any(exc > 0.1)
        dw_ie = -params.ei_balance_lr * (tr_inh[:, None] * exc[None, :])
        w_ie = jnp.where(any_pre, state.w_ie + dw_ie, state.w_ie)
        w_ie = jnp.maximum(w_ie, 0.0)
    else:
        w_ei, w_ie = state.w_ei, state.w_ie

    # --- Output inhibition with self-limiting driving force ---
    ref_drive = params.v_thresh_ref - params.e_inh
    driving = jnp.clip((v_e - params.e_inh) / jnp.maximum(ref_drive, 1.0), 0.0, None)
    g_total = g_a + g_b
    i_inh = g_total * driving

    new_state = IPoolState(
        inh=new_inh, w_ei=w_ei, w_ie=w_ie,
        g_gaba_a=g_a, g_gaba_b=g_b,
        trace_exc=tr_exc, trace_inh=tr_inh,
    )
    return IPoolOutput(state=new_state, i_inh=i_inh, inh_spikes=inh_spikes)


# =====================================================================
# Helpers
# =====================================================================


def ipool_da_gain(
    da_level: float | Array,
    *,
    ec50: float = 0.3, hill_n: float = 1.2, density: float = 0.4,
) -> Array:
    """Hill-equation D2 modulation of I→E gain (Seamans & Yang 2004).

    Matches ``receptor.py`` conventions; density reflects PV+ D2 density
    (~40% of MSN density).
    """
    da = jnp.clip(jnp.asarray(da_level, DTYPE), 0.0, 1.0)
    resp = (da ** hill_n) / (da ** hill_n + ec50 ** hill_n)
    return 1.0 + density * resp


def ipool_reset_transient(state: IPoolState, params: IPoolParams) -> IPoolState:
    """Zero dynamical state; preserve learned ``w_ei`` / ``w_ie``."""
    return IPoolState(
        inh=init_neuron_state(params.n_inh, v_rest=float(params.inh.v_rest)),
        w_ei=state.w_ei, w_ie=state.w_ie,
        g_gaba_a=jnp.zeros(params.n_exc, DTYPE),
        g_gaba_b=jnp.zeros(params.n_exc, DTYPE),
        trace_exc=jnp.zeros(params.n_exc, DTYPE),
        trace_inh=jnp.zeros(params.n_inh, DTYPE),
    )
