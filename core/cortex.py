"""Cortical area — canonical microcircuit (L4 / L2-3 / L5) in pure JAX.

References:
  Douglas & Martin (2004) — canonical cortical microcircuit
  Bastos et al. (2012)    — canonical microcircuit for predictive coding
  Markov et al. (2014)    — cortical interareal connectivity
  Rao & Ballard (1999)    — predictive coding in visual cortex
  Larkum (2013)           — L5 dendrite as perceptual gate

Architecture
------------
A single cortical area is a composition of three populations wired
together by local connections and local PV\u207a inhibition:

  ff_input \u2192 [L4 RS + IPool] \u2192 [L2/3 ErrorNeuron]   (PC: state vs error)
                                         \u2192 [L5 RS + IPool] \u2192 subcortical out
  td_prediction \u2192 L2/3 external_prediction           (FB from higher area)

Outputs (consumed by ``brain_graph`` for inter-area wiring):
  - ``ff_out``  = L2/3 error rate      \u2192 next area's L4 input (Markov SLN FF)
  - ``belief``  = L2/3 state rate      \u2192 working memory / internal readout
  - ``l5_rate`` = L5 spike-rate EMA    \u2192 BG striatum, cerebellar mossy,
                                         FB to lower area (as td_prediction)

Critical design choices
-----------------------
* Body-agnostic: no knowledge of joints, phonemes, pixels. Inputs and
  outputs are plain ``Array``\\s; modality-specific encoders live in
  ``sensory/``.
* Inter-area projections are NOT inside this module. ``brain_graph``
  owns matrix ``W_{A \u2192 B}`` of shape ``(n_l5_A, n_l23_state_B)`` for
  TD feedback, ``(n_l23_error_A, n_l4_B)`` for FF.
* Each layer reuses existing JAX primitives (``neuron_step``,
  ``ipool_step``, ``en_step``) \u2014 no new biophysics.
* L4 and L5 sparsity is set via IPool ``target_sparsity``; L2/3 sparsity
  emerges from PC competition (state vs error).
* Neuromodulation is threaded explicitly: ACh gates L2/3 gain,
  DA/NE scale IPool feedback and trace timescales through callers.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, BackendContext, split_key
from .state import NeuronParams, NeuronState, init_neuron_state
from .neuron import init_neuron_params, neuron_step
from .interneuron import (
    IPoolParams, IPoolState, init_ipool_params, init_ipool_state,
    ipool_step, ipool_reset_transient,
)
from .error_neuron import (
    ErrorNeuronParams, ErrorNeuronState,
    init_error_neuron_params, init_error_neuron_state,
    en_step, en_update_weights, en_receive_prediction,
    en_belief, en_prediction_error_rate, en_reset_transient,
)


# =====================================================================
# Params / State / IO containers
# =====================================================================


class CorticalAreaParams(eqx.Module):
    """Static parameters for a generic cortical area (no body coupling)."""

    # L4 (input granular)
    l4_ncfg: NeuronParams
    l4_ipool: IPoolParams
    l4_cond_scale: Array       # PSP-preserving boost for l4_w_in @ ff

    # L2/3 (predictive coding — ErrorNeuron handles its own weights)
    l23: ErrorNeuronParams

    # L5 (deep output)
    l5_ncfg: NeuronParams
    l5_ipool: IPoolParams
    l5_cond_scale: Array       # boost for l23_state \u2192 L5 projection

    # Channels / traces common
    e_exc: Array
    e_inh: Array
    nmda_decay: Array
    ampa_frac: Array
    rate_decay_l4: Array
    rate_decay_l5: Array

    # Sizes (static)
    input_size: int = eqx.field(static=True)
    n_l4: int = eqx.field(static=True)
    n_l23_state: int = eqx.field(static=True)
    n_l23_error: int = eqx.field(static=True)
    n_l5: int = eqx.field(static=True)
    l4_target_sparsity: float = eqx.field(static=True)
    l5_target_sparsity: float = eqx.field(static=True)


class CorticalAreaState(eqx.Module):
    # L4
    l4_nstate: NeuronState
    l4_ipool: IPoolState
    w_l4_in: Array            # (input_size, n_l4) excitatory
    g_nmda_l4: Array          # (n_l4,)
    rate_l4: Array            # (n_l4,) spike-rate EMA

    # L2/3 PC
    l23: ErrorNeuronState

    # L5
    l5_nstate: NeuronState
    l5_ipool: IPoolState
    w_l23_l5: Array           # (n_l23_state, n_l5) excitatory (L2/3 belief \u2192 L5)
    g_nmda_l5: Array
    rate_l5: Array

    # Plasticity traces (optional, for STDP on L4 & L5 weights)
    x_pre_l4: Array           # (input_size,)
    x_post_l4: Array          # (n_l4,)
    x_pre_l5: Array           # (n_l23_state,) i.e. L2/3 state rate trace
    x_post_l5: Array          # (n_l5,)


class CorticalInputs(NamedTuple):
    """Per-step drives to a cortical area.

    ``ff_input`` is treated as an excitatory drive (spike vector or rate
    vector in [0, 1]). ``td_prediction`` is a pre-projected (n_l23_state,)
    array injected into L2/3 via ``en_receive_prediction``.
    """

    ff_input: Array
    td_prediction: Array | None = None
    ach: Array | float = 0.5
    da: Array | float = 0.5
    ne: Array | float = 0.5
    receptor_gain: Array | float = 1.0


class CorticalOutput(NamedTuple):
    state: CorticalAreaState
    # Main externally consumed signals
    ff_out: Array          # (n_l23_error,) error rate \u2192 next area L4
    belief: Array          # (n_l23_state,) L2/3 state rate \u2192 WM / TD source
    l5_rate: Array         # (n_l5,) deep rate \u2192 BG, cerebellum, motor
    # Raw spikes for downstream primitives (e.g. oscillator phase reset)
    l4_spikes: Array
    l23_state_spikes: Array
    l23_error_spikes: Array
    l5_spikes: Array


# =====================================================================
# Helpers
# =====================================================================


def _psp_sigma(ncfg: NeuronParams, psp_mV: float, e_exc: float) -> float:
    """Half-normal sigma such that expected EPSP \u2248 ``psp_mV``."""
    g_L = float(ncfg.g_L)
    df = float(e_exc - ncfg.v_rest)
    return float(psp_mV * g_L / max(df, 1.0) / 0.7978845608)


def _rheobase_cond_scale(
    ncfg: NeuronParams,
    fan_in: int,
    *,
    e_exc: float = 0.0,
    ampa_frac: float = 0.75,
    input_rate: float = 0.05,
    headroom: float = 1.58,
) -> float:
    """Boost so expected pre-synaptic drive reaches AdEx rheobase.

    ``input_rate`` is the *expected fraction of presynaptic units firing
    per dt*, NOT this layer's own target output sparsity.
    """
    gap = float(abs(ncfg.v_thresh - ncfg.v_rest))
    i_rheo = float(ncfg.g_L) * (gap - float(ncfg.delta_t))
    df = float(e_exc - ncfg.v_rest)
    effective_df = float(e_exc - ncfg.v_thresh)
    mg_at_thr = 1.0 / (1.0 + 0.28 * float(jnp.exp(-0.062 * ncfg.v_thresh)))
    ampa_eff = ampa_frac + (1.0 - ampa_frac) * mg_at_thr
    psp = gap / 2.0
    mean_w = psp * float(ncfg.g_L) / max(df, 1.0)
    expected_active = max(1.0, fan_in * input_rate)
    denom = expected_active * mean_w * max(abs(effective_df), 1.0) * ampa_eff
    return headroom * i_rheo / max(denom, 1e-6)


def _nmda_mg_block(v: Array) -> Array:
    return 1.0 / (1.0 + 0.28 * jnp.exp(-0.062 * v))


# =====================================================================
# Init
# =====================================================================


def init_cortical_area_params(
    ctx: BackendContext,
    input_size: int,
    *,
    n_l4: int = 128,
    n_l23_state: int = 128,
    n_l23_error: int = 128,
    n_l5: int = 64,
    tau_m_l4: float = 20.0,
    tau_m_l5: float = 25.0,
    l4_target_sparsity: float = 0.1,
    l5_target_sparsity: float = 0.15,
    l4_expected_input_rate: float = 0.1,
    l5_expected_input_rate: float = 0.02,
    ach_ec50: float = 0.5,
    ach_hill_n: float = 1.5,
    tau_nmda: float = 100.0,
    ampa_nmda_ratio: float = 3.0,
    tau_rate: float = 20.0,
    e_exc: float = 0.0,
    e_inh: float = -75.0,
    v_rest: float = -70.0,
    v_thresh: float = -55.0,
    v_reset: float = -75.0,
    C_m: float = 281.0,
) -> CorticalAreaParams:
    """Build cortical area params from biophysical defaults."""
    # ---- L4 (RS pyramidal) ----
    g_L_l4 = C_m / tau_m_l4
    scale_l4 = g_L_l4 / 30.0
    l4_ncfg = init_neuron_params(
        ctx, v_rest=v_rest, v_thresh=v_thresh, v_reset=v_reset,
        C_m=C_m, g_L=g_L_l4, tau_w=144.0,
        a=4.0 * scale_l4, b=80.5 * scale_l4,
    )
    l4_ipool = init_ipool_params(ctx, n_l4, n_inh=max(2, n_l4 // 4))

    # ---- L2/3 PC (ErrorNeuron) ----
    l23 = init_error_neuron_params(
        ctx, n_input=n_l4,
        n_state=n_l23_state, n_error=n_l23_error,
        ach_ec50=ach_ec50, ach_hill_n=ach_hill_n,
    )

    # ---- L5 (RS pyramidal, slower adapt for bursting readout) ----
    g_L_l5 = C_m / tau_m_l5
    scale_l5 = g_L_l5 / 30.0
    l5_ncfg = init_neuron_params(
        ctx, v_rest=v_rest, v_thresh=v_thresh, v_reset=v_reset,
        C_m=C_m, g_L=g_L_l5, tau_w=144.0,
        a=4.0 * scale_l5, b=80.5 * scale_l5,
    )
    l5_ipool = init_ipool_params(ctx, n_l5, n_inh=max(2, n_l5 // 4))

    f = lambda x: jnp.asarray(x, DTYPE)
    ampa_frac = ampa_nmda_ratio / (1.0 + ampa_nmda_ratio)
    l4_cs = _rheobase_cond_scale(
        l4_ncfg, fan_in=input_size, e_exc=e_exc, ampa_frac=ampa_frac,
        input_rate=l4_expected_input_rate,
    )
    l5_cs = _rheobase_cond_scale(
        l5_ncfg, fan_in=n_l23_state, e_exc=e_exc, ampa_frac=ampa_frac,
        input_rate=l5_expected_input_rate,
    )
    return CorticalAreaParams(
        l4_ncfg=l4_ncfg, l4_ipool=l4_ipool, l4_cond_scale=f(l4_cs),
        l23=l23,
        l5_ncfg=l5_ncfg, l5_ipool=l5_ipool, l5_cond_scale=f(l5_cs),
        e_exc=f(e_exc), e_inh=f(e_inh),
        nmda_decay=f(ctx.decay(tau_nmda)),
        ampa_frac=f(ampa_frac),
        rate_decay_l4=f(ctx.decay(tau_rate)),
        rate_decay_l5=f(ctx.decay(tau_rate)),
        input_size=input_size,
        n_l4=n_l4,
        n_l23_state=n_l23_state,
        n_l23_error=n_l23_error,
        n_l5=n_l5,
        l4_target_sparsity=float(l4_target_sparsity),
        l5_target_sparsity=float(l5_target_sparsity),
    )


def init_cortical_area_state(
    key: PRNGKey, params: CorticalAreaParams, *, dtype=DTYPE,
) -> CorticalAreaState:
    """Initialise weights (half-normal, PSP-targeted) and zero traces."""
    gap4 = float(params.l4_ncfg.v_thresh - params.l4_ncfg.v_rest)
    psp4 = gap4 / 2.0
    sigma4 = _psp_sigma(params.l4_ncfg, psp4, float(params.e_exc))
    gap5 = float(params.l5_ncfg.v_thresh - params.l5_ncfg.v_rest)
    psp5 = gap5 / 2.0
    sigma5 = _psp_sigma(params.l5_ncfg, psp5, float(params.e_exc))

    k_l4w, k_l4i, k_l23, k_l5w, k_l5i = split_key(key, 5)

    w_l4_in = jnp.abs(
        jax.random.normal(k_l4w, (params.input_size, params.n_l4), dtype) * sigma4
    )
    w_l23_l5 = jnp.abs(
        jax.random.normal(k_l5w, (params.n_l23_state, params.n_l5), dtype) * sigma5
    )

    z = lambda shape: jnp.zeros(shape, dtype)
    return CorticalAreaState(
        l4_nstate=init_neuron_state(params.n_l4, v_rest=float(params.l4_ncfg.v_rest)),
        l4_ipool=init_ipool_state(
            k_l4i, params.l4_ipool, target_sparsity=params.l4_target_sparsity),
        w_l4_in=w_l4_in,
        g_nmda_l4=z(params.n_l4),
        rate_l4=z(params.n_l4),
        l23=init_error_neuron_state(k_l23, params.l23),
        l5_nstate=init_neuron_state(params.n_l5, v_rest=float(params.l5_ncfg.v_rest)),
        l5_ipool=init_ipool_state(
            k_l5i, params.l5_ipool, target_sparsity=params.l5_target_sparsity),
        w_l23_l5=w_l23_l5,
        g_nmda_l5=z(params.n_l5),
        rate_l5=z(params.n_l5),
        x_pre_l4=z(params.input_size),
        x_post_l4=z(params.n_l4),
        x_pre_l5=z(params.n_l23_state),
        x_post_l5=z(params.n_l5),
    )


# =====================================================================
# Step
# =====================================================================


def _pop_step(
    nstate: NeuronState, ncfg: NeuronParams, ipool_state: IPoolState,
    ipool_params: IPoolParams, ctx: BackendContext,
    pre: Array, w_in: Array, g_nmda: Array,
    *, cond_scale: Array, ampa_frac: Array, nmda_decay: Array,
    e_exc: Array, receptor_gain: Array, apply_stdp_ipool: bool,
):
    """Shared AdEx+IPool block used by L4 and L5.

    Returns (new_nstate, spikes, new_ipool, new_g_nmda).
    """
    # Conductance (nS)
    g_total = (pre @ w_in) * receptor_gain * cond_scale
    # NMDA slow EMA
    g_nmda_new = g_nmda * nmda_decay + (1.0 - nmda_decay) * g_total
    mg = _nmda_mg_block(nstate.v)
    g_syn = ampa_frac * g_total + (1.0 - ampa_frac) * g_nmda_new * mg
    i_syn = g_syn * (e_exc - nstate.v)

    new_n, spikes = neuron_step(nstate, ncfg, ctx, i_syn=i_syn, g_syn=g_syn)
    ip_out = ipool_step(
        ipool_state, ipool_params, ctx, spikes, new_n.v,
        apply_stdp=apply_stdp_ipool,
    )
    v_corr = ip_out.i_inh / ncfg.g_L
    v_new = jnp.clip(new_n.v - v_corr, -90.0, None)
    new_n = eqx.tree_at(lambda s: s.v, new_n, v_new)
    return new_n, spikes, ip_out.state, g_nmda_new


def cortical_area_step(
    state: CorticalAreaState,
    params: CorticalAreaParams,
    ctx: BackendContext,
    inputs: CorticalInputs,
    *,
    apply_stdp: bool = True,
) -> CorticalOutput:
    """One ``dt`` of canonical microcircuit dynamics.

    Sequence (per column, one dt):
      1. L4 integrates ``ff_input``  \u2192 spikes_l4
      2. L2/3 PC runs on spikes_l4 with optional ``td_prediction``
      3. L5 integrates L2/3 state rate \u2192 spikes_l5
      4. Rate EMAs updated; traces updated for optional STDP.
    """
    ff = inputs.ff_input.astype(DTYPE)
    rg = jnp.asarray(inputs.receptor_gain, DTYPE)
    ach = jnp.asarray(inputs.ach, DTYPE)

    # -- optional TD feedback injection --
    l23_state = state.l23
    if inputs.td_prediction is not None:
        l23_state = en_receive_prediction(l23_state, inputs.td_prediction)

    # -- (1) L4 --
    l4_n, l4_spk, l4_ip, l4_g_nmda = _pop_step(
        state.l4_nstate, params.l4_ncfg, state.l4_ipool, params.l4_ipool, ctx,
        pre=ff, w_in=state.w_l4_in, g_nmda=state.g_nmda_l4,
        cond_scale=params.l4_cond_scale, ampa_frac=params.ampa_frac,
        nmda_decay=params.nmda_decay, e_exc=params.e_exc,
        receptor_gain=rg, apply_stdp_ipool=apply_stdp,
    )

    # -- (2) L2/3 PC --
    en_out = en_step(
        l23_state, params.l23, ctx,
        l4_spk.astype(DTYPE), ach=ach, receptor_gain=rg,
    )

    # -- (3) L5 (L2/3 state spikes → deep output) --
    belief = en_belief(en_out.state)             # (n_l23_state,) rate EMA (readout)
    l23_state_drive = en_out.state_spikes.astype(DTYPE)
    l5_n, l5_spk, l5_ip, l5_g_nmda = _pop_step(
        state.l5_nstate, params.l5_ncfg, state.l5_ipool, params.l5_ipool, ctx,
        pre=l23_state_drive, w_in=state.w_l23_l5, g_nmda=state.g_nmda_l5,
        cond_scale=params.l5_cond_scale, ampa_frac=params.ampa_frac,
        nmda_decay=params.nmda_decay, e_exc=params.e_exc,
        receptor_gain=rg, apply_stdp_ipool=apply_stdp,
    )

    # -- (4) Rate EMAs --
    r4d = params.rate_decay_l4
    r5d = params.rate_decay_l5
    rate_l4 = state.rate_l4 * r4d + l4_spk * (1.0 - r4d)
    rate_l5 = state.rate_l5 * r5d + l5_spk * (1.0 - r5d)

    # -- (5) Traces (for optional STDP) --
    tr = ctx.decay(20.0)
    tr_a = jnp.asarray(tr, DTYPE)
    x_pre_l4 = state.x_pre_l4 * tr_a + ff
    x_post_l4 = state.x_post_l4 * tr_a + l4_spk
    x_pre_l5 = state.x_pre_l5 * tr_a + l23_state_drive
    x_post_l5 = state.x_post_l5 * tr_a + l5_spk

    new_state = CorticalAreaState(
        l4_nstate=l4_n, l4_ipool=l4_ip, w_l4_in=state.w_l4_in,
        g_nmda_l4=l4_g_nmda, rate_l4=rate_l4,
        l23=en_out.state,
        l5_nstate=l5_n, l5_ipool=l5_ip, w_l23_l5=state.w_l23_l5,
        g_nmda_l5=l5_g_nmda, rate_l5=rate_l5,
        x_pre_l4=x_pre_l4, x_post_l4=x_post_l4,
        x_pre_l5=x_pre_l5, x_post_l5=x_post_l5,
    )

    ff_out = en_prediction_error_rate(en_out.state)  # L2/3 error rate
    return CorticalOutput(
        state=new_state,
        ff_out=ff_out,
        belief=belief,
        l5_rate=rate_l5,
        l4_spikes=l4_spk,
        l23_state_spikes=en_out.state_spikes,
        l23_error_spikes=en_out.error_spikes,
        l5_spikes=l5_spk,
    )


# =====================================================================
# Learning (three-factor STDP on L4 input + L5 output + L2/3 PC)
# =====================================================================


def cortical_area_update(
    state: CorticalAreaState,
    params: CorticalAreaParams,
    *,
    modulator: float | Array = 0.0,      # e.g. RPE / global TD
    precision: Array | None = None,      # (n_l23_error,) astrocyte gain
    lr_l4: float = 5e-4,
    lr_l5: float = 5e-4,
    receptor_lr: float | Array = 1.0,
) -> CorticalAreaState:
    """Update all plastic weights in the area with three-factor STDP.

    - L4 feedforward (w_l4_in): Hebbian ``x_pre \u00b7 x_post`` gated by ``modulator``
    - L5 readout (w_l23_l5): same rule on belief \u2192 L5 trace
    - L2/3 bottom-up and top-down: delegated to ``en_update_weights``
    """
    m = jnp.asarray(modulator, DTYPE)
    # L4
    dw_l4 = lr_l4 * m * state.x_pre_l4[:, None] * state.x_post_l4[None, :]
    w_l4_new = jnp.maximum(state.w_l4_in + dw_l4, 0.0)
    # L5
    dw_l5 = lr_l5 * m * state.x_pre_l5[:, None] * state.x_post_l5[None, :]
    w_l5_new = jnp.maximum(state.w_l23_l5 + dw_l5, 0.0)
    # L2/3 PC
    l23_new = en_update_weights(
        state.l23, params.l23, m, precision=precision,
        receptor_lr=receptor_lr,
    )
    return eqx.tree_at(
        lambda s: (s.w_l4_in, s.w_l23_l5, s.l23),
        state,
        (w_l4_new, w_l5_new, l23_new),
    )


def cortical_area_reset_transient(
    state: CorticalAreaState,
    params: CorticalAreaParams,
) -> CorticalAreaState:
    """Reset membrane + traces + NMDA + rates; preserve all weights."""
    z = lambda shape: jnp.zeros(shape, DTYPE)
    return CorticalAreaState(
        l4_nstate=init_neuron_state(
            params.n_l4, v_rest=float(params.l4_ncfg.v_rest)),
        l4_ipool=ipool_reset_transient(state.l4_ipool, params.l4_ipool),
        w_l4_in=state.w_l4_in,
        g_nmda_l4=z(params.n_l4),
        rate_l4=z(params.n_l4),
        l23=en_reset_transient(state.l23, params.l23),
        l5_nstate=init_neuron_state(
            params.n_l5, v_rest=float(params.l5_ncfg.v_rest)),
        l5_ipool=ipool_reset_transient(state.l5_ipool, params.l5_ipool),
        w_l23_l5=state.w_l23_l5,
        g_nmda_l5=z(params.n_l5),
        rate_l5=z(params.n_l5),
        x_pre_l4=z(params.input_size),
        x_post_l4=z(params.n_l4),
        x_pre_l5=z(params.n_l23_state),
        x_post_l5=z(params.n_l5),
    )
