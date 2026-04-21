"""Predictive-coding error-neuron layer — pure JAX.

Bogacz (2017); Bastos et al. (2012); Rao & Ballard (1999).

Two AdEx populations implement a single cortical area under
continuous predictive coding:

- **Error neurons** (L4 stellate, fast τ≈4 ms) carry ε = input − g(μ)
  as a *signed* conductance: positive error drives via ``E_exc``,
  negative error via ``E_inh`` (Cl⁻ reversal). This replaces the
  current-mode implementation and keeps the drive self-limiting.
- **State neurons** (L2/3 pyramidal, slow τ≈20 ms) encode the belief
  μ; integrated from the bottom-up error conductance.

Weight matrices:
- ``w_in``  (n_input, n_error)  feedforward drive (Rao & Ballard)
- ``w_td``  (n_state, n_error)  generative model, state → predicted error
- ``w_bu``  (n_error, n_state)  bottom-up pathway, error → state update

Three-factor plasticity:
- ``Δw_bu = +lr · m_t · receptor_lr · precision · e_bu``
- ``Δw_td = −lr · m_t · receptor_lr · e_td``     (anti-Hebbian,
  Rao & Ballard 1999: top-down predictions learn to *reduce* error)

Differences from legacy:
- Causal ±20 ms spike-timing window replaced with standard exp-filtered
  traces (``tau=20 ms``). Same biological signal (pairing timing),
  lower compute + full JIT compatibility.
- External top-down prediction is a *state field* (``external_prediction``)
  not a mutable buffer; callers zero it explicitly if desired.
- ACh gain uses a pure ``hill_response`` helper (no hidden attribute).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, BackendContext, split_key
from .state import NeuronParams, NeuronState, init_neuron_state
from .neuron import init_neuron_params, neuron_step
from .free_energy import broadcast_precision


# =====================================================================
# Params / state
# =====================================================================


class ErrorNeuronParams(eqx.Module):
    """Static params for the state/error populations + learning rates."""

    state_np: NeuronParams
    error_np: NeuronParams

    e_exc: Array
    e_inh: Array

    # Plasticity
    w_bu_lr: Array
    w_td_lr: Array
    state_decay: Array     # eligibility decay on e_bu
    error_decay: Array     # eligibility decay on e_td
    trace_decay: Array     # exp-filter on spike traces
    rate_decay: Array      # EMA decay on rates

    # ACh Hill (defaults from legacy ErrorNeuronConfig)
    ach_ec50: Array
    ach_hill_n: Array
    ach_gain_min: Array
    ach_gain_max: Array

    # Static sizes
    n_input: int = eqx.field(static=True)
    n_state: int = eqx.field(static=True)
    n_error: int = eqx.field(static=True)


def init_error_neuron_params(
    ctx: BackendContext,
    n_input: int,
    *,
    n_state: int = 64,
    n_error: int = 64,
    tau_state: float = 20.0,
    tau_error: float = 4.0,
    v_rest: float = -70.0,
    v_thresh: float = -55.0,
    v_reset: float = -75.0,
    v_spike_cutoff: float = -30.0,
    delta_t: float = 2.0,
    C_m: float = 281.0,
    a: float = 4.0,
    b: float = 80.5,
    refrac_period_ms: float = 2.0,
    e_exc: float = 0.0,
    e_inh: float = -75.0,
    w_bu_lr: float = 5e-4,
    w_td_lr: float = 5e-4,
    tau_eligibility_bu: float = 20.0,
    tau_eligibility_td: float = 4.0,
    tau_trace: float = 20.0,
    tau_rate: float = 20.0,
    ach_ec50: float = 0.5,
    ach_hill_n: float = 1.5,
    ach_gain_min: float = 0.5,
    ach_gain_max: float = 1.5,
) -> ErrorNeuronParams:
    """State = slow pyramidal (tau_w=144), Error = fast FS-like (a=b=0)."""
    state_np = init_neuron_params(
        ctx,
        v_rest=v_rest, v_thresh=v_thresh, v_reset=v_reset,
        v_spike_cutoff=v_spike_cutoff, delta_t=delta_t,
        C_m=C_m, g_L=C_m / tau_state, tau_w=144.0,
        a=a, b=b, refrac_period_ms=refrac_period_ms,
    )
    error_np = init_neuron_params(
        ctx,
        v_rest=v_rest, v_thresh=v_thresh, v_reset=v_reset,
        v_spike_cutoff=v_spike_cutoff, delta_t=delta_t,
        C_m=C_m, g_L=C_m / tau_error, tau_w=tau_error,  # irrelevant (a=b=0)
        a=0.0, b=0.0, refrac_period_ms=refrac_period_ms,
    )
    f = lambda x: jnp.asarray(x, DTYPE)
    return ErrorNeuronParams(
        state_np=state_np, error_np=error_np,
        e_exc=f(e_exc), e_inh=f(e_inh),
        w_bu_lr=f(w_bu_lr), w_td_lr=f(w_td_lr),
        state_decay=f(ctx.decay(tau_eligibility_bu)),
        error_decay=f(ctx.decay(tau_eligibility_td)),
        trace_decay=f(ctx.decay(tau_trace)),
        rate_decay=f(ctx.decay(tau_rate)),
        ach_ec50=f(ach_ec50), ach_hill_n=f(ach_hill_n),
        ach_gain_min=f(ach_gain_min), ach_gain_max=f(ach_gain_max),
        n_input=n_input, n_state=n_state, n_error=n_error,
    )


class ErrorNeuronState(eqx.Module):
    """Dynamic state of a predictive-coding area."""

    state_nstate: NeuronState
    error_nstate: NeuronState

    w_in: Array            # (n_input, n_error)
    w_bu: Array            # (n_error, n_state)
    w_td: Array            # (n_state, n_error)

    # Eligibilities for three-factor updates
    e_bu: Array            # (n_error, n_state)
    e_td: Array            # (n_state, n_error)

    # Spike traces (exp-filtered) used to build eligibilities
    x_error: Array         # (n_error,)
    x_state: Array         # (n_state,)

    # Rate EMAs (population firing rates; used for readout + prediction)
    state_rate: Array      # (n_state,)
    error_rate: Array      # (n_error,)

    # Top-down prediction injected from a higher area (consumed once)
    external_prediction: Array   # (n_error,)


def _psp_scale(params: NeuronParams, psp_mV: float, e_exc: Array) -> float:
    """Half-normal σ that produces mean PSP amplitude ``psp_mV``."""
    g_L = float(params.g_L)
    df = float(e_exc - params.v_rest)
    g = psp_mV * g_L / max(df, 1.0)
    sqrt_2_over_pi = 0.7978845608028654
    return float(g / sqrt_2_over_pi)


def init_error_neuron_state(
    key: PRNGKey, params: ErrorNeuronParams,
    *,
    psp_ff_mV: float | None = None,
    psp_rec_mV: float | None = None,
    dtype=DTYPE,
) -> ErrorNeuronState:
    """Half-normal weights scaled to produce ~half-threshold PSPs."""
    gap = float(abs(params.state_np.v_thresh - params.state_np.v_rest))
    if psp_ff_mV is None:
        psp_ff_mV = gap / 2.0
    if psp_rec_mV is None:
        psp_rec_mV = gap / 2.0 * 1.3    # recurrent 1.3× feedforward

    sigma_in = _psp_scale(params.error_np, psp_ff_mV, params.e_exc)
    sigma_bu = _psp_scale(params.state_np, psp_rec_mV, params.e_exc)
    sigma_td = _psp_scale(params.error_np, psp_rec_mV, params.e_exc)

    k_in, k_bu, k_td = split_key(key, 3)
    w_in = jnp.abs(jax.random.normal(
        k_in, (params.n_input, params.n_error), dtype=dtype,
    ) * sigma_in)
    w_bu = jnp.abs(jax.random.normal(
        k_bu, (params.n_error, params.n_state), dtype=dtype,
    ) * sigma_bu)
    w_td = jnp.abs(jax.random.normal(
        k_td, (params.n_state, params.n_error), dtype=dtype,
    ) * sigma_td)

    return ErrorNeuronState(
        state_nstate=init_neuron_state(
            params.n_state, v_rest=float(params.state_np.v_rest)),
        error_nstate=init_neuron_state(
            params.n_error, v_rest=float(params.error_np.v_rest)),
        w_in=w_in, w_bu=w_bu, w_td=w_td,
        e_bu=jnp.zeros((params.n_error, params.n_state), dtype),
        e_td=jnp.zeros((params.n_state, params.n_error), dtype),
        x_error=jnp.zeros(params.n_error, dtype),
        x_state=jnp.zeros(params.n_state, dtype),
        state_rate=jnp.zeros(params.n_state, dtype),
        error_rate=jnp.zeros(params.n_error, dtype),
        external_prediction=jnp.zeros(params.n_error, dtype),
    )


# =====================================================================
# Step
# =====================================================================


class ErrorNeuronOutput(NamedTuple):
    state: ErrorNeuronState
    state_spikes: Array
    error_spikes: Array
    prediction_error: Array   # (n_error,) signed error conductance


def _ach_gain(params: ErrorNeuronParams, ach: Array) -> Array:
    ach_c = jnp.clip(ach, 0.0, 1.0)
    ach_n = ach_c ** params.ach_hill_n
    ec50_n = params.ach_ec50 ** params.ach_hill_n
    frac = ach_n / (ach_n + ec50_n + 1e-12)
    return params.ach_gain_min + (params.ach_gain_max - params.ach_gain_min) * frac


def en_step(
    state: ErrorNeuronState, params: ErrorNeuronParams, ctx: BackendContext,
    input_spikes: Array,
    *,
    ach: float | Array = 0.5,
    receptor_gain: float | Array = 1.0,
    consume_external_prediction: bool = True,
) -> ErrorNeuronOutput:
    """One dt: error neurons compute ε; state neurons integrate update.

    The caller passes ``ach`` (0–1, receptor drive) and a scalar
    ``receptor_gain`` that further multiplies the error drive (reflecting
    tonic receptor modulation). The external top-down prediction field
    is consumed and zeroed unless ``consume_external_prediction=False``.
    """
    inp = input_spikes.astype(DTYPE)
    ach_a = jnp.asarray(ach, DTYPE)
    rg = jnp.asarray(receptor_gain, DTYPE)
    gain = _ach_gain(params, ach_a)

    # --- Predicted error conductance from state rates + external TD ---
    prediction = state.state_rate @ state.w_td + state.external_prediction
    # --- Feedforward drive ---
    feedforward = inp @ state.w_in
    # --- Signed error (conductance difference) ---
    g_error = (gain * feedforward - prediction) * rg

    # --- Split into excitatory / inhibitory currents on error pop ---
    v_err = state.error_nstate.v
    pos_g = jnp.maximum(g_error, 0.0)
    neg_g = jnp.maximum(-g_error, 0.0)
    error_input = (
        pos_g * (params.e_exc - v_err)
        + neg_g * (params.e_inh - v_err)
    )
    error_g_syn = pos_g + neg_g

    new_error, error_spikes = neuron_step(
        state.error_nstate, params.error_np, ctx,
        i_syn=error_input, g_syn=error_g_syn,
    )

    # --- Bottom-up drive on state population ---
    g_bu = error_spikes @ state.w_bu           # (n_state,)
    state_input = g_bu * (params.e_exc - state.state_nstate.v)

    new_state_n, state_spikes = neuron_step(
        state.state_nstate, params.state_np, ctx,
        i_syn=state_input, g_syn=g_bu,
    )

    # --- Rate EMAs ---
    rd = params.rate_decay
    state_rate = state.state_rate * rd + state_spikes * (1.0 - rd)
    error_rate = state.error_rate * rd + error_spikes * (1.0 - rd)

    # --- Pre/post traces for eligibilities (exp-filter ⇔ causal window) ---
    x_err = state.x_error * params.trace_decay + error_spikes
    x_st = state.x_state * params.trace_decay + state_spikes

    # --- Eligibilities (Hebbian outer products, exp-filtered) ---
    e_bu = (
        state.e_bu * params.state_decay
        + x_err[:, None] * state_spikes[None, :]
    )
    e_td = (
        state.e_td * params.error_decay
        + x_st[:, None] * error_spikes[None, :]
    )

    # --- External prediction consumption ---
    ext_pred = jnp.where(
        consume_external_prediction,
        jnp.zeros_like(state.external_prediction),
        state.external_prediction,
    )

    new_s = ErrorNeuronState(
        state_nstate=new_state_n, error_nstate=new_error,
        w_in=state.w_in, w_bu=state.w_bu, w_td=state.w_td,
        e_bu=e_bu, e_td=e_td,
        x_error=x_err, x_state=x_st,
        state_rate=state_rate, error_rate=error_rate,
        external_prediction=ext_pred,
    )
    return ErrorNeuronOutput(
        state=new_s, state_spikes=state_spikes,
        error_spikes=error_spikes, prediction_error=g_error,
    )


# =====================================================================
# Learning
# =====================================================================


def en_update_weights(
    state: ErrorNeuronState, params: ErrorNeuronParams,
    modulation: float | Array,
    *,
    precision: Array | None = None,
    receptor_lr: float | Array = 1.0,
) -> ErrorNeuronState:
    """Three-factor learning on ``w_bu`` (Hebbian) and ``w_td`` (anti-Hebbian).

    ``precision`` optionally weights the bottom-up update per error
    neuron (broadcast from zone → ``n_error`` via ``broadcast_precision``).
    """
    m = jnp.asarray(modulation, DTYPE)
    rlr = jnp.asarray(receptor_lr, DTYPE)
    dw_bu = params.w_bu_lr * m * rlr * state.e_bu
    if precision is not None:
        prec = broadcast_precision(precision, params.n_error)
        dw_bu = dw_bu * prec[:, None]
    dw_td = -params.w_td_lr * m * rlr * state.e_td

    return eqx.tree_at(
        lambda s: (s.w_bu, s.w_td),
        state,
        (state.w_bu + dw_bu, state.w_td + dw_td),
    )


def en_receive_prediction(
    state: ErrorNeuronState, prediction: Array,
) -> ErrorNeuronState:
    """Store external top-down prediction for the next ``en_step``."""
    pred = prediction.astype(DTYPE)
    return eqx.tree_at(lambda s: s.external_prediction, state, pred)


def en_belief(state: ErrorNeuronState) -> Array:
    """Return current belief μ (state-neuron rate EMA)."""
    return state.state_rate


def en_prediction_error_rate(state: ErrorNeuronState) -> Array:
    """Error-neuron rate EMA, used as the biological PE signal."""
    return state.error_rate


def en_reset_transient(
    state: ErrorNeuronState, params: ErrorNeuronParams,
) -> ErrorNeuronState:
    """Clear dynamic state; preserve ``w_in`` / ``w_bu`` / ``w_td``."""
    return ErrorNeuronState(
        state_nstate=init_neuron_state(
            params.n_state, v_rest=float(params.state_np.v_rest)),
        error_nstate=init_neuron_state(
            params.n_error, v_rest=float(params.error_np.v_rest)),
        w_in=state.w_in, w_bu=state.w_bu, w_td=state.w_td,
        e_bu=jnp.zeros_like(state.e_bu),
        e_td=jnp.zeros_like(state.e_td),
        x_error=jnp.zeros(params.n_error, DTYPE),
        x_state=jnp.zeros(params.n_state, DTYPE),
        state_rate=jnp.zeros(params.n_state, DTYPE),
        error_rate=jnp.zeros(params.n_error, DTYPE),
        external_prediction=jnp.zeros(params.n_error, DTYPE),
    )
