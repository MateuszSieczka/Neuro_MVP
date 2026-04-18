"""Prefrontal working memory — pure JAX.

Goldman-Rakic (1995); O’Reilly & Frank (2006); Durstewitz et al. (2000);
Compte et al. (2000); Brette & Gerstner (2005); Feldmeyer et al. (2002).

Architecture:
- **Content neurons** (``n`` AdEx PFC pyramidals, τ_w=300 ms slow
  adaptation) with feedforward ``w_ff`` + dense recurrent ``w_lateral``.
  Conductance-based synapses: ``I = g · (E_exc − V)``.
- **Gate neurons** (``n_gate`` AdEx MSN-like, τ=25 ms) driven by
  ``ACh · DA · drive_calibrated``. Drive is derived from the AdEx
  rheobase so the population only fires above both thresholds
  (O’Reilly & Frank 2006 conjunction gate).
- Gate rate EMA ⇒ ``gate_signal ∈ [0, 1]`` that multiplicatively
  scales the ``w_ff`` conductance during content integration.

Differences from legacy:
- Astrocyte coupling dropped here; will re-emerge in Phase 2 cortex
  composition layer.
- Python ``if gate > 0.01`` branches replaced with always-on tensor ops
  (``gate`` factor already multiplies the update; the threshold was
  only a compute optimisation and is JIT-incompatible).
- Pre-mask uses the raw external spike vector (0/1) instead of the
  legacy ``ext > 0.1`` threshold — functionally identical for binary
  spikes.
- Lateral Hebbian learning is kept two-factor (no neuromodulation),
  matching Goldman-Rakic’s attractor-bootstrapping story, but its rate
  is now an explicit parameter.
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


class WMParams(eqx.Module):
    """Content + gate AdEx params + WM-specific scalars."""

    content: NeuronParams
    gate: NeuronParams

    # Conductance-based synapse
    e_exc: Array
    driving_force_exc: Array   # E_exc − V_rest

    # Lateral attractor
    lateral_strength: Array    # multiplies recurrent conductance
    lateral_lr: Array          # Hebbian lateral LR
    content_decay: Array       # low-pass on spikes for attractor trace

    # Feedforward three-factor STDP
    ff_lr: Array
    trace_decay: Array         # eligibility e decay
    pre_decay: Array
    post_decay: Array

    # Gate drive
    gate_drive: Array          # calibrated from AdEx rheobase
    gate_noise_std_mV: Array   # Destexhe 2003
    gate_max_rate_per_step: Array
    gate_rate_decay: Array

    # Static sizes
    n_in: int = eqx.field(static=True)
    n: int = eqx.field(static=True)
    n_gate: int = eqx.field(static=True)


def _rheobase_drive(
    gap_mV: float, delta_t: float, C_m: float, tau: float,
    a: float, ach_thr: float, da_thr: float,
) -> float:
    """Gate drive that barely reaches rheobase + adaptation equilibrium at
    both thresholds (ACh = ach_thr, DA = da_thr)."""
    g_L_eff = C_m / tau
    i_rheo = g_L_eff * (gap_mV - delta_t)
    w_adapt_eq = a * gap_mV
    denom = max(ach_thr, 0.01) * max(da_thr, 0.01)
    return float((i_rheo + w_adapt_eq) / denom)


def init_wm_params(
    ctx: BackendContext,
    n_in: int,
    n: int,
    *,
    n_gate: int = 32,
    # Content PFC AdEx (Durstewitz 2000)
    v_rest: float = -70.0,
    v_thresh: float = -55.0,
    v_reset: float = -75.0,
    v_spike_cutoff: float = -30.0,
    delta_t: float = 2.0,
    C_m: float = 281.0,
    g_L: float = 30.0,
    tau_w: float = 300.0,
    a_content: float = 2.0,
    b_content: float = 20.0,
    refrac_period_ms: float = 2.0,
    # Gate MSN AdEx (Humphries 2006)
    gate_tau: float = 25.0,
    gate_C_m: float = 281.0,
    gate_delta_t: float = 2.0,
    gate_v_spike_cutoff: float = -30.0,
    gate_tau_w: float = 144.0,
    gate_a: float = 4.0,
    gate_b: float = 80.5,
    gate_noise_mV: float = 1.0,
    gate_max_rate_hz: float = 40.0,
    # Synapse
    e_exc: float = 0.0,
    # Attractor + plasticity
    lateral_strength: float = 0.5,
    lateral_lr: float = 0.01,
    ff_lr: float = 0.01,
    tau_e: float = 20.0,
    tau_pre: float = 20.0,
    tau_post: float = 20.0,
    ach_gate_threshold: float = 0.5,
    da_gate_threshold: float = 0.4,
) -> WMParams:
    f = lambda x: jnp.asarray(x, DTYPE)

    content_p = init_neuron_params(
        ctx,
        v_rest=v_rest, v_thresh=v_thresh, v_reset=v_reset,
        v_spike_cutoff=v_spike_cutoff, delta_t=delta_t,
        C_m=C_m, g_L=g_L, tau_w=tau_w,
        a=a_content, b=b_content,
        refrac_period_ms=refrac_period_ms,
    )
    # Gate MSNs: g_L computed from C_m/tau so exp-Euler matches legacy.
    gate_g_L = gate_C_m / gate_tau
    gate_p = init_neuron_params(
        ctx,
        v_rest=v_rest, v_thresh=v_thresh, v_reset=v_reset,
        v_spike_cutoff=gate_v_spike_cutoff, delta_t=gate_delta_t,
        C_m=gate_C_m, g_L=gate_g_L, tau_w=gate_tau_w,
        a=gate_a, b=gate_b,
        refrac_period_ms=refrac_period_ms,
    )

    gap = abs(v_thresh - v_rest)
    drive = _rheobase_drive(
        gap, gate_delta_t, gate_C_m, gate_tau, gate_a,
        ach_gate_threshold, da_gate_threshold,
    )

    return WMParams(
        content=content_p,
        gate=gate_p,
        e_exc=f(e_exc),
        driving_force_exc=f(e_exc - v_rest),
        lateral_strength=f(lateral_strength),
        lateral_lr=f(lateral_lr),
        content_decay=f(ctx.decay(tau_w)),
        ff_lr=f(ff_lr),
        trace_decay=f(ctx.decay(tau_e)),
        pre_decay=f(ctx.decay(tau_pre)),
        post_decay=f(ctx.decay(tau_post)),
        gate_drive=f(drive),
        gate_noise_std_mV=f(gate_noise_mV),
        gate_max_rate_per_step=f(gate_max_rate_hz * ctx.dt / 1000.0),
        gate_rate_decay=f(ctx.decay(gate_tau)),
        n_in=n_in, n=n, n_gate=n_gate,
    )


class WMState(eqx.Module):
    """Content + gate AdEx states, weights, eligibility, content trace."""

    content: NeuronState
    gate: NeuronState
    gate_rate: Array          # (n_gate,) EMA of gate spikes
    gate_signal: Array        # scalar [0, 1]
    w_ff: Array               # (n_in, n) feedforward weights (nS)
    w_lateral: Array          # (n, n) dense recurrent (nS)
    e: Array                  # (n_in, n) eligibility trace
    x_pre: Array              # (n_in,)
    x_post: Array             # (n,)
    content_trace: Array      # (n,) low-pass spike trace


def init_wm_state(
    key: PRNGKey, params: WMParams,
    *,
    ff_psp_mV: float | None = None,
    lat_psp_mV: float | None = None,
    dtype=DTYPE,
) -> WMState:
    """Initialise WM with PSP-targeted half-normal weights.

    ``ff_psp_mV`` / ``lat_psp_mV`` default to ``gap/2`` and ``gap/3``
    respectively (thalamic → PFC unitary EPSPs 5–10 mV, Cruikshank
    et al. 2012; Gil & Bhatt 1999). Per-synapse conductance is derived
    from ``g = PSP · g_L / (E_exc − V_rest)`` and realised as a
    half-normal ``|N(0, σ)|`` with ``E[|w|] = g``. Gate neurons start
    in the up-state (V_T − 2 mV, Wilson & Kawaguchi 1996) so ACh·DA
    drive can fire them within 2–3 ms.
    """
    k_ff, k_lat = split_key(key, 2)
    n_in, n, n_gate = params.n_in, params.n, params.n_gate
    gap = float(abs(params.content.v_thresh - params.content.v_rest))
    if ff_psp_mV is None:
        ff_psp_mV = gap / 2.0
    if lat_psp_mV is None:
        lat_psp_mV = gap / 3.0
    # Half-normal: E[|N(0, σ)|] = σ·√(2/π) ≈ 0.7979·σ.
    sqrt_2_over_pi = float(jnp.sqrt(2.0 / jnp.pi))
    g_L = float(params.content.g_L)
    df = float(params.driving_force_exc)
    g_ff = ff_psp_mV * g_L / df
    g_lat = lat_psp_mV * g_L / df
    sigma_ff = g_ff / sqrt_2_over_pi
    sigma_lat = g_lat / sqrt_2_over_pi
    w_ff = jnp.abs(
        jax.random.normal(k_ff, (n_in, n), dtype=dtype) * sigma_ff
    )
    w_lat = jnp.abs(
        jax.random.normal(k_lat, (n, n), dtype=dtype) * sigma_lat
    )
    # No autapses.
    w_lat = w_lat * (1.0 - jnp.eye(n, dtype=dtype))

    content_state = init_neuron_state(n, v_rest=float(params.content.v_rest))
    gate_state = init_neuron_state(
        n_gate, v_rest=float(params.gate.v_thresh - 2.0),  # up-state
    )

    return WMState(
        content=content_state,
        gate=gate_state,
        gate_rate=jnp.zeros(n_gate, dtype=dtype),
        gate_signal=jnp.asarray(0.0, dtype),
        w_ff=w_ff,
        w_lateral=w_lat,
        e=jnp.zeros((n_in, n), dtype=dtype),
        x_pre=jnp.zeros(n_in, dtype=dtype),
        x_post=jnp.zeros(n, dtype=dtype),
        content_trace=jnp.zeros(n, dtype=dtype),
    )


# =====================================================================
# Step
# =====================================================================


class WMOutput(NamedTuple):
    state: WMState
    spikes: Array             # (n,) content spikes
    gate_signal: Array        # scalar


def _gate_step(
    state: WMState, params: WMParams, ctx: BackendContext,
    ach: Array, da: Array, key: PRNGKey,
) -> WMState:
    """Advance the AdEx MSN gate population and its rate EMA."""
    gp = params.gate
    # Scalar drive: ACh · DA · gate_drive, applied uniformly to gate pop.
    drive_scalar = ach * da * params.gate_drive
    drive = jnp.broadcast_to(drive_scalar, (params.n_gate,))
    # Membrane-noise current (Destexhe 2003): I_n ~ N(0, g_L · σ_V).
    noise = jax.random.normal(key, (params.n_gate,), dtype=DTYPE) * (
        gp.g_L * params.gate_noise_std_mV
    )
    i_syn = drive + noise
    g_syn = jnp.zeros_like(i_syn)  # drive is already current-mode here

    new_gate, gate_spikes = neuron_step(
        state.gate, gp, ctx, i_syn=i_syn, g_syn=g_syn,
    )
    # Population rate EMA → normalise to [0, 1].
    rate = (
        state.gate_rate * params.gate_rate_decay
        + gate_spikes * (1.0 - params.gate_rate_decay)
    )
    raw_signal = jnp.mean(rate)
    gate_signal = jnp.clip(
        raw_signal / jnp.maximum(params.gate_max_rate_per_step, 1e-8),
        0.0, 1.0,
    )
    return eqx.tree_at(
        lambda s: (s.gate, s.gate_rate, s.gate_signal),
        state,
        (new_gate, rate, gate_signal),
    )


def _content_step(
    state: WMState, params: WMParams, ctx: BackendContext,
    external_input: Array, receptor_gain: Array,
) -> WMOutput:
    """Advance the PFC content population under gated ff + attractor drive."""
    cp = params.content
    gate = state.gate_signal
    ext = external_input.astype(DTYPE)

    # Conductance-based input (gate-scaled, receptor-modulated).
    g_ff = gate * receptor_gain * (ext @ state.w_ff)            # (n,)
    I_ff = g_ff * (params.e_exc - state.content.v)

    # Recurrent attractor: lateral_strength · content_trace @ w_lat.
    g_rec = params.lateral_strength * (state.content_trace @ state.w_lateral)
    I_rec = g_rec * (params.e_exc - state.content.v)

    i_syn = I_ff + I_rec
    g_syn = g_ff + g_rec

    new_content, spikes = neuron_step(
        state.content, cp, ctx, i_syn=i_syn, g_syn=g_syn,
    )

    # STDP traces (pre on inputs, post on content spikes).
    x_pre = state.x_pre * params.pre_decay + ext * gate
    x_post = state.x_post * params.post_decay + spikes

    # Eligibility: decay + event-driven outer products, scaled by gate.
    e = state.e * params.trace_decay
    e = e + gate * (x_pre[:, None] * spikes[None, :])
    e = e + gate * (ext[:, None] * x_post[None, :])

    # Content trace (low-pass for attractor readout).
    ct = state.content_trace * params.content_decay + spikes

    new_state = eqx.tree_at(
        lambda s: (s.content, s.x_pre, s.x_post, s.e, s.content_trace),
        state,
        (new_content, x_pre, x_post, e, ct),
    )
    return WMOutput(state=new_state, spikes=spikes, gate_signal=gate)


def wm_step(
    state: WMState, params: WMParams, ctx: BackendContext,
    external_input: Array,
    ach: float | Array, da: float | Array,
    key: PRNGKey,
    receptor_gain: float | Array = 1.0,
) -> WMOutput:
    """One WM tick: gate population update → content AdEx + STDP traces."""
    ach_a = jnp.asarray(ach, DTYPE)
    da_a = jnp.asarray(da, DTYPE)
    rg = jnp.asarray(receptor_gain, DTYPE)
    state = _gate_step(state, params, ctx, ach_a, da_a, key)
    return _content_step(state, params, ctx, external_input, rg)


# =====================================================================
# Learning
# =====================================================================


def wm_update_ff(
    state: WMState, params: WMParams,
    m_t: float | Array,
    pred_error: Array,
    receptor_lr: float | Array = 1.0,
) -> WMState:
    """Three-factor STDP on ``w_ff``: Δw = lr·m_t·receptor_lr·e·ε(j).

    ``pred_error`` broadcasts over the post axis (``(n,)``).
    """
    m = jnp.asarray(m_t, DTYPE)
    rlr = jnp.asarray(receptor_lr, DTYPE)
    err = pred_error.astype(DTYPE)
    dw = params.ff_lr * m * rlr * state.e * err[None, :]
    return eqx.tree_at(lambda s: s.w_ff, state, state.w_ff + dw)


def wm_update_lateral(state: WMState, params: WMParams) -> WMState:
    """Two-factor Hebbian on ``w_lateral`` with row-max soft normalisation.

    Only active when at least two content neurons spiked (otherwise
    the outer product contributes nothing and we skip the normalisation
    scan to save compute — but still JIT-friendly via ``jnp.where``).
    """
    active = state.content.spikes
    # Hebbian outer, no autapses.
    dw = params.lateral_lr * (active[:, None] * active[None, :])
    n = params.n
    dw = dw * (1.0 - jnp.eye(n, dtype=DTYPE))
    w = state.w_lateral + dw
    # Soft row-max normalisation (keep rows with max > 1).
    row_max = jnp.max(w, axis=1, keepdims=True)
    scale = jnp.where(row_max > 1.0, row_max, jnp.asarray(1.0, DTYPE))
    w = w / scale
    w = w * (1.0 - jnp.eye(n, dtype=DTYPE))
    return eqx.tree_at(lambda s: s.w_lateral, state, w)


def wm_reset_transient(state: WMState, params: WMParams) -> WMState:
    """Clear dynamic state (V, traces, content); keep learned weights.

    JIT-safe: builds NeuronStates from tracer-valued ``v_rest`` rather
    than calling ``init_neuron_state`` (which casts to ``float``).
    """
    n, n_in, n_gate = params.n, params.n_in, params.n_gate
    cp = params.content
    gp = params.gate

    def _fresh(size, v_rest_arr):
        zeros = jnp.zeros(size, DTYPE)
        return NeuronState(
            v=jnp.full(size, v_rest_arr, dtype=DTYPE),
            w_adapt=zeros,
            refrac=jnp.zeros(size, dtype=jnp.int32),
            x_pre=zeros,
            x_post=zeros,
            spikes=zeros,
        )

    return eqx.tree_at(
        lambda s: (
            s.content, s.gate, s.gate_rate, s.gate_signal,
            s.e, s.x_pre, s.x_post, s.content_trace,
        ),
        state,
        (
            _fresh(n, cp.v_rest),
            _fresh(n_gate, gp.v_thresh - jnp.asarray(2.0, DTYPE)),
            jnp.zeros(n_gate, DTYPE),
            jnp.asarray(0.0, DTYPE),
            jnp.zeros((n_in, n), DTYPE),
            jnp.zeros(n_in, DTYPE),
            jnp.zeros(n, DTYPE),
            jnp.zeros(n, DTYPE),
        ),
    )
