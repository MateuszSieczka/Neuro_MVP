"""Prefrontal working memory — pure JAX.

Goldman-Rakic (1995); O’Reilly & Frank (2006); Durstewitz et al. (2000);
Compte et al. (2000); Wang (1999) & Wong-Wang (2006) — bump attractor
with explicit inhibition; Brette & Gerstner (2005); Feldmeyer et al.
(2002, 2006); Cruikshank et al. (2012); Song et al. (2005); Perin et
al. (2011); Buhl et al. (1995); Markram et al. (2004).

Architecture:
- **Content neurons** (``n`` AdEx PFC pyramidals, τ_w=300 ms slow
  adaptation) with sparse feedforward ``w_ff`` + sparse recurrent
  ``w_lateral``. Conductance-based synapses: ``I = g · (E − V)``.
- **Inhibitory interneurons** (``n_inhib`` AdEx fast-spiking pool,
  Markram 2004; τ=10 ms, a=b=0) driven by content spikes through
  ``w_ci``, projecting back onto the content pool through ``w_ic``
  with GABA-A reversal E_inh = −75 mV (Buhl 1995). This implements
  the Wang (1999) / Compte (2000) competitive inhibition loop —
  without it the dense excitatory recurrence would lock every
  content neuron into a uniform attractor regardless of the input,
  destroying pattern selectivity.
- **Gate neurons** (``n_gate`` AdEx MSN-like, τ=25 ms) driven by
  ``ACh · DA · drive_calibrated``. Drive is derived from the AdEx
  rheobase so the population only fires above both thresholds
  (O’Reilly & Frank 2006 conjunction gate).
- Gate rate EMA ⇒ ``gate_signal ∈ [0, 1]`` that multiplicatively
  scales the ``w_ff`` conductance during content integration.

Connectivity — sparse random (Dale-compliant):
- ``w_ff`` : Bernoulli(p=0.25) × |N(0, σ_ff)|  (Song 2005
  thalamocortical statistics).
- ``w_lateral`` : Bernoulli(p=0.15) × |N(0, σ_lat)|, no autapses
  (Perin 2011 L2/3-L2/3 pair statistics).
- ``w_ci`` / ``w_ic`` : dense half-normal — the FS pool forms a
  global inhibitory broadcast (Wang 1999 mean-field equivalent).

PSP calibration (Feldmeyer 2002, 2006; Cruikshank 2012):
- ``ff_psp_mV``   default **2.0** mV (single thalamic → L4 EPSP).
- ``lat_psp_mV``  default **0.5** mV (single L2/3 → L2/3 EPSP).
- ``ci_psp_mV``   default 2.0 mV (pyramidal → FS).
- ``ic_psp_mV``   default 3.0 mV (FS → pyramidal IPSP, Buhl 1995).

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
    """Content + gate + inhibitory AdEx params + WM-specific scalars."""

    content: NeuronParams
    gate: NeuronParams
    inhib: NeuronParams        # Fast-spiking PV+ interneurons

    # Conductance-based synapses
    e_exc: Array
    e_inh: Array               # GABA-A reversal (Buhl 1995)
    driving_force_exc: Array   # E_exc − V_rest
    driving_force_inh: Array   # V_rest − E_inh  (positive; used for g scaling)

    # Lateral attractor
    lateral_strength: Array    # multiplies recurrent conductance
    lateral_lr: Array          # Hebbian lateral LR
    content_decay: Array       # low-pass on spikes for attractor trace

    # Inhibitory pool
    inhib_trace_decay: Array   # EMA of inhib spikes used as smooth IPSC

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
    n_inhib: int = eqx.field(static=True)


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
    n_inhib: int | None = None,
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
    # Inhibitory FS AdEx (Markram 2004; Brette & Gerstner 2005 FS)
    inhib_tau: float = 10.0,
    inhib_C_m: float = 200.0,
    inhib_delta_t: float = 0.5,
    inhib_refrac_ms: float = 1.0,
    inhib_trace_tau_ms: float = 10.0,   # GABA-A decay (Buhl 1995)
    # Synapses
    e_exc: float = 0.0,
    e_inh: float = -75.0,               # GABA-A Cl− reversal
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

    if n_inhib is None:
        # Pyramidal : FS ratio ≈ 4:1 in neocortex (Markram 2004).
        n_inhib = max(1, n // 4)

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
    # FS interneurons: a=b=0 (no adaptation, Brette & Gerstner 2005
    # Tab. 1 FS), short membrane τ, shallow Δ_T → sharp spike.
    inhib_g_L = inhib_C_m / inhib_tau
    inhib_p = init_neuron_params(
        ctx,
        v_rest=v_rest, v_thresh=v_thresh, v_reset=v_reset,
        v_spike_cutoff=v_spike_cutoff, delta_t=inhib_delta_t,
        C_m=inhib_C_m, g_L=inhib_g_L, tau_w=inhib_tau,
        a=0.0, b=0.0,
        refrac_period_ms=inhib_refrac_ms,
    )

    gap = abs(v_thresh - v_rest)
    drive = _rheobase_drive(
        gap, gate_delta_t, gate_C_m, gate_tau, gate_a,
        ach_gate_threshold, da_gate_threshold,
    )

    return WMParams(
        content=content_p,
        gate=gate_p,
        inhib=inhib_p,
        e_exc=f(e_exc),
        e_inh=f(e_inh),
        driving_force_exc=f(e_exc - v_rest),
        driving_force_inh=f(v_rest - e_inh),
        lateral_strength=f(lateral_strength),
        lateral_lr=f(lateral_lr),
        content_decay=f(ctx.decay(tau_w)),
        inhib_trace_decay=f(ctx.decay(inhib_trace_tau_ms)),
        ff_lr=f(ff_lr),
        trace_decay=f(ctx.decay(tau_e)),
        pre_decay=f(ctx.decay(tau_pre)),
        post_decay=f(ctx.decay(tau_post)),
        gate_drive=f(drive),
        gate_noise_std_mV=f(gate_noise_mV),
        gate_max_rate_per_step=f(gate_max_rate_hz * ctx.dt / 1000.0),
        gate_rate_decay=f(ctx.decay(gate_tau)),
        n_in=n_in, n=n, n_gate=n_gate, n_inhib=int(n_inhib),
    )


class WMState(eqx.Module):
    """Content + gate + inhibitory AdEx states, weights, eligibility,
    content trace, inhibitory spike trace."""

    content: NeuronState
    gate: NeuronState
    inhib: NeuronState
    gate_rate: Array          # (n_gate,) EMA of gate spikes
    gate_signal: Array        # scalar [0, 1]
    w_ff: Array               # (n_in, n) feedforward weights (nS, sparse)
    w_lateral: Array          # (n, n) sparse recurrent (nS)
    w_ci: Array               # (n, n_inhib) content → inhibitory (nS)
    w_ic: Array               # (n_inhib, n) inhibitory → content (nS)
    e: Array                  # (n_in, n) eligibility trace
    x_pre: Array              # (n_in,)
    x_post: Array             # (n,)
    content_trace: Array      # (n,) low-pass spike trace
    inhib_trace: Array        # (n_inhib,) low-pass FS spike trace


def init_wm_state(
    key: PRNGKey, params: WMParams,
    *,
    ff_psp_mV: float | None = None,
    lat_psp_mV: float | None = None,
    ci_psp_mV: float | None = None,
    ic_psp_mV: float | None = None,
    ff_sparsity: float = 0.25,    # Song 2005 thalamocortical
    lat_sparsity: float = 0.15,   # Perin 2011 L2/3 pair connectivity
    dtype=DTYPE,
) -> WMState:
    """Initialise WM with PSP-targeted sparse half-normal weights.

    ``ff_psp_mV`` / ``lat_psp_mV`` default to biologically realistic
    unitary EPSP amplitudes — **2.0 mV** for thalamic → L4 (Cruikshank
    et al. 2012) and **0.5 mV** for L2/3 → L2/3 recurrent (Feldmeyer
    et al. 2006). These are orders-of-magnitude smaller than the
    legacy ``gap/2`` / ``gap/3`` defaults (7.5 / 5.0 mV), which drove
    every content neuron simultaneously and destroyed any input
    selectivity (see ``plan.md`` PFC-selectivity bug).

    ``ci_psp_mV`` / ``ic_psp_mV`` default to 2.0 mV (pyramidal → FS)
    and 3.0 mV (FS → pyramidal IPSP peak, Buhl 1995) respectively;
    these are dense (no sparsity mask) because the FS pool is a
    global broadcast (Wang 1999 mean-field).

    Connectivity is sparse random:
    - ``w_ff`` : ``Bernoulli(ff_sparsity) × |N(0, σ_ff)|`` rescaled
      by ``1/ff_sparsity`` so that total mean drive is invariant to
      sparsity (Song 2005).
    - ``w_lateral`` : same with ``lat_sparsity``; no autapses.

    Per-synapse conductance is derived from
    ``g = PSP · g_L / |driving_force|`` and realised as a half-normal
    ``|N(0, σ)|`` with ``E[|w|] = g``.
    Gate neurons start in the up-state (V_T − 2 mV, Wilson & Kawaguchi
    1996) so ACh·DA drive can fire them within 2–3 ms.
    """
    k_ff, k_lat, k_mff, k_mlat, k_ci, k_ic = split_key(key, 6)
    n_in, n, n_gate = params.n_in, params.n, params.n_gate
    n_inhib = params.n_inhib
    gap = float(abs(params.content.v_thresh - params.content.v_rest))
    if ff_psp_mV is None:
        ff_psp_mV = 2.0
    if lat_psp_mV is None:
        lat_psp_mV = 0.5
    if ci_psp_mV is None:
        ci_psp_mV = 2.0
    if ic_psp_mV is None:
        ic_psp_mV = 3.0
    # Half-normal: E[|N(0, σ)|] = σ·√(2/π) ≈ 0.7979·σ.
    sqrt_2_over_pi = float(jnp.sqrt(2.0 / jnp.pi))
    g_L_c = float(params.content.g_L)
    g_L_i = float(params.inhib.g_L)
    df_exc = float(params.driving_force_exc)
    df_inh = float(params.driving_force_inh)

    def _sigma(psp_mV: float, g_L: float, df: float) -> float:
        g = psp_mV * g_L / max(df, 1e-6)
        return g / sqrt_2_over_pi

    sigma_ff = _sigma(ff_psp_mV, g_L_c, df_exc)
    sigma_lat = _sigma(lat_psp_mV, g_L_c, df_exc)
    sigma_ci = _sigma(ci_psp_mV, g_L_i, df_exc)
    sigma_ic = _sigma(ic_psp_mV, g_L_c, df_inh)

    ff_sparsity = max(min(float(ff_sparsity), 1.0), 1e-3)
    lat_sparsity = max(min(float(lat_sparsity), 1.0), 1e-3)

    # --- Excitatory ff: sparse half-normal -------------------------
    raw_ff = jnp.abs(
        jax.random.normal(k_ff, (n_in, n), dtype=dtype) * sigma_ff
    )
    mask_ff = (
        jax.random.uniform(k_mff, (n_in, n), dtype=dtype) < ff_sparsity
    ).astype(dtype)
    w_ff = raw_ff * mask_ff / ff_sparsity
    # --- Recurrent lat: sparse, no autapses -----------------------
    raw_lat = jnp.abs(
        jax.random.normal(k_lat, (n, n), dtype=dtype) * sigma_lat
    )
    mask_lat = (
        jax.random.uniform(k_mlat, (n, n), dtype=dtype) < lat_sparsity
    ).astype(dtype)
    mask_lat = mask_lat * (1.0 - jnp.eye(n, dtype=dtype))
    w_lat = raw_lat * mask_lat / lat_sparsity
    # --- Content ↔ inhib: dense half-normal (Wang 1999 global) -----
    w_ci = jnp.abs(
        jax.random.normal(k_ci, (n, n_inhib), dtype=dtype) * sigma_ci
    )
    w_ic = jnp.abs(
        jax.random.normal(k_ic, (n_inhib, n), dtype=dtype) * sigma_ic
    )

    content_state = init_neuron_state(n, v_rest=float(params.content.v_rest))
    gate_state = init_neuron_state(
        n_gate, v_rest=float(params.gate.v_thresh - 2.0),  # up-state
    )
    inhib_state = init_neuron_state(
        n_inhib, v_rest=float(params.inhib.v_rest),
    )

    return WMState(
        content=content_state,
        gate=gate_state,
        inhib=inhib_state,
        gate_rate=jnp.zeros(n_gate, dtype=dtype),
        gate_signal=jnp.asarray(0.0, dtype),
        w_ff=w_ff,
        w_lateral=w_lat,
        w_ci=w_ci,
        w_ic=w_ic,
        e=jnp.zeros((n_in, n), dtype=dtype),
        x_pre=jnp.zeros(n_in, dtype=dtype),
        x_post=jnp.zeros(n, dtype=dtype),
        content_trace=jnp.zeros(n, dtype=dtype),
        inhib_trace=jnp.zeros(n_inhib, dtype=dtype),
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


def _inhib_step(
    state: WMState, params: WMParams, ctx: BackendContext,
) -> WMState:
    """Advance the AdEx FS interneuron pool.

    Driven by **current content spikes** via ``w_ci``; produces new
    inhibitory spikes whose low-pass ``inhib_trace`` is consumed by
    the next ``_content_step`` as a smooth IPSC drive.  This one-step
    causal lag (inhibition always derives from the *previous* content
    spike vector) is biologically realistic (axonal + synaptic delay
    ≈ 1 ms, Markram 2004) and keeps the kernel a simple feedforward
    sweep inside ``wm_step``.
    """
    ip = params.inhib
    # Drive: content spikes × w_ci → (n_inhib,). Conductance-based.
    g_drive = state.content.spikes @ state.w_ci
    I_drive = g_drive * (params.e_exc - state.inhib.v)
    new_inhib, inhib_spikes = neuron_step(
        state.inhib, ip, ctx, i_syn=I_drive, g_syn=g_drive,
    )
    # Low-pass trace — smooth IPSC (Buhl 1995 GABA-A τ ≈ 10 ms).
    trace = (
        state.inhib_trace * params.inhib_trace_decay
        + inhib_spikes * (1.0 - params.inhib_trace_decay)
    )
    return eqx.tree_at(
        lambda s: (s.inhib, s.inhib_trace),
        state,
        (new_inhib, trace),
    )


def _content_step(
    state: WMState, params: WMParams, ctx: BackendContext,
    external_input: Array, receptor_gain: Array,
) -> WMOutput:
    """Advance the PFC content population under gated ff + attractor +
    global inhibition."""
    cp = params.content
    gate = state.gate_signal
    ext = external_input.astype(DTYPE)

    # Conductance-based input (gate-scaled, receptor-modulated).
    g_ff = gate * receptor_gain * (ext @ state.w_ff)            # (n,)
    I_ff = g_ff * (params.e_exc - state.content.v)

    # Recurrent attractor: lateral_strength · content_trace @ w_lat.
    g_rec = params.lateral_strength * (state.content_trace @ state.w_lateral)
    I_rec = g_rec * (params.e_exc - state.content.v)

    # Global inhibition (Wang 1999): smooth IPSC from FS pool.
    g_inh = state.inhib_trace @ state.w_ic                      # (n,)
    I_inh = g_inh * (params.e_inh - state.content.v)            # negative at V > E_inh

    i_syn = I_ff + I_rec + I_inh
    g_syn = g_ff + g_rec + g_inh

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
    """One WM tick: gate update → FS inhib update (driven by previous
    content spikes) → content AdEx with ff + recurrent + inhibitory
    drive + STDP traces.

    Execution order encodes the biological delays: the FS pool
    integrates the *previous* content spike vector (≈ 1 ms lag), so
    the content step here already sees the freshly-updated
    ``inhib_trace`` as a competitive damping signal."""
    ach_a = jnp.asarray(ach, DTYPE)
    da_a = jnp.asarray(da, DTYPE)
    rg = jnp.asarray(receptor_gain, DTYPE)
    state = _gate_step(state, params, ctx, ach_a, da_a, key)
    state = _inhib_step(state, params, ctx)
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
    n_inhib = params.n_inhib
    cp = params.content
    gp = params.gate
    ip = params.inhib

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
            s.content, s.gate, s.inhib,
            s.gate_rate, s.gate_signal,
            s.e, s.x_pre, s.x_post,
            s.content_trace, s.inhib_trace,
        ),
        state,
        (
            _fresh(n, cp.v_rest),
            _fresh(n_gate, gp.v_thresh - jnp.asarray(2.0, DTYPE)),
            _fresh(n_inhib, ip.v_rest),
            jnp.zeros(n_gate, DTYPE),
            jnp.asarray(0.0, DTYPE),
            jnp.zeros((n_in, n), DTYPE),
            jnp.zeros(n_in, DTYPE),
            jnp.zeros(n, DTYPE),
            jnp.zeros(n, DTYPE),
            jnp.zeros(n_inhib, DTYPE),
        ),
    )
