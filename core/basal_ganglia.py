"""Basal ganglia — pure JAX port of D1/D2 dual-pathway action selection.

References:
  Frank (2005) "Dynamic dopamine modulation in the basal ganglia"
  Collins & Frank (2014) "Opponent actor learning (OpAL)"
  Shen, Flajolet, Greengard & Surmeier (2008) bidirectional D1/D2 STDP
  Taverna, Ilijic & Bhardwaj (2008) striatal lateral inhibition
  Bogacz & Gurney (2007) STN-GPe hyperdirect pathway
  Pezzulo et al. (2018) prefrontal → striatal EFE projection
  Niv et al. (2007) tonic DA tracks average reward rate
  Wilson & Kawaguchi (1996) MSN bistable dynamics

Architecture:
  - SNNDeepCritic: single AdEx pop, ventral striatum value estimator.
    Updated by VTA RPE (external) via three-factor STDP on ``e_h``.
  - D1D2Actor: two parallel AdEx pops of shape (action_dim,), each with
    population coding (``n_per_action`` MSNs per motor action).
    Selection is Gold & Shadlen spike-count WTA (``D1 - D2`` evidence).
    Learning uses the Collins & Frank asymmetric Hill rule:
      Δw_d1_LTP = +lr · (1 + d1_density · hill(DA)) · TD · e_d1   (TD>0)
      Δw_d1_LTD = -lr · ltd_ratio · (1 - d1_density · hill(DA)) · |TD| · e_d1  (TD<0)
      Δw_d2_LTP = +lr · (1 + d2_density · (1 - hill(DA))) · |TD| · e_d2  (TD<0)

Differences from legacy:
  - Bistable τ-switching removed (per-step τ_up/τ_down conditional was
    a marginal effect once exp-Euler stabilises the Up state; ncfg uses
    fixed tau_m_msn_up).
  - Continuous homeostatic scaling removed (Dale's law floor + column
    norm prune at init provide enough stability; add back if needed).
  - ATP / astrocyte coupling omitted (world_model is the cortex-side
    astrocyte client; BG is purely striatal).
  - Integration is one ``dt`` per call; callers run N substeps via
    ``jax.lax.scan`` and read spike-count evidence at the end.
  - Action selection returns ``net_evidence`` (pre-argmax) so callers
    can inject exploration noise, break ties, or argmax themselves.
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
from .receptor import hill_response


# =====================================================================
# Critic (ventral striatum)
# =====================================================================


def _rheobase_cond_scale(
    ncfg: NeuronParams,
    fan_in: int,
    *,
    e_exc: float = 0.0,
    ampa_frac: float = 0.75,
    target_rate: float = 0.05,
    headroom: float = 2.0,
) -> float:
    """Boost factor so expected steady-state input drives AdEx to rheobase.

    Follows legacy ``_derive_conductance_scale`` (Bogacz 2020 chap. 6):
      ``scale = headroom \u00b7 I_rheo / (N_active \u00b7 mean_w \u00b7 df \u00b7 ampa_eff)``
    with ``mean_w`` equal to the expected half-normal mean from the
    PSP-targeted init (``psp \u00b7 g_L / df``).
    """
    gap = float(abs(ncfg.v_thresh - ncfg.v_rest))
    i_rheo = float(ncfg.g_L) * (gap - float(ncfg.delta_t))
    df = float(e_exc - ncfg.v_rest)
    # Effective driving force over one \u03c4: (E_exc - V_thresh)
    effective_df = float(e_exc - ncfg.v_thresh)
    # Mg block at V_thresh (Jahr & Stevens 1990)
    mg_at_thr = 1.0 / (1.0 + 0.28 * float(jnp.exp(-0.062 * ncfg.v_thresh)))
    ampa_eff = ampa_frac + (1.0 - ampa_frac) * mg_at_thr
    # Half-normal mean from PSP init: w_mean = psp \u00b7 g_L / df \u00b7 \u221a(2/\u03c0)/0.798 = psp \u00b7 g_L / df
    psp = gap / 2.0
    mean_w = psp * float(ncfg.g_L) / max(df, 1.0)
    expected_active = max(1.0, fan_in * target_rate)
    denom = expected_active * mean_w * max(abs(effective_df), 1.0) * ampa_eff
    return headroom * i_rheo / max(denom, 1e-6)


class CriticParams(eqx.Module):
    """Static params for ``SNNDeepCritic``."""

    ncfg: NeuronParams
    ipool: IPoolParams
    e_exc: Array
    critic_lr: Array
    trace_decay: Array    # eligibility decay
    pre_decay: Array
    post_decay: Array
    rate_decay: Array
    nmda_decay: Array
    ampa_frac: Array
    cond_scale: Array     # forward-time conductance boost (AdEx rheobase)
    state_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)


def init_critic_params(
    ctx: BackendContext,
    state_size: int,
    *,
    hidden_size: int = 64,
    tau_m: float = 15.0,
    critic_lr: float = 0.03,
    tau_eligibility: float = 200.0,
    tau_trace: float = 20.0,
    tau_rate: float = 20.0,
    tau_nmda: float = 100.0,
    ampa_nmda_ratio: float = 3.0,
    e_exc: float = 0.0,
    v_rest: float = -70.0,
    v_thresh: float = -55.0,
    v_reset: float = -75.0,
    C_m: float = 281.0,
    n_interneurons: int = 0,
) -> CriticParams:
    """Build critic params. ``hidden_size`` AdEx neurons + optional inh pool."""
    g_L = C_m / tau_m
    # Adaptation scaling so that b/g_L preserves mV effect
    scale = g_L / 30.0
    ncfg = init_neuron_params(
        ctx, v_rest=v_rest, v_thresh=v_thresh, v_reset=v_reset,
        C_m=C_m, g_L=g_L, tau_w=144.0, a=4.0 * scale, b=80.5 * scale,
    )
    n_inh = max(1, n_interneurons if n_interneurons > 0 else hidden_size // 4)
    ipool = init_ipool_params(ctx, hidden_size, n_inh=n_inh)
    f = lambda x: jnp.asarray(x, DTYPE)
    ampa_frac = ampa_nmda_ratio / (1.0 + ampa_nmda_ratio)
    cond_scale = _rheobase_cond_scale(
        ncfg, fan_in=state_size, e_exc=e_exc, ampa_frac=ampa_frac,
    )
    return CriticParams(
        ncfg=ncfg, ipool=ipool, e_exc=f(e_exc),
        critic_lr=f(critic_lr),
        trace_decay=f(ctx.decay(tau_eligibility)),
        pre_decay=f(ctx.decay(tau_trace)),
        post_decay=f(ctx.decay(tau_trace)),
        rate_decay=f(ctx.decay(tau_rate)),
        nmda_decay=f(ctx.decay(tau_nmda)),
        ampa_frac=f(ampa_frac),
        cond_scale=f(cond_scale),
        state_size=state_size, hidden_size=hidden_size,
    )


class CriticState(eqx.Module):
    nstate: NeuronState
    ipool: IPoolState
    w_h: Array              # (state_size, hidden_size)
    g_nmda: Array           # (hidden_size,)
    activation: Array       # (hidden_size,) rate EMA
    e_h: Array              # (state_size, hidden_size) eligibility
    x_pre: Array            # (state_size,)
    x_post: Array           # (hidden_size,)


def _psp_scale(ncfg: NeuronParams, psp_mV: float, e_exc: float) -> float:
    g_L = float(ncfg.g_L)
    df = float(e_exc - ncfg.v_rest)
    return float(psp_mV * g_L / max(df, 1.0) / 0.7978845608)


def init_critic_state(
    key: PRNGKey, params: CriticParams,
    *, psp_mV: float | None = None, dtype=DTYPE,
) -> CriticState:
    gap = float(params.ncfg.v_thresh - params.ncfg.v_rest)
    psp = psp_mV if psp_mV is not None else gap / 2.0
    sigma = _psp_scale(params.ncfg, psp, float(params.e_exc))
    k_w, k_i = split_key(key, 2)
    w_h = jnp.abs(jax.random.normal(
        k_w, (params.state_size, params.hidden_size), dtype=dtype,
    ) * sigma)
    return CriticState(
        nstate=init_neuron_state(params.hidden_size, v_rest=float(params.ncfg.v_rest)),
        ipool=init_ipool_state(k_i, params.ipool, target_sparsity=0.15),
        w_h=w_h,
        g_nmda=jnp.zeros(params.hidden_size, dtype),
        activation=jnp.zeros(params.hidden_size, dtype),
        e_h=jnp.zeros((params.state_size, params.hidden_size), dtype),
        x_pre=jnp.zeros(params.state_size, dtype),
        x_post=jnp.zeros(params.hidden_size, dtype),
    )


def _nmda_mg_block(v: Array) -> Array:
    """Jahr & Stevens 1990 voltage-dep Mg²⁺ block."""
    return 1.0 / (1.0 + 0.28 * jnp.exp(-0.062 * v))


class CriticOutput(NamedTuple):
    state: CriticState
    spikes: Array
    activation: Array       # rate EMA (readout for V(s))


def critic_step(
    state: CriticState, params: CriticParams, ctx: BackendContext,
    state_spikes: Array,
    *, receptor_gain: float | Array = 1.0,
) -> CriticOutput:
    """One dt of critic dynamics (no learning; use ``critic_update`` after)."""
    inp = state_spikes.astype(DTYPE)
    rg = jnp.asarray(receptor_gain, DTYPE)

    # Pre trace
    x_pre = state.x_pre * params.pre_decay + (inp > 0.5).astype(DTYPE)

    # Synaptic conductance (nS)
    g_total = (inp @ state.w_h) * rg * params.cond_scale
    # NMDA EMA
    g_nmda = state.g_nmda * params.nmda_decay + (1.0 - params.nmda_decay) * g_total
    # AMPA + NMDA·B(V)
    mg = _nmda_mg_block(state.nstate.v)
    ampa = params.ampa_frac
    driving = params.e_exc - state.nstate.v
    i_syn = (ampa * g_total + (1.0 - ampa) * g_nmda * mg) * driving
    g_syn = ampa * g_total + (1.0 - ampa) * g_nmda * mg

    # AdEx step
    new_n, spikes = neuron_step(state.nstate, params.ncfg, ctx, i_syn=i_syn, g_syn=g_syn)

    # Inhibitory pool feedback (I->E)
    ipool_out = ipool_step(state.ipool, params.ipool, ctx, spikes, new_n.v)
    # Apply feedback current: new_n.v was already integrated; subtract a
    # small voltage correction proportional to i_inh / g_L (one-step
    # Euler approximation for I->E feedback between dt).
    v_corr = ipool_out.i_inh / params.ncfg.g_L
    v_post = jnp.clip(new_n.v - v_corr, -90.0, None)
    new_n = eqx.tree_at(lambda s: s.v, new_n, v_post)

    # Rate EMA
    activation = state.activation * params.rate_decay + spikes * (1.0 - params.rate_decay)

    # Post trace
    x_post = state.x_post * params.post_decay + spikes

    # Voltage-based eligibility (Clopath)
    v_norm = jnp.clip(
        (new_n.v - params.ncfg.v_rest) / (params.ncfg.v_thresh - params.ncfg.v_rest),
        0.0, 1.0,
    )
    v_centered = v_norm - jnp.mean(v_norm)
    e_compl = 1.0 - params.trace_decay
    e_h = (
        state.e_h * params.trace_decay
        + e_compl * inp[:, None] * v_centered[None, :]
    )

    new_s = CriticState(
        nstate=new_n, ipool=ipool_out.state, w_h=state.w_h,
        g_nmda=g_nmda, activation=activation,
        e_h=e_h, x_pre=x_pre, x_post=x_post,
    )
    return CriticOutput(state=new_s, spikes=spikes, activation=activation)


def critic_update(
    state: CriticState, params: CriticParams,
    td_error: float | Array,
    *, receptor_lr: float | Array = 1.0,
) -> CriticState:
    """Three-factor STDP: Δw_h = critic_lr · td · e_h."""
    td = jnp.asarray(td_error, DTYPE)
    rlr = jnp.asarray(receptor_lr, DTYPE)
    dw = params.critic_lr * rlr * td * state.e_h
    return eqx.tree_at(lambda s: s.w_h, state, state.w_h + dw)


def critic_reset_transient(
    state: CriticState, params: CriticParams,
) -> CriticState:
    return CriticState(
        nstate=init_neuron_state(params.hidden_size, v_rest=float(params.ncfg.v_rest)),
        ipool=ipool_reset_transient(state.ipool, params.ipool),
        w_h=state.w_h,
        g_nmda=jnp.zeros_like(state.g_nmda),
        activation=jnp.zeros_like(state.activation),
        e_h=jnp.zeros_like(state.e_h),
        x_pre=jnp.zeros_like(state.x_pre),
        x_post=jnp.zeros_like(state.x_post),
    )


# =====================================================================
# D1/D2 Actor (dorsal striatum)
# =====================================================================


class ActorParams(eqx.Module):
    """Static params for ``D1D2Actor``."""

    ncfg: NeuronParams
    ipool_d1: IPoolParams
    ipool_d2: IPoolParams
    e_exc: Array
    e_inh: Array
    actor_lr: Array
    ltd_ratio: Array
    d1_ec50: Array
    d1_hill_n: Array
    d1_density: Array
    d2_ec50: Array
    d2_hill_n: Array
    d2_density: Array
    d2_tonic_boost_max: Array
    d2_gain_comp: Array
    baseline_da: Array
    stn_strength: Array
    trace_decay: Array
    pre_decay: Array
    post_decay: Array
    rate_decay: Array
    nmda_decay: Array
    ampa_frac: Array
    cond_scale: Array         # forward-time conductance boost
    voltage_floor: Array
    down_state_factor: Array
    # Lateral (Taverna 2008): g_lat = p_conn · iPSP / driving_force
    g_lat_d1: Array
    g_lat_d2: Array
    g_lat_x: Array
    state_size: int = eqx.field(static=True)
    motor_dim: int = eqx.field(static=True)
    n_per_action: int = eqx.field(static=True)
    total_motor: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)


def init_actor_params(
    ctx: BackendContext,
    state_size: int,
    motor_dim: int,
    *,
    internal_dim: int = 0,
    n_per_action: int = 4,
    tau_m: float = 25.0,
    actor_lr: float = 3e-3,
    ltd_ratio: float = 0.7,
    baseline_da: float = 0.5,
    stn_strength: float = 0.5,
    d1_ec50: float = 0.5,
    d1_hill_n: float = 1.0,
    d1_density: float = 1.0,
    d2_ec50: float = 0.3,
    d2_hill_n: float = 1.2,
    d2_density: float = 0.6,
    d2_tonic_boost_max: float = 1.0,
    tau_eligibility: float = 500.0,
    tau_trace: float = 20.0,
    tau_rate: float = 25.0,
    tau_nmda: float = 100.0,
    ampa_nmda_ratio: float = 3.0,
    e_exc: float = 0.0,
    e_inh: float = -75.0,
    v_rest: float = -70.0,
    v_thresh: float = -55.0,
    v_reset: float = -75.0,
    C_m: float = 281.0,
    voltage_floor: float = 0.07,
    down_state_factor: float = 0.1,
) -> ActorParams:
    n_per_action = max(1, n_per_action)
    total_motor = motor_dim * n_per_action
    action_dim = total_motor + internal_dim

    g_L = C_m / tau_m
    scale = g_L / 30.0
    ncfg = init_neuron_params(
        ctx, v_rest=v_rest, v_thresh=v_thresh, v_reset=v_reset,
        C_m=C_m, g_L=g_L, tau_w=144.0, a=4.0 * scale, b=80.5 * scale,
    )
    ipool_d1 = init_ipool_params(ctx, action_dim, n_inh=max(2, action_dim // 2))
    ipool_d2 = init_ipool_params(ctx, action_dim, n_inh=max(2, action_dim // 2))

    # D2 gain compensation at baseline DA (Planert 2010: D1≈D2 in vivo)
    d1_resp_base = 1.0 / (1.0 + (d1_ec50 / max(baseline_da, 1e-6)) ** d1_hill_n)
    d1_mod_base = 1.0 + d1_density * d1_resp_base
    d2_resp_base = 1.0 / (1.0 + (d2_ec50 / max(baseline_da, 1e-6)) ** d2_hill_n)
    d2_mod_base = 1.0 - d2_density * d2_resp_base
    d2_tonic_base = 1.0 + d2_tonic_boost_max * (1.0 - baseline_da)
    d2_net_base = max(d2_mod_base * d2_tonic_base, 0.1)
    d2_gain_comp = d1_mod_base / d2_net_base

    # Lateral inhibition (Taverna 2008, Table 1)
    df_inh = v_thresh - e_inh   # ≈20 mV
    g_lat_d1 = 0.14 * 0.14 / df_inh
    g_lat_d2 = 0.11 * 0.12 / df_inh
    g_lat_x = 0.06 * 0.10 / df_inh

    f = lambda x: jnp.asarray(x, DTYPE)
    ampa_frac = ampa_nmda_ratio / (1.0 + ampa_nmda_ratio)
    cond_scale = _rheobase_cond_scale(
        ncfg, fan_in=state_size, e_exc=e_exc, ampa_frac=ampa_frac,
    )
    return ActorParams(
        ncfg=ncfg, ipool_d1=ipool_d1, ipool_d2=ipool_d2,
        e_exc=f(e_exc), e_inh=f(e_inh),
        actor_lr=f(actor_lr), ltd_ratio=f(ltd_ratio),
        d1_ec50=f(d1_ec50), d1_hill_n=f(d1_hill_n), d1_density=f(d1_density),
        d2_ec50=f(d2_ec50), d2_hill_n=f(d2_hill_n), d2_density=f(d2_density),
        d2_tonic_boost_max=f(d2_tonic_boost_max),
        d2_gain_comp=f(d2_gain_comp),
        baseline_da=f(baseline_da), stn_strength=f(stn_strength),
        trace_decay=f(ctx.decay(tau_eligibility)),
        pre_decay=f(ctx.decay(tau_trace)),
        post_decay=f(ctx.decay(tau_trace)),
        rate_decay=f(ctx.decay(tau_rate)),
        nmda_decay=f(ctx.decay(tau_nmda)),
        ampa_frac=f(ampa_frac),
        cond_scale=f(cond_scale),
        voltage_floor=f(voltage_floor),
        down_state_factor=f(down_state_factor),
        g_lat_d1=f(g_lat_d1), g_lat_d2=f(g_lat_d2), g_lat_x=f(g_lat_x),
        state_size=state_size,
        motor_dim=motor_dim, n_per_action=n_per_action,
        total_motor=total_motor, action_dim=action_dim,
    )


class ActorState(eqx.Module):
    nstate_d1: NeuronState
    nstate_d2: NeuronState
    ipool_d1: IPoolState
    ipool_d2: IPoolState
    w_d1: Array             # (state_size, action_dim)
    w_d2: Array             # (state_size, action_dim)
    g_nmda_d1: Array
    g_nmda_d2: Array
    rate_d1: Array
    rate_d2: Array
    e_d1: Array
    e_d2: Array
    x_pre: Array
    x_post_d1: Array
    x_post_d2: Array
    spike_count_d1: Array   # integer evidence accumulator (per dt)
    spike_count_d2: Array
    efe_g: Array            # (total_motor,) per-action EFE conductance
    epistemic_drive: Array  # scalar [0,1]
    da_level: Array         # phasic
    tonic_da: Array


def init_actor_state(
    key: PRNGKey, params: ActorParams,
    *, psp_mV: float | None = None, dtype=DTYPE,
) -> ActorState:
    gap = float(params.ncfg.v_thresh - params.ncfg.v_rest)
    psp = psp_mV if psp_mV is not None else gap / 2.0
    sigma = _psp_scale(params.ncfg, psp, float(params.e_exc))
    k1, k2, ki1, ki2 = split_key(key, 4)
    w_d1 = jnp.abs(jax.random.normal(
        k1, (params.state_size, params.action_dim), dtype=dtype,
    ) * sigma)
    w_d2 = jnp.abs(jax.random.normal(
        k2, (params.state_size, params.action_dim), dtype=dtype,
    ) * sigma)
    ad = params.action_dim
    z = lambda shape: jnp.zeros(shape, dtype)
    return ActorState(
        nstate_d1=init_neuron_state(ad, v_rest=float(params.ncfg.v_rest)),
        nstate_d2=init_neuron_state(ad, v_rest=float(params.ncfg.v_rest)),
        ipool_d1=init_ipool_state(ki1, params.ipool_d1, target_sparsity=0.05),
        ipool_d2=init_ipool_state(ki2, params.ipool_d2, target_sparsity=0.05),
        w_d1=w_d1, w_d2=w_d2,
        g_nmda_d1=z(ad), g_nmda_d2=z(ad),
        rate_d1=z(ad), rate_d2=z(ad),
        e_d1=z((params.state_size, ad)),
        e_d2=z((params.state_size, ad)),
        x_pre=z(params.state_size),
        x_post_d1=z(ad), x_post_d2=z(ad),
        spike_count_d1=z(ad), spike_count_d2=z(ad),
        efe_g=z(params.total_motor),
        epistemic_drive=jnp.asarray(0.0, dtype),
        da_level=jnp.asarray(0.5, dtype),
        tonic_da=jnp.asarray(float(params.baseline_da), dtype),
    )


# --- Modulation inputs bundle --------------------------------------


class ActorInputs(NamedTuple):
    """Per-step scalar inputs from neuromodulator / world model / EFE."""

    da: Array | float = 0.5
    tonic_da: Array | float = 0.5
    epistemic_drive: Array | float = 0.0
    efe_g: Array | None = None
    receptor_gain: Array | float = 1.0


def _hill(x: Array, ec50: Array, n: Array) -> Array:
    x_c = jnp.clip(x, 0.0, 1.0)
    xn = x_c ** n
    ec = ec50 ** n
    return xn / (xn + ec + 1e-12)


class ActorOutput(NamedTuple):
    state: ActorState
    spikes_d1: Array
    spikes_d2: Array
    net_evidence: Array     # (motor_dim,) spike-count D1 - D2 aggregated


def actor_step(
    state: ActorState, params: ActorParams, ctx: BackendContext,
    state_spikes: Array,
    inputs: ActorInputs = ActorInputs(),
) -> ActorOutput:
    """One dt of D1/D2 actor dynamics."""
    inp = state_spikes.astype(DTYPE)
    da = jnp.clip(jnp.asarray(inputs.da, DTYPE), 0.0, 1.0)
    tonic = jnp.clip(jnp.asarray(inputs.tonic_da, DTYPE), 0.0, 1.0)
    eps = jnp.clip(jnp.asarray(inputs.epistemic_drive, DTYPE), 0.0, 1.0)
    rg = jnp.asarray(inputs.receptor_gain, DTYPE)
    # EFE input: (motor_dim,) per-action conductance \u2192 expand to (total_motor,)
    # via ``repeat`` across the ``n_per_action`` MSN subpopulation.
    if inputs.efe_g is None:
        efe_g = jnp.zeros(params.total_motor, DTYPE)
    else:
        efe_in = jnp.asarray(inputs.efe_g, DTYPE)
        efe_g = jnp.repeat(jnp.clip(efe_in, 0.0, None), params.n_per_action)

    # Pre trace
    x_pre = state.x_pre * params.pre_decay + (inp > 0.5).astype(DTYPE)

    # Synaptic conductances
    g_d1 = (inp @ state.w_d1) * rg * params.cond_scale
    g_d2 = (inp @ state.w_d2) * rg * params.cond_scale * params.d2_gain_comp

    # NMDA EMA
    g_nmda_d1 = state.g_nmda_d1 * params.nmda_decay + (1.0 - params.nmda_decay) * g_d1
    g_nmda_d2 = state.g_nmda_d2 * params.nmda_decay + (1.0 - params.nmda_decay) * g_d2

    mg_d1 = _nmda_mg_block(state.nstate_d1.v)
    mg_d2 = _nmda_mg_block(state.nstate_d2.v)
    ampa = params.ampa_frac
    df_d1 = params.e_exc - state.nstate_d1.v
    df_d2 = params.e_exc - state.nstate_d2.v
    i_d1 = (ampa * g_d1 + (1.0 - ampa) * g_nmda_d1 * mg_d1) * df_d1
    i_d2 = (ampa * g_d2 + (1.0 - ampa) * g_nmda_d2 * mg_d2) * df_d2
    gsyn_d1 = ampa * g_d1 + (1.0 - ampa) * g_nmda_d1 * mg_d1
    gsyn_d2 = ampa * g_d2 + (1.0 - ampa) * g_nmda_d2 * mg_d2

    # D1 excitation / D2 inhibition (Frank 2005 Hill)
    d1_r = _hill(da, params.d1_ec50, params.d1_hill_n)
    d1_mod = 1.0 + params.d1_density * d1_r
    d2_r = _hill(da, params.d2_ec50, params.d2_hill_n)
    d2_mod = 1.0 - params.d2_density * d2_r
    d2_tonic = 1.0 + params.d2_tonic_boost_max * (1.0 - tonic)
    i_d1 = i_d1 * d1_mod
    i_d2 = i_d2 * d2_mod * d2_tonic

    # STN-GPe gate (HACK A): only suppress when DA below baseline
    da_deficit = jnp.maximum(params.baseline_da - da, 0.0)
    stn_factor = jnp.maximum(1.0 - params.stn_strength * da_deficit, 0.0)
    i_d1 = i_d1 * stn_factor
    i_d2 = i_d2 * stn_factor

    # Fast epistemic drive → D1 boost
    i_d1 = i_d1 * (1.0 + eps)

    # EFE conductance → D1 only, motor slice (Pezzulo 2018)
    efe_I = efe_g * (params.e_exc - state.nstate_d1.v[:params.total_motor])
    i_d1 = i_d1.at[:params.total_motor].add(efe_I)
    gsyn_d1 = gsyn_d1.at[:params.total_motor].add(efe_g)

    # AdEx steps
    new_d1, spk_d1 = neuron_step(
        state.nstate_d1, params.ncfg, ctx, i_syn=i_d1, g_syn=gsyn_d1,
    )
    new_d2, spk_d2 = neuron_step(
        state.nstate_d2, params.ncfg, ctx, i_syn=i_d2, g_syn=gsyn_d2,
    )

    # Inhibitory pool feedback (per pathway)
    ip_d1 = ipool_step(state.ipool_d1, params.ipool_d1, ctx, spk_d1, new_d1.v)
    ip_d2 = ipool_step(state.ipool_d2, params.ipool_d2, ctx, spk_d2, new_d2.v)
    v_d1 = new_d1.v - ip_d1.i_inh / params.ncfg.g_L
    v_d2 = new_d2.v - ip_d2.i_inh / params.ncfg.g_L
    v_d1 = jnp.clip(v_d1, -90.0, None)
    v_d2 = jnp.clip(v_d2, -90.0, None)

    # Cross-action lateral inhibition (Taverna 2008): only on motor slice
    npa = params.n_per_action
    M = params.motor_dim
    tm = params.total_motor
    d1_per_action = spk_d1[:tm].reshape(M, npa).sum(axis=1)
    d2_per_action = spk_d2[:tm].reshape(M, npa).sum(axis=1)
    total_d1 = d1_per_action.sum()
    total_d2 = d2_per_action.sum()
    # "others" = total - own; broadcast to (M, npa) then flatten
    d1_others = (total_d1 - d1_per_action)[:, None] * jnp.ones((M, npa), DTYPE)
    d2_others = (total_d2 - d2_per_action)[:, None] * jnp.ones((M, npa), DTYPE)
    d1_others_flat = d1_others.reshape(tm)
    d2_others_flat = d2_others.reshape(tm)
    # Apply driving force (E_inh - V), self-limiting at E_inh
    dv_d1 = params.g_lat_d1 * d1_others_flat * (params.e_inh - v_d1[:tm])
    dv_d2 = (
        params.g_lat_d2 * d2_others_flat * (params.e_inh - v_d2[:tm])
        + params.g_lat_x * d1_others_flat * (params.e_inh - v_d2[:tm])
    )
    v_d1 = v_d1.at[:tm].add(dv_d1)
    v_d2 = v_d2.at[:tm].add(dv_d2)
    v_d1 = jnp.clip(v_d1, -90.0, None)
    v_d2 = jnp.clip(v_d2, -90.0, None)
    new_d1 = eqx.tree_at(lambda s: s.v, new_d1, v_d1)
    new_d2 = eqx.tree_at(lambda s: s.v, new_d2, v_d2)

    # Rate EMAs
    rd = params.rate_decay
    rate_d1 = state.rate_d1 * rd + spk_d1 * (1.0 - rd)
    rate_d2 = state.rate_d2 * rd + spk_d2 * (1.0 - rd)

    # Post traces
    x_post_d1 = state.x_post_d1 * params.post_decay + spk_d1
    x_post_d2 = state.x_post_d2 * params.post_decay + spk_d2

    # Hybrid eligibility (HACK B): spike trace + voltage floor (Clopath)
    v_d1_norm = jnp.clip(
        (v_d1 - params.ncfg.v_rest) / (params.ncfg.v_thresh - params.ncfg.v_rest),
        0.0, 1.0,
    )
    v_d2_norm = jnp.clip(
        (v_d2 - params.ncfg.v_rest) / (params.ncfg.v_thresh - params.ncfg.v_rest),
        0.0, 1.0,
    )
    post_d1 = jnp.maximum(x_post_d1, params.voltage_floor * v_d1_norm)
    post_d2 = jnp.maximum(x_post_d2, params.voltage_floor * v_d2_norm)
    e_compl = 1.0 - params.trace_decay
    e_d1 = state.e_d1 * params.trace_decay + e_compl * inp[:, None] * post_d1[None, :]
    e_d2 = state.e_d2 * params.trace_decay + e_compl * inp[:, None] * post_d2[None, :]

    # Evidence accumulator
    sc_d1 = state.spike_count_d1 + spk_d1
    sc_d2 = state.spike_count_d2 + spk_d2

    # Net evidence per motor action (spike-count difference)
    ev_d1 = sc_d1[:tm].reshape(M, npa).sum(axis=1)
    ev_d2 = sc_d2[:tm].reshape(M, npa).sum(axis=1)
    net_ev = ev_d1 - ev_d2

    new_s = ActorState(
        nstate_d1=new_d1, nstate_d2=new_d2,
        ipool_d1=ip_d1.state, ipool_d2=ip_d2.state,
        w_d1=state.w_d1, w_d2=state.w_d2,
        g_nmda_d1=g_nmda_d1, g_nmda_d2=g_nmda_d2,
        rate_d1=rate_d1, rate_d2=rate_d2,
        e_d1=e_d1, e_d2=e_d2,
        x_pre=x_pre, x_post_d1=x_post_d1, x_post_d2=x_post_d2,
        spike_count_d1=sc_d1, spike_count_d2=sc_d2,
        efe_g=efe_g,
        epistemic_drive=eps,
        da_level=da,
        tonic_da=tonic,
    )
    return ActorOutput(state=new_s, spikes_d1=spk_d1, spikes_d2=spk_d2, net_evidence=net_ev)


# ---------------------- Action readout -----------------------------


def actor_select_action(
    state: ActorState, params: ActorParams, key: PRNGKey,
) -> Array:
    """Gold & Shadlen spike-count WTA with random tie-break.

    Returns scalar int32 in ``[0, motor_dim)``.
    """
    tm = params.total_motor
    ev_d1 = state.spike_count_d1[:tm].reshape(
        params.motor_dim, params.n_per_action).sum(axis=1)
    ev_d2 = state.spike_count_d2[:tm].reshape(
        params.motor_dim, params.n_per_action).sum(axis=1)
    net = ev_d1 - ev_d2
    max_ev = jnp.max(net)
    winners = (net == max_ev).astype(DTYPE)
    # Sample from winners uniformly (gives random tie-break)
    probs = winners / (winners.sum() + 1e-12)
    return jax.random.choice(key, params.motor_dim, p=probs).astype(jnp.int32)


def actor_reset_evidence(state: ActorState) -> ActorState:
    """Zero spike-count accumulators between decisions."""
    return eqx.tree_at(
        lambda s: (s.spike_count_d1, s.spike_count_d2),
        state,
        (jnp.zeros_like(state.spike_count_d1),
         jnp.zeros_like(state.spike_count_d2)),
    )


# ---------------------- Learning -----------------------------------


def _gate_eligibility_to_action(
    e: Array, action: Array, params: ActorParams,
) -> Array:
    """Multiply eligibility for non-chosen actions by ``down_state_factor``.

    Preserves full credit for the chosen action's MSN population;
    DOWN-state action channels keep only residual plasticity.
    """
    npa = params.n_per_action
    M = params.motor_dim
    tm = params.total_motor
    # Build (action_dim,) gating vector: 1.0 for chosen action block,
    # down_state_factor for other motor slots, 1.0 for internal slots
    idx = jnp.arange(tm)
    action_block = idx // npa
    motor_gate = jnp.where(
        action_block == action, 1.0, params.down_state_factor,
    ).astype(DTYPE)
    internal_gate = jnp.ones(params.action_dim - tm, DTYPE)
    gate = jnp.concatenate([motor_gate, internal_gate])
    return e * gate[None, :]


def actor_update(
    state: ActorState, params: ActorParams,
    td_error: float | Array, action: int | Array,
    *, receptor_lr: float | Array = 1.0,
) -> ActorState:
    """Collins & Frank asymmetric DA-modulated STDP (Shen 2008).

    TD > 0  (DA burst):  D1 LTP
    TD < 0  (DA dip):    D2 LTP + D1 LTD
    Non-chosen action eligibilities are down-weighted before applying.
    """
    td = jnp.asarray(td_error, DTYPE)
    rlr = jnp.asarray(receptor_lr, DTYPE)
    act = jnp.asarray(action, jnp.int32)
    base_lr = params.actor_lr * rlr

    da = jnp.clip(state.da_level, 0.0, 1.0)
    d1_r = _hill(da, params.d1_ec50, params.d1_hill_n)
    d2_r = _hill(da, params.d2_ec50, params.d2_hill_n)

    # Action-gate eligibility
    e_d1_g = _gate_eligibility_to_action(state.e_d1, act, params)
    e_d2_g = _gate_eligibility_to_action(state.e_d2, act, params)

    # D1 LTP for TD>0, LTD (· ltd_ratio) for TD<0 — signed TD handles
    # direction automatically; JIT-safe via jnp.where masks.
    td_pos = jnp.maximum(td, 0.0)
    td_neg = jnp.maximum(-td, 0.0)

    lr_d1_ltp = base_lr * (1.0 + params.d1_density * d1_r)
    lr_d1_ltd = base_lr * params.ltd_ratio * (1.0 - params.d1_density * d1_r)
    lr_d2_ltp = base_lr * (1.0 + params.d2_density * (1.0 - d2_r))

    dw_d1 = lr_d1_ltp * td_pos * e_d1_g - lr_d1_ltd * td_neg * e_d1_g
    dw_d2 = lr_d2_ltp * td_neg * e_d2_g

    # Dale's law: excitatory synapses >= 0
    w_d1 = jnp.maximum(state.w_d1 + dw_d1, 0.0)
    w_d2 = jnp.maximum(state.w_d2 + dw_d2, 0.0)
    return eqx.tree_at(
        lambda s: (s.w_d1, s.w_d2), state, (w_d1, w_d2),
    )


def actor_reset_transient(
    state: ActorState, params: ActorParams,
) -> ActorState:
    """Reset membrane + eligibility + traces; preserve ``w_d1``, ``w_d2``."""
    ad = params.action_dim
    z = lambda shape: jnp.zeros(shape, DTYPE)
    return ActorState(
        nstate_d1=init_neuron_state(ad, v_rest=float(params.ncfg.v_rest)),
        nstate_d2=init_neuron_state(ad, v_rest=float(params.ncfg.v_rest)),
        ipool_d1=ipool_reset_transient(state.ipool_d1, params.ipool_d1),
        ipool_d2=ipool_reset_transient(state.ipool_d2, params.ipool_d2),
        w_d1=state.w_d1, w_d2=state.w_d2,
        g_nmda_d1=z(ad), g_nmda_d2=z(ad),
        rate_d1=z(ad), rate_d2=z(ad),
        e_d1=z((params.state_size, ad)),
        e_d2=z((params.state_size, ad)),
        x_pre=z(params.state_size),
        x_post_d1=z(ad), x_post_d2=z(ad),
        spike_count_d1=z(ad), spike_count_d2=z(ad),
        efe_g=z(params.total_motor),
        epistemic_drive=jnp.asarray(0.0, DTYPE),
        da_level=jnp.asarray(0.5, DTYPE),
        tonic_da=jnp.asarray(float(params.baseline_da), DTYPE),
    )


def action_entropy(state: ActorState, params: ActorParams) -> Array:
    """Decision uncertainty in [0, 1] from winner – runner-up margin."""
    tm = params.total_motor
    ev_d1 = state.spike_count_d1[:tm].reshape(
        params.motor_dim, params.n_per_action).sum(axis=1)
    ev_d2 = state.spike_count_d2[:tm].reshape(
        params.motor_dim, params.n_per_action).sum(axis=1)
    net = ev_d1 - ev_d2
    sorted_ev = jnp.sort(net)[::-1]
    margin = sorted_ev[0] - sorted_ev[1]
    ev_range = sorted_ev[0] - sorted_ev[-1]
    uncertainty = jnp.where(
        ev_range > 1e-8, 1.0 - margin / (ev_range + 1e-8), 1.0,
    )
    return jnp.clip(uncertainty, 0.0, 1.0)
