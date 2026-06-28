"""Thalamus — relay nuclei + TRN (pure JAX).

References
----------
  Sherman & Guillery (2006) — Exploring the thalamus (ed. 2)
  Sherman (2016)            — Thalamus plays a central role in ongoing cortical functioning
  McCormick & Bal (1997)    — Sleep and arousal: thalamocortical mechanisms
  Crick (1984)              — Function of the thalamic reticular complex: searchlight
  Halassa & Kastner (2017)  — Thalamic functions in distributed cognitive control
  Pinault (2004)            — TRN: anatomical, functional and clinical overview
  Zhou, Grimm & Murray (2016)  — TC burst/tonic firing
  Destexhe & Sejnowski (2003) — Interactions between membrane conductances underlying thalamocortical slow-wave oscillations

Role (critical analysis)
------------------------
First-order relay nuclei gate modality-specific information flow into
cortex L4. Every relay nucleus (LGN, MGN, VPL, VPM, VA, VL, ...) shares
the SAME local circuit:
  sensory/motor afferent ──▶ TC cell ──▶ cortex L4
                    CT_L6 ──▶ TC cell      (excitatory modulation, gain)
                       TRN ──▶ TC cell      (inhibitory gating)
                   TC coll ──▶ TRN cell     (feedback collateral)
                    CT_L6 ──▶ TRN cell      (feedforward drive)

The difference between "LGN" and "VPM" is only what the afferent carries
— the circuit is identical. Therefore this module exposes a SINGLE
``RelayNucleus`` primitive; ``brain_graph`` instantiates named copies.

Burst / tonic switch
--------------------
TC cells (IB-type AdEx) show two firing modes driven by membrane
polarisation:
  • Tonic (depolarised, awake): faithful relay, high info transfer.
  • Burst (hyperpolarised): low-pass filtered bursts, alerting / salience.
Switch is controlled by a bias current that scales with ACh/NE
(McCormick & Bal 1997). A high-fidelity model needs T-type Ca²⁺; here
AdEx IB (a=4, b=150, tau_w=30) approximates burst behaviour when the
bias drops the membrane below V_rest, with acceptable accuracy for
system-level simulation.

TRN (thalamic reticular nucleus)
--------------------------------
Specialised GABAergic shell. We implement it as a *lean* FS pool similar
to ``IPoolState`` but with two excitatory inputs: TC collaterals AND
cortical L6 corticothalamic drive. Output is GABA current per relay
nucleus, with optional cross-nucleus mask (Crick searchlight).

Output to cortex
----------------
``relay_spikes`` ∈ {0,1}^{n_tc} is fed directly to L4 of the target
cortical area by ``brain_graph``. The caller projects via a learned
matrix (CT feedback sets the gain; the thalamus does NOT own that matrix).
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
# Relay nucleus (TC cells) — IB-type AdEx
# =====================================================================


class RelayParams(eqx.Module):
    """Static parameters for a single first-order relay nucleus."""

    tc: NeuronParams
    ampa_decay: Array          # AMPA conductance decay on TC cells
    gaba_decay: Array          # GABA conductance decay (TRN → TC)
    e_exc: Array               # 0 mV
    e_inh: Array               # -80 mV (Cl⁻, more negative than cortex IPool)

    # Burst/tonic mode: bias current = bias_gain·(ach + ne) − bias_base
    bias_base: Array           # pA offset — at (ach=0, ne=0) TC is hyperpol.
    bias_gain: Array           # pA per unit modulator

    # Rate trace decay
    rate_decay: Array

    n_afferent: int = eqx.field(static=True)
    n_ct: int = eqx.field(static=True)
    n_tc: int = eqx.field(static=True)


class RelayState(eqx.Module):
    tc: NeuronState
    w_af: Array          # (n_afferent, n_tc) afferent → TC (AMPA, learned static)
    w_ct: Array          # (n_ct, n_tc) cortex L6 → TC (AMPA, modulatory)
    g_ampa: Array        # (n_tc,) AMPA conductance
    g_gaba: Array        # (n_tc,) GABA from TRN
    rate_tc: Array       # (n_tc,) spike-rate EMA


def _relay_cond_scale(tc: NeuronParams, fan_in: int, e_exc: float,
                      expected_rate: float, raw_w_mean: float = 0.5,
                      headroom: float = 1.6) -> float:
    """Boost so ``expected_rate`` input drives TC cells to rheobase.

    Converts dimensionless raw weights (half-normal, mean ``raw_w_mean``)
    into nS so that
      active · (raw_w_mean · cond_scale) · (e_exc - v_rest) ≈ headroom · I_rheo.
    """
    gap = float(abs(tc.v_thresh - tc.v_rest))
    i_rheo = float(tc.g_L) * (gap - float(tc.delta_t))
    df = float(e_exc - tc.v_rest)
    expected = max(1.0, fan_in * expected_rate)
    denom = expected * raw_w_mean * max(df, 1.0)
    return headroom * i_rheo / max(denom, 1e-6)


def init_relay_params(
    ctx: BackendContext,
    n_afferent: int,
    *,
    n_tc: int = 64,
    n_ct: int | None = None,
    e_exc: float = 0.0,
    e_inh: float = -80.0,
    tau_ampa: float = 2.0,
    tau_gaba: float = 10.0,
    tau_rate: float = 30.0,
    bias_base: float = 30.0,
    bias_gain: float = 120.0,
) -> RelayParams:
    """Build relay params. TC cells use IB-type AdEx (Brette & Gerstner)."""
    n_ct_val = int(n_tc if n_ct is None else n_ct)
    tc = init_neuron_params(
        ctx,
        v_rest=-70.0, v_thresh=-55.0, v_reset=-75.0, v_spike_cutoff=-30.0,
        delta_t=2.0, C_m=281.0, g_L=30.0,
        tau_w=30.0, a=4.0, b=150.0,    # IB — bursting when hyperpolarised
        refrac_period_ms=2.0,
    )
    f = lambda x: jnp.asarray(x, DTYPE)
    return RelayParams(
        tc=tc,
        ampa_decay=f(ctx.decay(tau_ampa)),
        gaba_decay=f(ctx.decay(tau_gaba)),
        e_exc=f(e_exc), e_inh=f(e_inh),
        bias_base=f(bias_base), bias_gain=f(bias_gain),
        rate_decay=f(ctx.decay(tau_rate)),
        n_afferent=n_afferent, n_ct=n_ct_val, n_tc=n_tc,
    )


def init_relay_state(
    key: PRNGKey, params: RelayParams,
    *,
    w_af_mean: float = 0.5, w_af_cv: float = 0.3,
    w_ct_mean: float = 0.2, w_ct_cv: float = 0.3,
    expected_af_rate: float = 0.1,
    dtype=DTYPE,
) -> RelayState:
    """Half-normal afferent + CT weights, rheobase-scaled."""
    k_af, k_ct = split_key(key, 2)
    cond_scale = _relay_cond_scale(
        params.tc, params.n_afferent, float(params.e_exc), expected_af_rate,
        raw_w_mean=w_af_mean,
    )
    w_af = jnp.abs(
        jax.random.normal(k_af, (params.n_afferent, params.n_tc), dtype=dtype)
        * (w_af_mean * w_af_cv) + w_af_mean
    ) * jnp.asarray(cond_scale, dtype)
    # CT feedback is weaker (modulatory, Sherman & Guillery 2006)
    w_ct = jnp.abs(
        jax.random.normal(k_ct, (params.n_ct, params.n_tc), dtype=dtype)
        * (w_ct_mean * w_ct_cv) + w_ct_mean
    ) * jnp.asarray(cond_scale * 0.3, dtype)

    tc_state = init_neuron_state(params.n_tc, v_rest=float(params.tc.v_rest))
    z = lambda: jnp.zeros(params.n_tc, dtype)
    return RelayState(
        tc=tc_state, w_af=w_af, w_ct=w_ct,
        g_ampa=z(), g_gaba=z(), rate_tc=z(),
    )


# =====================================================================
# TRN — GABAergic shell
# =====================================================================


class TRNParams(eqx.Module):
    """Thalamic reticular nucleus parameters."""

    trn: NeuronParams
    ampa_decay: Array
    e_exc: Array
    e_inh: Array
    rate_decay: Array

    n_tc_total: int = eqx.field(static=True)      # all relay nuclei pooled
    n_ct: int = eqx.field(static=True)
    n_trn: int = eqx.field(static=True)


class TRNState(eqx.Module):
    trn: NeuronState
    w_tc_trn: Array     # (n_tc_total, n_trn) TC collaterals → TRN (AMPA)
    w_ct_trn: Array     # (n_ct, n_trn) L6 CT → TRN (AMPA)
    w_trn_tc: Array     # (n_trn, n_tc_total) TRN → TC (GABA, learned or fixed)
    g_ampa: Array       # (n_trn,) AMPA on TRN
    rate_trn: Array     # (n_trn,) spike-rate EMA


def init_trn_params(
    ctx: BackendContext, n_tc_total: int, n_ct: int,
    *, n_trn: int = 64,
    e_exc: float = 0.0, e_inh: float = -80.0,
    tau_ampa: float = 3.0, tau_rate: float = 30.0,
) -> TRNParams:
    """TRN neurons: fast-spiking-ish (a=0, b=20) — bursty when hyperpol."""
    trn = init_neuron_params(
        ctx,
        v_rest=-70.0, v_thresh=-55.0, v_reset=-75.0, v_spike_cutoff=-30.0,
        delta_t=2.0, C_m=281.0, g_L=50.0,   # a bit leakier than pyramidal
        tau_w=40.0, a=2.0, b=20.0, refrac_period_ms=2.0,
    )
    f = lambda x: jnp.asarray(x, DTYPE)
    return TRNParams(
        trn=trn,
        ampa_decay=f(ctx.decay(tau_ampa)),
        e_exc=f(e_exc), e_inh=f(e_inh),
        rate_decay=f(ctx.decay(tau_rate)),
        n_tc_total=n_tc_total, n_ct=n_ct, n_trn=n_trn,
    )


def init_trn_state(
    key: PRNGKey, params: TRNParams,
    *,
    w_tc_trn_mean: float = 0.3, w_ct_trn_mean: float = 0.4,
    w_trn_tc_mean: float = 0.15,
    expected_tc_rate: float = 0.05,
    dtype=DTYPE,
) -> TRNState:
    """Weights scaled so TC collaterals + L6 drive jointly reach rheobase."""
    k_tt, k_ct, k_back = split_key(key, 3)
    # Scale AMPA drive to TRN via rheobase of TRN cells (mean of both
    # pathways — rough but fine for a single-scalar calibration)
    avg_raw = 0.5 * (w_tc_trn_mean + w_ct_trn_mean)
    cond_scale = _relay_cond_scale(
        params.trn, params.n_tc_total + params.n_ct, float(params.e_exc),
        expected_tc_rate, raw_w_mean=avg_raw, headroom=1.2,
    )
    w_tc_trn = jnp.abs(
        jax.random.normal(k_tt, (params.n_tc_total, params.n_trn), dtype=dtype)
        * 0.3 + w_tc_trn_mean
    ) * jnp.asarray(cond_scale, dtype)
    w_ct_trn = jnp.abs(
        jax.random.normal(k_ct, (params.n_ct, params.n_trn), dtype=dtype)
        * 0.3 + w_ct_trn_mean
    ) * jnp.asarray(cond_scale, dtype)
    # TRN → TC is GABA — we'll use driving-force conversion in step,
    # so these are conductances (nS). No extra scale.
    w_trn_tc = jnp.abs(
        jax.random.normal(k_back, (params.n_trn, params.n_tc_total), dtype=dtype)
        * 0.2 + w_trn_tc_mean
    )
    trn_state = init_neuron_state(params.n_trn, v_rest=float(params.trn.v_rest))
    z = lambda: jnp.zeros(params.n_trn, dtype)
    return TRNState(
        trn=trn_state,
        w_tc_trn=w_tc_trn, w_ct_trn=w_ct_trn, w_trn_tc=w_trn_tc,
        g_ampa=z(), rate_trn=z(),
    )


# =====================================================================
# Combined step — one TRN serving a SINGLE relay nucleus (the common case)
# =====================================================================


class ThalamicOutput(NamedTuple):
    relay: RelayState
    trn: TRNState
    relay_spikes: Array        # (n_tc,) → cortex L4
    trn_spikes: Array          # (n_trn,) diagnostics
    relay_rate: Array          # (n_tc,) EMA
    trn_rate: Array            # (n_trn,) EMA


@eqx.filter_jit
def thalamic_step(
    relay: RelayState, relay_params: RelayParams,
    trn: TRNState, trn_params: TRNParams,
    ctx: BackendContext,
    afferent: Array,                      # (n_afferent,) spikes or rates
    ct_drive: Array,                      # (n_ct,) spikes from cortex L6
    *,
    ach: Array | float = 0.5,
    ne: Array | float = 0.5,
    afferent_gain: Array | float = 1.0,   # (n_tc,) or scalar — attention
) -> ThalamicOutput:
    """One dt of the full relay+TRN circuit (single nucleus).

    ACh + NE depolarise TC cells via the bias current, switching them
    from burst to tonic mode (McCormick & Bal 1997). The caller owns
    the neuromodulator scheduler — no hidden state here.

    ``afferent_gain`` is a per-TC multiplicative factor on the afferent
    AMPA conductance. This models pulvinar / top-down attention
    modulation of LGN/first-order relay responses (Saalmann 2012;
    Reynolds & Heeger 2009). Default 1.0 = no modulation.
    """
    af = afferent.astype(DTYPE)
    ct = ct_drive.astype(DTYPE)
    ach_s = jnp.asarray(ach, DTYPE)
    ne_s = jnp.asarray(ne, DTYPE)

    # ================ TRN layer ================
    # TRN receives TC collaterals (previous step's relay spikes) + CT
    tc_coll = relay.tc.spikes       # (n_tc,) already-emitted TC spikes
    g_ampa_trn = (
        trn.g_ampa * trn_params.ampa_decay
        + tc_coll @ trn.w_tc_trn
        + ct @ trn.w_ct_trn
    )
    i_trn_exc = g_ampa_trn * (trn_params.e_exc - trn.trn.v)

    new_trn_state, trn_spikes = neuron_step(
        trn.trn, trn_params.trn, ctx, i_syn=i_trn_exc, g_syn=g_ampa_trn,
    )

    # ================ Relay TC layer ================
    # TRN → TC GABAergic inhibition
    g_gaba = (
        relay.g_gaba * relay_params.gaba_decay
        + trn_spikes @ trn.w_trn_tc
    )
    i_tc_inh = g_gaba * (relay_params.e_inh - relay.tc.v)

    # Afferent + CT excitation (AMPA).  ``afferent_gain`` scales the
    # sensory afferent pathway (attention); CT remains ungated so the
    # cortical loop can override a silenced channel.
    af_gain = jnp.asarray(afferent_gain, DTYPE)
    g_ampa_tc = (
        relay.g_ampa * relay_params.ampa_decay
        + af_gain * (af @ relay.w_af)
        + ct @ relay.w_ct
    )
    i_tc_exc = g_ampa_tc * (relay_params.e_exc - relay.tc.v)

    # Bias current — burst/tonic switch
    # At (ach=ne=0) → i_bias = -bias_base (hyperpolarising → bursts)
    # At (ach=ne=1) → i_bias = 2·bias_gain - bias_base (depolarising → tonic)
    i_bias = relay_params.bias_gain * (ach_s + ne_s) - relay_params.bias_base

    i_tc = i_tc_exc + i_tc_inh + i_bias
    g_tc = g_ampa_tc + g_gaba

    new_tc_state, tc_spikes = neuron_step(
        relay.tc, relay_params.tc, ctx, i_syn=i_tc, g_syn=g_tc,
    )

    # ================ Rate EMAs ================
    rate_tc = relay.rate_tc * relay_params.rate_decay + (1.0 - relay_params.rate_decay) * tc_spikes
    rate_trn = trn.rate_trn * trn_params.rate_decay + (1.0 - trn_params.rate_decay) * trn_spikes

    new_relay = RelayState(
        tc=new_tc_state, w_af=relay.w_af, w_ct=relay.w_ct,
        g_ampa=g_ampa_tc, g_gaba=g_gaba, rate_tc=rate_tc,
    )
    new_trn = TRNState(
        trn=new_trn_state,
        w_tc_trn=trn.w_tc_trn, w_ct_trn=trn.w_ct_trn, w_trn_tc=trn.w_trn_tc,
        g_ampa=g_ampa_trn, rate_trn=rate_trn,
    )
    return ThalamicOutput(
        relay=new_relay, trn=new_trn,
        relay_spikes=tc_spikes, trn_spikes=trn_spikes,
        relay_rate=rate_tc, trn_rate=rate_trn,
    )


# =====================================================================
# Reset helpers
# =====================================================================


def relay_reset_transient(state: RelayState, params: RelayParams) -> RelayState:
    """Clear dynamical state; preserve ``w_af`` / ``w_ct``."""
    return RelayState(
        tc=init_neuron_state(params.n_tc, v_rest=float(params.tc.v_rest)),
        w_af=state.w_af, w_ct=state.w_ct,
        g_ampa=jnp.zeros(params.n_tc, DTYPE),
        g_gaba=jnp.zeros(params.n_tc, DTYPE),
        rate_tc=jnp.zeros(params.n_tc, DTYPE),
    )


def trn_reset_transient(state: TRNState, params: TRNParams) -> TRNState:
    """Clear dynamical state; preserve weights."""
    return TRNState(
        trn=init_neuron_state(params.n_trn, v_rest=float(params.trn.v_rest)),
        w_tc_trn=state.w_tc_trn, w_ct_trn=state.w_ct_trn,
        w_trn_tc=state.w_trn_tc,
        g_ampa=jnp.zeros(params.n_trn, DTYPE),
        rate_trn=jnp.zeros(params.n_trn, DTYPE),
    )
