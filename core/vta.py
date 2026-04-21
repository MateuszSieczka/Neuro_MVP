"""VTA dopaminergic circuit — pure JAX.

Eshel et al. (2015); Schultz (1997, 1998); Tobler et al. (2005);
Grace (1991); Schweighofer et al. (2008); Watabe-Uchida et al. (2017);
Bayer & Glimcher (2005); Frémaux & Gerstner (2016).

Circuit:
- **VP**    (inhibitory) carries ``V̂(s)`` captured at decision time.
- **PPTg**  (excitatory) carries ``γ_eff · V̂(s')`` — the PPTg synaptic
  τ implements temporal discount; 5-HT widens τ (Schweighofer 2008).
- **Reward** (excitatory) carries ``reward_gain · (r − b)`` where ``b`` is
  the long-term reward baseline (slow EMA, see below).
- **D2 autoreceptor** normalises RPE by a slow RMS trace (Tobler 2005).

The E/I balance ``RPE_raw = I_reward + I_ppTg − I_vp`` is algebraically
identical to classic baseline-subtracted TD ``(r − b) + γ V(s') − V(s)``
and is divided by the D2 adaptive gain to produce the broadcast RPE.
A homeostatic weight-norm bound prevents V estimates from diverging.

**Reward baseline (b)** — a slow EMA of the raw extrinsic reward kept
outside the critic. Bayer & Glimcher (2005) report that VTA dopamine
neurons encode RPE *relative to* an expected-reward signal that
adapts on a much slower timescale than per-state V̂; orbitofrontal /
amygdala circuits maintain that expectation. Frémaux & Gerstner
(2016) prove R-STDP requires reward-baseline subtraction for
stationary-task stability — without it the per-state critic alone has
to absorb the absolute reward level, which couples value-error and
policy-error and produces tonic-DA collapse on stationary tasks. The
baseline is ``b_{t+1} = decay·b_t + (1−decay)·r_t``; centering uses
the pre-update ``b_t`` so the system is causal.

Legacy notes:
- ``is_terminal`` is represented as a float mask (0/1) so the step is
  fully JIT-safe with no Python branching.
- Eligibility is the raw critic activation (gradient-consistent semi-
  gradient TD, Sutton & Barto 2018 §9.4).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, BackendContext


# =====================================================================
# Params / state
# =====================================================================


class VTAParams(eqx.Module):
    """Circuit hyperparameters + derived decays."""

    tau_ppTg: Array            # ms
    reward_gain: Array
    value_lr: Array
    auto_decay: Array          # per-step EMA decay for RMS
    baseline_decay: Array      # per-step EMA decay for reward baseline
    min_gain: Array
    readout_decay: Array
    dt: Array                  # ms per simulation step (for γ_eff)
    hidden_size: int = eqx.field(static=True)


def init_vta_params(
    ctx: BackendContext,
    hidden_size: int,
    *,
    tau_ppTg: float = 2488.0,
    reward_gain: float = 1.0,
    value_lr: float = 0.1,
    tau_autoreceptor: float = 10_000.0,
    # tau_reward_baseline: long-run E[r] tracker. Must be (a) slower
    # than the critic so V̂ carries state-specific structure, and (b)
    # slower than auto_rms so the D2 normaliser still tracks the
    # short-term variance of *centered* RPE. 30 s is the OFC/amygdala
    # appetitive-baseline timescale (Bayer & Glimcher 2005;
    # Padoa-Schioppa & Cai 2011).
    tau_reward_baseline: float = 30_000.0,
    # Baseline DA firing rate (Grace 1991: tonic VTA DA neurons fire
    # 3-8 Hz at rest). The D2 autoreceptor RMS normaliser must not
    # divide by zero when actual rewards are absent; the biophysical
    # floor is the tonic DA variance, which for an independent-spike
    # process at rate f with gain g over dt is f · g · dt. At 3 Hz,
    # gain 1.0, dt = 1 ms this gives 3e-3 — the minimum reward-signal
    # amplitude the circuit *can* resolve. Previously an empirical
    # 0.01; now derived.
    baseline_da_rate_hz: float = 3.0,
    readout_decay: float = 2e-5,
    dtype=DTYPE,
) -> VTAParams:
    f = lambda x: jnp.asarray(x, dtype)
    # SNR floor: tonic DA rate × reward_gain × dt (dt in ms → /1000).
    min_gain = baseline_da_rate_hz * reward_gain * (float(ctx.dt) / 1000.0)
    return VTAParams(
        tau_ppTg=f(tau_ppTg),
        reward_gain=f(reward_gain),
        value_lr=f(value_lr),
        auto_decay=f(ctx.decay(tau_autoreceptor)),
        baseline_decay=f(ctx.decay(tau_reward_baseline)),
        min_gain=f(min_gain),
        readout_decay=f(readout_decay),
        dt=f(ctx.dt),
        hidden_size=hidden_size,
    )


class VTAState(eqx.Module):
    """Weights + eligibility + stored V(s) snapshot + adaptive gain."""

    w_value: Array             # (hidden_size,)
    stored_v: Array            # scalar
    e_value: Array             # (hidden_size,)
    auto_rms: Array            # scalar, warm-start at 1.0
    reward_baseline: Array     # scalar, slow EMA of raw reward
    last_rpe: Array            # scalar diagnostic
    last_v_s_prime: Array      # scalar diagnostic
    last_gamma_eff: Array      # scalar diagnostic


def init_vta_state(
    key: PRNGKey, params: VTAParams, *, dtype=DTYPE
) -> VTAState:
    """Uniform ``[−v_std, +v_std]`` init with ``v_std = 1/√h``."""
    h = params.hidden_size
    v_std = 1.0 / jnp.sqrt(jnp.asarray(h, dtype))
    w = jax.random.uniform(key, (h,), minval=-v_std, maxval=v_std, dtype=dtype)
    z = jnp.asarray(0.0, dtype)
    return VTAState(
        w_value=w,
        stored_v=z,
        e_value=jnp.zeros(h, dtype=dtype),
        auto_rms=jnp.asarray(1.0, dtype),
        reward_baseline=z,
        last_rpe=z,
        last_v_s_prime=z,
        last_gamma_eff=jnp.asarray(0.99, dtype),
    )


# =====================================================================
# Phases: store → compute_rpe → update
# =====================================================================


def vta_store_prediction(
    state: VTAState, critic_activation: Array
) -> VTAState:
    """Freeze V̂(s) and its eligibility φ(s) at decision time."""
    act = critic_activation.astype(DTYPE)
    v_s = jnp.dot(act, state.w_value)
    return eqx.tree_at(
        lambda s: (s.stored_v, s.e_value),
        state,
        (v_s, act),
    )


class VTAOutput(NamedTuple):
    """RPE + updated state + diagnostics."""

    state: VTAState
    rpe: Array
    v_s: Array
    v_s_prime: Array
    gamma_eff: Array


def vta_compute_rpe(
    state: VTAState,
    params: VTAParams,
    critic_activation: Array,
    reward: float | Array,
    is_terminal: float | Array,
    serotonin: float | Array,
    n_substeps: int | Array,
) -> VTAOutput:
    """Compute the E/I-balanced and D2-normalised RPE for this transition.

    ``is_terminal`` is a float mask in ``[0, 1]`` (typically 0 or 1); it
    gates the PPTg excitation by ``(1 − is_terminal)``.
    """
    act_next = critic_activation.astype(DTYPE)
    r = jnp.asarray(reward, DTYPE)
    term = jnp.asarray(is_terminal, DTYPE)
    sero = jnp.clip(jnp.asarray(serotonin, DTYPE), 0.0, 2.0)
    n_sub = jnp.asarray(n_substeps, DTYPE)

    # Serotonin-modulated γ_eff (Schweighofer 2008).
    tau_eff = params.tau_ppTg * (1.0 + sero)
    gamma_eff = jnp.exp(-n_sub * params.dt / tau_eff)

    v_s_prime = jnp.dot(act_next, state.w_value)
    I_vp = state.stored_v
    I_ppTg = (1.0 - term) * gamma_eff * v_s_prime
    # Baseline-subtracted reward signal (Bayer & Glimcher 2005;
    # Frémaux & Gerstner 2016). The pre-update baseline is used so
    # the centering is causal; the baseline is then advanced.
    r_centered = r - state.reward_baseline
    I_reward = params.reward_gain * r_centered
    reward_baseline = (
        params.baseline_decay * state.reward_baseline
        + (1.0 - params.baseline_decay) * r
    )

    rpe_raw = I_reward + I_ppTg - I_vp

    # D2 autoreceptor RMS adaptation (Tobler 2005).
    auto_rms2 = (
        params.auto_decay * state.auto_rms ** 2
        + (1.0 - params.auto_decay) * rpe_raw ** 2
    )
    auto_rms = jnp.sqrt(auto_rms2)
    auto_gain = jnp.maximum(auto_rms, params.min_gain)
    rpe = rpe_raw / auto_gain

    new_state = eqx.tree_at(
        lambda s: (
            s.auto_rms, s.reward_baseline,
            s.last_rpe, s.last_v_s_prime, s.last_gamma_eff,
        ),
        state,
        (auto_rms, reward_baseline, rpe, v_s_prime, gamma_eff),
    )
    return VTAOutput(
        state=new_state,
        rpe=rpe,
        v_s=state.stored_v,
        v_s_prime=v_s_prime,
        gamma_eff=gamma_eff,
    )


def vta_update(
    state: VTAState, params: VTAParams, rpe: float | Array
) -> VTAState:
    """Three-factor semi-gradient TD on ``w_value`` + soft decay + norm bound.

    ``Δw = lr · RPE · e_value`` followed by a multiplicative decay
    ``(1 − readout_decay)`` (Bhatt 2009 protein turnover) and a norm cap
    derived from ``V_max ≤ w_norm_max · max||φ||``.
    """
    r = jnp.asarray(rpe, DTYPE)
    dw = params.value_lr * r * state.e_value
    w = (state.w_value + dw) * (1.0 - params.readout_decay)

    # Homeostatic norm bound.
    max_return = jnp.maximum(
        jnp.asarray(10.0, DTYPE),
        1.0 / jnp.maximum(jnp.asarray(1e-4, DTYPE), 1.0 - state.last_gamma_eff),
    )
    max_feat_norm = jnp.sqrt(jnp.asarray(params.hidden_size, DTYPE) / 4.0)
    w_norm_max = max_return / jnp.maximum(max_feat_norm, jnp.asarray(1.0, DTYPE))
    w_norm = jnp.linalg.norm(w)
    scale = jnp.where(w_norm > w_norm_max, w_norm_max / w_norm, jnp.asarray(1.0, DTYPE))
    w = w * scale

    return eqx.tree_at(lambda s: s.w_value, state, w)


def vta_reset_transient(state: VTAState, *, dtype=DTYPE) -> VTAState:
    """Clear the ``stored_v`` / ``e_value`` snapshots between episodes.

    Preserves ``w_value``, ``auto_rms`` and ``reward_baseline`` (all
    three are persistent across episodes \u2014 the baseline tracks the
    long-run reward expectation of the body in the world, which is
    *not* episode-bound).
    """
    h = state.w_value.shape[0]
    z = jnp.asarray(0.0, dtype)
    return eqx.tree_at(
        lambda s: (s.stored_v, s.e_value, s.last_rpe, s.last_v_s_prime),
        state,
        (z, jnp.zeros(h, dtype=dtype), z, z),
    )
