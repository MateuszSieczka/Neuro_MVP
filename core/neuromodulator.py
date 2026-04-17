"""Four-channel neuromodulatory system — pure JAX.

Doya (2002); Grace (1991); Niv et al. (2007); Tobler et al. (2005);
Berridge & Waterhouse (2003); Behrens et al. (2007).

Channels (all in [0, 1]):
- **Phasic DA**  — TD-error driven with Weber-Fechner adaptive gain
  (``da_gain = baseline_da / max(da_rms, baseline_da)``).
- **Tonic DA**   — minute-scale leaky integrator of σ(reward); tracks
  average reward rate, not |RPE| (Niv 2007).
- **ACh**        — novelty / uncertainty.
- **NE**         — global surprise (prediction-error magnitude).
- **5-HT**       — convex combo of world-stability (1 − mean|PE|) and
  behavioural-stability (TD-stability × reward-quality) with dorsal-raphe
  anatomical weights.

Differences from legacy:
- ``error`` / ``|TD|`` history deques (window 100) replaced by EMAs with
  τ = 100 steps — same steady-state mean, JIT-safe, no Python state.
- Consolidation gate: legacy hard-gates the ACC-stagnation attenuation to
  ``0.3 < tonic_da < 0.7`` (Frank 2005 crossover). Replaced with a smooth
  bell ``exp(−((tDA−0.5)/0.2)²)`` centred on the same crossover — same
  intent, differentiable.
- Per-region NE/ACh dropped here; will re-emerge in the brain-graph layer
  where region counts are known statically.
- The 5-HT←tonic-DA coupling via HT1A-like Hill kinetics is kept as a
  **phenomenological monotone** map, not a pharmacology claim (HT1A is an
  autoreceptor; this is only a bounded saturating function of tDA).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array, BackendContext


# =====================================================================
# Params / state
# =====================================================================


class NeuromodulatorParams(eqx.Module):
    """Decay constants (already τ→decay converted) + baselines + weights."""

    # Phasic decays
    da_decay: Array
    ach_decay: Array
    ne_decay: Array
    sero_decay: Array
    tonic_da_decay: Array

    # Baselines
    baseline_da: Array
    baseline_ach: Array
    baseline_ne: Array
    baseline_sero: Array
    baseline_tonic_da: Array

    # DA RMS adaptive gain (Tobler 2005)
    da_rms_decay: Array

    # 5-HT dorsal raphe weights
    sero_world_weight: Array
    sero_behavioral_weight: Array

    # ACC stagnation integrator (Behrens 2007)
    acc_pe_decay: Array

    # PE / TD history EMAs (replace 100-step deques)
    error_ema_decay: Array
    td_ema_decay: Array

    # Reward → 5-HT phenomenological Hill (monotone in tonic DA)
    sero_reward_ec50: Array
    sero_reward_hill_n: Array


def init_neuromodulator_params(
    ctx: BackendContext,
    *,
    tau_da: float = 200.0,
    tau_ach: float = 25.0,
    tau_ne: float = 75.0,
    tau_sero: float = 150.0,
    tau_tonic_da: float = 60_000.0,
    tau_acc: float = 30_000.0,
    baseline_da: float = 0.5,
    baseline_ach: float = 0.5,
    baseline_ne: float = 0.3,
    baseline_sero: float = 0.6,
    baseline_tonic_da: float = 0.0,
    da_rms_decay: float = 0.9999,
    sero_world_weight: float = 0.7,
    sero_behavioral_weight: float = 0.3,
    history_window_steps: float = 100.0,
    sero_reward_ec50: float = 0.4,
    sero_reward_hill_n: float = 1.0,
    dtype=DTYPE,
) -> NeuromodulatorParams:
    """All τ in ms, converted via ``ctx.decay``. ``history_window_steps`` is
    the equivalent-mean window for PE / TD EMAs.
    """
    f = lambda x: jnp.asarray(x, dtype)
    # EMA decay so that expected window ~ history_window_steps ⇒ decay = 1 - 1/N.
    ema = 1.0 - 1.0 / float(history_window_steps)
    return NeuromodulatorParams(
        da_decay=f(ctx.decay(tau_da)),
        ach_decay=f(ctx.decay(tau_ach)),
        ne_decay=f(ctx.decay(tau_ne)),
        sero_decay=f(ctx.decay(tau_sero)),
        tonic_da_decay=f(ctx.decay(tau_tonic_da)),
        baseline_da=f(baseline_da),
        baseline_ach=f(baseline_ach),
        baseline_ne=f(baseline_ne),
        baseline_sero=f(baseline_sero),
        baseline_tonic_da=f(baseline_tonic_da),
        da_rms_decay=f(da_rms_decay),
        sero_world_weight=f(sero_world_weight),
        sero_behavioral_weight=f(sero_behavioral_weight),
        acc_pe_decay=f(ctx.decay(tau_acc)),
        error_ema_decay=f(ema),
        td_ema_decay=f(ema),
        sero_reward_ec50=f(sero_reward_ec50),
        sero_reward_hill_n=f(sero_reward_hill_n),
    )


class NeuromodulatorState(eqx.Module):
    """Scalar leaves — all in [0, 1] except ``da_rms`` and ``td_abs_ema``
    which live on a larger signed/positive scale."""

    dopamine: Array
    tonic_da: Array
    acetylcholine: Array
    noradrenaline: Array
    serotonin: Array
    da_rms: Array
    error_ema: Array
    td_abs_ema: Array
    acc_pe_trace: Array


def init_neuromodulator_state(
    params: NeuromodulatorParams, *, dtype=DTYPE
) -> NeuromodulatorState:
    """Start at baselines with zeroed histories."""
    z = jnp.asarray(0.0, dtype)
    return NeuromodulatorState(
        dopamine=params.baseline_da.astype(dtype),
        tonic_da=params.baseline_tonic_da.astype(dtype),
        acetylcholine=params.baseline_ach.astype(dtype),
        noradrenaline=params.baseline_ne.astype(dtype),
        serotonin=params.baseline_sero.astype(dtype),
        da_rms=params.baseline_da.astype(dtype),
        error_ema=z,
        td_abs_ema=z,
        acc_pe_trace=z,
    )


# =====================================================================
# Step
# =====================================================================


def _ema(prev: Array, x: Array, decay: Array) -> Array:
    return prev * decay + x * (1.0 - decay)


def neuromodulator_step(
    state: NeuromodulatorState,
    params: NeuromodulatorParams,
    prediction_error: Array,
    td_error: float | Array = 0.0,
    reward: float | Array = 0.0,
    novelty: float | Array | None = None,
) -> NeuromodulatorState:
    """Advance the 4-channel system by one sim step.

    Parameters
    ----------
    prediction_error:
        ``(n,)`` or scalar — magnitude folded via ``clip(mean|PE|, 0, 1)``.
    td_error:
        Scalar TD error (signed). Drives phasic DA through Weber-Fechner
        adaptive coding.
    reward:
        Raw scalar reward in natural units; passed through σ(r) before
        integration into tonic DA.
    novelty:
        Optional scalar novelty signal for ACh. If ``None``, the PE
        magnitude is reused (legacy behaviour).
    """
    pe = prediction_error.astype(DTYPE)
    error_mag = jnp.clip(jnp.mean(jnp.abs(pe)), 0.0, 1.0)
    td = jnp.asarray(td_error, DTYPE)
    r = jnp.asarray(reward, DTYPE)

    # --- Phasic DA (Tobler 2005 adaptive coding) ----------------------
    da_rms2 = params.da_rms_decay * state.da_rms ** 2 + (
        1.0 - params.da_rms_decay
    ) * td ** 2
    da_rms = jnp.sqrt(da_rms2)
    da_gain = params.baseline_da / jnp.maximum(da_rms, params.baseline_da)
    rpe_signal = jnp.clip(params.baseline_da + da_gain * td, 0.0, 1.0)
    dopamine = _ema(state.dopamine, rpe_signal, params.da_decay)

    # --- ACh (novelty) ------------------------------------------------
    if novelty is None:
        nov_val = error_mag
    else:
        nov_val = jnp.clip(jnp.asarray(novelty, DTYPE), 0.0, 1.0)
    ach = _ema(state.acetylcholine, nov_val, params.ach_decay)

    # --- NE (global surprise) ----------------------------------------
    ne = _ema(state.noradrenaline, error_mag, params.ne_decay)

    # --- 5-HT (stability) --------------------------------------------
    error_ema = _ema(state.error_ema, error_mag, params.error_ema_decay)
    td_abs_ema = _ema(
        state.td_abs_ema, jnp.clip(jnp.abs(td), 0.0, 10.0), params.td_ema_decay
    )
    world_stability = jnp.clip(1.0 - error_ema, 0.0, 1.0)
    td_stability = 1.0 / (1.0 + td_abs_ema)
    # Phenomenological monotone f(tDA) ∈ [0, 1]. HT1A-like Hill shape only.
    tda_clip = jnp.clip(state.tonic_da, 0.0, 1.0)
    c_n = tda_clip ** params.sero_reward_hill_n
    ec_n = params.sero_reward_ec50 ** params.sero_reward_hill_n
    reward_quality = c_n / (c_n + ec_n + jnp.asarray(1e-10, DTYPE))
    behavioral_stability = td_stability * reward_quality
    stability = (
        params.sero_world_weight * world_stability
        + params.sero_behavioral_weight * behavioral_stability
    )
    sero = _ema(state.serotonin, stability, params.sero_decay)

    # --- Tonic DA (minute-scale average reward rate) ------------------
    r_clip = jnp.clip(r, -20.0, 20.0)
    reward_signal = 1.0 / (1.0 + jnp.exp(-r_clip))  # σ(reward)
    tonic_da = _ema(state.tonic_da, reward_signal, params.tonic_da_decay)

    # --- ACC PE trace (Behrens 2007) ---------------------------------
    acc = _ema(state.acc_pe_trace, error_mag, params.acc_pe_decay)

    # Clamp all [0,1] channels (numerical safety).
    clamp = lambda x: jnp.clip(x, 0.0, 1.0)
    return NeuromodulatorState(
        dopamine=clamp(dopamine),
        tonic_da=clamp(tonic_da),
        acetylcholine=clamp(ach),
        noradrenaline=clamp(ne),
        serotonin=clamp(sero),
        da_rms=da_rms,
        error_ema=error_ema,
        td_abs_ema=td_abs_ema,
        acc_pe_trace=clamp(acc),
    )


# =====================================================================
# Readouts
# =====================================================================


def learning_rate_modulation(state: NeuromodulatorState) -> Array:
    """Phasic DA → STDP learning-rate modulator."""
    return state.dopamine


def consolidation_gate(state: NeuromodulatorState) -> Array:
    """``tonic_da · 5-HT · (1 − bell(tDA) · acc_pe_trace)``.

    Smooth replacement for the legacy hard band-gate: the ACC-stagnation
    attenuation is strongest when tonic DA is near the D1/D2 crossover
    (≈0.5) and fades at the extremes, matching the Frank (2005)
    exploration/exploitation regimes without a discontinuity.
    """
    raw = state.tonic_da * state.serotonin
    bell = jnp.exp(-((state.tonic_da - 0.5) / 0.2) ** 2)
    attenuation = 1.0 - bell * state.acc_pe_trace
    return raw * attenuation


def bottom_up_gain(state: NeuromodulatorState) -> Array:
    """ACh → bottom-up / top-down balance (Hasselmo 2006)."""
    return state.acetylcholine


def competition_sharpness(state: NeuromodulatorState) -> Array:
    """NE → k-WTA / divisive-norm sharpness."""
    return state.noradrenaline


def planning_horizon(state: NeuromodulatorState) -> Array:
    """5-HT → temporal discount / planning depth (Doya 2002)."""
    return state.serotonin


def transmitter_vector(state: NeuromodulatorState) -> Array:
    """Pack levels into the ``(4,)`` ``(da, ach, ne, sero)`` vector consumed
    by :mod:`core.receptor`.
    """
    return jnp.stack(
        [state.dopamine, state.acetylcholine, state.noradrenaline, state.serotonin]
    )
