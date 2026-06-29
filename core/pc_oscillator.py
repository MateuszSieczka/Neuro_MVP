"""Theta/gamma oscillator with PAC — pure functional timing (§4).

Oscillations are the substrate's *timing* signal (plan §4 "oscillations
stay"): they do not change what is computed, only *when*.  Their one job
here is to phase-gate the hippocampus into separate **encoding** and
**retrieval** windows — the theta gate that §3 deferred to §4 (see
:mod:`core.pc_hippocampus`).

* **Theta** (the slow rhythm): its phase splits each cycle into an
  encoding half (θ < π) and a retrieval half (θ ≥ π) — Hasselmo, Bodelón
  & Wyble (2002): cortical input dominates on one phase (store the world),
  recurrent CA3 feedback on the other (recall from memory).  Gating
  encode vs complete on opposite phases stops new input overwriting a
  memory mid-recall.
* **Gamma** (the fast rhythm), nested in theta by **phase-amplitude
  coupling** (Lisman & Jensen 2013): gamma resets at each theta trough
  and its amplitude envelope peaks there, the canonical paces-within-a-
  cycle structure.

Neuromodulation (Doya 2002): NE speeds theta (arousal, faster sampling),
5-HT slows it (patience).  In SWS the rhythm clamps to a slow oscillation
and gamma is suppressed (Steriade 1993).

All timing is in **cognitive-step units** — the substrate has no
millisecond clock, so frequency is *cycles per step* and the phase
advances by ``2π·freq`` each call.  Pure ``jnp.where`` masking, so the
step is JIT-safe and self-contained (no shared spiking state).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array


TWO_PI = 2.0 * jnp.pi


# =====================================================================
# Params / state
# =====================================================================


class OscillatorParams(eqx.Module):
    """Theta/gamma constants — all frequencies in cycles per cognitive step."""

    theta_freq: Array          # base theta frequency (cycles/step)
    theta_min: Array
    theta_max: Array
    gamma_freq: Array          # gamma frequency (cycles/step)
    pac_depth: Array           # 0..1 depth of the theta→gamma envelope
    ne_theta_shift: Array      # Δ theta freq per unit NE  (> 0 ⇒ faster)
    sero_theta_shift: Array    # Δ theta freq per unit 5-HT (< 0 ⇒ slower)
    sws_freq: Array            # slow-wave clamp frequency


def init_oscillator_params(
    *,
    theta_freq: float = 0.10,
    theta_min: float = 0.05,
    theta_max: float = 0.20,
    gamma_freq: float = 0.70,
    pac_depth: float = 0.6,
    ne_theta_shift: float = 0.05,
    sero_theta_shift: float = -0.025,
    sws_freq: float = 0.02,
    dtype=DTYPE,
) -> OscillatorParams:
    """Theta/gamma in cycles/step (e.g. ``theta_freq=0.1`` ⇒ ~10-step cycle).

    The defaults give gamma ≈ 7× theta (a handful of gamma cycles nested
    per theta, as in cortex); only the *ratios* and the NE/5-HT signs
    matter, not absolute values (the substrate has no wall clock).
    """
    if not (theta_min <= theta_freq <= theta_max):
        raise ValueError("require theta_min ≤ theta_freq ≤ theta_max")
    if not (0.0 <= pac_depth <= 1.0):
        raise ValueError("pac_depth must be in [0, 1]")
    f = lambda x: jnp.asarray(x, dtype)
    return OscillatorParams(
        theta_freq=f(theta_freq), theta_min=f(theta_min), theta_max=f(theta_max),
        gamma_freq=f(gamma_freq), pac_depth=f(pac_depth),
        ne_theta_shift=f(ne_theta_shift), sero_theta_shift=f(sero_theta_shift),
        sws_freq=f(sws_freq),
    )


class OscillatorState(eqx.Module):
    """Theta + gamma phase ∈ [0, 2π) and the current gamma amplitude."""

    theta_phase: Array
    gamma_phase: Array
    gamma_amplitude: Array


def init_oscillator_state(*, dtype=DTYPE) -> OscillatorState:
    z = jnp.asarray(0.0, dtype)
    return OscillatorState(theta_phase=z, gamma_phase=z, gamma_amplitude=z)


# =====================================================================
# Step
# =====================================================================


class OscillatorOutput(NamedTuple):
    """Per-step cycle flags + the HC phase gates."""

    theta_reset: Array         # bool — theta cycle wrapped this step
    gamma_reset: Array         # bool — gamma cycle completed this step
    encoding_phase: Array      # bool — θ < π  (store window, Hasselmo 2002)
    retrieval_phase: Array     # bool — θ ≥ π  (recall window)
    in_up_state: Array         # bool — SWS Up state (only meaningful in SWS)


def oscillator_step(
    state: OscillatorState, params: OscillatorParams,
    *,
    ne_level: Array | float = 0.0,
    sero_level: Array | float = 0.0,
    sws_mode: Array | bool = False,
) -> tuple[OscillatorState, OscillatorOutput]:
    """Advance theta + gamma one cognitive step; return state + phase gates.

    Applies NE/5-HT frequency modulation, PAC (gamma resets at the theta
    trough with a cosine amplitude envelope) and the optional SWS clamp.
    """
    ne = jnp.asarray(ne_level, DTYPE)
    sero = jnp.asarray(sero_level, DTYPE)
    sws = jnp.asarray(sws_mode, dtype=jnp.bool_)

    # Effective theta frequency (branchless clamp + SWS override).
    theta_f_awake = jnp.clip(
        params.theta_freq + params.ne_theta_shift * ne + params.sero_theta_shift * sero,
        params.theta_min, params.theta_max,
    )
    theta_f = jnp.where(sws, params.sws_freq, theta_f_awake)
    gamma_f = jnp.where(sws, jnp.asarray(0.0, DTYPE), params.gamma_freq)

    d_theta = TWO_PI * theta_f
    d_gamma = TWO_PI * gamma_f

    old_theta = state.theta_phase
    theta_next = old_theta + d_theta
    gamma_next = state.gamma_phase + d_gamma

    # PAC: gamma phase-resets when theta crosses π (the trough).
    crossed_trough = (old_theta < jnp.pi) & (theta_next >= jnp.pi)
    gamma_next = jnp.where(crossed_trough, jnp.asarray(0.0, DTYPE), gamma_next)

    # Wrap both phases into [0, 2π).
    gamma_wrapped = gamma_next >= TWO_PI
    gamma_next = jnp.where(gamma_wrapped, gamma_next - TWO_PI, gamma_next)
    theta_wrapped = theta_next >= TWO_PI
    theta_next = jnp.where(theta_wrapped, theta_next - TWO_PI, theta_next)

    # Gamma amplitude envelope: (1 − cos θ)/2 ∈ [0, 1], peaks at θ = π.
    pac = params.pac_depth * (1.0 - jnp.cos(theta_next)) * 0.5
    gamma_amp_awake = 1.0 - params.pac_depth + pac
    gamma_amp = jnp.where(sws, jnp.asarray(0.0, DTYPE), gamma_amp_awake)

    new_state = OscillatorState(
        theta_phase=theta_next.astype(DTYPE),
        gamma_phase=gamma_next.astype(DTYPE),
        gamma_amplitude=gamma_amp.astype(DTYPE),
    )
    encoding = theta_next < jnp.pi
    output = OscillatorOutput(
        theta_reset=theta_wrapped,
        gamma_reset=crossed_trough | gamma_wrapped,
        encoding_phase=encoding,
        retrieval_phase=~encoding,
        in_up_state=sws & encoding,
    )
    return new_state, output
