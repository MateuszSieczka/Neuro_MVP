"""
Oscillator — pure functional theta/gamma pacemaker with PAC.

Replaces the legacy class-based ``ThetaGammaOscillator``.  All state
lives in ``OscillatorState`` (theta_phase, gamma_phase,
gamma_amplitude); parameters live in ``OscillatorParams``.

Theta (4–8 Hz): episodic encoding phase; gates memory I/O.
Gamma (30–100 Hz): local binding; paces k-WTA.

Phase-Amplitude Coupling (Lisman & Jensen 2013):
    Gamma phase-resets at each theta trough (θ = π).
    Gamma amplitude envelope = 1 − PAC + PAC · (1 − cos θ)/2
    ⇒ maximal gamma at θ=π (trough), minimal at θ=0 (peak).

Neuromodulation (scalar, per-step):
    NE  ↑  →  theta freq ↑  (arousal, faster cycling)
    5-HT↑  →  theta freq ↓  (patience, longer cycles)

SWS mode (``sws_mode=True``):
    Theta frequency clamped to ``sws_freq_hz`` (~1 Hz, Steriade 1993);
    gamma fully suppressed.  Up state = (θ < π), Down state = (θ ≥ π).

All branches use ``jnp.where`` masking — no Python conditionals, so
the step is jit-safe and SIMT-friendly.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array, BackendContext
from .state import OscillatorState


TWO_PI = 2.0 * jnp.pi


# ======================================================================
# Parameters
# ======================================================================


class OscillatorParams(eqx.Module):
    """Theta/gamma oscillator constants.

    All float32 scalars (traced leaves).  Allows runtime swap without
    retracing via ``eqx.tree_at``.
    """

    theta_freq_hz: Array       # base theta freq
    theta_min_hz: Array
    theta_max_hz: Array
    gamma_freq_hz: Array

    pac_depth: Array           # 0..1, depth of theta-gamma PAC envelope

    ne_theta_shift: Array      # Hz added per unit NE
    sero_theta_shift: Array    # Hz added per unit 5-HT (typically < 0)

    sws_freq_hz: Array         # slow-wave sleep theta-clamp freq


def init_oscillator_params(
    *,
    theta_freq_hz: float = 6.0,
    theta_min_hz: float = 4.0,
    theta_max_hz: float = 8.0,
    gamma_freq_hz: float = 40.0,
    pac_depth: float = 0.6,
    ne_theta_shift: float = 2.0,
    sero_theta_shift: float = -1.0,
    sws_freq_hz: float = 1.0,
) -> OscillatorParams:
    assert theta_min_hz <= theta_freq_hz <= theta_max_hz
    assert 0.0 <= pac_depth <= 1.0
    f = lambda x: jnp.asarray(x, DTYPE)
    return OscillatorParams(
        theta_freq_hz=f(theta_freq_hz),
        theta_min_hz=f(theta_min_hz),
        theta_max_hz=f(theta_max_hz),
        gamma_freq_hz=f(gamma_freq_hz),
        pac_depth=f(pac_depth),
        ne_theta_shift=f(ne_theta_shift),
        sero_theta_shift=f(sero_theta_shift),
        sws_freq_hz=f(sws_freq_hz),
    )


# ======================================================================
# Step
# ======================================================================


class OscillatorOutput(eqx.Module):
    """Per-step oscillator outputs used by downstream modules."""

    gamma_reset: Array      # bool scalar — gamma cycle completed this step
    theta_reset: Array      # bool scalar — theta cycle completed this step
    in_up_state: Array      # bool scalar — SWS Up state (only meaningful if sws_mode)
    encoding_phase: Array   # bool scalar — θ < π (Hasselmo 2005 encoding window)


def oscillator_step(
    state: OscillatorState,
    params: OscillatorParams,
    ctx: BackendContext,
    *,
    ne_level: Array | float = 0.0,
    sero_level: Array | float = 0.0,
    sws_mode: Array | bool = False,
) -> tuple[OscillatorState, OscillatorOutput]:
    """Advance theta + gamma phases by one ``ctx.dt``.

    Applies PAC (gamma reset at theta trough + cosine envelope) and
    optional SWS clamp.  Returns new state and cycle-completion flags.
    """
    dt_s = ctx.dt / 1000.0  # ms → s
    ne = jnp.asarray(ne_level, DTYPE)
    sero = jnp.asarray(sero_level, DTYPE)
    sws = jnp.asarray(sws_mode, dtype=jnp.bool_)

    # ── Effective theta frequency (branchless) ─────────────────────────
    theta_f_awake = jnp.clip(
        params.theta_freq_hz
        + params.ne_theta_shift * ne
        + params.sero_theta_shift * sero,
        params.theta_min_hz, params.theta_max_hz,
    )
    theta_f = jnp.where(sws, params.sws_freq_hz, theta_f_awake)
    # Gamma suppressed during SWS (Steriade 1993)
    gamma_f = jnp.where(sws, jnp.asarray(0.0, DTYPE), params.gamma_freq_hz)

    # ── Phase advance ──────────────────────────────────────────────────
    d_theta = TWO_PI * theta_f * dt_s
    d_gamma = TWO_PI * gamma_f * dt_s

    old_theta = state.theta_phase
    theta_next = old_theta + d_theta
    gamma_next = state.gamma_phase + d_gamma

    # ── PAC: gamma phase-resets when theta crosses π (trough) ─────────
    crossed_trough = (old_theta < jnp.pi) & (theta_next >= jnp.pi)
    gamma_next = jnp.where(crossed_trough, jnp.asarray(0.0, DTYPE), gamma_next)

    # ── Wrap phases to [0, 2π) ─────────────────────────────────────────
    gamma_wrapped_flag = gamma_next >= TWO_PI
    gamma_next = jnp.where(gamma_wrapped_flag, gamma_next - TWO_PI, gamma_next)

    theta_wrapped = theta_next >= TWO_PI
    theta_next = jnp.where(theta_wrapped, theta_next - TWO_PI, theta_next)

    # ── Gamma amplitude envelope (PAC) ─────────────────────────────────
    # (1 − cos θ)/2 ∈ [0,1]; peaks at θ=π (trough)
    pac = params.pac_depth * (1.0 - jnp.cos(theta_next)) * 0.5
    gamma_amp_awake = 1.0 - params.pac_depth + pac
    gamma_amp = jnp.where(sws, jnp.asarray(0.0, DTYPE), gamma_amp_awake)

    new_state = OscillatorState(
        theta_phase=theta_next.astype(DTYPE),
        gamma_phase=gamma_next.astype(DTYPE),
        gamma_amplitude=gamma_amp.astype(DTYPE),
    )

    gamma_reset = crossed_trough | gamma_wrapped_flag
    output = OscillatorOutput(
        gamma_reset=gamma_reset,
        theta_reset=theta_wrapped,
        in_up_state=sws & (theta_next < jnp.pi),
        encoding_phase=theta_next < jnp.pi,
    )
    return new_state, output
