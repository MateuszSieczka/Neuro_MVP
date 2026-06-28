"""Sleep-phase state machine — pure JAX.

Sleep is modelled as a three-way state machine over ``{WAKE, SWS, REM}``,
driven by two endogenous signals:

  * **ATP (sleep pressure)** — the normalised local glial/neuronal
    energy pool aggregated from :mod:`core.astrocyte`.  Low ATP
    triggers sleep onset; recovered ATP triggers wake.  The
    hysteresis gap between ``atp_to_sws`` and ``atp_to_wake`` is the
    biophysical analogue of the VLPO / orexin flip-flop
    (Saper et al. 2010).
  * **Elapsed phase duration** — time accumulated since the current
    phase began.  Used for the NREM↔REM alternation during a sleep
    bout: the first REM episode emerges after ~90 min of NREM
    (Nishida & Walker 2007); REM episodes last ~20 min before
    yielding back to NREM (Dement & Kleitman 1957).

No random number generator is consumed by ``sleep_step`` itself; phase
transitions are deterministic given ATP and the duration counter.
The ``rng`` field in :class:`SleepState` is held only for downstream
consumers (Phase 5B replay sampling / REM rollouts) so that sleep
logic owns its own randomness channel rather than mixing it with the
wake cognitive-step key.

References
----------
  Saper, Fuller, Pedersen, Lu, Scammell (2010)
      Sleep state switching.  *Neuron* 68, 1023–1042.
  Achermann & Borbély (1992)
      Mathematical models of sleep regulation. *J. Biol. Rhythms* 7.
  Nishida & Walker (2007)
      Daytime naps, motor memory consolidation and regionally
      specific sleep spindles.  *PLoS ONE* 2, e341.
  Dement & Kleitman (1957)
      Cyclic variations in EEG during sleep and their relation to
      eye movements, body motility, and dreaming.
  Steriade (1993)
      Cellular substrates of brain rhythms.
"""

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, BackendContext


# ======================================================================
# Phase enumeration
# ======================================================================


class SleepPhase(IntEnum):
    """Three-way sleep phase label.

    Stored inside :class:`SleepState` as a ``jnp.int32`` scalar so it
    fits cleanly into a JAX pytree and can be used as an index for
    ``jax.lax.switch`` (Phase 5B brain-cycle dispatch).
    """

    WAKE = 0
    SWS = 1
    REM = 2


# ======================================================================
# Params
# ======================================================================


class SleepParams(eqx.Module):
    """Endogenous sleep-transition thresholds.

    All thresholds are *biophysically calibrated* (see module
    docstring references) rather than tuned.  The WAKE↔sleep
    hysteresis is asymmetric: we fall asleep at lower ATP than we
    wake at, reproducing the flip-flop dynamics of the VLPO circuit
    (Saper 2010) and preventing sleep-onset chatter near threshold.
    """

    # Hysteresis thresholds on the cortical ATP field (normalised to
    # [0, 1] by :mod:`core.astrocyte`).
    atp_to_sws: Array            # WAKE → SWS when atp_mean < this
    atp_to_wake: Array           # {SWS, REM} → WAKE when atp_mean > this

    # Ultradian NREM↔REM alternation (only applied during sleep).
    tau_sws_to_rem_ms: Array     # SWS → REM after this long in SWS
    tau_rem_to_sws_ms: Array     # REM → SWS after this long in REM


def init_sleep_params(
    *,
    atp_to_sws: float = 0.3,
    atp_to_wake: float = 0.8,
    tau_sws_to_rem_ms: float = 90.0 * 60.0 * 1000.0,   # 90 min
    tau_rem_to_sws_ms: float = 20.0 * 60.0 * 1000.0,   # 20 min
) -> SleepParams:
    """Construct :class:`SleepParams` with biologically calibrated defaults.

    Defaults:

    * ``atp_to_sws = 0.3`` — sleep pressure threshold (Saper 2010 VLPO
      lower arm activation regime).
    * ``atp_to_wake = 0.8`` — wake-restoration threshold; the 0.5-wide
      hysteresis band (Achermann & Borbély 1992) prevents chattering.
    * ``tau_sws_to_rem = 90 min`` — first-REM latency of a nocturnal
      sleep bout (Nishida & Walker 2007).
    * ``tau_rem_to_sws = 20 min`` — mean REM episode duration
      (Dement & Kleitman 1957).
    """
    if not (0.0 < atp_to_sws < atp_to_wake <= 1.0):
        raise ValueError(
            "require 0 < atp_to_sws < atp_to_wake ≤ 1; got "
            f"atp_to_sws={atp_to_sws}, atp_to_wake={atp_to_wake}"
        )
    if tau_sws_to_rem_ms <= 0.0 or tau_rem_to_sws_ms <= 0.0:
        raise ValueError("tau_* must be positive")
    f = lambda x: jnp.asarray(x, DTYPE)
    return SleepParams(
        atp_to_sws=f(atp_to_sws),
        atp_to_wake=f(atp_to_wake),
        tau_sws_to_rem_ms=f(tau_sws_to_rem_ms),
        tau_rem_to_sws_ms=f(tau_rem_to_sws_ms),
    )


# ======================================================================
# State
# ======================================================================


class SleepState(eqx.Module):
    """Sleep-phase state pytree.

    Fields
    ------
    phase : int32 scalar
        One of :class:`SleepPhase`; use ``int(SleepPhase.X)`` when
        writing JAX comparisons.
    phase_duration_ms : float32 scalar
        Milliseconds elapsed since the phase last transitioned.
        Resets to 0.0 on every transition and increments by ``ctx.dt``
        per call to :func:`sleep_step`.
    rng : PRNGKey
        Dedicated randomness stream for the sleep subsystem (replay
        sampling, REM rollout noise).  Split it through ``split_key``
        inside the relevant step; do NOT share it with the wake key.
    """

    phase: Array
    phase_duration_ms: Array
    rng: Array


def init_sleep_state(
    key: PRNGKey,
    *,
    initial_phase: SleepPhase = SleepPhase.WAKE,
    dtype=DTYPE,
) -> SleepState:
    return SleepState(
        phase=jnp.asarray(int(initial_phase), jnp.int32),
        phase_duration_ms=jnp.asarray(0.0, dtype),
        rng=key,
    )


# ======================================================================
# Step
# ======================================================================


@eqx.filter_jit
def sleep_step(
    state: SleepState,
    params: SleepParams,
    ctx: BackendContext,
    atp_mean: Array | float,
) -> SleepState:
    """Advance the sleep state machine by one ``ctx.dt``.

    Branchless transition table (all comparisons evaluated,
    ``jnp.where`` selects the destination):

        WAKE → SWS  iff  atp_mean < atp_to_sws
        SWS  → REM  iff  phase_duration_ms ≥ tau_sws_to_rem
        SWS  → WAKE iff  atp_mean > atp_to_wake
        REM  → SWS  iff  phase_duration_ms ≥ tau_rem_to_sws
        REM  → WAKE iff  atp_mean > atp_to_wake

    Within a step at most ONE transition fires (conditions are ordered
    so that the duration-driven NREM↔REM flip loses to the ATP-driven
    wake transition when both are simultaneously active, which is the
    correct biological precedence: if the system is rested enough to
    wake, it wakes).

    The ``rng`` stream is consumed by a fresh split each step so that
    downstream samplers always pull an unused key; the state carries
    the advanced key forward.
    """
    atp = jnp.asarray(atp_mean, DTYPE)
    phase = state.phase
    dur_next = state.phase_duration_ms + ctx.dt

    WAKE = jnp.asarray(int(SleepPhase.WAKE), jnp.int32)
    SWS = jnp.asarray(int(SleepPhase.SWS), jnp.int32)
    REM = jnp.asarray(int(SleepPhase.REM), jnp.int32)

    is_wake = phase == WAKE
    is_sws = phase == SWS
    is_rem = phase == REM

    can_wake = atp > params.atp_to_wake
    fall_asleep = atp < params.atp_to_sws
    sws_to_rem_ready = dur_next >= params.tau_sws_to_rem_ms
    rem_to_sws_ready = dur_next >= params.tau_rem_to_sws_ms

    # Build next-phase by considering each current-phase branch.
    # (All branches are computed; ``jnp.where`` selects.)
    next_from_wake = jnp.where(fall_asleep, SWS, WAKE)
    # ATP-driven wake takes precedence over duration-driven NREM↔REM flip.
    next_from_sws = jnp.where(
        can_wake, WAKE,
        jnp.where(sws_to_rem_ready, REM, SWS),
    )
    next_from_rem = jnp.where(
        can_wake, WAKE,
        jnp.where(rem_to_sws_ready, SWS, REM),
    )

    next_phase = jnp.where(
        is_wake, next_from_wake,
        jnp.where(is_sws, next_from_sws, next_from_rem),
    )

    transitioned = next_phase != phase
    # Advance clock when phase is unchanged; reset on transition.
    next_duration = jnp.where(
        transitioned,
        jnp.asarray(0.0, DTYPE),
        dur_next.astype(DTYPE),
    )

    # Advance the sleep-local RNG so phase-5B samplers can fold_in
    # cycle counters without key reuse.
    new_rng, _ = jax.random.split(state.rng)

    return SleepState(
        phase=next_phase.astype(jnp.int32),
        phase_duration_ms=next_duration,
        rng=new_rng,
    )


def is_wake(state: SleepState) -> Array:
    """Return a bool scalar: ``True`` iff the agent is currently awake."""
    return state.phase == jnp.asarray(int(SleepPhase.WAKE), jnp.int32)


def is_sws(state: SleepState) -> Array:
    """Return a bool scalar: ``True`` iff the agent is currently in SWS."""
    return state.phase == jnp.asarray(int(SleepPhase.SWS), jnp.int32)


def is_rem(state: SleepState) -> Array:
    """Return a bool scalar: ``True`` iff the agent is currently in REM."""
    return state.phase == jnp.asarray(int(SleepPhase.REM), jnp.int32)
