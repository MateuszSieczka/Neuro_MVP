"""Sleep — offline free-energy minimisation on the graph (Faza U, §3 / §4).

Sleep is the substrate's **offline mode** (integration contract §0.4):
the very same relaxation + one rule that runs awake, but with the
afferent stream gone.  Two phases, both ending in ``pc_graph_relax`` →
``pc_graph_learn``:

* **SWS reverse replay** (:func:`sws_replay`) — *experience* replay.
  Stored wake experiences are reactivated most-recent-first (Wilson &
  McNaughton 1994); the sensory **and** motor nodes are clamped to the
  remembered ``(sensory, motor_belief)`` and the graph relaxes + learns,
  distilling the episode into the cortical generative model offline
  (McClelland, McNaughton & O'Reilly 1995 systems consolidation).

* **REM rollout** (:func:`rem_rollout`) — *generative* replay.  A deep
  cortical cause is sampled from its prior, the sensory node is left
  **unclamped**, and the generative edges render a fantasy observation
  top-down; the graph then re-explains that self-generated sample and
  learns, sharpening the generative manifold without external data
  (Hobson & McCarley 1977 activation-synthesis; van de Ven 2020
  generative replay).

The wake/sleep schedule is a three-way state machine over
``{WAKE, SWS, REM}`` (:func:`sleep_step`).  The legacy ATP / astrocyte
trigger is gone; sleep pressure is now the substrate's own quantity —
a low-pass of the **free energy** the wake cycle reports.  High
accumulated free energy (much unexplained, unconsolidated experience)
drives sleep onset; offline replay drives free energy back down and the
agent wakes.  One objective, asleep and awake.

References
----------
  Wilson & McNaughton (1994)            — SWS place-cell reverse replay.
  McClelland, McNaughton, O'Reilly (1995) — Complementary learning systems.
  Hobson & McCarley (1977)              — REM activation-synthesis.
  van de Ven, Siegelmann, Tolias (2020) — Generative replay for continual learning.
  Saper et al. (2010)                   — Flip-flop sleep switching (hysteresis).
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, split_key
from .pc_graph import (
    PCGraphParams, PCGraphState,
    pc_graph_clamp, pc_graph_relax, pc_graph_learn, REGION_INDEX,
)
from .pc_memory import (
    ReplayParams, ReplayState, replay_recent_indices, replay_gather,
)


# =====================================================================
# Phase enumeration + FSM
# =====================================================================


class SleepPhase(IntEnum):
    """Three-way sleep phase, stored as an int32 scalar in the state."""

    WAKE = 0
    SWS = 1
    REM = 2


class SleepParams(eqx.Module):
    """Free-energy-driven sleep-transition thresholds.

    Hysteresis (``fe_to_wake < fe_to_sws``) gives the flip-flop that
    prevents sleep-onset chatter (Saper 2010).  Sleep pressure is a
    low-pass of the reported free energy with smoothing ``pressure_alpha``
    (per cognitive step); ``tau_*`` set the ultradian SWS↔REM alternation
    in the same step units.
    """

    fe_to_sws: Array             # WAKE → SWS  when pressure > this
    fe_to_wake: Array            # {SWS,REM} → WAKE  when pressure < this
    pressure_alpha: Array        # EMA weight folding new FE into pressure
    tau_sws_to_rem: Array        # SWS → REM after this many steps in SWS
    tau_rem_to_sws: Array        # REM → SWS after this many steps in REM


def init_sleep_params(
    *,
    fe_to_sws: float = 1.0,
    fe_to_wake: float = 0.3,
    pressure_alpha: float = 0.05,
    tau_sws_to_rem: float = 40.0,
    tau_rem_to_sws: float = 15.0,
) -> SleepParams:
    """Sleep thresholds on the free-energy pressure signal.

    Defaults are in the substrate's free-energy units (calibrate to the
    wake cycle's typical ``free_energy``); the ratio ``fe_to_sws >
    fe_to_wake`` is what matters (the hysteresis band).  ``tau_*`` are in
    cognitive-step units, not milliseconds — the substrate has no wall
    clock.
    """
    if not (0.0 <= fe_to_wake < fe_to_sws):
        raise ValueError(
            "require 0 ≤ fe_to_wake < fe_to_sws; got "
            f"fe_to_wake={fe_to_wake}, fe_to_sws={fe_to_sws}"
        )
    if not (0.0 < pressure_alpha <= 1.0):
        raise ValueError("pressure_alpha must be in (0, 1]")
    if tau_sws_to_rem <= 0.0 or tau_rem_to_sws <= 0.0:
        raise ValueError("tau_* must be positive")
    f = lambda x: jnp.asarray(x, DTYPE)
    return SleepParams(
        fe_to_sws=f(fe_to_sws), fe_to_wake=f(fe_to_wake),
        pressure_alpha=f(pressure_alpha),
        tau_sws_to_rem=f(tau_sws_to_rem), tau_rem_to_sws=f(tau_rem_to_sws),
    )


class SleepState(eqx.Module):
    """Phase + elapsed duration + free-energy pressure + sleep-local RNG."""

    phase: Array                 # int32 scalar (SleepPhase)
    phase_duration: Array        # float scalar — steps since last transition
    pressure: Array              # float scalar — low-pass free energy
    rng: Array                   # dedicated key for replay / rollout sampling


def init_sleep_state(
    key: PRNGKey, *, initial_phase: SleepPhase = SleepPhase.WAKE, dtype=DTYPE,
) -> SleepState:
    return SleepState(
        phase=jnp.asarray(int(initial_phase), jnp.int32),
        phase_duration=jnp.asarray(0.0, dtype),
        pressure=jnp.asarray(0.0, dtype),
        rng=key,
    )


def sleep_step(
    state: SleepState, params: SleepParams, free_energy: Array | float,
) -> SleepState:
    """Advance the sleep FSM by one cognitive step given the reported FE.

    Branchless transition table (all conditions evaluated, ``jnp.where``
    selects), driven by the low-pass free-energy ``pressure``:

        WAKE → SWS   iff  pressure > fe_to_sws
        SWS  → REM   iff  duration ≥ tau_sws_to_rem
        SWS  → WAKE  iff  pressure < fe_to_wake
        REM  → SWS   iff  duration ≥ tau_rem_to_sws
        REM  → WAKE  iff  pressure < fe_to_wake

    The ATP-driven wake of the old flip-flop becomes pressure-driven: a
    rested (low-FE) system wakes, taking precedence over the duration
    flip.  The sleep-local RNG is advanced each step so replay samplers
    always pull a fresh key.
    """
    a = params.pressure_alpha
    pressure = (1.0 - a) * state.pressure + a * jnp.asarray(free_energy, DTYPE)
    dur_next = state.phase_duration + 1.0

    WAKE = jnp.asarray(int(SleepPhase.WAKE), jnp.int32)
    SWS = jnp.asarray(int(SleepPhase.SWS), jnp.int32)
    REM = jnp.asarray(int(SleepPhase.REM), jnp.int32)

    is_wake = state.phase == WAKE
    is_sws = state.phase == SWS

    can_wake = pressure < params.fe_to_wake
    fall_asleep = pressure > params.fe_to_sws
    sws_to_rem_ready = dur_next >= params.tau_sws_to_rem
    rem_to_sws_ready = dur_next >= params.tau_rem_to_sws

    next_from_wake = jnp.where(fall_asleep, SWS, WAKE)
    next_from_sws = jnp.where(
        can_wake, WAKE, jnp.where(sws_to_rem_ready, REM, SWS),
    )
    next_from_rem = jnp.where(
        can_wake, WAKE, jnp.where(rem_to_sws_ready, SWS, REM),
    )
    next_phase = jnp.where(
        is_wake, next_from_wake,
        jnp.where(is_sws, next_from_sws, next_from_rem),
    )

    transitioned = next_phase != state.phase
    next_duration = jnp.where(transitioned, jnp.asarray(0.0, DTYPE), dur_next)
    new_rng, _ = jax.random.split(state.rng)

    return SleepState(
        phase=next_phase.astype(jnp.int32),
        phase_duration=next_duration.astype(DTYPE),
        pressure=pressure.astype(DTYPE),
        rng=new_rng,
    )


def is_wake(state: SleepState) -> Array:
    return state.phase == jnp.asarray(int(SleepPhase.WAKE), jnp.int32)


def is_sws(state: SleepState) -> Array:
    return state.phase == jnp.asarray(int(SleepPhase.SWS), jnp.int32)


def is_rem(state: SleepState) -> Array:
    return state.phase == jnp.asarray(int(SleepPhase.REM), jnp.int32)


# =====================================================================
# Offline modes — SWS experience replay + REM generative rollout
# =====================================================================


def sws_replay(
    graph: PCGraphState, gparams: PCGraphParams,
    buffer: ReplayState, bparams: ReplayParams,
    *,
    sensory_idx: int | None = None,
    motor_idx: int | None = None,
    n_replay: int = 32,
    n_relax: int | None = None,
) -> PCGraphState:
    """One SWS consolidation pass over the ``n_replay`` most recent steps.

    Reverse-chronological (most recent first).  Each experience clamps the
    sensory and motor nodes to its stored ``(sensory, motor_belief)`` and
    the graph relaxes + learns by the one rule — the cortical model
    absorbs the episode offline.  Pure inference + the single Hebbian rule;
    no separate world-model or sequence update.
    """
    s_idx = REGION_INDEX["sensory"] if sensory_idx is None else int(sensory_idx)
    m_idx = REGION_INDEX["motor"] if motor_idx is None else int(motor_idx)

    idx = replay_recent_indices(buffer, bparams, int(n_replay))
    batch = replay_gather(buffer, idx)            # (n_replay, ·) most recent first

    def _body(g: PCGraphState, exp) -> tuple[PCGraphState, None]:
        sensory, motor, _fe = exp
        clamped = pc_graph_clamp(g, {s_idx: sensory, m_idx: motor})
        relaxed = pc_graph_relax(
            clamped, gparams, clamp=(s_idx, m_idx), n_steps=n_relax,
        )
        return pc_graph_learn(relaxed, gparams), None

    final, _ = jax.lax.scan(_body, graph, batch)
    return final


def rem_rollout(
    graph: PCGraphState, gparams: PCGraphParams, key: PRNGKey,
    *,
    sensory_idx: int | None = None,
    cause_idx: int | None = None,
    n_steps: int = 16,
    prior_std: float = 1.0,
    n_relax: int | None = None,
) -> PCGraphState:
    """``n_steps`` of generative replay seeded from the deep cortical prior.

    Each step: sample a cause ``c ~ N(0, prior_std)`` on the deep cortical
    node, clamp it and relax with the **sensory node free** so the
    generative edges render a fantasy observation top-down; then clamp
    that self-generated observation, free the cause, relax and learn.  The
    model thus rehearses and re-explains its own samples — offline
    free-energy minimisation with no external input (van de Ven 2020).
    """
    s_idx = REGION_INDEX["sensory"] if sensory_idx is None else int(sensory_idx)
    c_idx = REGION_INDEX["cortex_l3"] if cause_idx is None else int(cause_idx)
    cause_size = gparams.node_sizes[c_idx]
    std = jnp.asarray(prior_std, DTYPE)

    keys = jax.random.split(key, int(n_steps))

    def _body(g: PCGraphState, k: PRNGKey) -> tuple[PCGraphState, None]:
        # 1. Dream a deep cause; render it down to a fantasy observation
        #    (sensory free → generated by the top-down edges).
        cause = std * jax.random.normal(k, (cause_size,), DTYPE)
        seeded = pc_graph_clamp(g, {c_idx: cause})
        rendered = pc_graph_relax(seeded, gparams, clamp=(c_idx,), n_steps=n_relax)
        fantasy = rendered.mu[s_idx]
        # 2. Re-explain the self-generated observation and learn.
        clamped = pc_graph_clamp(rendered, {s_idx: fantasy})
        relaxed = pc_graph_relax(clamped, gparams, clamp=(s_idx,), n_steps=n_relax)
        return pc_graph_learn(relaxed, gparams), None

    final, _ = jax.lax.scan(_body, graph, keys)
    return final
