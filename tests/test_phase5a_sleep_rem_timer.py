"""Phase 5A — NREM↔REM ultradian alternation.

Inside a sleep bout the duration counter drives the alternation
between SWS and REM: the first REM episode emerges after ≈ 90 min of
NREM (Nishida & Walker 2007), and a REM episode lasts ≈ 20 min before
yielding back to SWS (Dement & Kleitman 1957).  This test runs
``sleep_step`` with a single timestep whose ``ctx.dt`` spans the
crossover moment, so one ``sleep_step`` call is enough to certify the
transition.

When ATP is high enough to wake the agent, the biological precedence
is that the ATP-driven WAKE transition wins over the duration-driven
NREM↔REM flip — the test also locks that ordering down.

References
----------
  Nishida & Walker (2007). Daytime naps, motor memory consolidation
      and regionally specific sleep spindles. *PLoS ONE* 2, e341.
  Dement & Kleitman (1957). Cyclic variations in EEG during sleep.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from core.backend import BackendContext
from core.sleep import (
    SleepPhase, SleepState,
    init_sleep_params, init_sleep_state, sleep_step,
)


def _ctx() -> BackendContext:
    return BackendContext(dt=1.0)


def _set_duration(state: SleepState, ms: float) -> SleepState:
    return eqx.tree_at(
        lambda s: s.phase_duration_ms, state, jnp.asarray(ms, jnp.float32),
    )


def test_sws_to_rem_after_first_rem_latency():
    """Staying in SWS past ``tau_sws_to_rem_ms`` flips to REM."""
    params = init_sleep_params()
    state = init_sleep_state(
        jax.random.PRNGKey(0), initial_phase=SleepPhase.SWS,
    )
    # One ctx.dt before the next step crosses the latency boundary.
    state = _set_duration(
        state,
        float(params.tau_sws_to_rem_ms) - 0.5,
    )

    # Sub-threshold ATP to rule out WAKE precedence.
    new_state = sleep_step(state, params, _ctx(), jnp.asarray(0.1))
    assert int(new_state.phase) == int(SleepPhase.REM)
    assert float(new_state.phase_duration_ms) == 0.0


def test_rem_to_sws_after_rem_duration():
    """A REM bout ends after ``tau_rem_to_sws_ms`` and returns to SWS."""
    params = init_sleep_params()
    state = init_sleep_state(
        jax.random.PRNGKey(0), initial_phase=SleepPhase.REM,
    )
    state = _set_duration(
        state,
        float(params.tau_rem_to_sws_ms) - 0.5,
    )

    new_state = sleep_step(state, params, _ctx(), jnp.asarray(0.1))
    assert int(new_state.phase) == int(SleepPhase.SWS)


def test_wake_takes_precedence_over_rem_timer_in_sws():
    """A fully rested SWS sleeper wakes, even if REM is due next."""
    params = init_sleep_params()
    state = init_sleep_state(
        jax.random.PRNGKey(0), initial_phase=SleepPhase.SWS,
    )
    state = _set_duration(
        state,
        float(params.tau_sws_to_rem_ms) + 100.0,
    )

    # Simultaneously: REM timer ready AND atp above wake threshold.
    # Biological ordering: if rested, wake.
    new_state = sleep_step(state, params, _ctx(), jnp.asarray(0.95))
    assert int(new_state.phase) == int(SleepPhase.WAKE)
