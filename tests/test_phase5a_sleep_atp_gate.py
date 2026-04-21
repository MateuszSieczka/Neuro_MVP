"""Phase 5A — VLPO flip-flop hysteresis on cortical ATP.

Structural invariant: the ``sleep_step`` transition table must
implement the asymmetric WAKE↔SWS hysteresis of Saper et al. (2010)'s
mutually-inhibitory VLPO / orexin flip-flop.  Below ``atp_to_sws``
sleep pressure dominates and the arousal system collapses to SWS;
once the local energy pool recovers above the *higher*
``atp_to_wake`` set-point the inhibition is lifted and the system
flips back to WAKE.  The gap between the two thresholds must suppress
any near-threshold chattering — this is exactly the behaviour this
test certifies.

References
----------
  Saper, Fuller, Pedersen, Lu, Scammell (2010).  Sleep state
      switching.  *Neuron* 68, 1023–1042.
  Achermann & Borbély (1992).  Mathematical models of sleep
      regulation.  *J. Biol. Rhythms* 7.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import BackendContext
from core.sleep import (
    SleepPhase,
    init_sleep_params, init_sleep_state, sleep_step,
)


def _ctx() -> BackendContext:
    # dt small enough that no single step exhausts the NREM→REM
    # 90-min duration budget.
    return BackendContext(dt=1.0)


def test_wake_to_sws_when_atp_below_threshold():
    """Low ATP below ``atp_to_sws`` drives WAKE → SWS in one step."""
    params = init_sleep_params()
    state = init_sleep_state(jax.random.PRNGKey(0))
    assert int(state.phase) == int(SleepPhase.WAKE)

    new_state = sleep_step(state, params, _ctx(), jnp.asarray(0.1))
    assert int(new_state.phase) == int(SleepPhase.SWS)
    # Transition resets the duration clock.
    assert float(new_state.phase_duration_ms) == 0.0


def test_sws_to_wake_requires_upper_threshold():
    """Only ATP strictly above ``atp_to_wake`` can wake the agent."""
    params = init_sleep_params()
    state = init_sleep_state(
        jax.random.PRNGKey(0), initial_phase=SleepPhase.SWS,
    )

    # ATP at 0.5 sits inside the hysteresis band: still asleep.
    band_state = sleep_step(state, params, _ctx(), jnp.asarray(0.5))
    assert int(band_state.phase) == int(SleepPhase.SWS)

    # ATP above atp_to_wake: flip back to WAKE.
    rested = sleep_step(state, params, _ctx(), jnp.asarray(0.95))
    assert int(rested.phase) == int(SleepPhase.WAKE)


def test_hysteresis_band_blocks_chatter():
    """Oscillating ATP within the hysteresis band never flips phase."""
    params = init_sleep_params()
    # Hysteresis band runs from atp_to_sws (0.3) up to atp_to_wake (0.8).
    # Any ATP inside that band must leave the current phase untouched.
    wake = init_sleep_state(jax.random.PRNGKey(0))
    sws = init_sleep_state(
        jax.random.PRNGKey(1), initial_phase=SleepPhase.SWS,
    )
    ctx = _ctx()
    for atp in (0.31, 0.5, 0.79):
        atp_arr = jnp.asarray(atp)
        assert int(sleep_step(wake, params, ctx, atp_arr).phase) == int(
            SleepPhase.WAKE
        )
        assert int(sleep_step(sws, params, ctx, atp_arr).phase) == int(
            SleepPhase.SWS
        )


def test_params_reject_non_hysteretic_config():
    """``atp_to_sws ≥ atp_to_wake`` is biophysically degenerate."""
    import pytest

    with pytest.raises(ValueError):
        init_sleep_params(atp_to_sws=0.9, atp_to_wake=0.3)
