"""Faza U — §3 sleep: offline free-energy minimisation on the graph.

Asserts the three claims of the sleep subsystem:

* the WAKE/SWS/REM FSM is driven by free-energy *pressure* (not ATP),
  with flip-flop hysteresis and duration-driven NREM↔REM alternation;
* SWS reverse replay consolidates stored experience — the same clamp →
  relax → learn, offline, lowers free energy on the replayed pattern;
* REM rollout learns purely generatively (sensory unclamped, samples
  from the model) and the samples are stochastic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.pc_graph import (
    init_region_graph, pc_graph_clamp, pc_graph_relax, graph_free_energy,
    REGION_INDEX,
)
from core.pc_memory import init_replay_params, init_replay_state, replay_store, Experience
from core.pc_sleep import (
    SleepPhase, init_sleep_params, init_sleep_state, sleep_step,
    is_wake, is_sws, is_rem, sws_replay, rem_rollout,
)


def _run(state, params, fe, k):
    for _ in range(k):
        state = sleep_step(state, params, fe)
    return state


# ---------------------------------------------------------------------
# the WAKE/SWS/REM state machine
# ---------------------------------------------------------------------


def test_fsm_falls_asleep_and_wakes_on_free_energy():
    p = init_sleep_params(fe_to_sws=1.0, fe_to_wake=0.3, pressure_alpha=0.3,
                          tau_sws_to_rem=20.0, tau_rem_to_sws=10.0)
    s = init_sleep_state(jax.random.PRNGKey(0))
    assert int(s.phase) == int(SleepPhase.WAKE)

    # High accumulated free energy → sleep onset.
    s = _run(s, p, 2.0, 30)
    assert bool(is_sws(s)) or bool(is_rem(s)), "high FE did not trigger sleep"

    # Free energy paid down → wake.
    s = _run(s, p, 0.0, 30)
    assert bool(is_wake(s)), "low FE did not restore wake"


def test_fsm_hysteresis_band_is_stable():
    p = init_sleep_params(fe_to_sws=1.0, fe_to_wake=0.3, pressure_alpha=0.5)
    # Pressure parked between the two thresholds must not flip either way.
    awake = _run(init_sleep_state(jax.random.PRNGKey(1)), p, 0.6, 40)
    assert bool(is_wake(awake)), "mid-band pressure should not start sleep"


def test_fsm_sws_to_rem_by_duration():
    p = init_sleep_params(fe_to_sws=1.0, fe_to_wake=0.3, pressure_alpha=0.5,
                          tau_sws_to_rem=15.0, tau_rem_to_sws=10.0)
    s = init_sleep_state(jax.random.PRNGKey(2), initial_phase=SleepPhase.SWS)
    # Hold pressure high enough not to wake; duration drives SWS → REM.
    s = _run(s, p, 0.6, 20)
    assert bool(is_rem(s)), "duration did not advance SWS → REM"


# ---------------------------------------------------------------------
# SWS reverse replay — offline consolidation lowers free energy
# ---------------------------------------------------------------------


def _fe_clamped(graph, gparams, s_idx, m_idx, sensory, motor):
    clamped = pc_graph_clamp(graph, {s_idx: sensory, m_idx: motor})
    relaxed = pc_graph_relax(clamped, gparams, clamp=(s_idx, m_idx), n_steps=20)
    return float(graph_free_energy(relaxed, gparams))


def test_sws_replay_consolidates_experience():
    gp, gs = init_region_graph(jax.random.PRNGKey(0), eta_mu=0.1, eta_w=0.05,
                               n_relax=20)
    s_idx, m_idx = REGION_INDEX["sensory"], REGION_INDEX["motor"]
    s_dim, m_dim = gp.node_sizes[s_idx], gp.node_sizes[m_idx]

    sensory = jax.random.normal(jax.random.PRNGKey(1), (s_dim,))
    motor = jax.random.normal(jax.random.PRNGKey(2), (m_dim,))

    bp = init_replay_params(capacity=8, sensory_size=s_dim, motor_size=m_dim)
    bs = init_replay_state(bp)
    for _ in range(8):                              # one pattern, repeated
        bs = replay_store(bs, bp, Experience(sensory, motor, jnp.asarray(1.0)))

    fe0 = _fe_clamped(gs, gp, s_idx, m_idx, sensory, motor)
    g = gs
    for _ in range(10):                            # several SWS passes
        g = sws_replay(g, gp, bs, bp, n_replay=8, n_relax=20)
    fe1 = _fe_clamped(g, gp, s_idx, m_idx, sensory, motor)

    assert fe1 < fe0, f"SWS replay did not consolidate: {fe0:.3f} → {fe1:.3f}"


# ---------------------------------------------------------------------
# REM rollout — generative, stochastic, no external data
# ---------------------------------------------------------------------


def test_rem_rollout_is_generative_and_stochastic():
    gp, gs = init_region_graph(jax.random.PRNGKey(0), eta_mu=0.1, eta_w=0.05,
                               n_relax=15)
    # Purely generative: no replay buffer, sensory never clamped to data.
    ga = rem_rollout(gs, gp, jax.random.PRNGKey(1), n_steps=8, n_relax=15)
    gb = rem_rollout(gs, gp, jax.random.PRNGKey(2), n_steps=8, n_relax=15)

    for w in ga.weights:
        assert jnp.all(jnp.isfinite(w)), "REM weights diverged"

    moved = sum(float(jnp.sum(jnp.abs(a - b)))
                for a, b in zip(ga.weights, gs.weights))
    assert moved > 0.0, "REM rollout did not learn from generated samples"

    # Different prior samples → different consolidation (it is sampling).
    diff = sum(float(jnp.sum(jnp.abs(a - b)))
               for a, b in zip(ga.weights, gb.weights))
    assert diff > 0.0, "REM rollout is not stochastic across seeds"
