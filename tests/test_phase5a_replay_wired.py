"""Phase 5A — replay buffer wired into the ActionBrain cognitive loop.

Structural invariant: every wake cognitive cycle writes the just-
closed ``(s_t, a_t, r_t, s_{t+1})`` transition into the prioritised
replay ring buffer (Schaul et al. 2016).  Phase 5B will consume these
during SWS for reverse-order memory consolidation (Wilson &
McNaughton 1994; Foster & Wilson 2006).  This test exercises the
wiring only — not the replay semantics themselves — so we keep the
loop short and use a non-visual sensory afferent.

The first cognitive cycle cannot form a full transition (there was
no previous ``s_t``), so we assert that after ``N`` cycles the buffer
holds exactly ``N`` valid entries starting from slot 0.

References
----------
  Schaul, Quan, Antonoglou & Silver (2016). Prioritized experience
      replay.  *ICLR*.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import BackendContext
from core.brain_graph import (
    init_action_brain_params, init_action_brain_state,
    action_brain_cognitive_step,
)
from core.replay_buffer import replay_size


def _build_brain(*, seed: int = 0, sensory_size: int = 8):
    ctx = BackendContext(dt=1.0)
    params = init_action_brain_params(
        ctx, sensory_size=sensory_size,
        n_body_actions=3, n_saccade_actions=2, substeps=4,
        replay_capacity=64,
    )
    state = init_action_brain_state(jax.random.PRNGKey(seed), params)
    return ctx, params, state


def test_replay_store_every_cycle():
    """N cognitive cycles ⇒ N valid slots in the ring buffer."""
    ctx, params, state = _build_brain()
    n_cycles = 10
    key = jax.random.PRNGKey(1)

    st = state
    sensory_seq = []
    for i in range(n_cycles):
        k_cycle, key = jax.random.split(key)
        # Use a different sensory vector each cycle so we can verify
        # that what was stored is the state the policy actually saw.
        s = jnp.full((params.sensory_size,), float(i), jnp.float32)
        sensory_seq.append(s)
        out = action_brain_cognitive_step(
            st, params, ctx, s,
            jnp.float32(0.0), jnp.float32(0.0), k_cycle,
        )
        st = out.state

    assert int(replay_size(st.replay)) == n_cycles
    valid = st.replay.valid
    # Valid entries are contiguous from slot 0 while buffer not full.
    assert bool(valid[:n_cycles].all())
    assert bool((~valid[n_cycles:]).all())


def test_replay_state_matches_observed_sensory():
    """``replay.state[i]`` equals the sensory vector fed at cycle i.

    The store-path writes ``exp.state = state.last_sensory`` — i.e.
    the *previous* decision cycle's sensory.  Cycle i=0 writes the
    zero-initialised ``last_sensory`` (no prior observation), so we
    assert matches only for i ≥ 1.
    """
    ctx, params, state = _build_brain()
    key = jax.random.PRNGKey(2)
    st = state

    sensories = []
    for i in range(6):
        k_cycle, key = jax.random.split(key)
        s = jnp.asarray(
            jnp.arange(params.sensory_size, dtype=jnp.float32) + float(i),
        )
        sensories.append(s)
        out = action_brain_cognitive_step(
            st, params, ctx, s,
            jnp.float32(0.0), jnp.float32(0.0), k_cycle,
        )
        st = out.state

    # Slot 0 holds the initial zero last_sensory (bootstrap);
    # slot i (i≥1) holds sensories[i-1].
    for i in range(1, 6):
        stored = st.replay.state[i]
        expected = sensories[i - 1]
        assert jnp.allclose(stored, expected), (
            f"slot {i}: stored={stored} expected={expected}"
        )


def test_replay_salience_at_least_floor():
    """Every stored entry carries salience ≥ ``salience_floor``."""
    ctx, params, state = _build_brain()
    key = jax.random.PRNGKey(3)
    st = state
    n_cycles = 12
    for i in range(n_cycles):
        k_cycle, key = jax.random.split(key)
        s = jax.random.normal(
            jax.random.fold_in(key, i),
            (params.sensory_size,), dtype=jnp.float32,
        )
        out = action_brain_cognitive_step(
            st, params, ctx, s,
            jnp.float32(0.5), jnp.float32(0.0), k_cycle,
        )
        st = out.state

    stored = st.replay.salience[:n_cycles]
    assert bool((stored >= params.replay.salience_floor).all())
    # And finite — no NaN / inf leaking from curiosity pipeline.
    assert bool(jnp.isfinite(stored).all())
