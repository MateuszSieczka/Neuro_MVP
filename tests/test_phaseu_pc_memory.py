"""Faza U — §3 memory primitives: replay buffer + episodic store.

Substrate-agnostic, JIT-safe stores backing sleep replay and the
hippocampal node group.  Asserts:

* the ring buffer wraps, caps its size, and yields recent-first indices;
* free-energy-prioritised sampling favours surprising experiences;
* the episodic store completes a value from its key (auto-association),
  gates writes on novelty + surprise, and evicts the least salient slot.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.pc_memory import (
    init_replay_params, init_replay_state, replay_store, Experience,
    replay_size, replay_recent_indices, replay_gather, replay_sample_indices,
    init_episodic_params, init_episodic_state,
    episodic_store, episodic_recall, episodic_size,
)


# ---------------------------------------------------------------------
# replay ring buffer
# ---------------------------------------------------------------------


def _store_n(state, params, n, *, s_dim, m_dim, fe=None):
    for i in range(n):
        f = float(i) if fe is None else float(fe[i])
        state = replay_store(
            state, params,
            Experience(jnp.ones(s_dim) * i, jnp.ones(m_dim) * (0.1 * i),
                       jnp.asarray(f)),
        )
    return state


def test_replay_ring_wraps_and_caps():
    p = init_replay_params(capacity=4, sensory_size=3, motor_size=2)
    s = init_replay_state(p)
    s = _store_n(s, p, 6, s_dim=3, m_dim=2)        # 6 writes into capacity 4

    assert int(replay_size(s)) == 4, "size must cap at capacity"
    # write_ptr wrapped: 6 % 4 == 2.
    assert int(s.write_ptr) == 2

    # Most-recent-first: last write was experience i=5 at slot (5 % 4)=1.
    idx = replay_recent_indices(s, p, 4)
    most_recent = replay_gather(s, idx)
    assert float(most_recent.sensory[0, 0]) == 5.0, "recent_indices not newest-first"
    assert float(most_recent.sensory[1, 0]) == 4.0


def test_replay_prioritised_favours_high_free_energy():
    p = init_replay_params(capacity=2, sensory_size=1, motor_size=1,
                           base_priority=0.01)
    s = init_replay_state(p)
    # Slot 0: low FE; slot 1: high FE.
    s = _store_n(s, p, 2, s_dim=1, m_dim=1, fe=[0.0, 100.0])

    idx = replay_sample_indices(s, p, jax.random.PRNGKey(0), 2000, prioritised=True)
    frac_high = float(jnp.mean((idx == 1).astype(jnp.float32)))
    assert frac_high > 0.9, f"surprise priority not honoured: {frac_high:.2f}"

    # Uniform mode ignores FE → roughly balanced.
    idu = replay_sample_indices(s, p, jax.random.PRNGKey(1), 2000, prioritised=False)
    frac_u = float(jnp.mean((idu == 1).astype(jnp.float32)))
    assert 0.4 < frac_u < 0.6, f"uniform sampling skewed: {frac_u:.2f}"


# ---------------------------------------------------------------------
# episodic store — pattern completion + gating
# ---------------------------------------------------------------------


def test_episodic_store_then_recall_completes_value():
    d = 8
    p = init_episodic_params(key_dim=d, value_dim=d, capacity=16,
                             similarity_thresh=0.8, gate_thresh=0.3)
    s = init_episodic_state(jax.random.PRNGKey(0), p)

    key_vec = jax.random.normal(jax.random.PRNGKey(1), (d,))
    value = jax.random.normal(jax.random.PRNGKey(2), (d,))
    out = episodic_store(s, p, key_vec, value, gate=1.0)
    assert bool(out.stored)
    assert int(episodic_size(out.state)) == 1

    # Recall with a noisy version of the key returns the stored value.
    noisy = key_vec + 0.05 * jax.random.normal(jax.random.PRNGKey(3), (d,))
    rec = episodic_recall(out.state, p, noisy)
    assert float(jnp.linalg.norm(rec.value - value)) < 1e-5, "completion wrong value"
    assert float(rec.similarity) > 0.8


def test_episodic_novelty_and_gate_block_writes():
    d = 8
    p = init_episodic_params(key_dim=d, value_dim=d, capacity=16,
                             similarity_thresh=0.8, gate_thresh=0.3)
    s = init_episodic_state(jax.random.PRNGKey(0), p)
    key_vec = jax.random.normal(jax.random.PRNGKey(1), (d,))

    # Below-gate surprise → no write.
    low = episodic_store(s, p, key_vec, key_vec, gate=0.1)
    assert not bool(low.stored)
    assert int(episodic_size(low.state)) == 0

    # First real write stores; an identical cue is no longer novel.
    first = episodic_store(s, p, key_vec, key_vec, gate=1.0)
    second = episodic_store(first.state, p, key_vec, key_vec, gate=1.0)
    assert bool(first.stored) and not bool(second.stored)
    assert int(episodic_size(second.state)) == 1


def test_episodic_evicts_least_salient_when_full():
    d = 8
    cap = 3
    # Denser DG code (dg_k > 1) + orthogonal basis keys ⇒ each episode is
    # cleanly separated and unambiguously novel.
    p = init_episodic_params(key_dim=d, value_dim=d, capacity=cap,
                             dg_sparsity=0.2, similarity_thresh=0.9,
                             gate_thresh=0.1)
    s = init_episodic_state(jax.random.PRNGKey(0), p)

    basis = jnp.eye(d)
    # Fill with distinct keys at increasing salience 0.2, 0.5, 0.9.
    sal = [0.2, 0.5, 0.9]
    for i in range(cap):
        s = episodic_store(s, p, basis[i], basis[i], gate=sal[i]).state
    assert int(episodic_size(s)) == cap

    # A new high-salience episode evicts the least salient (slot 0).
    out = episodic_store(s, p, basis[cap], basis[cap], gate=1.0)
    assert bool(out.stored)
    assert int(episodic_size(out.state)) == cap, "capacity must hold"
    # The 0.2-salience episode is gone; the 0.9 one survives.
    survived = episodic_recall(out.state, p, basis[2])
    assert float(survived.similarity) > 0.9, "high-salience memory wrongly evicted"
    evicted = episodic_recall(out.state, p, basis[0])
    assert float(evicted.similarity) < 0.9, "least-salient episode not evicted"
