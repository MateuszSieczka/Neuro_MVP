"""Experience replay buffer — pure JAX.

Fixed-capacity struct-of-arrays ring buffer with JIT-safe writes and
sampling. This module intentionally exports *data primitives only*:
the orchestration of SWS reverse-replay / REM forward-replay is the
responsibility of the brain-graph layer (Phase 2) since it depends on
basal-ganglia + world-model coupling.

Stored fields per experience:
- ``state``        continuous state vector at step t
- ``action``       int32 scalar
- ``reward``       float scalar (external reward)
- ``next_state``   continuous state vector at step t+1
- ``prediction_error``  scalar PE from world model (for salience)
- ``done``         bool / float 0.0 or 1.0
- ``salience``     scalar ∈ [0, 1] (sampling weight)
- ``recorded_da``  dopamine level at encoding (for phase-matched replay)

Sampling: priority ~ salience + base_prob, via ``jax.random.choice``
with explicit probability weights. Consolidation fields
(``replay_count``, ``consolidated``) support optional prioritised decay.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey


# =====================================================================
# Params / state
# =====================================================================


class ReplayParams(eqx.Module):
    """Static shape / schedule parameters."""

    capacity: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    base_sample_prob: float = eqx.field(static=True)
    salience_floor: float = eqx.field(static=True)


def init_replay_params(
    capacity: int, state_size: int,
    *, base_sample_prob: float = 0.05, salience_floor: float = 0.05,
) -> ReplayParams:
    return ReplayParams(
        capacity=capacity, state_size=state_size,
        base_sample_prob=base_sample_prob,
        salience_floor=salience_floor,
    )


class ReplayState(eqx.Module):
    """Struct-of-arrays ring buffer."""

    state: Array            # (capacity, state_size)
    action: Array           # (capacity,) int32
    reward: Array           # (capacity,)
    next_state: Array       # (capacity, state_size)
    prediction_error: Array # (capacity,)
    done: Array             # (capacity,)
    salience: Array         # (capacity,)
    recorded_da: Array      # (capacity,)
    replay_count: Array     # (capacity,) int32
    consolidated: Array     # (capacity,) bool
    valid: Array            # (capacity,) bool
    write_ptr: Array        # scalar int32


def init_replay_state(params: ReplayParams, dtype=DTYPE) -> ReplayState:
    C, S = params.capacity, params.state_size
    return ReplayState(
        state=jnp.zeros((C, S), dtype),
        action=jnp.zeros(C, jnp.int32),
        reward=jnp.zeros(C, dtype),
        next_state=jnp.zeros((C, S), dtype),
        prediction_error=jnp.zeros(C, dtype),
        done=jnp.zeros(C, dtype),
        salience=jnp.zeros(C, dtype),
        recorded_da=jnp.zeros(C, dtype),
        replay_count=jnp.zeros(C, jnp.int32),
        consolidated=jnp.zeros(C, dtype=bool),
        valid=jnp.zeros(C, dtype=bool),
        write_ptr=jnp.asarray(0, jnp.int32),
    )


class Experience(NamedTuple):
    """Per-step experience record (arrays with leading batch optional)."""

    state: Array
    action: Array          # int32 scalar
    reward: Array          # scalar
    next_state: Array
    prediction_error: Array
    done: Array
    salience: Array
    recorded_da: Array


# =====================================================================
# Store / sample
# =====================================================================


def replay_store(
    state: ReplayState, params: ReplayParams, exp: Experience,
) -> ReplayState:
    """Append one experience at ``write_ptr`` (ring buffer)."""
    ptr = state.write_ptr
    sal = jnp.maximum(exp.salience.astype(DTYPE), params.salience_floor)
    return ReplayState(
        state=state.state.at[ptr].set(exp.state.astype(DTYPE)),
        action=state.action.at[ptr].set(exp.action.astype(jnp.int32)),
        reward=state.reward.at[ptr].set(exp.reward.astype(DTYPE)),
        next_state=state.next_state.at[ptr].set(exp.next_state.astype(DTYPE)),
        prediction_error=state.prediction_error.at[ptr].set(
            exp.prediction_error.astype(DTYPE)),
        done=state.done.at[ptr].set(exp.done.astype(DTYPE)),
        salience=state.salience.at[ptr].set(sal),
        recorded_da=state.recorded_da.at[ptr].set(
            exp.recorded_da.astype(DTYPE)),
        replay_count=state.replay_count.at[ptr].set(0),
        consolidated=state.consolidated.at[ptr].set(False),
        valid=state.valid.at[ptr].set(True),
        write_ptr=(ptr + 1) % params.capacity,
    )


def replay_size(state: ReplayState) -> Array:
    """Number of valid entries (scalar int32)."""
    return state.valid.sum().astype(jnp.int32)


def replay_sample_indices(
    state: ReplayState, params: ReplayParams,
    key: PRNGKey, n: int,
    *, prioritised: bool = True,
) -> Array:
    """Sample ``n`` indices weighted by salience (+ uniform floor).

    If ``prioritised=False`` samples uniformly over valid slots.
    """
    valid_f = state.valid.astype(DTYPE)
    if prioritised:
        weights = (
            valid_f
            * (state.salience + params.base_sample_prob)
            * (1.0 - 0.7 * state.consolidated.astype(DTYPE))
        )
    else:
        weights = valid_f
    total = weights.sum()
    probs = jnp.where(total > 0, weights / (total + 1e-12), valid_f / (valid_f.sum() + 1e-12))
    return jax.random.choice(
        key, params.capacity, shape=(n,), replace=True, p=probs,
    )


def replay_gather(state: ReplayState, idx: Array) -> Experience:
    """Gather a batch of experiences by index."""
    return Experience(
        state=state.state[idx],
        action=state.action[idx],
        reward=state.reward[idx],
        next_state=state.next_state[idx],
        prediction_error=state.prediction_error[idx],
        done=state.done[idx],
        salience=state.salience[idx],
        recorded_da=state.recorded_da[idx],
    )


def replay_mark_replayed(
    state: ReplayState, idx: Array,
    *, consolidation_threshold: int = 3,
) -> ReplayState:
    """Increment ``replay_count`` and flag consolidated above threshold."""
    new_counts = state.replay_count.at[idx].add(1)
    consolidated = new_counts >= consolidation_threshold
    return eqx.tree_at(
        lambda s: (s.replay_count, s.consolidated),
        state, (new_counts, consolidated),
    )


def replay_recent_indices(
    state: ReplayState, params: ReplayParams, n: int,
) -> Array:
    """Return the last ``n`` *valid* write positions in descending order.

    Useful for SWS reverse replay: the most recent trajectory goes first.
    ``n`` must be <= capacity; invalid slots are replaced by the ring
    position before them.
    """
    ptr = state.write_ptr
    offsets = jnp.arange(n, dtype=jnp.int32)
    idx = (ptr - 1 - offsets) % params.capacity
    return idx


def replay_clear(state: ReplayState, params: ReplayParams) -> ReplayState:
    """Empty the buffer (keep shapes)."""
    return init_replay_state(params)
