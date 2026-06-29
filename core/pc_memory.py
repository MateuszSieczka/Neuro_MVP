"""Memory primitives for offline replay + episodic recall (Faza U, §3).

Two substrate-agnostic, JIT-safe stores that the predictive-coding graph
does not carry itself but that sleep (:mod:`core.pc_sleep`) and the
hippocampal node group (:mod:`core.pc_hippocampus`) build on:

* :class:`ReplayBuffer` — a fixed-capacity struct-of-arrays **ring
  buffer** of wake experiences ``(sensory, motor_belief, free_energy)``.
  Free energy is the stored salience: surprising experiences (high
  ``F``) are replayed preferentially (Schaul et al. 2016, |TD|-error
  prioritised replay, here at the free-energy timescale).  SWS reverse
  replay consumes it most-recent-first (Wilson & McNaughton 1994).

* :class:`EpisodicStore` — a content-addressable **one-shot** store: a
  cue is pattern-separated into a sparse DG code (random projection +
  top-k, Rolls 2013) and matched by cosine similarity for completion.
  Writes are gated by a novelty / surprise scalar and evict the least
  salient slot when full (McClelland, McNaughton & O'Reilly 1995 fast
  hippocampal weights).  Generic in key/value width so the hippocampal
  node group can use it auto-associatively (key = value = HC belief).

Both store the substrate's own quantities — a flat ``sensory`` rate
vector, the canonical pre-``tanh`` ``motor_belief``, the scalar free
energy ``F`` — and nothing from the discarded discrete/spiking world
(no integer action, no external reward, no neuromodulator level).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey


# =====================================================================
# Replay buffer — ring of (sensory, motor_belief, free_energy)
# =====================================================================


class ReplayParams(eqx.Module):
    """Static shape + the uniform sampling floor."""

    capacity: int = eqx.field(static=True)
    sensory_size: int = eqx.field(static=True)
    motor_size: int = eqx.field(static=True)
    base_priority: float = eqx.field(static=True)


def init_replay_params(
    capacity: int, sensory_size: int, motor_size: int,
    *, base_priority: float = 0.05,
) -> ReplayParams:
    """``base_priority`` is the uniform floor added to the free-energy
    sampling weight so a low-surprise experience is still replayed
    occasionally (avoids starving consolidated memories)."""
    return ReplayParams(
        capacity=int(capacity), sensory_size=int(sensory_size),
        motor_size=int(motor_size), base_priority=float(base_priority),
    )


class ReplayState(eqx.Module):
    """Struct-of-arrays ring buffer."""

    sensory: Array          # (capacity, sensory_size)
    motor_belief: Array     # (capacity, motor_size) — pre-tanh command
    free_energy: Array      # (capacity,) — salience / replay priority
    valid: Array            # (capacity,) bool
    write_ptr: Array        # scalar int32


def init_replay_state(params: ReplayParams, *, dtype=DTYPE) -> ReplayState:
    C = params.capacity
    return ReplayState(
        sensory=jnp.zeros((C, params.sensory_size), dtype),
        motor_belief=jnp.zeros((C, params.motor_size), dtype),
        free_energy=jnp.zeros(C, dtype),
        valid=jnp.zeros(C, dtype=bool),
        write_ptr=jnp.asarray(0, jnp.int32),
    )


class Experience(NamedTuple):
    """One wake step worth remembering for offline replay."""

    sensory: Array          # (sensory_size,)
    motor_belief: Array     # (motor_size,) pre-tanh
    free_energy: Array      # scalar


def replay_store(
    state: ReplayState, params: ReplayParams, exp: Experience,
) -> ReplayState:
    """Append one experience at ``write_ptr`` (overwrites oldest)."""
    ptr = state.write_ptr
    return ReplayState(
        sensory=state.sensory.at[ptr].set(exp.sensory.astype(DTYPE)),
        motor_belief=state.motor_belief.at[ptr].set(exp.motor_belief.astype(DTYPE)),
        free_energy=state.free_energy.at[ptr].set(
            jnp.asarray(exp.free_energy, DTYPE)),
        valid=state.valid.at[ptr].set(True),
        write_ptr=(ptr + 1) % params.capacity,
    )


def replay_size(state: ReplayState) -> Array:
    """Number of valid entries (scalar int32)."""
    return state.valid.sum().astype(jnp.int32)


def replay_sample_indices(
    state: ReplayState, params: ReplayParams, key: PRNGKey, n: int,
    *, prioritised: bool = True,
) -> Array:
    """Sample ``n`` slot indices, weighted by free energy when prioritised.

    Surprise-prioritised (Schaul 2016): weight ``= F + base_priority`` on
    valid slots.  Falls back to uniform over valid slots when
    ``prioritised`` is off or every weight is zero.  Samples with
    replacement so a buffer holding fewer than ``n`` entries still returns
    ``n`` indices (duplicate replay is sign-idempotent under the one rule).
    """
    valid_f = state.valid.astype(DTYPE)
    if prioritised:
        weights = valid_f * (state.free_energy + params.base_priority)
    else:
        weights = valid_f
    total = weights.sum()
    probs = jnp.where(
        total > 0,
        weights / (total + 1e-12),
        valid_f / (valid_f.sum() + 1e-12),
    )
    return jax.random.choice(
        key, params.capacity, shape=(n,), replace=True, p=probs,
    )


def replay_recent_indices(
    state: ReplayState, params: ReplayParams, n: int,
) -> Array:
    """The last ``n`` written slots, most-recent first (SWS reverse replay).

    Wilson & McNaughton (1994): NREM reactivates the most recent
    trajectory in reverse order.  ``n`` must be ≤ capacity; when the
    buffer is not yet full the indices wrap into still-invalid slots, so
    callers gate on :func:`replay_size` if exact recency matters.
    """
    offsets = jnp.arange(n, dtype=jnp.int32)
    return (state.write_ptr - 1 - offsets) % params.capacity


def replay_gather(state: ReplayState, idx: Array) -> Experience:
    """Gather a batch of experiences by index."""
    return Experience(
        sensory=state.sensory[idx],
        motor_belief=state.motor_belief[idx],
        free_energy=state.free_energy[idx],
    )


def replay_clear(params: ReplayParams) -> ReplayState:
    """Empty the buffer (keep shapes)."""
    return init_replay_state(params)


# =====================================================================
# Episodic store — DG pattern-separated, content-addressable, one-shot
# =====================================================================


class EpisodicParams(eqx.Module):
    """Static sizes + the novelty gate / sparsity hyper-params."""

    similarity_thresh: Array     # cosine ≥ this ⇒ not novel ⇒ no write
    gate_thresh: Array           # write only when the surprise gate ≥ this
    capacity: int = eqx.field(static=True)
    key_dim: int = eqx.field(static=True)
    value_dim: int = eqx.field(static=True)
    dg_dim: int = eqx.field(static=True)
    dg_k: int = eqx.field(static=True)


def init_episodic_params(
    key_dim: int, value_dim: int,
    *,
    capacity: int = 256,
    dg_expansion_factor: int = 5,
    dg_sparsity: float = 0.05,
    similarity_thresh: float = 0.85,
    gate_thresh: float = 0.3,
) -> EpisodicParams:
    """``dg_dim = key_dim · expansion``; ``dg_k`` is the top-k active count.

    A cue is novel when its best cosine match is below
    ``similarity_thresh``; a write also requires the surprise gate to
    exceed ``gate_thresh`` (the substrate's free-energy / novelty signal,
    replacing the discarded NE gate)."""
    dg_dim = int(key_dim) * int(dg_expansion_factor)
    dg_k = max(1, int(dg_sparsity * dg_dim))
    f = lambda x: jnp.asarray(x, DTYPE)
    return EpisodicParams(
        similarity_thresh=f(similarity_thresh),
        gate_thresh=f(gate_thresh),
        capacity=int(capacity), key_dim=int(key_dim), value_dim=int(value_dim),
        dg_dim=dg_dim, dg_k=dg_k,
    )


class EpisodicState(eqx.Module):
    """Fixed-capacity SoA store + the fixed random DG projection."""

    dg_projection: Array         # (key_dim, dg_dim)
    keys: Array                  # (capacity, dg_dim) sparse binary DG codes
    values: Array                # (capacity, value_dim)
    saliences: Array             # (capacity,) gate value at encoding
    valid: Array                 # (capacity,) bool
    write_ptr: Array             # scalar int32 — next empty slot, capped


def init_episodic_state(
    key: PRNGKey, params: EpisodicParams, *, dtype=DTYPE,
) -> EpisodicState:
    """Gaussian DG projection with unit-variance columns (Rolls 2013)."""
    dg = jax.random.normal(
        key, (params.key_dim, params.dg_dim), dtype=dtype,
    ) / jnp.sqrt(jnp.asarray(params.key_dim, dtype))
    C = params.capacity
    return EpisodicState(
        dg_projection=dg,
        keys=jnp.zeros((C, params.dg_dim), dtype),
        values=jnp.zeros((C, params.value_dim), dtype),
        saliences=jnp.zeros(C, dtype),
        valid=jnp.zeros(C, dtype=bool),
        write_ptr=jnp.asarray(0, jnp.int32),
    )


def dg_encode(state: EpisodicState, params: EpisodicParams, cue: Array) -> Array:
    """``cue → sparse binary DG code`` via top-k thresholding (separation)."""
    projected = cue.astype(DTYPE) @ state.dg_projection
    threshold = jnp.sort(projected)[::-1][params.dg_k - 1]
    return (projected >= threshold).astype(DTYPE)


def _cos_sim(a: Array, B: Array, valid: Array) -> Array:
    """Cosine similarity of ``a`` against each valid row of ``B`` (else −1)."""
    a_norm = jnp.linalg.norm(a) + 1e-8
    B_norm = jnp.linalg.norm(B, axis=1) + 1e-8
    sim = (B @ a) / (B_norm * a_norm)
    return jnp.where(valid, sim, jnp.asarray(-1.0, DTYPE))


class StoreOutput(NamedTuple):
    state: EpisodicState
    stored: Array                # bool scalar
    slot: Array                  # int32 scalar — slot used


def episodic_store(
    state: EpisodicState, params: EpisodicParams,
    key_input: Array, value: Array, gate: Array | float,
) -> StoreOutput:
    """Novelty- and gate-gated one-shot write (JIT-safe, branchless).

    Stores ``value`` under the DG code of ``key_input`` when the cue is
    novel *and* the surprise ``gate`` clears ``gate_thresh``.  Picks the
    first empty slot, else evicts the least-salient slot (interference
    forgetting).  ``gate`` is recorded as the slot's salience.
    """
    g = jnp.asarray(gate, DTYPE)
    dg = dg_encode(state, params, key_input)
    sims = _cos_sim(dg, state.keys, state.valid)
    novel = jnp.max(sims) < params.similarity_thresh
    stored = novel & (g >= params.gate_thresh)

    empty_mask = ~state.valid
    any_empty = jnp.any(empty_mask)
    empty_slot = jnp.argmax(empty_mask.astype(jnp.int32))
    sal_key = jnp.where(state.valid, state.saliences, jnp.asarray(jnp.inf, DTYPE))
    least_salient = jnp.argmin(sal_key)
    slot = jnp.where(any_empty, empty_slot, least_salient).astype(jnp.int32)

    def _w_row(arr, row):
        return jnp.where(stored, arr.at[slot].set(row), arr)

    def _w_scalar(arr, val):
        return jnp.where(stored, arr.at[slot].set(val), arr)

    new_state = EpisodicState(
        dg_projection=state.dg_projection,
        keys=_w_row(state.keys, dg),
        values=_w_row(state.values, value.astype(DTYPE)),
        saliences=_w_scalar(state.saliences, g),
        valid=_w_scalar(state.valid, jnp.asarray(True)),
        write_ptr=jnp.minimum(
            state.write_ptr + stored.astype(jnp.int32) * any_empty.astype(jnp.int32),
            jnp.asarray(params.capacity, jnp.int32),
        ),
    )
    return StoreOutput(state=new_state, stored=stored, slot=slot)


class RecallOutput(NamedTuple):
    value: Array                 # (value_dim,) best-match stored value
    similarity: Array            # scalar cosine of the match (−1 ⇒ empty)


def episodic_recall(
    state: EpisodicState, params: EpisodicParams, cue: Array,
) -> RecallOutput:
    """Pattern completion: return the stored value of the best DG match.

    The cue is pattern-separated, matched by cosine against valid keys,
    and the most similar slot's value is returned (Treves & Rolls 1994
    auto-associative completion).  ``similarity`` is ``−1`` on an empty
    store; callers gate on it when a positive match is required.
    """
    dg = dg_encode(state, params, cue)
    sims = _cos_sim(dg, state.keys, state.valid)
    best = jnp.argmax(sims)
    return RecallOutput(value=state.values[best], similarity=sims[best])


def episodic_size(state: EpisodicState) -> Array:
    """Number of currently valid episodes."""
    return state.valid.sum().astype(jnp.int32)
