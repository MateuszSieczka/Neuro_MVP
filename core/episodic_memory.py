"""Hippocampal episodic memory — pure JAX.

Rolls (2013) “Pattern completion and separation in the hippocampus”;
O’Neill et al. (2010); McClelland et al. (1995).

Fixed-capacity structure-of-arrays:
- ``keys``          (capacity, dg_dim)  DG-sparse encodings
- ``states``        (capacity, state_dim)
- ``next_states``   (capacity, state_dim)
- ``actions``       (capacity,)          integer (one-hot indices fit here;
                                          continuous actions stored separately)
- ``rewards``       (capacity,)
- ``saliences``     (capacity,)
- ``replay_counts`` (capacity,)
- ``valid``         (capacity,)          bool mask
- ``write_ptr``     scalar               next eligible empty slot

Legacy behaviour preserved:
- NE-gated storage (skip when ``ne_level < ne_threshold``).
- Novelty check via cosine similarity against all valid stored DG keys.
- Interference-based forgetting: overwrite the most-similar
  *non-consolidated* memory (``replay_count < consolidation_threshold``);
  if none qualify, overwrite the least-salient slot.

JAX differences:
- Python ``list.append`` → scalar ``write_ptr`` that saturates at
  ``capacity``; subsequent writes go through the interference rule.
- The whole write path is JIT-safe (no Python conditions).
- ``try_store`` returns a new state plus a boolean indicator.
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


class EpisodicParams(eqx.Module):
    """Static hyperparameters + fixed sizes."""

    ne_threshold: Array
    similarity_thresh: Array    # (1 − similarity) → novelty
    dg_sparsity: Array
    consolidation_threshold: Array  # scalar int replay count
    capacity: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    dg_dim: int = eqx.field(static=True)
    dg_k: int = eqx.field(static=True)


def init_episodic_params(
    state_dim: int,
    *,
    capacity: int = 500,
    ne_threshold: float = 0.3,
    similarity_thresh: float = 0.85,
    dg_sparsity: float = 0.05,
    dg_expansion_factor: int = 5,
    consolidation_threshold: int = 3,
) -> EpisodicParams:
    dg_dim = state_dim * dg_expansion_factor
    dg_k = max(1, int(dg_sparsity * dg_dim))
    f = lambda x: jnp.asarray(x, DTYPE)
    return EpisodicParams(
        ne_threshold=f(ne_threshold),
        similarity_thresh=f(similarity_thresh),
        dg_sparsity=f(dg_sparsity),
        consolidation_threshold=jnp.asarray(consolidation_threshold, jnp.int32),
        capacity=capacity, state_dim=state_dim,
        dg_dim=dg_dim, dg_k=dg_k,
    )


class EpisodicState(eqx.Module):
    """Fixed-capacity SoA buffer + DG projection matrix."""

    dg_projection: Array         # (state_dim, dg_dim)
    keys: Array                  # (capacity, dg_dim)
    states: Array                # (capacity, state_dim)
    next_states: Array           # (capacity, state_dim)
    actions: Array               # (capacity,) int32
    rewards: Array               # (capacity,)
    saliences: Array             # (capacity,)
    replay_counts: Array         # (capacity,) int32
    valid: Array                 # (capacity,) bool
    write_ptr: Array             # scalar int32 — index of next empty slot


def init_episodic_state(
    key: PRNGKey, params: EpisodicParams, *, dtype=DTYPE,
) -> EpisodicState:
    """Gaussian DG projection with unit-variance columns (Rolls 2013)."""
    dg = jax.random.normal(
        key, (params.state_dim, params.dg_dim), dtype=dtype,
    ) / jnp.sqrt(jnp.asarray(params.state_dim, dtype))
    C = params.capacity
    return EpisodicState(
        dg_projection=dg,
        keys=jnp.zeros((C, params.dg_dim), dtype=dtype),
        states=jnp.zeros((C, params.state_dim), dtype=dtype),
        next_states=jnp.zeros((C, params.state_dim), dtype=dtype),
        actions=jnp.zeros(C, dtype=jnp.int32),
        rewards=jnp.zeros(C, dtype=dtype),
        saliences=jnp.zeros(C, dtype=dtype),
        replay_counts=jnp.zeros(C, dtype=jnp.int32),
        valid=jnp.zeros(C, dtype=jnp.bool_),
        write_ptr=jnp.asarray(0, jnp.int32),
    )


# =====================================================================
# DG encoding
# =====================================================================


def dg_encode(state: EpisodicState, params: EpisodicParams, x: Array) -> Array:
    """``state → sparse binary DG code`` via top-k thresholding."""
    projected = x.astype(DTYPE) @ state.dg_projection
    # Top-k threshold: the (k)-th largest value.
    sorted_desc = jnp.sort(projected)[::-1]
    threshold = sorted_desc[params.dg_k - 1]
    return (projected >= threshold).astype(DTYPE)


def _cos_sim(a: Array, B: Array, valid: Array) -> Array:
    """Cosine sim of ``a`` against each row of ``B``, masked by ``valid``.

    Invalid rows return ``-1`` (can never beat a valid positive sim).
    """
    a_norm = jnp.linalg.norm(a) + 1e-8
    B_norm = jnp.linalg.norm(B, axis=1) + 1e-8
    sim = (B @ a) / (B_norm * a_norm)
    return jnp.where(valid, sim, jnp.asarray(-1.0, DTYPE))


# =====================================================================
# Storage
# =====================================================================


class StoreOutput(NamedTuple):
    state: EpisodicState
    stored: Array                # bool scalar
    slot: Array                  # int32 scalar — index used (−0 if no-op)


def try_store(
    state: EpisodicState, params: EpisodicParams,
    s: Array, a: int | Array, r: float | Array, s_next: Array,
    ne_level: float | Array,
) -> StoreOutput:
    """NE-gated, novelty-gated, interference-forgetting write.

    Decision tree (all JIT-safe via masks):
      1. ``stored = (ne ≥ ne_threshold) AND novel``.
      2. ``slot =`` first empty if any; else most-similar non-consolidated;
         else least-salient slot overall.
      3. If ``stored``: write all SoA fields at ``slot``.

    Returns the new state plus a boolean indicator and the slot used.
    """
    ne = jnp.asarray(ne_level, DTYPE)
    s_f = s.astype(DTYPE)
    s_next_f = s_next.astype(DTYPE)
    a_int = jnp.asarray(a, jnp.int32)
    r_f = jnp.asarray(r, DTYPE)

    key = dg_encode(state, params, s_f)
    sims = _cos_sim(key, state.keys, state.valid)
    max_sim = jnp.max(sims)
    novel = max_sim < params.similarity_thresh

    ne_gate = ne >= params.ne_threshold
    stored = ne_gate & novel

    # Pick slot.
    empty_mask = ~state.valid
    any_empty = jnp.any(empty_mask)
    empty_slot = jnp.argmax(empty_mask.astype(jnp.int32))

    non_consolidated = state.replay_counts < params.consolidation_threshold
    cand_mask = state.valid & non_consolidated
    # Sim subset: valid∩non-consolidated; others ⇒ -1.
    cand_sims = jnp.where(cand_mask, sims, jnp.asarray(-1.0, DTYPE))
    any_candidate = jnp.any(cand_mask)
    cand_slot = jnp.argmax(cand_sims)

    # Fallback: least-salient overall valid slot.
    sal_key = jnp.where(state.valid, state.saliences, jnp.asarray(jnp.inf, DTYPE))
    least_salient = jnp.argmin(sal_key)

    slot = jnp.where(
        any_empty, empty_slot,
        jnp.where(any_candidate, cand_slot, least_salient),
    ).astype(jnp.int32)

    # Masked write: if stored, overwrite the slot; else identity update.
    def _set_row(arr: Array, row: Array, idx: Array) -> Array:
        return arr.at[idx].set(row)

    def _set_scalar(arr: Array, val: Array, idx: Array) -> Array:
        return arr.at[idx].set(val)

    keys_new = jnp.where(
        stored, _set_row(state.keys, key, slot), state.keys,
    )
    states_new = jnp.where(
        stored, _set_row(state.states, s_f, slot), state.states,
    )
    nexts_new = jnp.where(
        stored, _set_row(state.next_states, s_next_f, slot), state.next_states,
    )
    actions_new = jnp.where(
        stored, _set_scalar(state.actions, a_int, slot), state.actions,
    )
    rewards_new = jnp.where(
        stored, _set_scalar(state.rewards, r_f, slot), state.rewards,
    )
    sal_val = jnp.clip(ne, 0.0, 1.0)
    sals_new = jnp.where(
        stored, _set_scalar(state.saliences, sal_val, slot), state.saliences,
    )
    # Reset replay count on overwrite.
    rc_new = jnp.where(
        stored,
        _set_scalar(state.replay_counts, jnp.asarray(0, jnp.int32), slot),
        state.replay_counts,
    )
    valid_new = jnp.where(
        stored, _set_scalar(state.valid, jnp.asarray(True), slot), state.valid,
    )
    # write_ptr tracks empty-slot count (monotone, capped at capacity).
    wp_new = jnp.minimum(
        state.write_ptr + stored.astype(jnp.int32) * any_empty.astype(jnp.int32),
        jnp.asarray(params.capacity, jnp.int32),
    )

    new_state = eqx.tree_at(
        lambda st: (
            st.keys, st.states, st.next_states, st.actions, st.rewards,
            st.saliences, st.replay_counts, st.valid, st.write_ptr,
        ),
        state,
        (
            keys_new, states_new, nexts_new, actions_new, rewards_new,
            sals_new, rc_new, valid_new, wp_new,
        ),
    )
    return StoreOutput(state=new_state, stored=stored, slot=slot)


# =====================================================================
# Recall + consolidation
# =====================================================================


class RecallOutput(NamedTuple):
    indices: Array           # (top_k,)
    similarities: Array      # (top_k,)


def recall(
    state: EpisodicState, params: EpisodicParams,
    cue: Array, top_k: int = 1,
) -> RecallOutput:
    """Return ``top_k`` slot indices sorted by cosine similarity.

    Invalid rows score ``-1``; callers must gate on ``similarities`` if
    they need a positive match.
    """
    cue_key = dg_encode(state, params, cue)
    sims = _cos_sim(cue_key, state.keys, state.valid)
    # Use negative sort to get descending order.
    neg_sorted_idx = jnp.argsort(-sims)
    idx = neg_sorted_idx[:top_k]
    return RecallOutput(indices=idx, similarities=sims[idx])


def mark_replayed(state: EpisodicState, idx: Array) -> EpisodicState:
    """Increment ``replay_counts[idx]`` (for consolidation bookkeeping)."""
    rc = state.replay_counts.at[idx].add(1)
    return eqx.tree_at(lambda s: s.replay_counts, state, rc)


def episodic_size(state: EpisodicState) -> Array:
    """Number of currently valid episodes."""
    return jnp.sum(state.valid.astype(jnp.int32))


def episodic_clear(state: EpisodicState, params: EpisodicParams) -> EpisodicState:
    """Drop all episodes; keep the DG projection."""
    C = params.capacity
    return eqx.tree_at(
        lambda s: (
            s.keys, s.states, s.next_states, s.actions, s.rewards,
            s.saliences, s.replay_counts, s.valid, s.write_ptr,
        ),
        state,
        (
            jnp.zeros((C, params.dg_dim), DTYPE),
            jnp.zeros((C, params.state_dim), DTYPE),
            jnp.zeros((C, params.state_dim), DTYPE),
            jnp.zeros(C, jnp.int32),
            jnp.zeros(C, DTYPE),
            jnp.zeros(C, DTYPE),
            jnp.zeros(C, jnp.int32),
            jnp.zeros(C, jnp.bool_),
            jnp.asarray(0, jnp.int32),
        ),
    )
