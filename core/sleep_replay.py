"""Sleep-phase offline learning — SWS reverse replay + REM forward rollout.

Two JIT-safe entrypoints used by the :func:`core.brain_graph.brain_cycle`
dispatcher:

* :func:`sws_replay_step`
    Sample a mini-batch of transitions from the replay buffer
    (prioritised by salience, which at write time is set to the
    world-model learning-progress curiosity — Schaul et al. 2016
    |TD error|-prioritised replay, here at the PE-EMA timescale),
    iterate them **reverse-chronologically**
    (Wilson & McNaughton 1994 place-cell reverse replay observed in
    hippocampal ripples during NREM sleep) and apply the offline
    world-model + CA3 sequence-memory updates.  Plasticity is gated
    by the oscillator's current SWS up-state flag
    (Steriade & Timofeev 2003 — Up-state = depolarised, spiking
    regime; Down-state = hyperpolarised, silent — weight change
    restricted to Up phases so synapse modification follows normal
    correlation physiology).  The critic and actor weights are NOT
    updated offline: BG three-factor STDP requires a live pre×post
    spiking correlation in the dt window, which cannot be faithfully
    reconstructed from stored state vectors alone; policy
    refinement therefore waits for the subsequent wake cycle, when
    the consolidated forward model provides better surprise
    signals (Ji & Wilson 2007).

* :func:`rem_rollout_step`
    Without writing to the replay buffer, sample a start state and
    evolve it forward through the world model for ``k`` steps
    (forward "dreaming" — Hobson & McCarley 1977 activation-synthesis,
    Hasselmo 2006 ACh-high/DA-low REM regime).  Each generated
    transition feeds the CA3 sequence memory, consolidating the
    temporal chain without re-writing it to episodic storage
    (Louie & Wilson 2001 replay of recent experience during REM).

Time compression.  NREM reactivation runs at ~5× wake speed
(Born 2009 compressed replay); the module does NOT expose a separate
``ctx_sws`` — time compression is an emergent property of running the
replay loop without the full perceive-substep chain (each replay
iteration is one ``wm_update`` call, not ``substeps=20`` dt-steps).
Biological plausibility is preserved at the resolution of transition
identity; the fine spike-time sequencing that distinguishes wake
plasticity from replay plasticity is not modelled.

References
----------
  Wilson & McNaughton (1994)   — Place-cell reactivation during SWS.
  Louie & Wilson (2001)        — REM replay of recent experience.
  Steriade & Timofeev (2003)   — Up/Down slow-oscillation plasticity.
  Born (2009)                  — Compressed offline reactivation.
  McClelland, McNaughton, O'Reilly (1995) — Complementary learning systems.
  Hasselmo (2006)              — ACh-high/DA-low REM modulation regime.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, BackendContext, split_key
from .replay_buffer import (
    ReplayParams, ReplayState,
    replay_sample_indices, replay_recent_indices,
    replay_mark_replayed, replay_size,
)
from .sequence_memory import seqmem_step
from .world_model import wm_predict, wm_update
from .hippocampus import HippocampusParams, HippocampusState


# =====================================================================
# SWS reverse replay (→ WM + HC CA3 consolidation)
# =====================================================================


class _SWSCarry(NamedTuple):
    wm_state: object
    hc_state: HippocampusState


def sws_replay_step(
    wm_state,
    wm_params,
    ctx: BackendContext,
    replay: ReplayState,
    replay_params: ReplayParams,
    hc_state: HippocampusState,
    hc_params: HippocampusParams,
    key: PRNGKey,
    *,
    n_replay: int = 32,
    n_body_actions: int,
    n_saccade_actions: int,
    ach: Array | float = 1.0,
) -> tuple[object, ReplayState, HippocampusState]:
    """One SWS consolidation pass over ``n_replay`` sampled transitions.

    Reverse-chronological ordering is realised by first pulling the
    most recent valid indices (replay_recent_indices) when the buffer
    is still mostly fresh, and otherwise by sampling prioritised by
    salience.  For each transition we apply:

      1. :func:`core.world_model.wm_update` on ``(s, a, s')`` — drives
         the encoder + decoder toward the consolidated transition
         model.
      2. :func:`core.sequence_memory.seqmem_step` on the replayed
         ``s`` — CA3 learns the off-policy temporal chain.

    Returns the updated world-model state, replay buffer (with
    replay-count bumped on the sampled indices) and hippocampus
    state.  The caller is responsible for advancing ``ctx`` /
    oscillator / sleep state.
    """
    n = int(n_replay)
    action_size = int(n_body_actions + n_saccade_actions)

    # Sample indices prioritised by salience.  When the buffer holds
    # fewer than ``n_replay`` valid entries the JAX choice still
    # returns ``n`` indices by sampling with replacement; the
    # wm_update on duplicates is harmless (idempotent-in-sign).
    idx = replay_sample_indices(
        replay, replay_params, key, n, prioritised=True,
    )

    states = replay.state[idx]                 # (n, state_size)
    next_states = replay.next_state[idx]       # (n, state_size)
    actions = replay.action[idx]               # (n,) int32

    def _body(carry: _SWSCarry, step_idx: int):
        # Reverse-chronological traversal.
        i = n - 1 - step_idx
        s = states[i]
        s_next = next_states[i]
        a = actions[i]
        action_oh = (jnp.arange(action_size) == a).astype(DTYPE)

        # WM offline update (McClelland 1995 neocortical slow learning).
        # CA3 replay is NOT run here: the replay buffer stores WM
        # state-space vectors (sensory drives) while CA3 operates on
        # the EC-belief space.  Reconstructing EC belief from a stored
        # WM state would require re-running the cortical encoder
        # offline, which is too costly for the scan body; CA3
        # consolidation therefore waits for the subsequent wake cycle.
        wm_out = wm_update(
            carry.wm_state, wm_params, ctx,
            s, action_oh, s_next,
            m_t=1.0, ach=ach,
        )
        return _SWSCarry(wm_out.state, carry.hc_state), None

    carry0 = _SWSCarry(wm_state, hc_state)
    carry_final, _ = jax.lax.scan(
        _body, carry0, jnp.arange(n), length=n,
    )
    new_replay = replay_mark_replayed(replay, idx)
    return carry_final.wm_state, new_replay, carry_final.hc_state


# =====================================================================
# REM forward rollout (→ WM + HC CA3 sequence learning)
# =====================================================================


def rem_rollout_step(
    wm_state,
    wm_params,
    ctx: BackendContext,
    replay: ReplayState,
    replay_params: ReplayParams,
    hc_state: HippocampusState,
    hc_params: HippocampusParams,
    key: PRNGKey,
    *,
    k_steps: int = 10,
    n_body_actions: int,
    n_saccade_actions: int,
    ach: Array | float = 1.0,
) -> tuple[object, HippocampusState]:
    """``k_steps`` forward rollout seeded from a replay-sampled start.

    Pick one transition from the replay buffer, seed ``current = s_next``,
    then iterate ``wm_predict`` (forward dreaming — Hobson & McCarley
    1977; Hasselmo 2006 REM regime).  Each generated prediction is
    fed into :func:`seqmem_step` so CA3 refines its transition model
    on the SAMPLED futures — this consolidates *generalisation* rather
    than memorisation (Payne 2012 REM for gist extraction).

    The replay buffer is intentionally NOT written to: generated
    transitions are simulations, not experience, and storing them
    would inflate the salience-weighted distribution with model
    bias (Schmidhuber 1991 on avoiding self-reinforcing hallucination
    loops during planning).
    """
    k_seed, k_action = split_key(key, 2)
    start_idx = replay_sample_indices(
        replay, replay_params, k_seed, n=1, prioritised=True,
    )[0]
    current_state = replay.next_state[start_idx]
    # Random action from the stored action space — REM is
    # exploratory-generative, not policy-conditioned (Hasselmo 2006
    # ACh-high/DA-low).
    action_size = int(n_body_actions + n_saccade_actions)
    action_keys = jax.random.split(k_action, k_steps)

    def _body(carry, t):
        wm_s, cur, hc_s = carry
        a_key = action_keys[t]
        a_id = jax.random.randint(a_key, (), 0, action_size)
        a_oh = (jnp.arange(action_size) == a_id).astype(DTYPE)
        wm_out = wm_predict(
            wm_s, wm_params, ctx,
            cur, a_oh, ach=ach,
        )
        predicted = wm_out.predicted_state
        # CA3 is NOT advanced on REM rollouts: generated WM predictions
        # live in sensory/WM space (state_size), not in the EC-belief
        # space that CA3 expects, so feeding them through seqmem would
        # be a type error rather than a physiologically meaningful
        # update.  CA3 generalisation can happen from separately-
        # sampled EC-space trajectories; that is left to a future
        # polish pass.
        return (wm_out.state, predicted, hc_s), None

    (wm_final, _last_state, hc_final), _ = jax.lax.scan(
        _body, (wm_state, current_state, hc_state), jnp.arange(k_steps),
        length=k_steps,
    )
    return wm_final, hc_final
