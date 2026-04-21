"""Learning pipeline — pure plasticity primitives shared by wake + sleep.

One decision cycle emits four plasticity updates:

* critic (ventral striatum) — TD value update modulated by RPE.
* actor_body / actor_saccade — parallel D1/D2 loops each receiving the
  same RPE augmented with a modality-specific epistemic bonus.
* cortex — three-factor STDP commit with RPE as the neuromodulator.
* attention — Hebbian top-down gain learning.

Factoring these four calls out of
:func:`core.brain_graph.action_brain_cognitive_step` gives a single
surface that :mod:`core.sleep` (Phase 5B) can invoke with
replay-sampled (s, a, r, s') tuples.  The functions are intentionally
*thin wrappers* around the region-local ``*_update`` / ``*_learn``
primitives — the brain-graph layer still owns the composition rule
(what counts as the "RPE modulator" for each region); this module
just encapsulates the call sites so wake-learning and replay-learning
cannot drift apart.

References
----------
  Sutton & Barto (2018)            — TD(0), actor–critic, eligibility.
  Frank (2005); Collins & Frank (2014) — D1/D2 OpAL decomposition.
  Reynolds & Heeger (2009)         — normalisation model of attention.
  Dayan & Yu (2003)                — dual epistemic/extrinsic composition.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array
from .basal_ganglia import (
    ActorParams, ActorState, actor_update,
    CriticParams, CriticState, critic_update,
)
from .cortex import (
    CorticalAreaParams, CorticalAreaState, cortical_area_update,
)
from .attention import (
    AttentionParams, AttentionState, attention_learn,
)


# ======================================================================
# Critic
# ======================================================================


def critic_learn_step(
    state: CriticState,
    params: CriticParams,
    rpe: Array | float,
) -> CriticState:
    """Apply one TD update to the ventral-striatum critic.

    Thin wrapper around :func:`core.basal_ganglia.critic_update`.
    Kept in :mod:`core.learning_pipeline` so Phase 5B's replay
    learner calls exactly the same entrypoint as the wake cycle.
    """
    return critic_update(state, params, jnp.asarray(rpe, DTYPE))


# ======================================================================
# Dual actors (body + saccade) — parallel BG loops
# ======================================================================


def actors_learn_step(
    body_state: ActorState,
    body_params: ActorParams,
    saccade_state: ActorState,
    saccade_params: ActorParams,
    *,
    rpe: Array | float,
    body_bonus: Array | float = 0.0,
    saccade_bonus: Array | float = 0.0,
) -> tuple[ActorState, ActorState]:
    """Apply RPE + modality bonus to each parallel actor.

    The same extrinsic RPE drives both loops (shared VTA broadcast);
    credit is assigned per-domain through the INDEPENDENT eligibility
    traces each actor has committed at the end of the previous decision
    window (Alexander, DeLong & Strick 1986 — parallel segregated
    cortico–BG–thalamo–cortical circuits; Tan 1993 — independent
    learners with a shared global reward).

    ``body_bonus`` is the z-scored curiosity drive (Friston 2017 EFE
    epistemic term over transition surprise); ``saccade_bonus`` is the
    z-scored sensory info-gain (Itti & Baldi 2009 Bayesian surprise).
    Both are already unit-variance via
    :mod:`core.precision_bus.precision_standardize` so additive
    composition is well-scaled regardless of raw signal magnitude.
    """
    rpe_arr = jnp.asarray(rpe, DTYPE)
    body_rpe = rpe_arr + jnp.asarray(body_bonus, DTYPE)
    saccade_rpe = rpe_arr + jnp.asarray(saccade_bonus, DTYPE)
    new_body = actor_update(body_state, body_params, body_rpe)
    new_saccade = actor_update(saccade_state, saccade_params, saccade_rpe)
    return new_body, new_saccade


# ======================================================================
# Cortex — three-factor STDP commit
# ======================================================================


def cortex_learn_step(
    state: CorticalAreaState,
    params: CorticalAreaParams,
    modulator: Array | float,
) -> CorticalAreaState:
    """Commit cortical eligibility traces under the current RPE.

    The eligibility traces were built by ``cortical_area_step`` during
    the perceive loop (pre × post correlations over ``substeps`` dt);
    :func:`core.cortex.cortical_area_update` multiplies them by the
    scalar modulator (RPE at wake; recorded-DA at replay) and by
    the receptor-derived plasticity gain to produce a three-factor
    weight update (Schultz 1998; Reynolds & Wickens 2002).
    """
    return cortical_area_update(
        state, params, modulator=jnp.asarray(modulator, DTYPE),
    )


# ======================================================================
# Attention — top-down Hebbian
# ======================================================================


def attention_learn_step(
    state: AttentionState,
    params: AttentionParams,
    *,
    assoc_activity: Array,
    column_mean_rates: Array,
    gains: Array,
) -> AttentionState:
    """Hebbian update of attention top-down weights.

    See Reynolds & Heeger (2009) normalisation model and Feldman &
    Friston (2010) precision-weighting rationale.
    """
    return attention_learn(
        state, params,
        assoc_activity=assoc_activity,
        column_mean_rates=column_mean_rates,
        gains=gains,
    )
