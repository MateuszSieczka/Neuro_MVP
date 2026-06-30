"""Active inference on the PC graph — action as inference (Faza U, U.5).

Replaces M1 node-perturbation REINFORCE (``core/m1.py``) and the BG
actor-critic with one principle: **action minimises (expected) free
energy** (Friston 2010, 2017).  No policy gradient, no separate critic.

Two faces of the same rule, both on the shared graph + the one
learning rule:

* **Continuous control — predictions, not commands** (Adams, Shipp &
  Friston 2013).  A motor node is given a *flat prior* (action carries no
  prior preference); a preferred outcome is clamped on an outcome node;
  relaxing the graph infers the motor value that, through the generative
  (forward) model, would produce the preferred outcome.  The body is then
  driven to fulfil that prediction.  ``pc_act_infer``.

  The forward model is acquired self-supervised: random commands →
  observe realised outcome → learn the motor→outcome edge with the one
  rule (``pc_act_learn_forward``).  This is canonical babbling preceding
  goal-directed reach (Oller 1980; von Hofsten 2004; plan §U.5) — and it
  is *required*: an untrained forward model makes the inferred command
  explode (verified), so babble first, reach second.

* **Discrete policy selection — argmin EFE** (Friston 2017 "BG as policy
  precision").  Given candidate policies' pragmatic (goal/reward) and
  epistemic (information-gain) values, pick the policy minimising
  ``G = −pragmatic + ambiguity − β·epistemic`` (``efe_select``).  The
  epistemic term is curiosity; ``β`` is NE-modulated.

Neuromodulators are precision controllers (Yu & Dayan 2005; FitzGerald
2015; Parr & Friston 2017): ``scale_node_precision`` is the hook by which
ACh (sensory Π), DA (reward/policy Π) and NE (volatility → β) gate the
graph, rather than being separate modules.

Hierarchical goals come for free: a preference clamped on a *deep* node
propagates down to motor through the generative hierarchy (Friston,
Pezzulo et al. 2018) — depth of the same graph, not new machinery.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array
from .free_energy import expected_free_energy
from .pc_graph import (
    PCGraphParams, PCGraphState,
    pc_graph_clamp, pc_graph_relax, pc_graph_learn, pc_graph_predictions,
)

# The forward-model update clamps both the action and the outcome and relaxes
# the *intermediate* causes — chiefly the cerebellar hidden layer of the
# motor→cerebellum→outcome forward model — before the Hebbian step.  That
# hidden cause is a genuine latent (a single direct edge it is not), so it
# must settle as deeply as ordinary inference; ``None`` ⇒ the graph's own
# ``n_relax``.  (A short fixed settle would leave the hidden layer unsettled
# and the forward model mis-fit — the reason a 1-step settle silently broke
# reach when the model was a single linear edge dressed up as a forward model.)
DEFAULT_FORWARD_SETTLE_STEPS = None


# =====================================================================
# Precision control (neuromodulation hook) + flat action priors
# =====================================================================


def scale_node_precision(
    state: PCGraphState, node_idx: int, gain: float | Array,
) -> PCGraphState:
    """Multiply a node's precision Π by ``gain`` (neuromodulatory control).

    ACh ↑ sensory-node Π (attention / perceptual gain), DA ↑ reward-node
    Π, NE sets volatility.  This is how neuromodulators enter the unified
    substrate — as precision gains, not separate rules (Parr & Friston
    2017).
    """
    pi = list(state.pi)
    pi[node_idx] = (pi[node_idx] * jnp.asarray(gain, DTYPE)).astype(DTYPE)
    return eqx.tree_at(lambda s: s.pi, state, tuple(pi))


def set_action_prior(
    state: PCGraphState, motor_idx: int, precision: float = 0.0,
) -> PCGraphState:
    """Give an action node a flat prior — Π → ``precision`` (default 0).

    Action variables in active inference carry no prior preference: they
    are inferred purely to satisfy the preferred outcome (Friston 2010).
    A nonzero prior precision would regularise the inferred command toward
    0 and bias reaching, so the prior is genuinely **flat** (Π = 0).  This
    is admissible only because the relaxation is curvature-preconditioned
    *and* the action node has a child (the forward-model edge
    ``motor→cerebellum``): its inference curvature ``L = Π + φ'²·Σ_child W²Π``
    is then strictly positive even at Π = 0, so the step stays finite — the
    old ``1e-3`` floor (a magic stabiliser) is no longer needed.  Pair with
    ``fixed_pi_nodes`` (the node must not have its precision learned back,
    :func:`core.pc_graph.pc_graph_learn`), else the first learning step
    overwrites this flat prior.
    """
    pi = list(state.pi)
    pi[motor_idx] = jnp.full_like(pi[motor_idx], jnp.asarray(precision, DTYPE))
    return eqx.tree_at(lambda s: s.pi, state, tuple(pi))


# =====================================================================
# Continuous control — action as inference (predictions, not commands)
# =====================================================================


class ActInferOutput(NamedTuple):
    state: PCGraphState         # relaxed graph (beliefs at the AIF equilibrium)
    command: Array              # (motor_dim,) inferred motor belief
    predicted_outcome: Array    # (outcome_dim,) model's predicted outcome


def pc_act_infer(
    state: PCGraphState, params: PCGraphParams,
    motor_idx: int, outcome_idx: int,
    preference: Array,
    *,
    preference_mask: Array | None = None,
    observations: dict | None = None,
    free_nodes: tuple[int, ...] | None = None,
    n_steps: int | None = None,
) -> ActInferOutput:
    """Infer the motor command that would realise ``preference``.

    Clamps the outcome node to the preferred outcome (and any
    ``observations`` such as the current sensory afferent), then relaxes
    the graph with the motor node free.  The relaxed motor belief is the
    command (predictions-not-commands); the body fulfils it downstream.

    ``preference_mask`` makes the preference *partial*: a per-dimension
    boolean over the outcome node, ``True`` = pinned to ``preference``,
    ``False`` = free.  A goal is usually a partial specification (e.g.
    "zero target-error" on the goal channels while proprioception is left
    to be inferred); the unpinned dimensions relax from their current
    belief.  ``None`` pins the whole outcome node.

    ``free_nodes`` restricts inference to the **action pathway** — only
    these nodes (the motor node and its forward-model hidden layer, the
    cerebellum) relax; every other node is held at its current perceptual
    belief.  This is the defining move of active inference: *perception is
    fixed, action varies* (Friston 2010).  Without it the clamped goal is
    an outcome the whole generative model relaxes to explain, and the
    cortical/world-model causes simply re-explain the preferred outcome
    among themselves — driving the outcome error to zero **without moving
    the motor command** (explaining-away).  Holding perception leaves the
    forward model ``motor→cerebellum→outcome`` as the *only* path that can
    satisfy the goal, so the command is actually inferred.  ``None`` (the
    default) frees every non-clamped node (whole-graph inference, the
    behaviour used by hierarchical-goal inference and the unit tests).

    Assumes the motor node has a flat prior (:func:`set_action_prior`)
    and the forward-model edges are trained (:func:`pc_act_learn_forward`).
    """
    clamp_values = {
        idx: jnp.asarray(val, DTYPE) for idx, val in (observations or {}).items()
    }
    whole_clamp = list(clamp_values.keys())
    clamp_masks: dict[int, Array] = {}

    pref = jnp.asarray(preference, DTYPE)
    if preference_mask is None:
        clamp_values[outcome_idx] = pref
        whole_clamp.append(outcome_idx)
    else:
        mask = jnp.asarray(preference_mask, bool)
        # Pin only the masked dimensions; leave the rest at their belief.
        clamp_values[outcome_idx] = jnp.where(mask, pref, state.mu[outcome_idx])
        clamp_masks[outcome_idx] = mask

    if free_nodes is not None:
        # Hold every node outside the action pathway at its current belief —
        # they keep their (perceptual) μ, so only the action pathway relaxes
        # to satisfy the goal.  The outcome node is excluded: it is the goal
        # carrier (pinned whole, or partially via the mask with free dims).
        free = set(int(j) for j in free_nodes)
        already = set(whole_clamp) | set(clamp_masks)
        whole_clamp.extend(
            j for j in range(params.n_nodes)
            if j not in free and j != outcome_idx and j not in already
        )

    clamped = pc_graph_clamp(state, clamp_values)
    relaxed = pc_graph_relax(
        clamped, params,
        clamp=tuple(whole_clamp),
        clamp_masks=clamp_masks or None,
        n_steps=n_steps,
    )
    preds = pc_graph_predictions(relaxed, params)
    return ActInferOutput(
        state=relaxed,
        command=relaxed.mu[motor_idx],
        predicted_outcome=preds[outcome_idx],
    )


def pc_act_learn_forward(
    state: PCGraphState, params: PCGraphParams,
    motor_idx: int, outcome_idx: int,
    command: Array, realised_outcome: Array,
    *,
    free_nodes: tuple[int, ...] | None = None,
    n_relax: int | None = DEFAULT_FORWARD_SETTLE_STEPS,
    update_precision: bool = True,
) -> PCGraphState:
    """Learn the forward model from a realised (command → outcome) pair.

    Clamps the motor node to the executed command and the outcome node to
    the realised outcome, relaxes the intermediate causes (the cerebellar
    hidden layer) for ``n_relax`` settling steps — ``None`` ⇒ the graph's
    ``n_relax``, since that hidden cause is inferred like any latent — then
    applies the one rule, descending ``½‖realised − forward(command)‖²``
    along the ``motor→cerebellum→outcome`` path.  Used during babbling
    (random commands) and continuously during reaching.

    ``free_nodes`` confines the settle to the **action pathway** (the
    forward-model hidden layer), holding every other node — so the forward
    model learns the *full* command→reafference map instead of only the
    residual the perceptual hierarchy leaves unexplained.  Without it the
    cortical/world-model causes (free) co-explain the clamped reafference
    during babbling and the ``cerebellum→outcome`` edge learns almost
    nothing, so there is no accurate model left to invert at reach.  ``None``
    (the default) frees every non-clamped node (the behaviour used by the
    unit tests and any caller that wants perception to learn here too).
    """
    steps = params.n_relax if n_relax is None else int(n_relax)
    clamped = pc_graph_clamp(
        state, {motor_idx: jnp.asarray(command, DTYPE),
                outcome_idx: jnp.asarray(realised_outcome, DTYPE)},
    )
    whole_clamp = [motor_idx, outcome_idx]
    if free_nodes is not None:
        free = set(int(j) for j in free_nodes)
        clamp_set = {motor_idx, outcome_idx}
        whole_clamp.extend(
            j for j in range(params.n_nodes)
            if j not in free and j not in clamp_set
        )
    relaxed = pc_graph_relax(
        clamped, params, clamp=tuple(whole_clamp), n_steps=steps,
    )
    return pc_graph_learn(relaxed, params, update_precision=update_precision)


# =====================================================================
# Discrete policy selection — argmin expected free energy
# =====================================================================


def pc_efe(
    pragmatic: float | Array, epistemic: float | Array,
    *, ambiguity: float | Array = 0.0, epistemic_weight: float | Array = 1.0,
) -> Array:
    """Expected free energy ``G`` of a policy (lower = better)."""
    return expected_free_energy(pragmatic, epistemic, ambiguity, epistemic_weight)


class PolicyChoice(NamedTuple):
    index: Array        # argmin policy index
    G: Array            # (n_policies,) expected free energy per policy


def efe_select(
    pragmatic_values: Array, epistemic_values: Array,
    *,
    ambiguity: float | Array = 0.0,
    epistemic_weight: float | Array = 1.0,
) -> PolicyChoice:
    """Choose the policy with minimal expected free energy (Friston 2017).

    ``pragmatic_values`` / ``epistemic_values`` are per-candidate vectors
    (goal-progress and information-gain).  ``epistemic_weight`` (β, NE-
    modulated) trades exploitation against exploration; with β = 0 the
    choice is greedy-pragmatic, with large β it is curiosity-driven.
    """
    G = expected_free_energy(
        pragmatic_values, epistemic_values, ambiguity, epistemic_weight,
    )
    return PolicyChoice(index=jnp.argmin(G), G=G)


def epistemic_value(
    state: PCGraphState, node_idx: int,
    *,
    learning_progress: float | Array | None = None,
    lp_weight: float | Array = 1.0,
) -> Array:
    """Expected information gain at a node — the exploration term of EFE.

    Default (``learning_progress=None``): the inverse-precision proxy
    ``mean(1/Π)`` — low-precision (uncertain) outcomes carry the most to
    learn (Friston 2017 epistemic value / salience).  Cheap, but
    noise-blind: irreducible noise also reads as low precision (the
    "noisy-TV" trap).

    When a ``learning_progress`` signal is supplied (the world model's
    ``pe_long − pe_short`` from :func:`core.pc_neuromod.neuromod_curiosity`,
    Oudeyer 2007), its rectified value is added with weight ``lp_weight``:
    curiosity is then drawn to regions whose error is actually *falling*
    (genuinely learnable), not merely uncertain.  Additive and opt-in, so
    existing callers are unchanged.
    """
    inv_precision = jnp.mean(1.0 / (state.pi[node_idx] + jnp.asarray(1e-6, DTYPE)))
    if learning_progress is None:
        return inv_precision
    lp = jnp.maximum(jnp.asarray(learning_progress, DTYPE), 0.0)
    return inv_precision + jnp.asarray(lp_weight, DTYPE) * lp
