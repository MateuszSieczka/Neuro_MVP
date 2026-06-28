"""Graph-driven brain — the cognitive cycle as relaxation (Faza U, U.3).

A complete brain whose one decision cycle is **free-energy relaxation on
the region graph**, not a hand-coded sequence of region calls.
``pc_brain_cognitive_step`` does only:

    1. clamp the sensory node to the afferent observation,
    2. relax the whole graph (order emerges from error flow, Salvatori
       2022; Incremental PC, Salvatori 2024),
    3. read the motor command off the relaxed motor node,
    4. learn every edge with the one rule (ΔW = η·Π·ε·φ(μ)).

Adding a region or a projection changes the graph passed to
:func:`core.pc_graph.init_region_graph`; this cognitive step does not
change — the property a hand-coded region sequence never had.

Scope / integration
-------------------
The step takes a flat ``sensory`` afferent vector and returns a bounded
``joint_command`` (``tanh`` of the motor node, ∈ [−1, 1]): a
substrate-agnostic interface a body adapter can drive directly (clamp
sensory → step → apply command; the embodiment adapter is the next,
external build — plan §12).

Goal-directed motor *action* (driving the motor node to a preferred
outcome by inference + expected-free-energy selection) is U.5
(:func:`pc_brain_act`); this step covers perception, action read-out and
perceptual learning — the relaxation that replaces the sequence.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey
from .pc_graph import (
    PCGraphParams, PCGraphState,
    init_region_graph, pc_graph_clamp, pc_graph_relax,
    pc_graph_learn, graph_free_energy, REGION_INDEX,
)
from .pc_active import (
    set_action_prior, pc_act_infer, pc_act_learn_forward,
    scale_node_precision, epistemic_value,
)


class PCBrainParams(eqx.Module):
    """Region graph + the read-out node indices (static)."""

    graph: PCGraphParams

    sensory_idx: int = eqx.field(static=True)
    motor_idx: int = eqx.field(static=True)
    value_idx: int = eqx.field(static=True)
    policy_idx: int = eqx.field(static=True)
    cortex_top_idx: int = eqx.field(static=True)

    @property
    def sensory_dim(self) -> int:
        return self.graph.node_sizes[self.sensory_idx]

    @property
    def motor_dim(self) -> int:
        return self.graph.node_sizes[self.motor_idx]


class PCBrainState(eqx.Module):
    """Just the graph state — μ beliefs + edge weights + precision."""

    graph: PCGraphState


def init_pc_brain(
    key: PRNGKey,
    *,
    sensory_size: int,
    motor_size: int,
    **graph_kwargs,
) -> tuple[PCBrainParams, PCBrainState]:
    """Build a graph-driven brain with all regions as nodes (one rule)."""
    gp, gs = init_region_graph(
        key, sensory_size=sensory_size, motor_size=motor_size, **graph_kwargs,
    )
    # Action node carries a flat prior (active inference: the command is
    # inferred to satisfy preferences, it has no prior preference of its
    # own — Friston 2010).  Set once at construction (U.5).
    gs = set_action_prior(gs, REGION_INDEX["motor"])
    params = PCBrainParams(
        graph=gp,
        sensory_idx=REGION_INDEX["sensory"],
        motor_idx=REGION_INDEX["motor"],
        value_idx=REGION_INDEX["value"],
        policy_idx=REGION_INDEX["policy"],
        cortex_top_idx=REGION_INDEX["cortex_l3"],
    )
    return params, PCBrainState(graph=gs)


class PCBrainOutput(NamedTuple):
    state: PCBrainState
    joint_command: Array       # (motor_dim,) ∈ [−1, 1]
    value: Array               # scalar — expected value node belief
    policy: Array              # (policy_dim,) — policy node belief (logits)
    free_energy: Array         # scalar — global objective at relaxation
    belief: Array              # (cortex_top,) — deep cortical cause
    epistemic: Array           # scalar — sensory info-gain (curiosity drive)


def pc_brain_cognitive_step(
    state: PCBrainState,
    params: PCBrainParams,
    sensory: Array,
    *,
    n_relax: int | None = None,
    learn: bool = True,
    precision_gains: dict[int, float] | None = None,
) -> PCBrainOutput:
    """One cycle: clamp sensory → relax graph → read motor → learn.

    The action emerges from the relaxed motor node; the deep cortical
    node carries the abstract cause; ``free_energy`` is the single
    objective everything minimised this cycle.  Set ``learn=False`` to
    run pure inference (e.g. evaluation / held-out probing).

    ``precision_gains`` maps node index → multiplicative precision gain:
    the neuromodulatory hook (ACh ↑ sensory Π = attention, DA ↑ reward Π;
    Parr & Friston 2017).  Gains modulate this cycle's inference and the
    weight of its error in learning; they do not persist (precision is an
    EMA, restored by the learning update).  ``epistemic`` reports the
    sensory node's information gain (mean inverse precision) — the
    exploration term active inference would feed to :func:`efe_select`.
    """
    g = state.graph
    s_idx = params.sensory_idx

    g_mod = g
    if precision_gains:
        for idx, gain in precision_gains.items():
            g_mod = scale_node_precision(g_mod, idx, gain)

    clamped = pc_graph_clamp(g_mod, {s_idx: sensory.astype(DTYPE)})
    relaxed = pc_graph_relax(
        clamped, params.graph, clamp=(s_idx,), n_steps=n_relax,
    )
    fe = graph_free_energy(relaxed, params.graph)

    motor_belief = relaxed.mu[params.motor_idx]
    joint_command = jnp.tanh(motor_belief).astype(DTYPE)
    value = jnp.mean(relaxed.mu[params.value_idx])
    policy = relaxed.mu[params.policy_idx]
    belief = relaxed.mu[params.cortex_top_idx]

    if learn:
        new_graph = pc_graph_learn(relaxed, params.graph)
    else:
        # Pure inference: advance only the beliefs μ; weights and the
        # (possibly modulated) precision of the incoming state are left
        # untouched, so a probe never mutates the model.
        new_graph = eqx.tree_at(lambda s: s.mu, g, relaxed.mu)

    epistemic = epistemic_value(new_graph, s_idx)

    return PCBrainOutput(
        state=PCBrainState(graph=new_graph),
        joint_command=joint_command,
        value=value,
        policy=policy,
        free_energy=fe,
        belief=belief,
        epistemic=epistemic,
    )


# =====================================================================
# Active inference: goal-directed action (U.5)
# =====================================================================


def pc_brain_act(
    state: PCBrainState,
    params: PCBrainParams,
    preferred_sensory: Array,
    *,
    n_relax: int | None = None,
) -> Array:
    """Infer the command realising a preferred sensory outcome (no REINFORCE).

    Active inference: clamp the *preferred* proprioceptive reafference on
    the sensory node, relax with the (flat-prior) motor node free; the
    motor belief that explains the preference through the motor→sensory
    forward model is the command (Adams, Shipp & Friston 2013 —
    "predictions, not commands").  Requires a trained forward model
    (:func:`pc_brain_learn_forward` during babbling).
    """
    out = pc_act_infer(
        state.graph, params.graph,
        motor_idx=params.motor_idx, outcome_idx=params.sensory_idx,
        preference=preferred_sensory, n_steps=n_relax,
    )
    return jnp.tanh(out.command).astype(DTYPE)


def pc_brain_learn_forward(
    state: PCBrainState,
    params: PCBrainParams,
    command: Array,
    realised_sensory: Array,
) -> PCBrainState:
    """Self-supervised forward-model update from a (command → reafference) pair.

    The motor→sensory edge learns by the one rule; used during babbling
    (random commands) so the active-inference action of
    :func:`pc_brain_act` has a forward model to invert.  ``command`` is
    the executed motor belief (pre-``tanh``).
    """
    new_graph = pc_act_learn_forward(
        state.graph, params.graph,
        motor_idx=params.motor_idx, outcome_idx=params.sensory_idx,
        command=command, realised_outcome=realised_sensory,
    )
    return PCBrainState(graph=new_graph)
