"""Faza U — krok U.5: active inference replaces REINFORCE.

``core.pc_active`` makes action a consequence of free-energy
minimisation, not policy gradient.  Asserts the U.5 claims:

* **babble → reach with no REINFORCE**: a forward model learnt from
  random commands (one rule) lets action-as-inference drive a plant to a
  target — the capability M1 node-perturbation was for, now from local
  PC inference;
* **predictions, not commands**: clamping a preferred outcome and
  relaxing infers the command whose predicted outcome matches the
  preference;
* **argmin EFE policy selection**: greedy when β = 0, curiosity-driven
  when the epistemic term is up-weighted;
* the graph-driven brain produces a goal-sensitive command via active
  inference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from core.pc_graph import (
    init_pc_graph_params, init_pc_graph_state, pc_graph_predictions,
)
from core.pc_active import (
    set_action_prior, pc_act_infer, pc_act_learn_forward,
    efe_select, pc_efe,
)
from core.pc_brain import (
    init_pc_brain, pc_brain_act, pc_brain_learn_forward,
)


# ---------------------------------------------------------------------
# babble → reach, no REINFORCE  (THE capability)
# ---------------------------------------------------------------------


def test_babble_then_reach_no_reinforce():
    d = 4
    p = init_pc_graph_params(
        (d, d), ((1, 0),),                 # motor(1) → outcome(0)
        act="linear", eta_mu=0.3, eta_w=0.05, n_relax=60,
    )
    k1, k2, k3, kb = jax.random.split(jax.random.PRNGKey(0), 4)
    s = init_pc_graph_state(k1, p)
    A = jax.random.normal(k2, (d, d)) * 0.6 + jnp.eye(d)   # the body (plant)
    qstar = jax.random.normal(k3, (d,))
    s = set_action_prior(s, 1)

    # Phase 1 — canonical babbling: random commands, learn forward model.
    for kk in jax.random.split(kb, 300):
        cmd = jax.random.normal(kk, (d,)) * 0.5
        realised = A @ cmd
        s = pc_act_learn_forward(
            s, p, 1, 0, cmd, realised, update_precision=False,
        )
    assert float(jnp.linalg.norm(s.weights[0] - A)) < 0.3, "forward model not learnt"

    # Phase 2 — reach by action inference (no policy gradient anywhere).
    out = pc_act_infer(s, p, 1, 0, qstar, n_steps=150)
    realised = A @ out.command
    rel_err = float(jnp.linalg.norm(qstar - realised) / jnp.linalg.norm(qstar))
    assert rel_err < 0.1, f"active-inference reach failed: rel err {rel_err:.3f}"


# ---------------------------------------------------------------------
# predictions, not commands
# ---------------------------------------------------------------------


def test_action_as_inference_matches_preference():
    d = 5
    p = init_pc_graph_params(
        (d, d), ((1, 0),), act="linear", eta_mu=0.3, n_relax=300,
    )
    s = init_pc_graph_state(jax.random.PRNGKey(1), p)
    # Known forward model = identity → inferred command should equal the
    # preferred outcome (and so should the predicted outcome).
    s = eqx.tree_at(lambda z: z.weights, s, (jnp.eye(d),))
    s = set_action_prior(s, 1, precision=1e-4)

    pref = jax.random.normal(jax.random.PRNGKey(2), (d,))
    out = pc_act_infer(s, p, 1, 0, pref, n_steps=300)
    rel = float(jnp.linalg.norm(out.predicted_outcome - pref) / jnp.linalg.norm(pref))
    assert rel < 0.05, f"inferred action does not realise preference: rel {rel:.3f}"


# ---------------------------------------------------------------------
# argmin expected free energy
# ---------------------------------------------------------------------


def test_efe_policy_selection():
    pragmatic = jnp.array([1.0, 0.5, 0.2])     # policy 0 best for the goal
    epistemic = jnp.array([0.0, 0.0, 2.0])     # policy 2 most informative

    greedy = efe_select(pragmatic, epistemic, epistemic_weight=0.0)
    assert int(greedy.index) == 0, "β=0 should pick the pragmatic policy"

    curious = efe_select(pragmatic, epistemic, epistemic_weight=5.0)
    assert int(curious.index) == 2, "high β should pick the epistemic policy"

    # G is lower (better) for the chosen policy.
    assert float(curious.G[2]) < float(curious.G[0])
    # Scalar EFE: more pragmatic value → lower G.
    assert float(pc_efe(2.0, 0.0)) < float(pc_efe(1.0, 0.0))


# ---------------------------------------------------------------------
# graph-driven brain: goal-directed command via active inference
# ---------------------------------------------------------------------


def test_pc_brain_act_is_goal_directed():
    sensory, motor = 6, 3
    params, state = init_pc_brain(
        jax.random.PRNGKey(3),
        sensory_size=sensory, motor_size=motor,
        eta_mu=0.2, eta_w=3e-2, n_relax=30,
    )
    # Synthetic body: reafference = B @ command.  Babble to learn the
    # motor→sensory forward model.
    B = jax.random.normal(jax.random.PRNGKey(4), (sensory, motor)) * 0.5
    for kk in jax.random.split(jax.random.PRNGKey(5), 200):
        cmd = jax.random.normal(kk, (motor,)) * 0.5
        state = pc_brain_learn_forward(state, params, cmd, B @ cmd)

    pref_a = jnp.ones(sensory) * 0.5
    pref_b = -jnp.ones(sensory) * 0.5
    cmd_a = pc_brain_act(state, params, pref_a, n_relax=80)
    cmd_b = pc_brain_act(state, params, pref_b, n_relax=80)

    assert cmd_a.shape == (motor,)
    assert jnp.all(jnp.abs(cmd_a) <= 1.0)
    assert float(jnp.sum(jnp.abs(cmd_a - cmd_b))) > 1e-3, (
        "active-inference command not goal-sensitive"
    )
