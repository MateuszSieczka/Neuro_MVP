"""Faza U — §6 temporal credit: the dynamic edge ``w_dyn`` and its uses.

A temporal edge is a generative edge whose source is the *previous*
cycle's belief ``μ_prev`` (dynamic / generalized predictive coding,
Friston 2008).  It is the one-rule substrate of temporal credit — no
second plasticity mechanism — and it subsumes three things that earlier
milestones deferred:

* the sequence-memory transition rule → a ``world_model``/sensory
  transition edge that learns to predict the *next* sensory state;
* the VTA TD target → a ``value(t−1)→value(t)`` edge whose own prediction
  error *is* the temporal-difference error (the bootstrap is the edge's
  prediction, not a separate TD update);
* the DA reward-prediction-error proxy → the DA channel now reads that
  value-node ε directly (closes the §4 deferral).

These tests assert: the edge is inert when absent (static graphs
unchanged), the carry is a true 1-cycle delay, a transition edge learns a
moving sequence and ablating it degrades next-state prediction, the value
edge produces a TD-like error, and the DA channel tracks that ε.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from core.pc_graph import (
    init_pc_graph_params, init_pc_graph_state,
    pc_graph_clamp, pc_graph_errors, pc_graph_learn, pc_graph_roll,
    pc_graph_step, _temporal_predictions,
    init_region_graph, REGION_INDEX,
)
from core.pc_neuromod import (
    init_neuromod_params, init_neuromod_state, neuromod_step,
)


def _onehot(i: int, n: int) -> jnp.ndarray:
    return jnp.eye(n, dtype=jnp.float32)[i]


# ---------------------------------------------------------------------
# the edge is purely additive — absent ⇒ nothing changes
# ---------------------------------------------------------------------


def test_region_graph_temporal_opt_in_is_additive():
    """Default region graph has no temporal edges; the flag adds exactly two."""
    gp0, gs0 = init_region_graph(jax.random.PRNGKey(0))
    gp1, gs1 = init_region_graph(jax.random.PRNGKey(0), temporal_edges=True)

    # Spatial topology is identical; temporal edges are the only addition.
    assert gp0.n_edges == gp1.n_edges == 15
    assert gp0.n_dyn_edges == 0 and len(gs0.w_dyn) == 0
    assert gp1.n_dyn_edges == 2 and len(gs1.w_dyn) == 2

    # The two temporal self-edges are the value TD bootstrap and the
    # world-model sensory-transition cause — named nodes, not magic indices.
    V, WM = REGION_INDEX["value"], REGION_INDEX["world_model"]
    assert set(gp1.dyn_edges) == {(V, V), (WM, WM)}

    # μ_prev exists on both but is a zeroed, inert carry at init.
    assert all(float(jnp.sum(jnp.abs(m))) == 0.0 for m in gs1.mu_prev)


def test_first_cycle_temporal_prediction_is_zero():
    """μ_prev = 0 and φ(0) = 0 ⇒ a fresh temporal edge predicts nothing.

    The first cycle of any run is therefore identical to the static graph;
    the temporal edge only speaks once the carry holds a real belief.
    """
    p = init_pc_graph_params((4,), (), dyn_edges=((0, 0),), act="tanh")
    s = init_pc_graph_state(jax.random.PRNGKey(1), p)
    temporal0 = _temporal_predictions(s.mu_prev, s.w_dyn, p)
    assert float(jnp.sum(jnp.abs(temporal0[0]))) == 0.0

    # After the carry advances to a non-zero belief, the edge drives ≠ 0.
    s = eqx.tree_at(lambda st: st.mu, s, (jnp.ones(4),))
    s = pc_graph_roll(s)                         # μ_prev ← μ = ones
    temporal1 = _temporal_predictions(s.mu_prev, s.w_dyn, p)
    assert float(jnp.sum(jnp.abs(temporal1[0]))) > 0.0


def test_roll_is_a_one_cycle_delay():
    """``pc_graph_roll`` carries exactly the last cycle's μ, nothing older."""
    p = init_pc_graph_params((3,), (), dyn_edges=((0, 0),), act="linear")
    s = init_pc_graph_state(jax.random.PRNGKey(2), p)
    a, b = jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])

    s = pc_graph_roll(eqx.tree_at(lambda st: st.mu, s, (a,)))
    assert jnp.allclose(s.mu_prev[0], a)
    s = pc_graph_roll(eqx.tree_at(lambda st: st.mu, s, (b,)))
    assert jnp.allclose(s.mu_prev[0], b)         # only t−1 survives, not t−2


# ---------------------------------------------------------------------
# sequence memory → one transition edge that learns the next state
# ---------------------------------------------------------------------


def test_transition_edge_learns_moving_sequence_and_ablation_degrades():
    """A sensory self-edge learns a moving bump; removing it kills prediction.

    The deferred sequence-memory transition rule, as one ``w_dyn`` edge:
    the bump at position ``p`` at ``t−1`` must predict position ``p+1`` at
    ``t``.  Linear activation so the edge converges to the shift operator;
    precision held fixed to isolate the weight rule.
    """
    D = 5
    p = init_pc_graph_params((D,), (), dyn_edges=((0, 0),), act="linear", eta_w=0.3)
    s = init_pc_graph_state(jax.random.PRNGKey(3), p)

    # Train over many periods of the deterministic shift sequence.
    for _ in range(80):
        for t in range(D):
            out = pc_graph_step(
                s, p, {0: _onehot(t, D)}, n_steps=1, update_precision=False,
            )
            s = out.state

    # One-step-ahead generative prediction from each position.
    def pred_err(weights_dyn):
        err = 0.0
        for t in range(D):
            pred = weights_dyn[0] @ _onehot(t, D)        # linear φ
            err += float(jnp.sum((pred - _onehot((t + 1) % D, D)) ** 2))
            assert int(jnp.argmax(pred)) == (t + 1) % D, "wrong next position"
        return err / D

    trained = pred_err(s.w_dyn)
    ablated = init_pc_graph_state(jax.random.PRNGKey(3), p).w_dyn  # untrained edge
    err_ablated = sum(
        float(jnp.sum((ablated[0] @ _onehot(t, D) - _onehot((t + 1) % D, D)) ** 2))
        for t in range(D)
    ) / D

    assert trained < 0.1, f"transition edge did not learn the sequence: {trained:.3f}"
    assert err_ablated > 5 * trained, "ablating the temporal edge must degrade prediction"


# ---------------------------------------------------------------------
# VTA TD → a value temporal edge whose ε is the TD error
# ---------------------------------------------------------------------


def test_value_temporal_edge_is_a_td_error():
    """value(t−1)→value(t): predictable reward ⇒ ε→0, surprise ⇒ ε spikes.

    The value node's own prediction error against the temporal edge is the
    TD error — vanishing once the reward becomes predictable (the bootstrap
    explains it) and re-appearing on an unexpected reward.  A graph with no
    temporal edge cannot predict and keeps a full-size error throughout.
    """
    p = init_pc_graph_params((1,), (), dyn_edges=((0, 0),), act="linear", eta_w=0.5)
    s = init_pc_graph_state(jax.random.PRNGKey(4), p)

    def cycle(state, r):
        clamped = pc_graph_clamp(state, {0: jnp.array([r])})
        eps = float(pc_graph_errors(clamped, p)[0][0])     # r − w·r_prev
        learned = pc_graph_roll(pc_graph_learn(clamped, p, update_precision=False))
        return learned, eps

    # Predictable reward: the TD error must decay toward zero.
    eps_first = None
    for _ in range(40):
        s, eps = cycle(s, 1.0)
        if eps_first is None:
            eps_first = eps
    eps_steady = eps
    # Surprising reward: a jump the bootstrap did not expect ⇒ large RPE.
    _, eps_surprise = cycle(s, 2.0)

    assert eps_first > 0.5, "first reward is unexplained (no prior prediction)"
    assert abs(eps_steady) < 0.1, "predictable reward must drive the TD error to ~0"
    assert eps_surprise > 0.5, "an unexpected reward must produce a large RPE"

    # Ablation: no temporal edge ⇒ the value error never shrinks.
    p0 = init_pc_graph_params((1,), (), act="linear", eta_w=0.5)
    s0 = init_pc_graph_state(jax.random.PRNGKey(4), p0)
    for _ in range(40):
        clamped = pc_graph_clamp(s0, {0: jnp.array([1.0])})
        s0 = pc_graph_roll(pc_graph_learn(clamped, p0, update_precision=False))
    eps_ablated = float(pc_graph_errors(
        pc_graph_clamp(s0, {0: jnp.array([1.0])}), p0)[0][0])
    assert eps_ablated > 5 * abs(eps_steady), "without a temporal edge ε cannot fall"


# ---------------------------------------------------------------------
# DA channel reads the value-node temporal ε (closes §4)
# ---------------------------------------------------------------------


def test_da_tracks_value_temporal_epsilon():
    """Same value belief, different temporal ε ⇒ different DA.

    The discriminator from the old Δ-value proxy: here the *current* value
    belief is identical in both cases (μ = 1), so a level/Δ-based DA would
    fire equally; only an ε-based DA distinguishes a predicted reward
    (ε = 0) from a surprising one (ε = 1).
    """
    p = init_pc_graph_params((1,), (), dyn_edges=((0, 0),), act="linear")
    s = init_pc_graph_state(jax.random.PRNGKey(5), p)

    def with_fields(mu, mu_prev, w):
        return eqx.tree_at(
            lambda st: (st.mu, st.mu_prev, st.w_dyn), s,
            ((jnp.array([mu]),), (jnp.array([mu_prev]),), (jnp.array([[w]]),)),
        )

    predicted = with_fields(1.0, 1.0, 1.0)   # temporal pred = 1 → ε = 0
    surprising = with_fields(1.0, 1.0, 0.0)  # temporal pred = 0 → ε = 1

    np_ = init_neuromod_params(value_idx=0, sensory_idx=0, wm_idx=0)
    ns = init_neuromod_state(np_)
    da_pred = float(neuromod_step(ns, np_, predicted, p).da)
    da_surp = float(neuromod_step(ns, np_, surprising, p).da)
    assert da_surp > da_pred, "DA must track the value node's temporal RPE, not its level"


# ---------------------------------------------------------------------
# integration: the real region graph trains with temporal edges live
# ---------------------------------------------------------------------


def test_region_graph_temporal_trains_under_one_rule():
    """Full region graph (temporal on) runs a sequence; temporal weights move."""
    gp, gs = init_region_graph(
        jax.random.PRNGKey(6), temporal_edges=True, eta_w=1e-2, n_relax=20,
    )
    s_idx = REGION_INDEX["sensory"]
    D = gp.node_sizes[s_idx]
    w_dyn0 = gs.w_dyn

    s = gs
    for t in range(24):
        phase = 2.0 * jnp.pi * (t % 8) / 8.0
        obs = jnp.sin(phase + jnp.arange(D, dtype=jnp.float32))   # moving pattern
        out = pc_graph_step(s, gp, {s_idx: obs}, n_steps=20)
        s = out.state
        assert jnp.isfinite(out.free_energy), "temporal region graph FE not finite"

    for w in s.w_dyn:
        assert jnp.all(jnp.isfinite(w)), "temporal weight blew up"
    moved = sum(
        float(jnp.sum(jnp.abs(a - b))) for a, b in zip(s.w_dyn, w_dyn0)
    )
    assert moved > 0.0, "the one rule did not update the temporal edges"
