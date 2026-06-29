"""Faza U — §4 neuromodulators as precision controllers + curiosity.

Asserts the four channels move with their graph-native drivers, that the
read-outs land in the existing substrate hooks (``precision_gains`` →
``scale_node_precision``; NE → EFE β; learning progress → ``epistemic_value``),
and that the curiosity term measurably changes the exploration value (the
plan §U.5 ablation).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from core.pc_graph import (
    init_region_graph, pc_graph_clamp, pc_graph_relax, REGION_INDEX,
)
from core.pc_neuromod import (
    init_neuromod_params, init_neuromod_state, neuromod_step,
    neuromod_precision_gains, neuromod_beta, neuromod_curiosity, neuromod_horizon,
)
from core.pc_active import scale_node_precision, efe_select, epistemic_value


def _relaxed(seed=0, sensory=None):
    gp, gs = init_region_graph(jax.random.PRNGKey(seed))
    s = gp.node_sizes[REGION_INDEX["sensory"]]
    obs = jnp.ones(s) if sensory is None else sensory
    clamped = pc_graph_clamp(gs, {REGION_INDEX["sensory"]: obs})
    return gp, pc_graph_relax(clamped, gp, clamp=(REGION_INDEX["sensory"],))


def _set_node(graph, idx, value):
    mu = list(graph.mu)
    mu[idx] = value.astype(mu[idx].dtype)
    return eqx.tree_at(lambda s: s.mu, graph, tuple(mu))


# ---------------------------------------------------------------------
# Channels track their drivers
# ---------------------------------------------------------------------


def test_ach_rises_with_sensory_novelty():
    np_ = init_neuromod_params()
    gp, relaxed = _relaxed(sensory=jnp.full(12, 5.0))   # large, unexplained input
    quiet_gp, quiet = _relaxed(sensory=jnp.zeros(12))
    ach_loud = neuromod_step(init_neuromod_state(np_), np_, relaxed, gp).ach
    ach_quiet = neuromod_step(init_neuromod_state(np_), np_, quiet, quiet_gp).ach
    assert float(ach_loud) > float(ach_quiet), "ACh should rise with sensory novelty"


def test_da_rises_when_value_belief_increases():
    np_ = init_neuromod_params()
    gp, relaxed = _relaxed()
    state = init_neuromod_state(np_)
    # First step fixes value_prev; then raise the value-node belief → +RPE.
    state = neuromod_step(state, np_, relaxed, gp)
    higher = _set_node(relaxed, REGION_INDEX["value"],
                       relaxed.mu[REGION_INDEX["value"]] + 2.0)
    da_up = neuromod_step(state, np_, higher, gp).da
    da_flat = neuromod_step(state, np_, relaxed, gp).da
    assert float(da_up) > float(da_flat), "DA should track positive value RPE"


def test_curiosity_positive_when_world_model_error_falls():
    """LP = pe_long − pe_short > 0 while the wm error is decreasing (mastering).

    Learning progress needs both timescales settled: the long EMA holds the
    historical (high) error baseline, the short EMA tracks the now-lower
    current error.  Warm both at high error, then drop it.
    """
    np_ = init_neuromod_params()
    gp, relaxed = _relaxed()
    state = init_neuromod_state(np_)
    wm = REGION_INDEX["world_model"]
    high = _set_node(relaxed, wm, jnp.full(gp.node_sizes[wm], 4.0))
    low = _set_node(relaxed, wm, jnp.zeros(gp.node_sizes[wm]))
    for _ in range(200):                       # both EMAs converge to high error
        state = neuromod_step(state, np_, high, gp)
    for _ in range(10):                        # error drops: short falls below long
        state = neuromod_step(state, np_, low, gp)
    assert float(neuromod_curiosity(state)) > 0.0, "falling wm error ⇒ positive LP"


# ---------------------------------------------------------------------
# Read-outs land in the existing hooks
# ---------------------------------------------------------------------


def test_precision_gains_drive_scale_node_precision():
    np_ = init_neuromod_params()
    gp, relaxed = _relaxed(sensory=jnp.full(12, 4.0))
    state = neuromod_step(init_neuromod_state(np_), np_, relaxed, gp)
    gains = neuromod_precision_gains(state, np_)
    assert set(gains) == {REGION_INDEX["sensory"], REGION_INDEX["value"]}
    # Gains are accepted by the precision hook and change the node's Π.
    s_idx = REGION_INDEX["sensory"]
    scaled = scale_node_precision(relaxed, s_idx, gains[s_idx])
    assert scaled.pi[s_idx].shape == relaxed.pi[s_idx].shape
    assert float(jnp.mean(scaled.pi[s_idx])) != float(jnp.mean(relaxed.pi[s_idx]))


def test_ne_beta_widens_exploration():
    """High NE ⇒ larger β ⇒ EFE selection favours the epistemic option."""
    np_ = init_neuromod_params()
    calm = init_neuromod_state(np_)
    # Force a high-NE state (volatile world) vs the calm baseline.
    aroused = eqx.tree_at(lambda s: s.ne, calm, jnp.asarray(0.9, calm.ne.dtype))
    beta_hi = neuromod_beta(aroused, np_)
    beta_lo = neuromod_beta(calm, np_)
    assert float(beta_hi) > float(beta_lo)
    # Two policies: A pragmatic, B epistemic. Large β should pick B.
    pragmatic = jnp.array([1.0, 0.0])
    epistemic = jnp.array([0.0, 1.0])
    pick_hi = efe_select(pragmatic, epistemic, epistemic_weight=beta_hi).index
    assert int(pick_hi) == 1, "aroused (high β) explores the epistemic policy"


def test_horizon_monotone_in_serotonin():
    np_ = init_neuromod_params()
    lo = eqx.tree_at(lambda s: s.sero, init_neuromod_state(np_), jnp.asarray(0.0))
    hi = eqx.tree_at(lambda s: s.sero, init_neuromod_state(np_), jnp.asarray(1.0))
    assert float(neuromod_horizon(hi, np_)) > float(neuromod_horizon(lo, np_))


def test_learning_progress_augments_epistemic_value():
    """Plan §U.5 ablation: adding LP changes the exploration term."""
    gp, relaxed = _relaxed()
    base = float(epistemic_value(relaxed, REGION_INDEX["world_model"]))
    with_lp = float(epistemic_value(relaxed, REGION_INDEX["world_model"],
                                    learning_progress=0.7))
    assert with_lp > base, "positive learning progress raises epistemic value"
    # Negative / zero LP is rectified — never lowers the base value.
    none_lp = float(epistemic_value(relaxed, REGION_INDEX["world_model"],
                                    learning_progress=-1.0))
    assert abs(none_lp - base) < 1e-6
