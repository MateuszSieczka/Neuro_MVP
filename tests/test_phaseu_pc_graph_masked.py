"""Partial (per-dimension) clamping on the PC graph.

The substrate primitive a *partial preference* needs: pin a subset of a
node's dimensions (a goal on some channels, an occluded observation) while
the rest are inferred.  Asserts:

* whole-node clamping is unchanged — ``clamp=(j,)`` ≡ an all-True mask;
* a partial mask holds exactly its dimensions and frees the others;
* held dimensions still drive their predictors (the point of a goal clamp);
* a node cannot be both whole-clamped and partially clamped.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from core.pc_graph import (
    init_pc_graph_params, init_pc_graph_state,
    pc_graph_clamp, pc_graph_relax,
)


def _graph():
    # motor(node 1, size 3) → outcome(node 0, size 4); linear so the math is
    # transparent.  node 0 has no outgoing edge → its free dims move under ε_0.
    p = init_pc_graph_params((4, 3), ((1, 0),), act="linear", n_relax=8)
    s = init_pc_graph_state(jax.random.PRNGKey(0), p)
    s = pc_graph_clamp(s, {0: jnp.arange(4, dtype=jnp.float32),
                           1: jnp.ones(3)})
    return p, s


def test_whole_node_clamp_equals_all_true_mask():
    p, s = _graph()
    by_tuple = pc_graph_relax(s, p, clamp=(0,), n_steps=8)
    by_mask = pc_graph_relax(
        s, p, clamp_masks={0: jnp.ones(4, bool)}, n_steps=8,
    )
    for a, b in zip(by_tuple.mu, by_mask.mu):
        assert jnp.allclose(a, b), "all-True mask must equal whole-node clamp"


def test_partial_mask_holds_only_masked_dims():
    p, s = _graph()
    mask = jnp.array([True, False, True, False])
    out = pc_graph_relax(s, p, clamp_masks={0: mask}, n_steps=8)
    before, after = s.mu[0], out.mu[0]
    # Held dims unchanged; free dims moved (node 0 carries a nonzero error).
    assert jnp.allclose(after[mask], before[mask]), "masked dims must be held"
    assert not jnp.allclose(after[~mask], before[~mask]), "free dims must move"


def test_held_dims_drive_their_predictors():
    p, s = _graph()
    # Pin the whole outcome node away from the prediction → its parent (motor)
    # must move to reduce the error it sources.
    mask = jnp.ones(4, bool)
    out = pc_graph_relax(s, p, clamp_masks={0: mask}, n_steps=8)
    assert not jnp.allclose(out.mu[1], s.mu[1]), (
        "a pinned observation must drive the node that predicts it"
    )


def test_double_clamp_rejected():
    p, s = _graph()
    with pytest.raises(ValueError, match="both a whole-node clamp"):
        pc_graph_relax(s, p, clamp=(0,), clamp_masks={0: jnp.ones(4, bool)})
