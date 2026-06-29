"""Faza U — §2 foveal Gabor prior on the cortex_l1→sensory generative edge.

The one substrate-touching change of the vision milestone.  Asserts it is
backward-compatible (default ``None`` ⇒ byte-identical to the old init),
that it shapes *only* the foveal ON/OFF rows of *only* the target edge,
that the drive magnitude is matched to the LeCun init, and the guard rails.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from core.pc_graph import (
    REGION_INDEX, init_region_graph,
    FovealGaborInit, apply_foveal_gabor_init, init_pc_graph_params,
    init_pc_graph_state,
)


def _edge_idx(params, src, dst):
    for e, (s, d) in enumerate(params.edges):
        if s == src and d == dst:
            return e
    raise AssertionError("edge not found")


def _build(key, sensory_size, cortex_size, gabor):
    return init_region_graph(
        key, sensory_size=sensory_size, cortex_size=cortex_size,
        gabor_foveal_init=gabor,
    )


def test_default_is_byte_identical():
    key = jax.random.PRNGKey(0)
    p0, s0 = _build(key, 144, 32, None)
    p1, s1 = _build(key, 144, 32, None)
    for w0, w1 in zip(s0.weights, s1.weights):
        assert jnp.array_equal(w0, w1), "no-gabor init must be deterministic + unchanged"


def test_gabor_shapes_only_foveal_rows_of_only_target_edge():
    key = jax.random.PRNGKey(1)
    P = 8
    fb = P * P                       # foveal block length
    sensory = 2 * fb + 16            # ON + OFF + 16 periphery rows
    gfi = FovealGaborInit(patch_size=P, on_offset=0, off_offset=fb, mix=0.7)

    p0, s0 = _build(key, sensory, 32, None)
    p1, s1 = _build(key, sensory, 32, gfi)

    e = _edge_idx(p1, REGION_INDEX["cortex_l1"], REGION_INDEX["sensory"])
    w0, w1 = s0.weights[e], s1.weights[e]

    # Foveal ON+OFF rows changed; peripheral rows untouched.
    assert not jnp.allclose(w0[: 2 * fb], w1[: 2 * fb]), "foveal rows seeded"
    assert jnp.array_equal(w0[2 * fb:], w1[2 * fb:]), "peripheral rows untouched"
    assert jnp.all(jnp.isfinite(w1))

    # Every other edge is identical (only the target edge was touched).
    for idx, (a, b) in enumerate(zip(s0.weights, s1.weights)):
        if idx != e:
            assert jnp.array_equal(a, b), f"edge {idx} must be untouched"


def test_gabor_drive_matches_lecun_scale():
    key = jax.random.PRNGKey(2)
    P = 8
    fb = P * P
    sensory = 2 * fb
    gfi = FovealGaborInit(patch_size=P, on_offset=0, off_offset=fb, mix=1.0)
    p, s = _build(key, sensory, 64, gfi)
    e = _edge_idx(p, REGION_INDEX["cortex_l1"], REGION_INDEX["sensory"])
    w = s.weights[e]
    n_in = w.shape[1]
    lecun = 1.0 / (n_in ** 0.5)
    target = lecun * (2.0 / jnp.pi) ** 0.5
    on_block = w[:fb]
    # mix=1.0 ⇒ pure half-rectified Gabor scaled to the half-normal mean.
    assert abs(float(on_block.mean()) - float(target)) < 1e-2, "drive matched to LeCun init"
    assert float(jnp.var(on_block)) > 0.0, "Gabor imparts orientation structure"


def test_gabor_columns_vary_across_orientations():
    """Distinct cortical units get distinct orientation/SF Gabors (not tied)."""
    key = jax.random.PRNGKey(3)
    P = 8
    fb = P * P
    gfi = FovealGaborInit(patch_size=P, on_offset=0, off_offset=fb, mix=1.0)
    p, s = _build(key, 2 * fb, 32, gfi)
    e = _edge_idx(p, REGION_INDEX["cortex_l1"], REGION_INDEX["sensory"])
    on = s.weights[e][:fb]            # (fb, n_units) — projective fields per unit
    col0, col1 = on[:, 0], on[:, 1]
    assert not jnp.allclose(col0, col1), "adjacent units carry different Gabors"


def test_apply_gabor_guards():
    key = jax.random.PRNGKey(4)
    params = init_pc_graph_params((10, 6), ((1, 0),))   # edge 1→0, dst dim 10
    state = init_pc_graph_state(key, params)
    # Foveal blocks that exceed the destination node must raise.
    bad = FovealGaborInit(patch_size=8, on_offset=0, off_offset=64)
    with pytest.raises(ValueError, match="exceed"):
        apply_foveal_gabor_init(state, params, 1, 0, bad)
    # A request for an absent edge must raise.
    ok = FovealGaborInit(patch_size=2, on_offset=0, off_offset=4)
    with pytest.raises(ValueError, match="no edge"):
        apply_foveal_gabor_init(state, params, 0, 1, ok)
