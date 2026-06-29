"""Faza U — §2 vision adapter: retina → LGN → the sensory clamp.

Asserts the retina's defining property (scale-invariance: a fixed-length
afferent regardless of input resolution), the LGN normalisation range +
tonic floor, the composed ``vision_encode`` dimensionality, and that a
brain sized to the afferent accepts the encoded image as its sensory clamp.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sensory.retina import RetinaConfig, init_retina_state, retina_step
from sensory.lgn import lgn_normalize, LGN_BASELINE
from sensory.vision import (
    init_vision_params, init_vision_state, vision_encode, init_vision_brain,
)
from core.pc_brain import pc_brain_cognitive_step


def _blob_image(key, h, w):
    """A smooth synthetic greyscale image in [0, 1]."""
    img = jax.random.uniform(key, (h, w))
    # light blur so DoG has structure, not pure noise
    k = jnp.array([0.25, 0.5, 0.25])
    img = jnp.apply_along_axis(lambda r: jnp.convolve(r, k, "same"), 1, img)
    return jnp.clip(img, 0.0, 1.0)


def test_retina_afferent_is_scale_invariant():
    cfg = RetinaConfig(fovea_size=8, n_pyramid=2, periphery_tile=4)
    st = init_retina_state(cfg)
    fix = jnp.array([0.5, 0.5])
    img_small = _blob_image(jax.random.PRNGKey(0), 32, 32)
    img_big = _blob_image(jax.random.PRNGKey(1), 96, 128)
    _, s_small = retina_step(st, cfg, img_small, fix)
    _, s_big = retina_step(st, cfg, img_big, fix)
    a_small, a_big = s_small.as_afferent(), s_big.as_afferent()
    assert a_small.shape == (cfg.afferent_size,)
    assert a_big.shape == (cfg.afferent_size,), "afferent size independent of input res"
    assert jnp.all(jnp.isfinite(a_small)) and jnp.all(jnp.isfinite(a_big))
    assert float(a_small.min()) >= 0.0 and float(a_small.max()) <= 1.0


def test_fovea_block_layout_matches_afferent():
    cfg = RetinaConfig(fovea_size=8, n_pyramid=2, periphery_tile=4)
    # ON then OFF foveal blocks lead the afferent, each fovea_block long.
    assert cfg.fovea_block == 64
    assert cfg.afferent_size == 2 * 64 + 2 * 2 * 16 + 16


def test_lgn_lifts_floor_and_bounds_range():
    # Sparse rectified-DoG-like afferent: mostly zeros, a few edges.
    afferent = jnp.zeros(200).at[jnp.array([3, 50, 120])].set(0.8)
    out = lgn_normalize(afferent)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert float(out.min()) > 0.0, "tonic baseline lifts the silent channels off the floor"
    assert abs(float(out.min()) - LGN_BASELINE) < 1e-3, "silent channels sit at the tonic floor"
    assert float(out.mean()) > float(afferent.mean()), "gain control raises the operating point"


def test_vision_encode_dims_and_clamps_brain():
    vp = init_vision_params(RetinaConfig(fovea_size=8, n_pyramid=2, periphery_tile=4))
    rst = init_vision_state(vp)
    img = _blob_image(jax.random.PRNGKey(2), 48, 48)
    new_rst, afferent = vision_encode(rst, vp, img, jnp.array([0.5, 0.5]))
    assert afferent.shape == (vp.afferent_size,)
    # A brain sized to the afferent takes it as the sensory clamp.
    _, bp, bs = init_vision_brain(jax.random.PRNGKey(3), vp, motor_size=4, gabor=False)
    assert bp.sensory_dim == vp.afferent_size
    out = pc_brain_cognitive_step(bs, bp, afferent, learn=False)
    assert jnp.isfinite(out.free_energy), "encoded image relaxes to a finite free energy"
    assert out.belief.shape == (bp.graph.node_sizes[bp.cortex_top_idx],)
