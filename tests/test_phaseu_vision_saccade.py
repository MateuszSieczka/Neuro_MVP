"""Faza U — §2 active saccades + the rebuilt synthetic visual-grid validation.

Two things the milestone must show on the live substrate:

1. **Saccade info-gain drives fixation** — the fovea moves to the most
   surprising location (Bayesian surprise = sensory ``|ε|``), selected by
   ``efe_select`` (argmin expected free energy).
2. **The deep cortical node carries the cause** — clamping the encoded
   image and relaxing leaves a cell-distinctive belief on ``cortex_l3``
   (the rebuilt ``visual_grid`` from §1, on the new interface), and
   learning on a clamped image reduces free energy.

Synthetic images only (no MuJoCo); small retina to stay CPU-cheap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sensory.retina import RetinaConfig
from sensory.vision import (
    init_vision_params, init_vision_state, vision_encode,
    init_vision_brain, saccade_info_gain, select_fixation,
)
from core.pc_brain import pc_brain_cognitive_step


_CFG = RetinaConfig(fovea_size=8, n_pyramid=2, periphery_tile=4)


def _checker(h, w, block):
    yy, xx = jnp.mgrid[0:h, 0:w]
    return (((yy // block) + (xx // block)) % 2).astype(jnp.float32)


def _grating(h, w, theta, freq):
    yy, xx = jnp.mgrid[0:h, 0:w].astype(jnp.float32) / max(h, w)
    proj = jnp.cos(theta) * xx + jnp.sin(theta) * yy
    return (0.5 + 0.5 * jnp.sin(2.0 * jnp.pi * freq * proj)).astype(jnp.float32)


_CHECKER_FIX = jnp.array([0.25, 0.25])   # (x, y) = top-left quadrant centre
_FLAT_FIX = jnp.array([0.75, 0.75])      # bottom-right (bland) quadrant centre


def _patchy_image():
    """High-contrast checker in the top-left quadrant, flat grey elsewhere."""
    img = jnp.full((64, 64), 0.5, jnp.float32)
    img = img.at[0:32, 0:32].set(_checker(32, 32, 2))
    return img


def test_saccade_info_gain_higher_on_surprising_region():
    """Against a flat prior the surprise signal is bottom-up saliency: the
    fovea is drawn to the high-contrast patch, not the bland region."""
    vp = init_vision_params(_CFG)
    rst = init_vision_state(vp)
    img = _patchy_image()
    _, bp, bs = init_vision_brain(jax.random.PRNGKey(0), vp, motor_size=4, gabor=True)
    g_checker = saccade_info_gain(bs, bp, rst, vp, img, _CHECKER_FIX)
    g_flat = saccade_info_gain(bs, bp, rst, vp, img, _FLAT_FIX)
    assert float(g_checker) > float(g_flat), \
        "the structured patch violates the flat prior more than the bland region"


def test_select_fixation_picks_most_informative():
    vp = init_vision_params(_CFG)
    rst = init_vision_state(vp)
    img = _patchy_image()
    _, bp, bs = init_vision_brain(jax.random.PRNGKey(1), vp, motor_size=4, gabor=True)
    candidates = jnp.array([
        _CHECKER_FIX,           # checker (informative)
        _FLAT_FIX,              # bland
        [0.75, 0.25],           # bland
        [0.25, 0.75],           # bland
    ])
    choice = select_fixation(bs, bp, rst, vp, img, candidates)
    assert int(choice.index) == 0, "fovea moves to the checker quadrant"
    assert int(jnp.argmax(choice.info_gain)) == 0
    assert jnp.allclose(choice.fixation, candidates[0])


def test_deep_cortical_node_carries_the_cause():
    """Rebuilt visual_grid: distinct cells → distinct cortex_l3 beliefs."""
    vp = init_vision_params(_CFG)
    _, bp, bs = init_vision_brain(jax.random.PRNGKey(2), vp, motor_size=4, gabor=True)
    fix = jnp.array([0.5, 0.5])

    # Four visually distinct "grid cells".
    cells = [
        _grating(64, 64, 0.0, 4.0),
        _grating(64, 64, jnp.pi / 2, 4.0),
        _grating(64, 64, jnp.pi / 4, 8.0),
        _checker(64, 64, 4),
    ]
    beliefs = []
    for c in cells:
        rst = init_vision_state(vp)
        _, aff = vision_encode(rst, vp, c, fix)
        out = pc_brain_cognitive_step(bs, bp, aff, learn=False)
        beliefs.append(out.belief)

    # The deep node separates the cells (each pair of beliefs differs).
    for i in range(len(beliefs)):
        for j in range(i + 1, len(beliefs)):
            assert not jnp.allclose(beliefs[i], beliefs[j], atol=1e-4), \
                f"cells {i},{j} must get distinct deep causes"

    # The cause is deterministic: same cell → same belief.
    rst = init_vision_state(vp)
    _, aff0 = vision_encode(rst, vp, cells[0], fix)
    again = pc_brain_cognitive_step(bs, bp, aff0, learn=False)
    assert jnp.allclose(again.belief, beliefs[0], atol=1e-5)


def test_learning_on_clamped_image_reduces_free_energy():
    vp = init_vision_params(_CFG)
    _, bp, bs = init_vision_brain(jax.random.PRNGKey(3), vp, motor_size=4, gabor=True)
    rst = init_vision_state(vp)
    img = _grating(64, 64, jnp.pi / 3, 6.0)
    _, aff = vision_encode(rst, vp, img, jnp.array([0.5, 0.5]))

    fe0 = float(pc_brain_cognitive_step(bs, bp, aff, learn=False).free_energy)
    state = bs
    for _ in range(15):
        out = pc_brain_cognitive_step(state, bp, aff, learn=True)
        state = out.state
    fe1 = float(pc_brain_cognitive_step(state, bp, aff, learn=False).free_energy)
    assert fe1 < fe0, "the substrate learns the clamped image (free energy falls)"
