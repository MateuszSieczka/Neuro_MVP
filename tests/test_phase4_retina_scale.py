"""Phase 4 — retina scale invariance.

Core claim: ``retina_step`` maps any square image to the same fixed-size
afferent vector. This is the structural invariant that lets the rest of
the visual pipeline stay body-agnostic (Rodieck 1998; Felleman & Van
Essen 1991).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from sensory import RetinaConfig, init_retina_state, retina_step


@pytest.mark.parametrize("hw", [64, 128, 256, 512, 1024])
def test_retina_shape_invariant_across_scales(hw: int) -> None:
    cfg = RetinaConfig()
    state = init_retina_state(cfg)
    key = jax.random.PRNGKey(hw)
    img = jax.random.uniform(key, (hw, hw), minval=0.0, maxval=1.0)

    _, sample = retina_step(state, cfg, img, jnp.array([0.5, 0.5], jnp.float32))
    aff = sample.as_afferent()

    assert aff.shape == (cfg.afferent_size,), (
        f"expected ({cfg.afferent_size},), got {aff.shape}"
    )
    assert jnp.isfinite(aff).all()
    assert float(aff.min()) >= 0.0
    assert float(aff.max()) <= 1.0 + 1e-6


def test_retina_afferent_size_matches_cfg_arithmetic() -> None:
    cfg = RetinaConfig()
    expected = (
        2 * cfg.fovea_size * cfg.fovea_size
        + 2 * cfg.n_pyramid * cfg.periphery_tile * cfg.periphery_tile
        + cfg.periphery_tile * cfg.periphery_tile
    )
    assert cfg.afferent_size == expected


def test_retina_signal_present() -> None:
    """A non-trivial image must yield a non-constant afferent vector.

    Rodieck (1965) DoG extracts local luminance contrasts; any image with
    spatial structure should produce variance in both ON and OFF
    channels.
    """
    cfg = RetinaConfig()
    state = init_retina_state(cfg)
    size = 64
    yy, xx = jnp.mgrid[0:size, 0:size].astype(jnp.float32)
    # Vertical half-plane step (classic ON/OFF edge test).
    edge = (xx > size / 2).astype(jnp.float32)
    _, sample = retina_step(state, cfg, edge, jnp.array([0.5, 0.5], jnp.float32))

    aff = sample.as_afferent()
    assert float(aff.std()) > 0.0, "afferent must not be constant"
    # On a bright-on-left-dark-on-right edge the ON channel fires on
    # the bright side of the luminance step and OFF on the dark side, so
    # both channels must carry non-zero activity.
    assert float(sample.fovea_on.sum()) > 0.0
    assert float(sample.fovea_off.sum()) > 0.0
