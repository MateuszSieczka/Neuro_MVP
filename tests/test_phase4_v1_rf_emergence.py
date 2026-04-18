"""Phase 4 — V1 Gabor initialisation sanity.

Plan: ``sensory/v1.py`` — learned sparse coding area; initialised with
Gabor-like weights (Hubel & Wiesel 1962; Jones & Palmer 1987) as a
starting prior. RF sharpening through STDP is a long-timescale
phenomenon (Olshausen & Field 1996); we only assert the *starting
condition* is consistent with the V1 prior.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from core.backend import DEFAULT
from core.cortex import CorticalInputs, cortical_area_step
from sensory import (
    RetinaConfig,
    init_v1_params,
    init_v1_state,
    init_retina_state,
    retina_step,
    lgn_normalize,
)


def test_v1_builds_with_expected_shapes() -> None:
    cfg = RetinaConfig()
    params = init_v1_params(DEFAULT, cfg)
    state = init_v1_state(jax.random.PRNGKey(0), params, cfg)

    assert params.input_size == cfg.afferent_size
    assert state.w_l4_in.shape == (cfg.afferent_size, params.n_l4)
    assert jnp.isfinite(state.w_l4_in).all()
    # Half-normal init → strictly non-negative weights (cortex invariant).
    assert float(state.w_l4_in.min()) >= 0.0


def test_v1_gabor_init_produces_zero_mean_receptive_fields() -> None:
    """Gabor filters are zero-mean; stored as ON/OFF split they must
    preserve that balance (ON mean ≈ OFF mean) on the foveal channels.
    """
    cfg = RetinaConfig()
    params = init_v1_params(DEFAULT, cfg)
    state = init_v1_state(jax.random.PRNGKey(1), params, cfg, gabor_init=True)

    P = cfg.fovea_size * cfg.fovea_size
    on_block = state.w_l4_in[:P]
    off_block = state.w_l4_in[P : 2 * P]

    on_mean = float(on_block.mean())
    off_mean = float(off_block.mean())
    # Both blocks positive; balanced within factor 2 (random mixing adds
    # asymmetry but Gabor zero-mean property dominates).
    assert on_mean > 0.0 and off_mean > 0.0
    ratio = on_mean / off_mean
    assert 0.5 < ratio < 2.0, f"ON/OFF imbalance: {ratio}"


def test_v1_responds_to_lgn_afferent() -> None:
    """Full retina → LGN → V1 smoke: after 20 substeps the cortex must
    reach non-zero belief on at least some inputs. This is the same
    cadence as ``brain_graph.minimal_brain_step``.
    """
    cfg = RetinaConfig()
    params = init_v1_params(DEFAULT, cfg)
    state = init_v1_state(jax.random.PRNGKey(2), params, cfg)

    rstate = init_retina_state(cfg)
    size = 64
    yy, xx = jnp.mgrid[0:size, 0:size].astype(jnp.float32)
    img = 0.5 + 0.4 * jnp.sin(2 * jnp.pi * 3.0 * xx / size)
    _, sample = retina_step(rstate, cfg, img, jnp.array([0.5, 0.5], jnp.float32))
    aff = lgn_normalize(sample.as_afferent())

    inp = CorticalInputs(
        ff_input=aff, td_prediction=None,
        ach=jnp.asarray(0.5), da=jnp.asarray(0.5), ne=jnp.asarray(0.5),
        receptor_gain=jnp.asarray(1.0),
    )

    def body(c, _):
        out = cortical_area_step(c, params, DEFAULT, inp)
        return out.state, out.belief.max()

    _, belief_trace = jax.lax.scan(body, state, None, length=20)
    assert jnp.isfinite(belief_trace).all()
    # A grating (strong oriented signal) must recruit at least one L4 unit.
    assert float(belief_trace.max()) > 0.0
