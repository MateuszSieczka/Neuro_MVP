"""Phase 4 — auditory pathway skeleton.

Plan: cochleogram 64 mel bands + MGN gain control + A1 cortical area.
Tests assert structural invariants — actual tuning curves are learned.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from core.backend import DEFAULT
from core.cortex import CorticalInputs, cortical_area_step
from sensory import (
    CochleaConfig,
    Cochleogram,
    cochlea_step,
    mgn_normalize,
    init_a1_params,
    init_a1_state,
)


def _tone(freq_hz: float, cfg: CochleaConfig, seed: int = 0) -> jnp.ndarray:
    t = jnp.arange(cfg.window_size, dtype=jnp.float32) / cfg.sample_rate
    tone = 0.5 * jnp.sin(2 * jnp.pi * freq_hz * t)
    noise = 0.02 * jax.random.normal(jax.random.PRNGKey(seed), (cfg.window_size,))
    return tone + noise


def test_cochlea_shape_invariant() -> None:
    cfg = CochleaConfig()
    wave = _tone(1000.0, cfg, seed=0)
    cg = cochlea_step(cfg, wave, jnp.array([0.5, 1.0], jnp.float32))
    assert isinstance(cg, Cochleogram)
    assert cg.bands.shape == (cfg.n_bands, cfg.n_frames)
    assert cg.as_afferent().shape == (cfg.afferent_size,)
    assert jnp.isfinite(cg.bands).all()


@pytest.mark.parametrize("freq_hz,expect_low", [
    (200.0, True),    # low tone → peak in lower mel bands
    (4000.0, False),  # high tone → peak in upper mel bands
])
def test_cochleogram_frequency_selectivity(freq_hz: float, expect_low: bool) -> None:
    """A pure tone must concentrate energy in a subset of mel bands
    consistent with its frequency (Stevens & Volkmann 1940)."""
    cfg = CochleaConfig()
    wave = _tone(freq_hz, cfg)
    cg = cochlea_step(cfg, wave, jnp.array([0.5, 1.0], jnp.float32))
    peak_band = int(cg.bands.sum(axis=1).argmax())

    if expect_low:
        assert peak_band < cfg.n_bands // 2, f"200 Hz should land low, got band {peak_band}"
    else:
        assert peak_band > cfg.n_bands // 2, f"4 kHz should land high, got band {peak_band}"


def test_mgn_normalisation_matches_cortical_operating_range() -> None:
    cfg = CochleaConfig()
    wave = _tone(1000.0, cfg)
    cg = cochlea_step(cfg, wave, jnp.array([0.5, 1.0], jnp.float32))
    aff = mgn_normalize(cg.as_afferent())
    # Same operating range as LGN (target_mean=0.25, baseline=0.15).
    mean = float(aff.mean())
    assert 0.05 < mean < 0.6, f"MGN mean {mean} outside cortical range"
    assert float(aff.min()) >= 0.0
    assert float(aff.max()) <= 1.0 + 1e-6


def test_a1_end_to_end_tone_response() -> None:
    cfg = CochleaConfig()
    params = init_a1_params(DEFAULT, cfg)
    state = init_a1_state(jax.random.PRNGKey(0), params)

    wave = _tone(1000.0, cfg)
    cg = cochlea_step(cfg, wave, jnp.array([0.5, 1.0], jnp.float32))
    aff = mgn_normalize(cg.as_afferent())

    inp = CorticalInputs(
        ff_input=aff, td_prediction=None,
        ach=jnp.asarray(0.5), da=jnp.asarray(0.5), ne=jnp.asarray(0.5),
        receptor_gain=jnp.asarray(1.0),
    )

    def body(c, _):
        out = cortical_area_step(c, params, DEFAULT, inp)
        return out.state, out.belief.max()

    _, trace = jax.lax.scan(body, state, None, length=20)
    assert jnp.isfinite(trace).all()
    # Pure tone with MGN-normalised drive must recruit at least one unit.
    assert float(trace.max()) > 0.0
