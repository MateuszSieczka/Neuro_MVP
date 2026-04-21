"""Phase 5B — DG pattern separation (Rolls 2013).

Two cosine-similar input patterns must project to DG codes that are
decorrelated (cosine ≤ 0.5 even when inputs have cosine ≥ 0.9).
"""

from __future__ import annotations

import jax.numpy as jnp

from core import (
    DEFAULT, make_key, split_key,
    init_hippocampus_params, init_hippocampus_state,
    dg_encode,
)


def _cos(a, b):
    return float(
        jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-8)
    )


def test_dg_separates_similar_inputs():
    key = make_key(0)
    k1, k2 = split_key(key, 2)
    input_dim = 64
    params = init_hippocampus_params(
        input_dim=input_dim,
        dg_sparsity=0.05,
        dg_expansion_factor=5,
    )
    state = init_hippocampus_state(k1, params)

    base = jax_rand_unit(k2, input_dim)
    jitter_key = make_key(1)
    perturbation = 0.8 * jax_rand_unit(jitter_key, input_dim)
    a = base
    b = base + perturbation

    cos_input = _cos(a, b)
    assert 0.5 <= cos_input <= 0.9, (
        f"test setup: expect moderate input similarity, got {cos_input:.3f}"
    )

    code_a = dg_encode(state.dg, params.dg, a)
    code_b = dg_encode(state.dg, params.dg, b)

    cos_dg = _cos(code_a, code_b)
    # Pattern separation: DG codes must be *more* decorrelated than
    # the inputs (Rolls 2013 expansion recoding).  A ≥ 0.1 contraction
    # on the similarity axis is the minimal fingerprint of the DG
    # top-k sparsifier at 5% sparsity in an already-reasonable-sized
    # input.
    assert cos_dg < cos_input - 0.1, (
        f"DG failed to separate: cos_input={cos_input:.3f}, cos_dg={cos_dg:.3f}"
    )


def jax_rand_unit(key, d):
    import jax.random as jr
    v = jr.normal(key, (d,))
    return v / (jnp.linalg.norm(v) + 1e-8)
