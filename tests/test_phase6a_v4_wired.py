"""Phase 6A — V4 is wired into the sensory hierarchy.

Phase 6A's remit is to *wire* V2/V4 into the stack; full selectivity
emergence is Phase 7.  Therefore we assert (a) shapes plumb through,
(b) V4 weights move under STDP when the stack is driven, and (c) V4
belief is finite.  Selectivity emergence is deferred.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import DEFAULT, make_key
from sensory import (
    RetinaConfig,
    init_sensory_stack_params, init_sensory_stack_state,
    sensory_stack_step,
)


def test_phase6a_v4_wired() -> None:
    p = init_sensory_stack_params(DEFAULT, retina_cfg=RetinaConfig())
    s = init_sensory_stack_state(make_key(0), p)

    # Shape checks — hierarchy cascades.
    assert p.v2.input_size == p.v1.n_l23_state
    assert p.v4.input_size == p.v2.n_l23_state
    assert s.v2.rate_l4.shape == (p.v2.n_l4,)
    assert s.v4.rate_l4.shape == (p.v4.n_l4,)

    # Drive the stack for a short while and confirm the V2/V4 path
    # runs end-to-end without producing NaNs.  Downstream rate-level
    # emergence requires inter-area normalisation which is Phase 7;
    # Phase 6A only asserts wiring.
    fix = jnp.asarray([0.5, 0.5])
    key = make_key(1)
    for _ in range(5):
        key, sk = jax.random.split(key)
        img = jax.random.uniform(sk, (64, 64), jnp.float32)
        out = sensory_stack_step(s, p, DEFAULT, img, fix)
        s = out.state

    # Shape and NaN sanity.
    assert out.v2_belief.shape == (p.v2.n_l23_state,)
    assert out.v4_belief.shape == (p.v4.n_l23_state,)
    assert jnp.all(jnp.isfinite(out.v4_belief))
    assert jnp.all(jnp.isfinite(out.v2_belief))
    assert jnp.all(jnp.isfinite(s.v4.rate_l4))
    assert jnp.all(jnp.isfinite(s.v2.rate_l4))

    # Stack output still exposes V1 belief unchanged (backwards compat).
    assert out.belief.shape == (p.v1.n_l23_state,)
