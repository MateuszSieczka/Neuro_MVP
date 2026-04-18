"""Phase 4 — V1→V2→V4/IT visual hierarchy shape plumbing.

Assert only structural compatibility: each area's input size must match
the preceding area's ``n_l23_error`` channel, and feedforward projection
shapes must line up so the hierarchy can be scanned end-to-end.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import DEFAULT
from core.cortex import CorticalInputs, cortical_area_step
from sensory import (
    RetinaConfig,
    init_v1_params, init_v1_state,
    init_v2_params, init_v2_state,
    init_v4it_params, init_v4it_state,
)


def test_ventral_hierarchy_shapes_compose() -> None:
    ret = RetinaConfig()
    v1p = init_v1_params(DEFAULT, ret)
    v2p = init_v2_params(DEFAULT, input_size=v1p.n_l23_error)
    v4p = init_v4it_params(DEFAULT, input_size=v2p.n_l23_error)

    assert v2p.input_size == v1p.n_l23_error
    assert v4p.input_size == v2p.n_l23_error
    # Sizes decreasing along the hierarchy (Felleman & Van Essen 1991).
    assert v1p.n_l4 >= v2p.n_l4 >= v4p.n_l4


def test_ventral_hierarchy_can_forward_propagate() -> None:
    ret = RetinaConfig()
    v1p = init_v1_params(DEFAULT, ret)
    v2p = init_v2_params(DEFAULT, input_size=v1p.n_l23_error)
    v4p = init_v4it_params(DEFAULT, input_size=v2p.n_l23_error)

    v1s = init_v1_state(jax.random.PRNGKey(0), v1p, ret)
    v2s = init_v2_state(jax.random.PRNGKey(1), v2p)
    v4s = init_v4it_state(jax.random.PRNGKey(2), v4p)

    aff = 0.25 * jnp.ones(ret.afferent_size, jnp.float32)

    def step_area(st, p, ff):
        inp = CorticalInputs(
            ff_input=ff, td_prediction=None,
            ach=jnp.asarray(0.5), da=jnp.asarray(0.5), ne=jnp.asarray(0.5),
            receptor_gain=jnp.asarray(1.0),
        )
        return cortical_area_step(st, p, DEFAULT, inp)

    out_v1 = step_area(v1s, v1p, aff)
    out_v2 = step_area(v2s, v2p, out_v1.ff_out)
    out_v4 = step_area(v4s, v4p, out_v2.ff_out)

    assert out_v1.ff_out.shape == (v1p.n_l23_error,)
    assert out_v2.ff_out.shape == (v2p.n_l23_error,)
    assert out_v4.belief.shape == (v4p.n_l4,)
    assert jnp.isfinite(out_v4.belief).all()
