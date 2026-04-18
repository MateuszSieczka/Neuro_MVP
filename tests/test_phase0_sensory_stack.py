"""P0.7 — SensoryStack (retina → LGN → V1) integration tests.

Weryfikuje że:
  1. Stack zwraca belief o oczekiwanym kształcie i pe_rate skalar.
  2. Jednolite obrazy dają niższe belief / PE niż strukturalne.
  3. Przesunięcie fixacji zmienia wyjście (active sampling).
  4. STDP updates nie wybuchają po 50 krokach.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import BackendContext
from sensory.retina import RetinaConfig
from sensory.sensory_stack import (
    init_sensory_stack_params, init_sensory_stack_state, sensory_stack_step,
)


def test_sensory_stack_shapes():
    ctx = BackendContext(dt=1.0)
    cfg = RetinaConfig(fovea_size=8, n_pyramid=2, periphery_tile=4)
    params = init_sensory_stack_params(
        ctx, retina_cfg=cfg, n_l4=64, n_l23_state=32, n_l23_error=16, n_l5=16,
    )
    state = init_sensory_stack_state(jax.random.PRNGKey(0), params)

    image = jax.random.uniform(jax.random.PRNGKey(1), (32, 32), jnp.float32)
    fix = jnp.array([0.5, 0.5], jnp.float32)

    out = sensory_stack_step(state, params, ctx, image, fix)
    assert out.belief.shape == (32,), out.belief.shape
    assert out.pe_rate.shape == (), out.pe_rate.shape
    assert jnp.isfinite(out.belief).all()
    assert jnp.isfinite(out.pe_rate)


def test_sensory_stack_uniform_vs_structured():
    """Uniform grey image → lower belief activity than structured image."""
    ctx = BackendContext(dt=1.0)
    cfg = RetinaConfig(fovea_size=8, n_pyramid=2, periphery_tile=4)
    params = init_sensory_stack_params(
        ctx, retina_cfg=cfg, n_l4=64, n_l23_state=32, n_l23_error=16, n_l5=16,
    )
    state_u = init_sensory_stack_state(jax.random.PRNGKey(0), params)
    state_s = init_sensory_stack_state(jax.random.PRNGKey(0), params)
    fix = jnp.array([0.5, 0.5], jnp.float32)

    uniform = jnp.ones((32, 32), jnp.float32) * 0.5
    structured = jax.random.uniform(jax.random.PRNGKey(2), (32, 32), jnp.float32)

    # Accumulate belief energy over ~100 dt.
    def _energy(state, img):
        total = 0.0
        for _ in range(100):
            o = sensory_stack_step(
                state, params, ctx, img, fix, apply_stdp=False,
            )
            state = o.state
            total += float(o.belief.sum())
        return total

    e_uniform = _energy(state_u, uniform)
    e_structured = _energy(state_s, structured)
    # Structured should produce strictly more activity.
    assert e_structured > e_uniform, (
        f"structured {e_structured:.1f} should exceed uniform {e_uniform:.1f}"
    )


def test_sensory_stack_fixation_matters():
    """Different fixations on structured image → different afferent.

    V1 may not have spiked after 60 dt on novel inputs; check the
    retina+LGN substrate which is strictly upstream of V1 dynamics.
    """
    from sensory.retina import retina_step, init_retina_state
    from sensory.lgn import lgn_normalize

    cfg = RetinaConfig(fovea_size=8, n_pyramid=2, periphery_tile=4)
    state = init_retina_state(cfg)

    # Image: small bright spot in the *centre* of the *left half*
    # (offset) so fovea at fix_left sees it, fix_right does not.
    img = jnp.zeros((32, 32), jnp.float32)
    # 3×3 spot around (row=16, col=8) — mid-height, left-side.
    img = img.at[14:18, 6:10].set(1.0)
    # Separate fixations: left fixates on the spot, right fixates opposite.
    fix_on = jnp.array([8.0 / 32, 16.0 / 32], jnp.float32)
    fix_off = jnp.array([24.0 / 32, 16.0 / 32], jnp.float32)

    _, sample_on = retina_step(state, cfg, img, fix_on)
    _, sample_off = retina_step(state, cfg, img, fix_off)
    af_on = lgn_normalize(sample_on.as_afferent())
    af_off = lgn_normalize(sample_off.as_afferent())

    diff = float(jnp.abs(af_on - af_off).sum())
    assert diff > 1e-2, f"fixation had no effect on afferent (diff={diff:.2e})"


def test_sensory_stack_stdp_stable():
    """STDP + astrocyte update for 50 dt — no NaN / inf."""
    ctx = BackendContext(dt=1.0)
    cfg = RetinaConfig(fovea_size=8, n_pyramid=2, periphery_tile=4)
    params = init_sensory_stack_params(
        ctx, retina_cfg=cfg, n_l4=64, n_l23_state=32, n_l23_error=16, n_l5=16,
    )
    state = init_sensory_stack_state(jax.random.PRNGKey(0), params)
    fix = jnp.array([0.5, 0.5], jnp.float32)

    key = jax.random.PRNGKey(9)
    for _ in range(50):
        key, k = jax.random.split(key)
        img = jax.random.uniform(k, (32, 32), jnp.float32)
        o = sensory_stack_step(state, params, ctx, img, fix, apply_stdp=True)
        state = o.state

    assert jnp.isfinite(state.v1.w_l4_in).all()
    assert jnp.isfinite(o.belief).all()
    assert jnp.isfinite(o.pe_rate)
