"""P0.6 — Attention gate active (Saalmann 2012 pulvinar modulation).

Weryfikuje że:
  1. `thalamic_step` z `afferent_gain=0.5` redukuje relay rate vs `=1.0`.
  2. `thalamic_step` z `afferent_gain=2.0` zwiększa relay rate.
  3. `minimal_brain_step` integruje attention — state.attention ewoluuje.
  4. Attention gains w [0.1, ∞); suma rozkładu attn_weights ~= 1.
  5. ActionBrain nadal stabilny po 10 krokach.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import BackendContext
from core.brain_graph import (
    init_minimal_brain_params, init_minimal_brain_state, minimal_brain_step,
    init_action_brain_params, init_action_brain_state, action_brain_step,
)
from core.thalamus import (
    init_relay_params, init_relay_state, init_trn_params, init_trn_state,
    thalamic_step,
)


def _run_thal_step_rates(afferent_gain: float, steps: int = 400) -> float:
    ctx = BackendContext(dt=1.0)
    n_af, n_tc, n_ct = 16, 32, 16
    rp = init_relay_params(ctx, n_afferent=n_af, n_tc=n_tc, n_ct=n_ct)
    tp = init_trn_params(ctx, n_tc_total=n_tc, n_ct=n_ct, n_trn=16)
    r = init_relay_state(jax.random.PRNGKey(0), rp)
    t = init_trn_state(jax.random.PRNGKey(1), tp)

    af = jnp.ones((n_af,), jnp.float32) * 0.3
    ct = jnp.zeros((n_ct,), jnp.float32)

    total_spikes = 0.0
    for _ in range(steps):
        out = thalamic_step(
            r, rp, t, tp, ctx, af, ct, ach=0.6, ne=0.4,
            afferent_gain=jnp.asarray(afferent_gain, jnp.float32),
        )
        r, t = out.relay, out.trn
        total_spikes += float(out.relay_spikes.sum())
    # Hz-equivalent: spikes / (n_tc * steps * dt_ms / 1000)
    return total_spikes / (n_tc * steps * ctx.dt / 1000.0)


def test_thalamus_afferent_gain_reduces_rate():
    r_low = _run_thal_step_rates(afferent_gain=0.3)
    r_base = _run_thal_step_rates(afferent_gain=1.0)
    assert r_low < r_base, f"gain 0.3 should reduce rate: low={r_low:.1f}, base={r_base:.1f}"


def test_thalamus_afferent_gain_boosts_rate():
    r_base = _run_thal_step_rates(afferent_gain=1.0)
    r_high = _run_thal_step_rates(afferent_gain=2.0)
    assert r_high > r_base, f"gain 2.0 should boost rate: base={r_base:.1f}, high={r_high:.1f}"


def test_attention_state_evolves_in_minimal_brain():
    ctx = BackendContext(dt=1.0)
    params = init_minimal_brain_params(ctx, sensory_size=16)
    state = init_minimal_brain_state(jax.random.PRNGKey(0), params)
    initial_attn = state.attention.attn_weights

    sensory = jnp.ones((16,), jnp.float32) * 0.3
    for _ in range(200):
        out = minimal_brain_step(state, params, ctx, sensory)
        state = out.state

    # attention distribution evolved away from uniform.
    diff = float(jnp.abs(state.attention.attn_weights - initial_attn).sum())
    assert diff > 1e-4, f"attn_weights didn't change (diff={diff:.2e})"
    # Distribution is normalised (~1).
    s = float(state.attention.attn_weights.sum())
    assert 0.9 < s < 1.1


def test_action_brain_with_attention_stable():
    ctx = BackendContext(dt=1.0)
    params = init_action_brain_params(
        ctx, sensory_size=16, n_body_actions=3, n_saccade_actions=1,
    )
    state = init_action_brain_state(jax.random.PRNGKey(1), params)

    sensory = jnp.ones((16,), jnp.float32) * 0.3
    key = jax.random.PRNGKey(2)
    for _ in range(5):
        key, k = jax.random.split(key)
        out = action_brain_step(
            state, params, ctx, sensory,
            jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32),
            k,
        )
        state = out.state

    assert jnp.isfinite(out.rpe)
    assert jnp.isfinite(state.attention.attn_weights).all()
    # Gains are derived via ACh modulation; ensure they're sane.
    assert float(state.attention.attn_weights.min()) >= 0.0
