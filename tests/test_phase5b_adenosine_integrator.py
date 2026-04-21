"""Phase 5B — adenosine integrator (Porkka-Heiskanen 1997).

Exponential rise during wake toward 1.0 and exponential fall during
SWS toward 0.0 at the prescribed τ's; values stay bounded to [0, 1].
"""

from __future__ import annotations

import jax.numpy as jnp

from core import (
    DEFAULT, init_neuromodulator_params, init_neuromodulator_state,
    adenosine_update, make_key,
)


def test_adenosine_rises_during_wake_and_falls_during_sws():
    ctx = DEFAULT
    # Short τ's so a few hundred ticks are enough to saturate.
    params = init_neuromodulator_params(
        ctx,
        tau_adenosine_rise_ms=200.0,
        tau_adenosine_fall_ms=100.0,
    )
    state = init_neuromodulator_state(params)

    # 400 ticks awake → should rise well above 0.5 (τ_rise = 200 ms,
    # ctx.dt = 1 ms → ~2 τ to reach 1−e⁻² ≈ 0.86).
    for _ in range(400):
        state = adenosine_update(state, params, is_awake=True)
    rose = float(state.adenosine)
    assert 0.7 <= rose <= 1.0, f"adenosine rise broken: {rose:.4f}"

    # 400 ticks SWS → should fall well below 0.2 (τ_fall = 100 ms,
    # ctx.dt = 1 ms → ~4 τ, residual ≈ e⁻⁴ ≈ 0.018 of the peak).
    for _ in range(400):
        state = adenosine_update(state, params, is_awake=False)
    fell = float(state.adenosine)
    assert fell < 0.2, f"adenosine fall broken: {fell:.4f}"

    # Bounds preserved.
    assert 0.0 <= fell <= 1.0
    assert 0.0 <= rose <= 1.0
