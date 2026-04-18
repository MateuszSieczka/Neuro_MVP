"""P0.9 \u2014 Boredom drive via slow PE-EMA \u2192 NE bonus.

Yu & Dayan (2005): NE encodes unexpected uncertainty.  We operationalise
*boredom* as the gap between a slow long-term PE baseline and the
current PE \u2014 when the world becomes *more* predictable than average,
ReLU(pe_long \u2212 pe_now) > 0 and adds a NE bonus that raises
exploration gain.  Weight 0.05 per plan; not task-tuned.

Tests:
  1. pe_long is stored and rises under sustained PE.
  2. With low sustained PE first, then zero PE, boredom > 0 and NE is
     higher than without the bonus.
  3. Zero PE forever -> pe_long=0, boredom=0, NE at baseline (no drift).
  4. Constant PE -> pe_long converges to PE, boredom -> 0 eventually.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import BackendContext
from core.neuromodulator import (
    init_neuromodulator_params, init_neuromodulator_state,
    neuromodulator_step,
)


def _run(pe_sequence, *, boredom_weight=0.05, dt=1.0):
    ctx = BackendContext(dt=dt)
    params = init_neuromodulator_params(
        ctx, boredom_weight=boredom_weight, tau_pe_long=1_000.0,
    )
    state = init_neuromodulator_state(params)
    pe_longs = []
    nes = []
    for pe in pe_sequence:
        pe_arr = jnp.asarray([pe], jnp.float32)
        state = neuromodulator_step(state, params, pe_arr)
        pe_longs.append(float(state.pe_long))
        nes.append(float(state.noradrenaline))
    return state, pe_longs, nes


def test_pe_long_stored_and_rises():
    state, pe_longs, _ = _run([0.5] * 500)
    assert pe_longs[0] < pe_longs[-1]
    assert pe_longs[-1] > 0.15  # slow EMA, still rising towards 0.5


def test_boredom_lifts_ne_above_baseline_free_run():
    """After sustained high PE then sudden drop, NE should be elevated
    vs a run with boredom disabled."""
    seq = [0.8] * 2000 + [0.0] * 500
    _, _, nes_on = _run(seq, boredom_weight=0.2)
    _, _, nes_off = _run(seq, boredom_weight=0.0)
    # Over the low-PE tail, boredom-on should keep NE higher.
    tail_on = sum(nes_on[-200:]) / 200.0
    tail_off = sum(nes_off[-200:]) / 200.0
    assert tail_on > tail_off + 1e-3, (
        f"boredom bonus should raise NE: on={tail_on:.4f}, off={tail_off:.4f}"
    )


def test_zero_pe_forever_keeps_pe_long_zero():
    state, pe_longs, nes = _run([0.0] * 500)
    assert pe_longs[-1] < 1e-4
    # NE should hover near baseline 0.3.
    assert 0.25 < nes[-1] < 0.35


def test_constant_pe_converges_boredom_to_zero():
    # If pe == pe_long (both approach the same constant), boredom = 0.
    state, pe_longs, _ = _run([0.3] * 5000)
    assert abs(pe_longs[-1] - 0.3) < 0.05
