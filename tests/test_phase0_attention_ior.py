"""Phase 0 micro-test: spatial-attention inhibition of return (IOR).

Plan (P0.11): "attention_step + attention_learn → po fixacji rejonu jego
gain spada o ≥30% (IOR działa)."

Protocol (Posner & Cohen 1984):
  1. **Fixation phase**: drive attention to a single column TARGET for
     600 dt with strong bottom-up saliency. ``ior_trace[TARGET]`` grows
     toward 1.0 (τ_IOR = 400 ms by default).
  2. **Disengage phase**: present zero stimulus for 80 dt so that
     ``attn_weights`` smoothes back toward uniform (smoothing τ ≈ 10 ms),
     while ``ior_trace`` decays slowly and remains substantially elevated.
  3. **Re-probe**: present the SAME bottom-up stimulus at TARGET. Compare
     the resulting attention weight at TARGET against a baseline probe
     run on a fresh, never-fixated attention state. The attended weight
     after IOR build-up must be ≥ 30% lower than baseline.

We additionally check the mechanistic preconditions: ior_trace must
grow during fixation and remain elevated after disengage (otherwise
the test is vacuous).

References
----------
- Posner & Cohen (1984) — IOR after ~300 ms.
- Reynolds & Heeger (2009) — divisive normalisation gain model.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from core.backend import BackendContext
from core.attention import (
    init_attention_params, init_attention_state, attention_step,
)


N_COL = 16
N_ASSOC = 8
TARGET = 5
FIX_STEPS = 600     # > ior_tau (400 ms) so trace approaches saturation
DISENGAGE_STEPS = 80   # > smoothing_tau (~10 ms), << ior_tau (400 ms)
PLAN_DROP_THRESHOLD = 0.30


def _build():
    ctx = BackendContext(dt=1.0)
    params = init_attention_params(ctx)
    return ctx, params


def _drive(state, params, bu, n_steps):
    assoc = jnp.zeros(N_ASSOC, jnp.float32)

    def step(carry, _):
        out = attention_step(
            carry, params, assoc, bottom_up_errors=bu,
            global_ach=1.0, ne_level=0.5,
        )
        return out.state, None

    state, _ = jax.lax.scan(step, state, jnp.arange(n_steps))
    return state


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ior_suppresses_recently_attended_region(seed):
    """After fixating column TARGET, re-probing it produces ≥30% lower
    attention weight than a baseline probe on a fresh state.
    """
    ctx, params = _build()
    state0 = init_attention_state(jax.random.PRNGKey(seed), N_ASSOC, N_COL)
    bu_target = jnp.zeros(N_COL, jnp.float32).at[TARGET].set(1.0)
    bu_off = jnp.zeros(N_COL, jnp.float32)
    assoc = jnp.zeros(N_ASSOC, jnp.float32)

    # 1) Fixation builds IOR trace at TARGET.
    state_fix = _drive(state0, params, bu_target, FIX_STEPS)
    ior_after_fix = float(state_fix.ior_trace[TARGET])
    assert ior_after_fix > 0.5, (
        f"seed={seed}: ior_trace[TARGET]={ior_after_fix:.3f} did not build "
        "up during fixation; IOR mechanism is broken"
    )

    # 2) Disengage: attn smoothes back, IOR decays slowly.
    state_dis = _drive(state_fix, params, bu_off, DISENGAGE_STEPS)
    ior_after_dis = float(state_dis.ior_trace[TARGET])
    attn_after_dis = float(state_dis.attn_weights[TARGET])
    assert ior_after_dis > 0.4, (
        f"seed={seed}: ior_trace decayed too fast during disengage "
        f"({ior_after_dis:.3f}); IOR window too short"
    )
    assert attn_after_dis < 0.2, (
        f"seed={seed}: attn_weights[TARGET]={attn_after_dis:.3f} did not "
        "smooth back toward uniform during disengage; smoothing too slow"
    )

    # 3) Re-probe vs fresh baseline.
    out_probe = attention_step(
        state_dis, params, assoc, bottom_up_errors=bu_target,
        global_ach=1.0, ne_level=0.5,
    )
    out_baseline = attention_step(
        state0, params, assoc, bottom_up_errors=bu_target,
        global_ach=1.0, ne_level=0.5,
    )

    attn_probe = float(out_probe.attn_distribution[TARGET])
    attn_base = float(out_baseline.attn_distribution[TARGET])
    drop = (attn_base - attn_probe) / attn_base
    assert drop >= PLAN_DROP_THRESHOLD, (
        f"seed={seed}: IOR drop only {drop*100:.1f}%; plan requires "
        f"≥{PLAN_DROP_THRESHOLD*100:.0f}%. "
        f"attn_baseline={attn_base:.4f} attn_probe={attn_probe:.4f} "
        f"ior_trace_at_probe={ior_after_dis:.3f}"
    )


def test_ior_does_not_suppress_unattended_columns():
    """Columns that were never attended must NOT be suppressed by IOR
    (ior_trace[other] ≈ 0).  Otherwise the suppression is global, not
    return-specific.
    """
    ctx, params = _build()
    state0 = init_attention_state(jax.random.PRNGKey(0), N_ASSOC, N_COL)
    bu_target = jnp.zeros(N_COL, jnp.float32).at[TARGET].set(1.0)
    state_fix = _drive(state0, params, bu_target, FIX_STEPS)

    other_cols = jnp.array(
        [i for i in range(N_COL) if i != TARGET], dtype=jnp.int32
    )
    other_traces = jax.device_get(state_fix.ior_trace[other_cols])
    assert float(other_traces.max()) < 0.05, (
        f"Non-target IOR traces grew unexpectedly: max={float(other_traces.max()):.3f}"
    )
    target_trace = float(state_fix.ior_trace[TARGET])
    assert target_trace > 10.0 * float(other_traces.max() + 1e-6), (
        f"IOR not selective: target_trace={target_trace:.3f}, "
        f"max other={float(other_traces.max()):.3f}"
    )
