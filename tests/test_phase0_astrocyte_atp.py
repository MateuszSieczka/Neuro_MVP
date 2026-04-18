"""Phase 0 micro-test: astrocyte ATP depletion / recovery / threshold-shift.

Plan (P0.11): "sustained 50 Hz spiking layer → ATP spada do <0.3 atp_max
po 5000 dt; threshold_shift > 2 mV."

Calibration
-----------
The astrocyte ATP pool represents the **local** glial/synaptic
energy pool (Aubert & Costalat 2005; Rangaraju et al. 2014), NOT
the whole-brain reservoir (Attwell 2001, τ ≈ 200 s). Defaults are
chosen so that at r = 0.05 spikes/ms (50 Hz) the dynamic
equilibrium is ATP ≈ 0.1·atp_max, with time-constant τ ≈ 2 s —
which makes the P0.11 plan-literal criterion reachable without any
test-side hacks.

The tests cover four independent properties:
  * **Plan-literal criterion** at 50 Hz / 5000 dt.
  * **Mechanism at extreme drive** (rate = 5, 50 000 dt): the ODE
    collapses the pool to ≤ 0.05, threshold_shift > 2 mV, leak_gain
    > 1.4.
  * **Recovery in silence** from a fully-depleted state.
  * **No spurious decay** at baseline silence.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from core.backend import BackendContext
from core.astrocyte import (
    init_astrocyte_params, AstrocyteState, astrocyte_step,
    threshold_shift, leak_gain,
)


N_ZONES = 4


def _build():
    ctx = BackendContext(dt=1.0)
    params = init_astrocyte_params(ctx)
    return ctx, params


def _fresh_state(params):
    return AstrocyteState(
        calcium=jnp.zeros(N_ZONES, jnp.float32),
        d_serine=jnp.zeros(N_ZONES, jnp.float32),
        atp=jnp.full((N_ZONES,), float(params.atp_max), jnp.float32),
    )


def _scan(state, params, ctx, rate, n_steps):
    rates = jnp.full(N_ZONES, float(rate), jnp.float32)

    def step(carry, _):
        return astrocyte_step(carry, params, ctx, rates), None

    final, _ = jax.lax.scan(step, state, jnp.arange(n_steps))
    return final


def test_atp_depletes_under_extreme_drive():
    """Sustained extreme drive (rate=5, 50 000 dt) → ATP ≤ 0.05.

    Confirms the depletion ODE ``dATP = regen·(max−ATP) − cost·rate``
    actually runs and that, given enough drive·time, the pool collapses
    (would correspond to severe metabolic stress, Magistretti 2006).
    """
    ctx, params = _build()
    state = _fresh_state(params)
    final = _scan(state, params, ctx, rate=5.0, n_steps=50_000)
    atp = float(final.atp.mean())
    assert atp <= 0.05, f"ATP only fell to {atp:.4f}; pool not depleting"


def test_threshold_shift_and_leak_gain_rise_with_depletion():
    """Once ATP is depleted, V_T must rise (>2 mV) and g_L must rise
    (×≥1.4) — the metabolic-distress neuromodulation pathway
    (Attwell 2001, Magistretti 2006).
    """
    ctx, params = _build()
    state = _scan(_fresh_state(params), params, ctx, rate=5.0, n_steps=50_000)
    ts = float(threshold_shift(state, params).mean())
    lg = float(leak_gain(state, params).mean())
    assert ts > 2.0, f"threshold_shift {ts:.2f} mV ≤ 2 mV — depletion has no effect"
    assert lg > 1.4, f"leak_gain {lg:.3f} ≤ 1.4 — depletion has no effect"


def test_atp_recovers_when_silenced():
    """From a fully-depleted state (ATP ≈ 0), ≥ 10 000 dt of silence
    must produce strict, monotonic recovery toward atp_max.
    Recovery is slow (τ ≈ 200 s) so we only require ATP > 0 — the
    recovery ODE branch is on.
    """
    ctx, params = _build()
    depleted = _scan(_fresh_state(params), params, ctx, rate=5.0, n_steps=50_000)
    atp_dep = float(depleted.atp.mean())
    assert atp_dep < 0.05

    recovered = _scan(depleted, params, ctx, rate=0.0, n_steps=10_000)
    atp_rec = float(recovered.atp.mean())
    assert atp_rec > atp_dep + 1e-3, (
        f"No recovery observed: depleted={atp_dep:.4f}, "
        f"after 10000 dt silence={atp_rec:.4f}"
    )

    # And much longer silence recovers further (mechanism is monotonic).
    longer = _scan(depleted, params, ctx, rate=0.0, n_steps=50_000)
    atp_long = float(longer.atp.mean())
    assert atp_long > atp_rec, (
        f"ATP not monotonic in silence: 10k={atp_rec:.4f}, 50k={atp_long:.4f}"
    )


def test_atp_unaffected_at_baseline_silence():
    """A never-fired state must keep ATP ≈ atp_max (no spurious leak)."""
    ctx, params = _build()
    state = _scan(_fresh_state(params), params, ctx, rate=0.0, n_steps=5_000)
    atp = float(state.atp.mean())
    assert abs(atp - float(params.atp_max)) < 1e-4


def test_plan_literal_50hz_5000dt_atp_below_0_3():
    """P0.11 plan-literal: 50 Hz × 5000 dt → ATP < 0.3, TS > 2 mV."""
    ctx, params = _build()
    final = _scan(_fresh_state(params), params, ctx, rate=0.05, n_steps=5_000)
    atp = float(final.atp.mean())
    ts = float(threshold_shift(final, params).mean())
    assert atp < 0.3, f"plan threshold violated: ATP={atp:.4f}"
    assert ts > 2.0, f"plan threshold violated: TS={ts:.4f} mV"
