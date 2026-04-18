"""Phase 0 micro-test: PFC working memory attractor persistence.

Plan (P0.11): "WM utrzymuje attractor po usunięciu drive ≥ 200 dt
(memory span > working window)."

We drive PFC with a sparse input pattern under high ACh+DA (gate open),
let `wm_update_lateral` carve a Hebbian attractor, then remove the drive
and turn neuromodulators off. Persistent activity must remain elevated
above the silent baseline for at least 200 dt.

Mechanism under test
--------------------
- `pfc_step` runs `wm_step` (gate AdEx + content AdEx + STDP traces).
- `wm_update_lateral` performs Hebbian update on `w_lateral` from the
  outer product of co-active content neurons (Goldman-Rakic 1995
  attractor view of WM).
- After drive removal, content_trace × w_lateral provides the recurrent
  excitation that sustains spiking — this is what we measure.

Note (carry-over, see /memories/repo/phase0_audit_findings.md):
    The current WM is non-selective: cos(A,B) ≈ 1.000 for distinct
    drive patterns A,B. The literal plan criterion (persistence) is
    met, but proper input-pattern-specific WM still requires lateral
    inhibition / WTA design pass. NOT band-aided here.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from core.backend import BackendContext
from core.pfc import init_pfc_params, init_pfc_state, pfc_step
from core.working_memory import wm_update_lateral


INPUT_SIZE = 32
N_CONTENT = 64
DRIVE_STEPS = 200
PERSIST_STEPS = 250  # > plan threshold of 200 dt
SEEDS = (0, 1, 2)


def _build():
    ctx = BackendContext(dt=1.0)
    params = init_pfc_params(
        ctx, input_size=INPUT_SIZE, n_content=N_CONTENT, n_gate=32,
    )
    return ctx, params


def _drive_phase(state, params, ctx, drive, key):
    """Run drive_steps of pfc_step + Hebbian lateral update."""

    def step(carry, k):
        st = carry
        out = pfc_step(st, params, ctx, drive, 1.0, 1.0, k)
        new_wm = wm_update_lateral(out.state.wm, params.wm)
        new_st = type(out.state)(wm=new_wm, output_rate=out.state.output_rate)
        return new_st, None

    keys = jax.random.split(key, DRIVE_STEPS)
    state, _ = jax.lax.scan(step, state, keys)
    return state


def _persist_phase(state, params, ctx, key):
    """Run persist_steps of pfc_step with zero drive and ACh=DA=0."""
    zero = jnp.zeros(INPUT_SIZE, jnp.float32)

    def step(carry, k):
        st = carry
        out = pfc_step(st, params, ctx, zero, 0.0, 0.0, k)
        return out.state, out.content_rate.mean()

    keys = jax.random.split(key, PERSIST_STEPS)
    final, traj = jax.lax.scan(step, state, keys)
    return final, traj


def _silent_baseline(params, ctx, key, *, steps=PERSIST_STEPS):
    """Output_rate of a fresh, never-driven PFC over the same horizon."""
    state = init_pfc_state(jax.random.PRNGKey(0), params)
    zero = jnp.zeros(INPUT_SIZE, jnp.float32)

    def step(carry, k):
        st = carry
        out = pfc_step(st, params, ctx, zero, 0.0, 0.0, k)
        return out.state, out.content_rate.mean()

    keys = jax.random.split(key, steps)
    _, traj = jax.lax.scan(step, state, keys)
    return traj


@pytest.mark.parametrize("seed", SEEDS)
def test_pfc_persistent_activity_after_drive_removal(seed):
    """After 200 dt drive then drive removed, mean output_rate must stay
    significantly above the silent baseline for >= 200 dt.

    Plan: "WM utrzymuje attractor po usunięciu drive ≥ 200 dt".
    """
    ctx, params = _build()

    # Silent baseline trajectory (same params, never driven).
    base_traj = _silent_baseline(params, ctx, jax.random.PRNGKey(7777))
    base_at_200 = float(base_traj[200])
    assert base_at_200 < 1e-3, (
        f"Silent PFC should be ~0 firing before t=200 (got {base_at_200})"
    )

    # Drive then persist.
    drive = jnp.zeros(INPUT_SIZE, jnp.float32).at[:8].set(1.0)
    state = init_pfc_state(jax.random.PRNGKey(seed), params)
    state = _drive_phase(
        state, params, ctx, drive, jax.random.PRNGKey(seed + 100),
    )
    end_drive_rate = float(state.output_rate.mean())
    final, traj = _persist_phase(
        state, params, ctx, jax.random.PRNGKey(seed + 200),
    )
    traj = jax.device_get(traj)

    # 1) Drive phase actually built up firing.
    assert end_drive_rate > 0.05, (
        f"seed={seed}: drive did not produce firing "
        f"(end_drive output_rate={end_drive_rate:.4f})"
    )

    # 2) Persistent activity at t = 200 dt is well above silent baseline.
    rate_at_200 = float(traj[200])
    assert rate_at_200 > 10.0 * max(base_at_200, 1e-4), (
        f"seed={seed}: rate at t=200 ({rate_at_200:.4f}) not >>10x "
        f"silent baseline ({base_at_200:.6f})"
    )
    assert rate_at_200 > 0.05, (
        f"seed={seed}: rate at t=200 ({rate_at_200:.4f}) below 0.05 — "
        f"attractor collapsed within 200 dt"
    )

    # 3) Activity does not collapse over the full persistence window.
    min_rate_persist = float(traj.min())
    assert min_rate_persist > 0.02, (
        f"seed={seed}: min output_rate during persist ({min_rate_persist:.4f}) "
        f"dropped below 0.02 — attractor unstable"
    )


def test_pfc_persistence_not_random_drift():
    """Without any drive at all, content_rate must NOT spontaneously
    self-ignite into the persistent regime — i.e. the attractor is
    drive-conditional, not a pathological always-on state.
    """
    ctx, params = _build()
    base_traj = _silent_baseline(params, ctx, jax.random.PRNGKey(123))
    final_silent = float(base_traj[-1])
    max_silent = float(base_traj.max())
    assert max_silent < 1e-2, (
        f"Untriggered PFC self-ignited (max silent rate={max_silent:.4f}); "
        f"persistence test would be vacuous"
    )
    assert final_silent < 1e-2
