"""P0.3 — weryfikacja że astrocyta jest aktywnie podpięta do cortex.

Cel: udowodnić że przy sustained input:
  1. Ca²⁺ rośnie (zone-wise EMA of rate²).
  2. ATP spada (spike cost > regen).
  3. threshold_shift rośnie (ATP depletion → V_T wyżej).
  4. Reset_transient faktycznie zeruje astrocytę.

Astrocyta ma powolne τ (Ca ~5 s, ATP ~200 s) — testy używają dłuższego
horyzontu niż cortex-alive.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import BackendContext
from core.cortex import (
    CorticalInputs, init_cortical_area_params, init_cortical_area_state,
    cortical_area_step, cortical_area_reset_transient,
)
from core.astrocyte import threshold_shift as astro_threshold_shift


def _run(drive: float, n_steps: int, seed: int = 0):
    ctx = BackendContext(dt=1.0)
    params = init_cortical_area_params(ctx, input_size=16)
    state = init_cortical_area_state(jax.random.PRNGKey(seed), params)
    inp = jnp.full((16,), drive, jnp.float32)
    inputs = CorticalInputs(ff_input=inp)

    def scan_fn(s, _):
        out = cortical_area_step(s, params, ctx, inputs, apply_ipool_stdp=False)
        astro = out.state.astrocyte
        return out.state, (astro.calcium.mean(), astro.atp.mean())

    final, (ca_hist, atp_hist) = jax.lax.scan(scan_fn, state, None, length=n_steps)
    return params, state, final, ca_hist, atp_hist


def test_astrocyte_calcium_rises_with_activity():
    """High drive → sustained L4/L5 firing → Ca²⁺ integrator climbs."""
    _, _, final, ca_hist, _ = _run(drive=0.3, n_steps=2000)
    ca_start = float(ca_hist[100])
    ca_end = float(ca_hist[-1])
    # Minimal monotonic rise: late Ca² clearly above early.
    assert ca_end > ca_start * 1.5 + 1e-4, (
        f"Ca²⁺ did not rise: start={ca_start:.4g} end={ca_end:.4g}"
    )
    assert float(final.astrocyte.calcium.max()) > 0.0


def test_astrocyte_atp_depletes_under_load():
    """Sustained activity consumes ATP faster than regen."""
    _, _, final, _, atp_hist = _run(drive=0.3, n_steps=2000)
    atp_start = float(atp_hist[100])
    atp_end = float(atp_hist[-1])
    # Demand exceeds regen; small but measurable drop.
    assert atp_end < atp_start - 1e-4, (
        f"ATP did not deplete: start={atp_start:.4f} end={atp_end:.4f}"
    )
    # Still non-negative (clipped in astrocyte_step).
    assert float(final.astrocyte.atp.min()) >= 0.0


def test_astrocyte_threshold_shift_nonzero_after_activity():
    """Depleted ATP → positive V_T shift (neurons harder to fire)."""
    ctx = BackendContext(dt=1.0)
    params, _, final, _, _ = _run(drive=0.3, n_steps=2000)
    shift = astro_threshold_shift(final.astrocyte, params.astrocyte)
    assert float(shift.max()) > 0.0, "threshold_shift stayed zero"
    # ATP range [0,1], shift = k·(1-ATP) with k=10 mV; at moderate load
    # we expect small but non-trivial shift (> 0.01 mV).
    assert float(shift.max()) < params.astrocyte.atp_threshold_shift + 1e-3


def test_astrocyte_reset_transient_zeros_state():
    """End-of-episode reset must clear Ca and restore full ATP."""
    params, _, final, _, _ = _run(drive=0.3, n_steps=500)
    assert float(final.astrocyte.calcium.max()) > 0.0  # truly activated
    reset = cortical_area_reset_transient(final, params)
    assert float(reset.astrocyte.calcium.max()) == 0.0
    assert float(reset.astrocyte.d_serine.max()) == 0.0
    # ATP restored to atp_max (default 1.0).
    atp_max = float(params.astrocyte.atp_max)
    assert abs(float(reset.astrocyte.atp.min()) - atp_max) < 1e-5


def test_astrocyte_silent_without_activity():
    """No input → no spikes → Ca stays ~0, ATP near max."""
    _, _, final, ca_hist, atp_hist = _run(drive=0.0, n_steps=500)
    assert float(ca_hist[-1]) < 1e-3
    # Small ATP drain from stochastic firing is OK; expect close to max.
    assert float(atp_hist[-1]) > 0.99
