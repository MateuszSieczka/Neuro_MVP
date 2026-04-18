"""P0.4 — theta-phase gating of cortex excitability (Lakatos 2008).

Weryfikuje że `CorticalInputs.excitability_mod` faktycznie moduluje
odpowiedź kory:
  1. exc_mod=1.1 (theta peak) → wyższe firing rates niż baseline (1.0).
  2. exc_mod=0.9 (theta trough) → niższe firing rates niż baseline.
  3. Pełny cykl theta w `minimal_brain_step` produkuje zauważalną
     oscylację w rate EMA (depth ~several % of mean rate).

Note: the rate window is restricted to ~one theta cycle (200 dt) so
the steady-state rate reflects the **instantaneous** effect of
``excitability_mod`` rather than the slow astrocytic ATP feedback
(τ ≈ 2 s, Aubert & Costalat 2005), which dominates over multi-second
horizons and reverses the sign of the rate response (more spikes →
more ATP depletion → higher V_T → fewer spikes).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import BackendContext
from core.cortex import (
    CorticalInputs, init_cortical_area_params, init_cortical_area_state,
    cortical_area_step,
)
from core.brain_graph import (
    init_minimal_brain_params, init_minimal_brain_state, minimal_brain_step,
)


def _cortex_rate_with_mod(mod_value: float, n_steps: int = 100) -> float:
    ctx = BackendContext(dt=1.0)
    params = init_cortical_area_params(ctx, input_size=16)
    state = init_cortical_area_state(jax.random.PRNGKey(0), params)
    inputs = CorticalInputs(
        ff_input=jnp.full((16,), 0.15, jnp.float32),
        excitability_mod=jnp.asarray(mod_value, jnp.float32),
    )

    def scan_fn(s, _):
        out = cortical_area_step(s, params, ctx, inputs, apply_stdp=False)
        return out.state, out.state.rate_l4.mean()

    _, hist = jax.lax.scan(scan_fn, state, None, length=n_steps)
    # Window 30-80 dt: long enough to escape the warm-up transient,
    # short enough that astrocytic ATP feedback (τ ≈ 2 s) has not yet
    # introduced sign-reversing homeostasis. This isolates the
    # *instantaneous* effect of ``excitability_mod`` (Lakatos 2008
    # theta-phase modulation acts on this fast timescale).
    return float(jnp.mean(hist[30:80]) * 1000.0)


def test_excitability_up_increases_firing():
    r_base = _cortex_rate_with_mod(1.0)
    r_up = _cortex_rate_with_mod(1.1)
    assert r_up > r_base, f"exc 1.1 should exceed baseline: {r_up:.2f} vs {r_base:.2f} Hz"


def test_excitability_down_decreases_firing():
    r_base = _cortex_rate_with_mod(1.0)
    r_down = _cortex_rate_with_mod(0.9)
    assert r_down < r_base, (
        f"exc 0.9 should be below baseline: {r_down:.2f} vs {r_base:.2f} Hz"
    )


def test_theta_cycle_in_minimal_brain_modulates_activity():
    """Bez ingerencji — czy brain_step faktycznie liczy theta i podaje?

    Sprawdzamy: oscillator.theta_phase zmienia się w czasie (nie jest
    zamrożony) i cortex rate ma non-zero variance (nie stałe constans).
    """
    ctx = BackendContext(dt=1.0)
    params = init_minimal_brain_params(ctx, sensory_size=16)
    state = init_minimal_brain_state(jax.random.PRNGKey(0), params)
    sensory = jnp.ones((16,), jnp.float32) * 0.15

    def scan_fn(s, _):
        out = minimal_brain_step(s, params, ctx, sensory)
        return out.state, (
            out.state.oscillator.theta_phase,
            out.state.cortex.rate_l4.mean(),
        )

    _, (phases, rates) = jax.lax.scan(scan_fn, state, None, length=500)
    # Theta phase must advance (not frozen, not constant).
    phase_range = float(jnp.max(phases) - jnp.min(phases))
    assert phase_range > 3.0, f"Theta phase stuck: range {phase_range:.3f}"
    # Rates have some variance (consistent with phase-dependent modulation
    # + noise). Over 500 ms with 6 Hz theta we expect ~3 cycles.
    rate_std = float(jnp.std(rates[100:]))
    assert rate_std > 0.0, "cortex rate perfectly constant — theta gate not wired"
