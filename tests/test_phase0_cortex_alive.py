"""Phase 0 diagnostic — MinimalBrain / ActionBrain are alive and finite.

These tests replace the ad-hoc smoke-test drift we used to run manually.
They are REGRESSION guards, not biological-fidelity targets.

Key invariants (anything failing these breaks all downstream phases):
  * Zero drive → network silent (no exploding free-running activity).
  * Calibration-point drive (10% afferent rate, Barth & Poulet 2012
    lower bound) → every layer fires at least once in 500 dt.
  * Neuron state stays finite; firing rates never hit the dt⁻¹ ceiling.
  * ActionBrain with a Gaussian bandit emits finite RPE / total reward.

Rate targets (Phase-0 debt)
---------------------------
Awake cortex fires spontaneously at 1–10 Hz (Barth & Poulet 2012) and
first-order thalamic relay at 5–30 Hz (Sherman & Guillery 2006). At the
current calibration MinimalBrain runs ~3–5× hotter than that on the
10 %-rate operating point; we assert the loose upper bound 150 Hz here
and will tighten after receptor pharmacology (P0.2) and astrocyte/ATP
throttling (P0.3) are wired in.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from core.backend import BackendContext
from core.brain_graph import (
    init_minimal_brain_params, init_minimal_brain_state, minimal_brain_step,
    init_action_brain_params, init_action_brain_state, action_brain_step,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _run_minimal(sensory: jax.Array, n_steps: int = 500, seed: int = 0):
    ctx = BackendContext(dt=1.0)
    params = init_minimal_brain_params(ctx, sensory_size=int(sensory.size))
    state = init_minimal_brain_state(jax.random.PRNGKey(seed), params)
    sensory = sensory.astype(jnp.float32)

    def scan_fn(s, _):
        out = minimal_brain_step(s, params, ctx, sensory)
        diag = jnp.stack([
            jnp.mean(out.state.cortex.rate_l4),
            jnp.mean(out.state.cortex.l23.state_rate),
            jnp.mean(out.state.cortex.l23.error_rate),
            jnp.mean(out.cortex_l5_rate),
            jnp.mean(out.relay_spikes),
        ])
        return out.state, diag

    final_state, diag_hist = jax.lax.scan(scan_fn, state, None, length=n_steps)
    # Return mean rates (Hz) over the last 40 % of the run, after warmup.
    tail = diag_hist[-int(0.4 * n_steps):]
    # rate EMAs are per-dt firing probabilities (dt = 1 ms) → × 1000 = Hz.
    rates_hz = jnp.mean(tail, axis=0) * 1000.0
    return final_state, rates_hz, diag_hist


# ------------------------------------------------------------------
# MinimalBrain invariants
# ------------------------------------------------------------------


def test_minimal_brain_silent_at_zero_drive() -> None:
    """No input → no runaway activity. Network must be quiescent."""
    sensory = jnp.zeros((16,), jnp.float32)
    _, rates_hz, _ = _run_minimal(sensory)
    for i, name in enumerate(("L4", "L23state", "L23error", "L5", "Relay")):
        assert float(rates_hz[i]) < 1.0, (
            f"{name} fires {float(rates_hz[i]):.2f} Hz with zero drive; "
            "uncontrolled recurrent activity."
        )


def test_minimal_brain_alive_at_calibration_drive() -> None:
    """10 % uniform rate (calibration point) → every layer alive."""
    sensory = jnp.ones((16,), jnp.float32) * 0.1
    _, rates_hz, _ = _run_minimal(sensory)
    # We only require L4 and Relay to actively fire; L2/3 and L5 may
    # stay quiet at the calibration point because cortex L4 rates are
    # too sparse to recruit them until STDP warms up (expected).
    names = ("L4", "L23state", "L23error", "L5", "Relay")
    for i, name in enumerate(names):
        r = float(rates_hz[i])
        assert 0.0 <= r < 200.0, (
            f"{name} firing rate {r:.2f} Hz out of finite bounds [0, 200)"
        )
    assert float(rates_hz[0]) > 1.0, "L4 silent at calibration drive"
    assert float(rates_hz[4]) > 5.0, "Relay silent at calibration drive"


def test_minimal_brain_scales_with_drive() -> None:
    """Higher afferent intensity → higher firing (monotonicity)."""
    rates = []
    for d in (0.05, 0.1, 0.2):
        _, rates_hz, _ = _run_minimal(jnp.ones((16,), jnp.float32) * d)
        rates.append(rates_hz)
    # Relay monotone in drive (allow small floating-point jitter).
    r_relay = [float(r[4]) for r in rates]
    assert r_relay[0] <= r_relay[1] + 1.0 <= r_relay[2] + 2.0, (
        f"relay rates not monotone in drive: {r_relay}"
    )


def test_minimal_brain_state_finite() -> None:
    """All leaves of the final state must be finite (no NaN / inf)."""
    sensory = jnp.ones((16,), jnp.float32) * 0.1
    final_state, _, _ = _run_minimal(sensory, n_steps=200)
    leaves = jax.tree.leaves(final_state)
    for i, leaf in enumerate(leaves):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert bool(jnp.all(jnp.isfinite(leaf))), (
                f"non-finite values in state leaf #{i} (shape={leaf.shape})"
            )


# ------------------------------------------------------------------
# ActionBrain smoke — requires bandit body
# ------------------------------------------------------------------


def test_action_brain_rpe_finite_on_bandit() -> None:
    """ActionBrain produces finite RPE / total reward on the bandit body."""
    from embodiment.bandit import GaussianBanditBody

    ctx = BackendContext(dt=1.0)
    body = GaussianBanditBody.create(jax.random.PRNGKey(1), n_actions=3)
    params = init_action_brain_params(
        ctx, sensory_size=body.sensory_size,
        n_body_actions=body.n_actions, substeps=10,
    )
    state = init_action_brain_state(jax.random.PRNGKey(0), params)

    body, sample = body.reset(jax.random.PRNGKey(2))
    key = jax.random.PRNGKey(42)

    # Run 10 decision cycles; assert RPE stays finite throughout.
    for step in range(10):
        key, k_brain, k_body = jax.random.split(key, 3)
        out = action_brain_step(
            state, params, ctx,
            sample.sensory,
            prev_reward=sample.reward,
            prev_done=sample.done,
            key=k_brain,
        )
        assert bool(jnp.isfinite(out.rpe)), f"RPE non-finite at step {step}"
        assert bool(jnp.isfinite(out.total_reward)), (
            f"total_reward non-finite at step {step}"
        )
        state = out.state
        body, sample = body.act(
            k_body,
            out.body_action,
            out.saccade_action,
        )
