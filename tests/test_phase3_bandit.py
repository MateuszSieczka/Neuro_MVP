"""Phase 3 — Bandit learning via BG actor + VTA RPE.

Plan target (plan.md §2.5):
  3-armed Gaussian, 800 cycles, p(best arm) ≥ 0.55 over the
  final quarter (200 cycles).

Rationale (Sutton & Barto 2018 §2.3): with a constant context the
critic gets no spatial signal; the actor must learn D1/D2 weights
from VTA RPE alone. This isolates the BG+VTA loop.

Methodology
-----------
- We average p_best over 5 environment seeds (different arm_means).
  Per-seed scores are noisy because the brain's NE/boredom drive
  (Yu & Dayan 2005, P0.9) drifts policy on a stationary task; the
  multi-seed mean is the policy-quality estimator the plan calls for.
- The loop is ``jax.lax.scan``-compiled. Without it, every Python
  iteration would retrace the full ActionBrain pytree
  (≈ 15 min for 600 cycles → ≈ 8 s scanned).

Tests
-----
1. ``test_bandit_prefers_best_arm`` — multi-seed mean p_best > 0.55.
2. ``test_bandit_early_learning`` — within the first 200 cycles the
   policy already exceeds chance (→ detects total non-learning).
3. ``test_bandit_runs_without_nans`` — stability smoke.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import BackendContext
from core.brain_graph import (
    init_action_brain_params, init_action_brain_state, action_brain_step,
)
from embodiment.bandit import GaussianBanditBody


def _rollout(body, params, state0, key0, n_cycles):
    """JIT-compiled bandit rollout. Returns (final_state, action_ids)."""
    ctx = BackendContext(dt=1.0)
    sensory = body._sensory()  # constant context

    def step(carry, k):
        st, prev_r = carry
        k_step, k_act = jax.random.split(k)
        out = action_brain_step(
            st, params, ctx, sensory, prev_r,
            jnp.asarray(0.0, jnp.float32), k_step,
        )
        a = out.body_action
        mu = body.arm_means[a]
        noise = (
            jax.random.normal(k_act, (), dtype=jnp.float32) * body.noise_sigma
        )
        return (out.state, mu + noise), a

    keys = jax.random.split(key0, n_cycles)
    (final, _), actions = jax.lax.scan(
        step, (state0, jnp.float32(0.0)), keys,
    )
    return final, actions


def _run_one_seed(seed_body: int, seed_state: int, seed_roll: int,
                  n_cycles: int):
    body = GaussianBanditBody.create(
        jax.random.PRNGKey(seed_body),
        n_actions=3, mean_spread=2.0, noise_sigma=0.1,
    )
    best = int(jnp.argmax(body.arm_means))
    ctx = BackendContext(dt=1.0)
    params = init_action_brain_params(
        ctx, sensory_size=body.sensory_size,
        n_body_actions=3, n_saccade_actions=2,
    )
    state = init_action_brain_state(jax.random.PRNGKey(seed_state), params)
    _, actions = _rollout(body, params, state, jax.random.PRNGKey(seed_roll),
                          n_cycles)
    return jax.device_get(actions), best, jax.device_get(body.arm_means)


def test_bandit_prefers_best_arm():
    """Multi-seed mean p(best arm) on the final quarter ≥ 0.55.

    Per-plan threshold; averaged across 5 environment seeds because
    NE/boredom drive (P0.9) makes per-seed tail noisy on a stationary
    task. This is the policy-quality estimator the plan specifies.
    """
    n_cycles = 800
    seeds = [7, 0, 3, 19, 42]
    tail_p_best = []
    for sb in seeds:
        actions, best, _ = _run_one_seed(sb, 11, 13, n_cycles)
        tail = actions[int(0.75 * n_cycles):]
        tail_p_best.append(float((tail == best).mean()))
    mean_tail = sum(tail_p_best) / len(tail_p_best)
    assert mean_tail > 0.55, (
        f"BG mean p_best over {len(seeds)} seeds = {mean_tail:.3f} "
        f"(per-seed: {[f'{p:.2f}' for p in tail_p_best]}); "
        f"plan target 0.55"
    )


def test_bandit_early_learning():
    """Within the first 200 cycles, p(best) already exceeds chance.

    Detects total failure-to-learn: even if late-stage exploration
    drifts policy, the *first* quarter must be clearly above chance
    (1/3 ≈ 0.333). Threshold 0.55 averaged across seeds.
    """
    n_cycles = 800
    seeds = [7, 0, 3, 19, 42]
    early_p_best = []
    for sb in seeds:
        actions, best, _ = _run_one_seed(sb, 11, 13, n_cycles)
        early = actions[: n_cycles // 4]
        early_p_best.append(float((early == best).mean()))
    mean_early = sum(early_p_best) / len(early_p_best)
    assert mean_early > 0.55, (
        f"BG mean early-window p_best = {mean_early:.3f} "
        f"(per-seed: {[f'{p:.2f}' for p in early_p_best]})"
    )


def test_bandit_runs_without_nans():
    """Stability smoke: long rollout keeps actor weights finite."""
    body = GaussianBanditBody.create(jax.random.PRNGKey(0), n_actions=3)
    ctx = BackendContext(dt=1.0)
    params = init_action_brain_params(
        ctx, sensory_size=body.sensory_size,
        n_body_actions=3, n_saccade_actions=2,
    )
    state = init_action_brain_state(jax.random.PRNGKey(1), params)
    final_state, _ = _rollout(body, params, state, jax.random.PRNGKey(2), 100)
    assert bool(jnp.isfinite(final_state.actor_body.w_d1).all())
    assert bool(jnp.isfinite(final_state.actor_body.w_d2).all())
