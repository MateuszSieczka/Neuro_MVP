"""Phase 3 — World-model (inferior-olive proxy) learns the dynamics.

Plan target (plan.md §2.5):
  world_model MSE spada ≥ 30% po 1500 krokach na GridWorld.

Rationale (Wolpert & Kawato 1998 internal model; Apps & Strata 2015
inferior olive as PE comparator):
  In our architecture the dense forward model in ``core/world_model.py``
  plays the role of the inferior olive: it predicts ``s_{t+1}`` from
  ``(s_t, a_t)``, and the prediction error becomes the climbing-fibre
  signal driving cerebellum learning.  If the WM does not actually
  reduce its error over time, the whole intrinsic-reward + cerebellum
  loop is broken.

Test compares the mean squared prediction error of the first 200
cycles to the mean of the last 200, requiring a ≥ 30% reduction.
This holds across seeds (measured: 47–69%; threshold 30% per-seed).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from core.backend import BackendContext
from core.brain_graph import (
    init_action_brain_params, init_action_brain_state,
    action_brain_cognitive_step,
)
from embodiment.gridworld import GridWorldBody


def _make_body():
    return GridWorldBody.create(
        size=5, goal=(99, 99),  # unreachable — no terminations
        goal_reward=0.0, step_cost=0.0, max_steps=10 ** 9,
    )


def _rollout_mse(n_cycles: int, seed: int):
    ctx = BackendContext(dt=1.0)
    body = _make_body()
    params = init_action_brain_params(
        ctx, sensory_size=body.sensory_size,
        n_body_actions=body.n_actions, n_saccade_actions=2, substeps=4,
    )
    state = init_action_brain_state(jax.random.PRNGKey(seed), params)
    body, sample0 = body.reset(jax.random.PRNGKey(seed + 1))

    b, st = body, state
    prev_r, prev_d = sample0.reward, sample0.done
    mse_list = []
    keys = jax.random.split(jax.random.PRNGKey(seed + 2), n_cycles)
    for i in range(n_cycles):
        k_step, k_act = jax.random.split(keys[i])
        out = action_brain_cognitive_step(
            st, params, ctx, b._encode(b.pos), prev_r, prev_d, k_step,
        )
        new_b, smp = b.act(k_act, out.body_action, jnp.int32(0))
        mse_list.append(float(jnp.mean(out.state.world_model.prediction_error ** 2)))
        b, st = new_b, out.state
        prev_r, prev_d = smp.reward, smp.done

    return jnp.array(mse_list)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_world_model_mse_drops(seed):
    """WM MSE in last 200 cycles is at least 30% lower than first 200."""
    mse = _rollout_mse(n_cycles=1500, seed=seed)
    early = float(mse[:200].mean())
    late = float(mse[-200:].mean())
    drop_frac = (early - late) / max(early, 1e-9)
    assert drop_frac >= 0.30, (
        f"seed={seed}: WM MSE early={early:.4f} late={late:.4f} "
        f"drop={drop_frac * 100:.1f}% (target ≥ 30%)"
    )


def test_world_model_mse_drops_consistently_across_seeds():
    """Mean drop across 3 seeds clearly above the 30% threshold."""
    drops = []
    for seed in (0, 1, 2):
        mse = _rollout_mse(n_cycles=1500, seed=seed)
        early = float(mse[:200].mean())
        late = float(mse[-200:].mean())
        drops.append((early - late) / max(early, 1e-9))
    mean_drop = sum(drops) / len(drops)
    assert mean_drop >= 0.40, (
        f"mean WM MSE drop across seeds = {mean_drop * 100:.1f}% "
        f"(per-seed: {[f'{d * 100:.1f}%' for d in drops]})"
    )
