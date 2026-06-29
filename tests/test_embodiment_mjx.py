"""MJX arm + babble→reach drivers — the embodiment milestone (§1).

Requires MuJoCo: skipped where it is absent (Windows dev boxes), run in the
Colab notebook / a MuJoCo CI.  These are smoke tests (shapes / finiteness /
the goal wiring), not a learned-reach convergence check — convergence is a
training run, not a unit test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("mujoco", reason="MJX arm needs MuJoCo")

from embodiment.babbling import BabbleConfig, run_babbling
from embodiment.mjx_arm_body import default_arm_config
from embodiment.reach import ReachConfig, build_reacher, run_reach


def _tiny_reacher(key):
    """A deliberately small arm + brain so the smoke tests compile fast."""
    cfg = default_arm_config(n_cells_per_joint=6, n_target_cells=6, max_steps=8)
    return build_reacher(key, arm_cfg=cfg, eta_mu=0.1, eta_w=2e-2, n_relax=10)


def test_arm_body_drives_a_matching_brain():
    params, _, body = _tiny_reacher(jax.random.PRNGKey(0))
    # The factory sizes the brain from the body — they must agree.
    assert body.sensory_size == params.sensory_dim
    assert body.motor_dim == params.motor_dim


def test_reach_goal_pins_only_target_error_channels():
    _, _, body = _tiny_reacher(jax.random.PRNGKey(0))
    pref, mask = body.reach_goal()
    assert pref.shape == (body.sensory_size,)
    assert int(jnp.sum(mask)) == 2 * body.cfg.n_target_cells   # exactly the goal axes


def test_babble_then_reach_smoke():
    params, state, body = _tiny_reacher(jax.random.PRNGKey(1))
    babble = run_babbling(
        state, params, body, jax.random.PRNGKey(2),
        BabbleConfig(n_cycles=16, target_refresh=8, forward_settle_steps=1),
    )
    assert babble.command_traj.shape == (16, body.motor_dim)

    result = run_reach(
        babble.brain_state, params, babble.body, jax.random.PRNGKey(3),
        ReachConfig(max_steps=8, act_relax_steps=10),
    )
    assert result.dists.shape == (8,)
    assert jnp.all(jnp.isfinite(result.dists))
    assert result.success.dtype == jnp.bool_
