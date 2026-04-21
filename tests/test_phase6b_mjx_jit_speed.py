"""Phase 6B — JIT speed: 1000 wake cycles under 60 s (Windows tolerance).

Skipped unless mujoco + mjx are installed (Colab GPU is the
reference platform for Phase 6B).
"""
from __future__ import annotations

import time

import pytest
pytest.importorskip("mujoco")
pytest.importorskip("mujoco.mjx")

import jax
import jax.numpy as jnp

from core.backend import DEFAULT, make_key
from embodiment.reacher_env import build_reacher
from embodiment.mjx_run_loop import run_reach_episode


def test_phase6b_mjx_jit_speed() -> None:
    params, state, body = build_reacher(make_key(0))

    # Warm-up: one short episode to trigger JIT compiles.
    key = make_key(1)
    res = run_reach_episode(
        state, params, DEFAULT, body, key, max_steps=10,
    )
    state, body = res.brain_state, res.body

    # Timed run: 1000 cycles of target reaching, no reset between.
    t0 = time.perf_counter()
    res = run_reach_episode(
        state, params, DEFAULT, body, make_key(2),
        max_steps=1000, reset_body=False,
    )
    dt = time.perf_counter() - t0

    # Windows/CPU tolerance: 60 s; Colab GPU should come in under 30 s.
    assert dt < 60.0, f"1000 cycles took {dt:.1f} s (>60 s budget)"
    assert jnp.all(jnp.isfinite(res.rewards))
