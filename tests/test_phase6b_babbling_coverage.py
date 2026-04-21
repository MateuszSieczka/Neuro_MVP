"""Phase 6B — motor babbling covers ≥ 40 % of the 2-D workspace."""
from __future__ import annotations

import pytest
pytest.importorskip("mujoco")
pytest.importorskip("mujoco.mjx")

import numpy as np

from core.backend import DEFAULT, make_key
from embodiment.reacher_env import build_reacher
from embodiment.mjx_run_loop import run_babbling


def test_phase6b_babbling_coverage() -> None:
    params, state, body = build_reacher(make_key(0))

    # Use a short run for tests; notebook uses 30k cycles.  Coverage is
    # measured on a 20×20 grid over the reachable annulus.
    n_cycles = 2000
    res = run_babbling(
        state, params, DEFAULT, body, make_key(1),
        n_cycles=n_cycles, ou_tau=20.0, ou_sigma=0.6,
        target_refresh=500,
    )

    half = body.cfg.workspace_half
    tips = np.asarray(res.tip_traj)
    # Bin into a 20×20 grid over [-half, half]^2.
    bins = 20
    hx = np.clip(((tips[:, 0] + half) / (2 * half) * bins).astype(int), 0, bins - 1)
    hy = np.clip(((tips[:, 1] + half) / (2 * half) * bins).astype(int), 0, bins - 1)
    visited = set(zip(hx.tolist(), hy.tolist()))
    # The reachable annulus is roughly π(r_max²−r_min²) / (2·half)².  For
    # 0.45/0.15 annulus and half=0.5 that's ~57 % of the square.
    # We assert ≥ 40 % of a *feasibility-corrected* target.
    annulus_area = np.pi * (0.45 ** 2 - 0.15 ** 2)
    square_area = (2 * half) ** 2
    feasible_cells = max(1, int(round(bins * bins * annulus_area / square_area)))
    coverage = len(visited) / feasible_cells
    assert coverage >= 0.4, (
        f"Babbling covered only {coverage:.2f} of reachable cells "
        f"(visited={len(visited)}, feasible={feasible_cells})"
    )
