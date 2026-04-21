"""Phase 5B — CA1 mismatch comparator → basal-forebrain ACh boost
(McGaughy 2008; Lisman & Grace 2005).

After the agent has "seen" a stable pattern A twice in a row, the
hippocampus_step's ``mismatch`` output should be small on a third
presentation of A and large when a novel pattern X is shown instead.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import math

from core import (
    make_key, split_key,
    init_hippocampus_params, init_hippocampus_state,
    hippocampus_step,
)


def _unit(key, d):
    v = jr.normal(key, (d,))
    return v / (jnp.linalg.norm(v) + 1e-8)


def test_ca1_mismatch_is_high_on_novel_and_low_on_familiar():
    k0 = make_key(0)
    k1, kA, kX = split_key(k0, 3)
    input_dim = 48
    params = init_hippocampus_params(input_dim=input_dim)
    state = init_hippocampus_state(k1, params)

    A = _unit(kA, input_dim)
    X = _unit(kX, input_dim)
    # θ = 3π/2 → rec_gate = 1, enc_gate = 0 — recall is maximal, the
    # comparator is most meaningful, and no new DG storage contaminates
    # the test.
    theta_rec = jnp.asarray(1.5 * math.pi, jnp.float32)

    # Burn in a few cycles so CA3 has a non-trivial predicted_next
    # for A.
    for _ in range(4):
        state = hippocampus_step(
            state, params, A,
            theta_phase=theta_rec, ne_level=0.8,
        ).state

    # The CA1 comparator compares `ca1_prev_recall` (the stored output
    # from the previous cycle) against the live `ec_in`.  To test the
    # comparator directly we now overwrite `ca1_prev_recall` with A
    # and then call the step twice: once with ec_in = A (familiar),
    # once with ec_in = X (novel).
    import equinox as eqx
    state_pin_A = eqx.tree_at(
        lambda s: s.ca1_prev_recall, state, A,
    )

    out_familiar = hippocampus_step(
        state_pin_A, params, A,
        theta_phase=theta_rec, ne_level=0.0,
    )
    out_novel = hippocampus_step(
        state_pin_A, params, X,
        theta_phase=theta_rec, ne_level=0.0,
    )

    fam = float(out_familiar.mismatch)
    nov = float(out_novel.mismatch)
    assert nov > fam, (
        f"CA1 mismatch failed: familiar={fam:.4f} >= novel={nov:.4f}"
    )
    # Meaningful gap: novel must be at least 2× the familiar mismatch
    # (McGaughy 2008 — novelty releases ACh on the order of baseline).
    assert nov >= 2.0 * max(fam, 1e-3), (
        f"CA1 mismatch gap too small: familiar={fam:.4f}, novel={nov:.4f}"
    )
