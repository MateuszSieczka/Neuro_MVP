"""Faza U — §4 spatial attention as per-slice sensory precision gain.

Asserts divisive-normalisation competition (the surprising field wins),
inhibition-of-return (a held field is suppressed over time), uniform
saliency → no attention, and that the ``(sensory_dim,)`` gain vector is
accepted by the existing ``precision_gains`` hook of the cognitive step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.pc_brain import init_pc_brain, pc_brain_cognitive_step
from core.pc_attention import (
    init_attention_params, init_attention_state,
    attention_step, attention_precision_gains,
)


def test_surprising_field_wins_and_others_suppressed():
    ap = init_attention_params()
    st = init_attention_state(6)
    sal = jnp.array([3.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    out = attention_step(st, ap, sal)
    assert int(jnp.argmax(out.gains)) == 0, "most surprising field gets most Π"
    assert float(out.gains[0]) > 1.0 > float(out.gains[1]), "winner up, losers below 1"
    assert float(jnp.min(out.gains)) >= float(ap.gain_floor) - 1e-6


def test_uniform_saliency_gives_no_attention():
    ap = init_attention_params()
    st = init_attention_state(8)
    out = attention_step(st, ap, jnp.ones(8))
    assert float(jnp.max(jnp.abs(out.gains - 1.0))) < 1e-4, "flat saliency ⇒ flat Π"


def test_inhibition_of_return_favours_fresh_field():
    """Posner & Cohen (1984): a cued (recently attended) field loses to a
    fresh field of equal saliency — attention is inhibited from returning."""
    ap = init_attention_params(tau_ior=5.0)   # fast IOR so the effect shows quickly
    state = init_attention_state(4)
    # Cue: attend field 0 for a while, building inhibition there.
    for _ in range(20):
        out = attention_step(state, ap, jnp.array([2.0, 0.0, 0.0, 0.0]))
        state = out.state
    assert float(state.ior_trace[0]) > 0.0, "field 0 accrued inhibition-of-return"
    # Target: equal saliency on the inhibited field 0 and the fresh field 1.
    out = attention_step(state, ap, jnp.array([1.0, 1.0, 0.0, 0.0]))
    assert float(out.gains[1]) > float(out.gains[0]), \
        "the fresh field wins over the recently attended one"


def test_gain_vector_feeds_cognitive_step_hook():
    key = jax.random.PRNGKey(0)
    bp, bs = init_pc_brain(key, sensory_size=12, motor_size=8)
    ap = init_attention_params()
    ast = init_attention_state(bp.sensory_dim)
    # Relax once on a real observation so sensory error (saliency) is nonzero.
    sensory = jax.random.normal(jax.random.PRNGKey(1), (12,))
    relaxed = pc_brain_cognitive_step(bs, bp, sensory, learn=False)
    _, gains = attention_precision_gains(ast, ap, relaxed.state.graph, bp.graph)
    assert gains[bp.sensory_idx].shape == (12,), "per-slice gain over sensory node"
    # The array gain flows through precision_gains without error.
    out = pc_brain_cognitive_step(bs, bp, sensory, precision_gains=gains)
    assert jnp.isfinite(out.free_energy), "array precision_gains accepted"
