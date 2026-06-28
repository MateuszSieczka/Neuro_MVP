"""Faza U — krok U.3: cykl poznawczy jako relaksacja grafu.

``core.pc_brain`` replaces the hand-coded ``action_brain_cognitive_step``
sequence with free-energy relaxation on the region graph.  One cycle is
clamp(sensory) → relax → read motor → learn (one rule).  These tests
assert the closure properties:

* the step runs end to end with finite, bounded outputs (drop-in for the
  MJX driver: same flat sensory in, ``tanh`` joint command out);
* repeated exposure to a fixed observation lowers the global free energy
  (the graph *learns to predict* its afferent — perception works);
* ``learn=False`` is pure inference (no weight change);
* the motor read-out depends on the sensory input (action is driven by
  the relaxed graph, not constant).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.pc_brain import init_pc_brain, pc_brain_cognitive_step
from core.pc_graph import REGION_INDEX


def _brain(seed=0, **kw):
    return init_pc_brain(
        jax.random.PRNGKey(seed),
        sensory_size=8, motor_size=3,
        eta_mu=0.05, eta_w=2e-2, n_relax=30, **kw,
    )


def test_cognitive_step_runs_and_is_bounded():
    params, state = _brain()
    sensory = jax.random.normal(jax.random.PRNGKey(1), (params.sensory_dim,))
    out = pc_brain_cognitive_step(state, params, sensory)

    assert out.joint_command.shape == (params.motor_dim,)
    assert jnp.all(jnp.abs(out.joint_command) <= 1.0), "command outside [-1,1]"
    assert jnp.isfinite(out.value)
    assert jnp.isfinite(out.free_energy)
    assert jnp.all(jnp.isfinite(out.belief))
    assert jnp.isfinite(out.epistemic), "epistemic info-gain not finite"


def test_precision_gain_modulates_inference():
    """The neuromodulatory precision hook changes the relaxed cycle."""
    params, state = _brain(seed=7)
    sensory = jax.random.normal(jax.random.PRNGKey(8), (params.sensory_dim,))

    base = pc_brain_cognitive_step(state, params, sensory, learn=False)
    gated = pc_brain_cognitive_step(
        state, params, sensory, learn=False,
        precision_gains={REGION_INDEX["sensory"]: 10.0},
    )
    # Boosting sensory precision sharpens the attended error → the global
    # objective and the motor read-out shift; pure inference leaves weights.
    assert float(jnp.sum(jnp.abs(gated.joint_command - base.joint_command))) > 1e-5, (
        "precision gain had no effect on inference"
    )
    moved = sum(
        float(jnp.sum(jnp.abs(a - b)))
        for a, b in zip(gated.state.graph.weights, state.graph.weights)
    )
    assert moved == 0.0, "precision-gated inference mutated weights"


def test_repeated_exposure_lowers_free_energy():
    """Graph learns to predict a fixed afferent → perceptual FE drops."""
    params, state = _brain(seed=2)
    sensory = jax.random.normal(jax.random.PRNGKey(3), (params.sensory_dim,))

    fe0 = float(pc_brain_cognitive_step(state, params, sensory, learn=False).free_energy)
    for _ in range(200):
        out = pc_brain_cognitive_step(state, params, sensory)
        state = out.state
    feN = float(pc_brain_cognitive_step(state, params, sensory, learn=False).free_energy)

    assert feN < fe0 * 0.7, f"perception did not learn: FE {fe0:.4f} → {feN:.4f}"


def test_inference_only_does_not_change_weights():
    params, state = _brain(seed=4)
    sensory = jax.random.normal(jax.random.PRNGKey(5), (params.sensory_dim,))
    out = pc_brain_cognitive_step(state, params, sensory, learn=False)
    moved = sum(
        float(jnp.sum(jnp.abs(a - b)))
        for a, b in zip(out.state.graph.weights, state.graph.weights)
    )
    assert moved == 0.0, "learn=False changed weights"


def test_motor_readout_depends_on_sensory():
    params, state = _brain(seed=6)
    # Train a little so weights are non-trivial.
    for _ in range(50):
        s = jax.random.normal(jax.random.PRNGKey(_), (params.sensory_dim,))
        state = pc_brain_cognitive_step(state, params, s).state

    s_a = jnp.ones(params.sensory_dim) * 0.8
    s_b = -jnp.ones(params.sensory_dim) * 0.8
    cmd_a = pc_brain_cognitive_step(state, params, s_a, learn=False).joint_command
    cmd_b = pc_brain_cognitive_step(state, params, s_b, learn=False).joint_command
    assert float(jnp.sum(jnp.abs(cmd_a - cmd_b))) > 1e-4, (
        "motor read-out ignores the sensory input"
    )
