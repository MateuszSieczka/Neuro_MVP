"""Phase 6A — cerebellum receives motor prediction error and learns.

When ``bypass_m1=False``, motor PE flows into the cerebellum climbing
fibre channel, so after many cycles the cerebellum's granule→Purkinje
weights should have changed (Marr–Albus–Ito LTD).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from core import init_action_brain_params, init_action_brain_state
from core.brain_graph import action_brain_cognitive_step
from core.backend import DEFAULT, make_key


def test_phase6a_cerebellum_motor_pe() -> None:
    p = init_action_brain_params(
        DEFAULT,
        sensory_size=12, n_body_actions=4, n_saccade_actions=4,
        bypass_m1=False,
    )
    s = init_action_brain_state(make_key(0), p)
    w_init = s.cerebellum.w_gp

    key = make_key(1)
    for t in range(40):
        key, sk = jax.random.split(key)
        sensory = jax.random.uniform(sk, (p.sensory_size,), jnp.float32)
        out = action_brain_cognitive_step(
            s, p, DEFAULT, sensory,
            prev_reward=float(jnp.sin(0.3 * t)),
            prev_done=0.0, key=sk,
        )
        s = out.state

    dw = s.cerebellum.w_gp - w_init
    assert jnp.all(jnp.isfinite(s.cerebellum.w_gp))
    # Weights moved (motor-PE is reaching the cerebellum).
    assert float(jnp.linalg.norm(dw)) > 1e-6
    # State tracks proprioception.
    n_joints = p.proprio.n_joints
    assert s.last_joint_angles.shape == (n_joints,)
    assert s.last_joint_velocities.shape == (n_joints,)
    assert jnp.all(jnp.isfinite(s.last_joint_angles))
