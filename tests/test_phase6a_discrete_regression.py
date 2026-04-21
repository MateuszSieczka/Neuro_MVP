"""Phase 6A — regression safety: ``bypass_m1=True`` (default) yields the
exact same action_brain cognitive trajectory as the pre-6A code path.

We assert structural invariants that downstream discrete-action code
depends on, rather than bit-identity with a frozen trace (bit-identity
is checked implicitly by the unchanged discrete phase-0/3/4/5 tests).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from core import init_action_brain_params, init_action_brain_state
from core.brain_graph import action_brain_cognitive_step
from core.backend import DEFAULT, make_key


def test_phase6a_discrete_regression() -> None:
    p = init_action_brain_params(
        DEFAULT,
        sensory_size=12, n_body_actions=4, n_saccade_actions=4,
        # bypass_m1 defaults to True — regression path.
    )
    assert p.bypass_m1 is True
    s0 = init_action_brain_state(make_key(0), p)
    s = s0

    key = make_key(1)
    body_actions = []
    for t in range(25):
        key, sk = jax.random.split(key)
        sensory = jax.random.uniform(sk, (p.sensory_size,), jnp.float32)
        out = action_brain_cognitive_step(
            s, p, DEFAULT, sensory,
            prev_reward=0.0, prev_done=0.0, key=sk,
        )
        s = out.state
        body_actions.append(int(out.body_action))

    # Discrete action still in range.
    for a in body_actions:
        assert 0 <= a < p.n_body_actions
    # M1 state exists but is untouched (bypass path).
    assert jnp.allclose(
        s.m1.motor_readout, s0.m1.motor_readout
    ), "bypass_m1=True must not mutate M1 readout"
    assert jnp.allclose(
        s.m1.last_joint_command, s0.m1.last_joint_command
    ), "bypass_m1=True must not mutate last_joint_command"
