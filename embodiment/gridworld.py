"""Minimal grid world — test BG action selection in a spatial task.

Semantics
---------
- ``size × size`` grid, integer positions.
- Four cardinal actions (N=0, E=1, S=2, W=3); invalid moves leave the
  agent in place (boundary reflection with zero reward).
- One goal cell; stepping into it yields ``goal_reward`` and ends the
  episode.
- Each non-terminal step costs ``step_cost`` (default 0 — we don't
  want shaping to replace real learning).
- Sensory is concatenation of
  * row one-hot, col one-hot (size ``2·size``);
  * Δx, Δy to goal, each clipped to ``[-1, 1]`` and rescaled to ``[0, 1]``;
  that gives ``2·size + 2`` features.

Phase 3 rationale
-----------------
GridWorld requires the actor to use state information (position-
conditioned policy); the bandit does not. Tests the whole
cortex → BG → VTA → neuromodulator loop.

The body is deterministic; stochasticity is injected only by the
brain's own action selection.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from core.backend import DTYPE, Array, PRNGKey

from .body_interface import BodyInterface, SensorySample, gauss_pop_encode


_DIRS = jnp.asarray(
    # N,   E,  S,  W
    [[-1, 0], [0, 1], [1, 0], [0, -1]], jnp.int32,
)


class GridWorldBody(eqx.Module, BodyInterface):
    """Deterministic 2-D grid world with a single goal."""

    pos: Array                                # (2,) int32 (row, col)
    goal: Array                               # (2,) int32
    step_cost: Array                          # scalar
    goal_reward: Array                        # scalar
    size: int = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    step_idx: Array = eqx.field(default=None)  # scalar int32
    sensory_size: int = eqx.field(static=True, default=0)
    n_actions: int = eqx.field(static=True, default=4)

    @classmethod
    def create(
        cls,
        *,
        size: int = 5,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] = (4, 4),
        step_cost: float = 0.0,
        goal_reward: float = 1.0,
        max_steps: int = 50,
    ) -> "GridWorldBody":
        pos = jnp.asarray(start, jnp.int32)
        g = jnp.asarray(goal, jnp.int32)
        # Gaussian population code of row and col, plus Δx/Δy.
        sensory_size = 2 * size + 2
        return cls(
            pos=pos, goal=g,
            step_cost=jnp.asarray(step_cost, DTYPE),
            goal_reward=jnp.asarray(goal_reward, DTYPE),
            size=int(size), max_steps=int(max_steps),
            step_idx=jnp.asarray(0, jnp.int32),
            sensory_size=int(sensory_size),
            n_actions=4,
        )

    # --- sensory encoding --------------------------------------------

    def _encode(self, pos: Array) -> Array:
        """Gaussian place-code of ``(row, col)`` + Δxy hint (Pouget 2000)."""
        row_norm = pos[0].astype(DTYPE) / max(1, self.size - 1)
        col_norm = pos[1].astype(DTYPE) / max(1, self.size - 1)
        row = gauss_pop_encode(row_norm, self.size)
        col = gauss_pop_encode(col_norm, self.size)
        delta = (self.goal - pos).astype(DTYPE)
        d = jnp.clip(delta, -1.0, 1.0) * 0.5 + 0.5
        return jnp.concatenate([row, col, d]).astype(DTYPE)

    # --- Body interface ---------------------------------------------

    def reset(self, key: PRNGKey) -> tuple["GridWorldBody", SensorySample]:
        new_body = eqx.tree_at(
            lambda b: (b.pos, b.step_idx),
            self,
            (jnp.zeros(2, jnp.int32), jnp.asarray(0, jnp.int32)),
        )
        sample = SensorySample(
            sensory=new_body._encode(new_body.pos),
            reward=jnp.asarray(0.0, DTYPE),
            done=jnp.asarray(0.0, DTYPE),
            info={"pos": new_body.pos},
        )
        return new_body, sample

    def act(
        self,
        key: PRNGKey,
        body_action: Array,
        saccade_action: Array,
    ) -> tuple["GridWorldBody", SensorySample]:
        del saccade_action   # plain GridWorld has no visual sensor
        a = jnp.clip(jnp.asarray(body_action, jnp.int32), 0, 3)
        d = _DIRS[a]
        new_pos = jnp.clip(self.pos + d, 0, self.size - 1)
        on_goal = jnp.all(new_pos == self.goal)
        step_idx = self.step_idx + 1
        time_out = step_idx >= self.max_steps
        done = jnp.logical_or(on_goal, time_out).astype(DTYPE)
        reward = jnp.where(on_goal, self.goal_reward, -self.step_cost)
        new_body = eqx.tree_at(
            lambda b: (b.pos, b.step_idx),
            self, (new_pos, step_idx),
        )
        sample = SensorySample(
            sensory=new_body._encode(new_pos),
            reward=reward.astype(DTYPE),
            done=done,
            info={"pos": new_pos, "on_goal": on_goal, "time_out": time_out},
        )
        return new_body, sample
