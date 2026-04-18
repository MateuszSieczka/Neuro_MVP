"""Visual grid world — grid world where the brain sees the world through a retina.

Phase 4 rationale
-----------------
Same 2-D grid layout as :class:`embodiment.GridWorldBody` (four
cardinal actions, single goal cell), but the sensory channel is now
the output of :mod:`sensory.retina` applied to a per-cell texture.
This gives the brain structured visual afferents that differ from
cell to cell, forcing the cortex to learn a position-conditioned
state representation from pixels.

Design decisions
----------------
- **Static textures**: one bitmap per grid cell, pre-generated at
  construction time with a cell-dependent random seed. The textures
  contain a mixture of an oriented grating (cell-specific orientation
  and spatial frequency) and a blob (cell-specific location), so each
  cell produces a visually distinct retinal afferent.
- **Fixed fovea**: Phase 4.0 keeps the fixation point at the centre
  of the current cell's image. Saccades are added as a second action
  head of the BG actor in Phase 4.1.
- **Deterministic transitions**: no environmental stochasticity —
  all randomness comes from the brain's own action selection. This
  is essential for reproducible scaling tests.
- **Rendering resolution is adjustable** via ``tex_size``. Doubling
  or halving it must NOT change the sensory vector size because
  the retina absorbs resolution (Phase 4 scale-invariance property).

The texture generator is deliberately not a photograph or a
curated dataset: Olshausen & Field-style V1 emergence wants natural
image statistics, but for a first integration smoke test a
synthetic textured grid is enough. Natural-image sources are
planned for Phase 4.2 tests.

References
----------
- Olshausen & Field (1996) — natural image statistics drive V1 RFs.
- Findlay & Walker (1999) — saccades as goal-directed actions
  (relevant once Phase 4.1 adds the saccade head).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from core.backend import DTYPE, Array, PRNGKey

from sensory import RetinaConfig, RetinaState, init_retina_state, retina_step
from sensory.lgn import lgn_normalize

from .body_interface import BodyInterface, SensorySample


_DIRS = jnp.asarray(
    [[-1, 0], [0, 1], [1, 0], [0, -1]], jnp.int32,
)


# ---------------------------------------------------------------------
# Saccade action space
# ---------------------------------------------------------------------
#
# The brain's oculomotor BG loop emits indices in
# ``[0, core.SACCADE_ACTION_DIM)``. VisualGridBody interprets them as
# follows:
#
#   index 0      : recentre fovea to (0.5, 0.5)
#   indices 1..8 : relative fixation shift in one of eight directions
#                  (N, NE, E, SE, S, SW, W, NW) with step
#                  ``SACCADE_STEP`` in normalised image coordinates.
#
# Saccade amplitude (``SACCADE_STEP`` ≈ 15 % of the image) is chosen
# within the typical human range of 2-20° for exploratory saccades
# (Rayner 1998). Fixations are clipped to ``[SACCADE_MARGIN,
# 1-SACCADE_MARGIN]`` so the foveal patch never falls off the image.

SACCADE_STEP: float = 0.15
SACCADE_MARGIN: float = 0.05
FIXATION_CENTRE: tuple[float, float] = (0.5, 0.5)

# Relative (Δx, Δy) for indices 1..8; index 0 is handled specially.
_SACCADE_DELTAS = jnp.asarray(
    [
        [0.0,  0.0],    # 0: centre (delta unused, recentre by fiat)
        [0.0, -1.0],    # 1: N
        [1.0, -1.0],    # 2: NE
        [1.0,  0.0],    # 3: E
        [1.0,  1.0],    # 4: SE
        [0.0,  1.0],    # 5: S
        [-1.0, 1.0],    # 6: SW
        [-1.0, 0.0],    # 7: W
        [-1.0, -1.0],   # 8: NW
    ],
    DTYPE,
)


def _apply_saccade(
    fixation_xy: Array, saccade_action: Array,
) -> Array:
    """Map a saccade index to a new fixation point in ``[0, 1]^2``."""
    idx = jnp.clip(jnp.asarray(saccade_action, jnp.int32), 0, 8)
    centre = jnp.asarray(FIXATION_CENTRE, DTYPE)
    delta = _SACCADE_DELTAS[idx] * jnp.asarray(SACCADE_STEP, DTYPE)
    shifted = jnp.clip(
        fixation_xy + delta, SACCADE_MARGIN, 1.0 - SACCADE_MARGIN,
    )
    # Index 0 recentres; all other indices shift relatively.
    is_recentre = idx == 0
    return jnp.where(is_recentre, centre, shifted).astype(DTYPE)


# ---------------------------------------------------------------------
# Synthetic texture generator
# ---------------------------------------------------------------------


def _cell_texture(key: PRNGKey, size: int) -> Array:
    """Generate a ``(size, size)`` greyscale texture in ``[0, 1]``.

    Each texture is a mixture of:
    - an oriented sinusoidal grating with a random orientation and
      spatial frequency drawn from biologically plausible ranges
      (orientations uniform on ``[0, pi)``, sf in cycles/image
      ``[2, 8]``);
    - a Gaussian blob at a random location;
    - low-amplitude white noise.

    The mixture is clipped to ``[0, 1]`` so the retina can treat the
    output as contrast.
    """
    kor, ksf, kb, kn = jax.random.split(key, 4)
    yy, xx = jnp.mgrid[0:size, 0:size].astype(DTYPE) / jnp.asarray(size - 1, DTYPE)
    theta = jax.random.uniform(kor, (), DTYPE, 0.0, jnp.pi)
    sf = jax.random.uniform(ksf, (), DTYPE, 2.0, 8.0)
    phase = jnp.cos(theta) * xx + jnp.sin(theta) * yy
    grating = 0.5 + 0.5 * jnp.sin(2.0 * jnp.pi * sf * phase)

    bx = jax.random.uniform(kb, (), DTYPE, 0.2, 0.8)
    by = jax.random.uniform(kb, (), DTYPE, 0.2, 0.8)
    blob = jnp.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2.0 * 0.06 ** 2))

    noise = 0.02 * jax.random.normal(kn, (size, size), DTYPE)
    tex = 0.5 * grating + 0.5 * blob + noise
    return jnp.clip(tex, 0.0, 1.0).astype(DTYPE)


def _make_cell_bank(key: PRNGKey, n_cells: int, tex_size: int) -> Array:
    """Pre-generate ``n_cells`` textures of shape ``(tex_size, tex_size)``.

    The bank is stored in the body as a frozen JAX array; cell
    lookup is a static-index gather keyed by ``row * size + col``.
    """
    keys = jax.random.split(key, n_cells)

    def _one(k):
        return _cell_texture(k, tex_size)

    bank = jax.vmap(_one)(keys)                    # (n_cells, tex, tex)
    return bank.astype(DTYPE)


# ---------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------


class VisualGridBody(eqx.Module, BodyInterface):
    """Grid world whose sensory channel is the retinal afferent vector.

    The body stores a static bank of per-cell textures. On every step
    the texture of the agent's current cell is fed through the retina
    with the fovea fixed at ``fixation_xy`` — a ``(2,)`` array in
    normalised image coordinates ``[0, 1]^2`` updated by the brain's
    oculomotor BG loop via the ``saccade_action`` argument of
    :meth:`act`. A fresh body starts at the image centre.
    """

    pos: Array                                     # (2,) int32
    goal: Array                                    # (2,) int32
    step_idx: Array                                # int32 scalar
    step_cost: Array                               # DTYPE scalar
    goal_reward: Array                             # DTYPE scalar
    tex_bank: Array                                # (size*size, tex, tex)
    retina_state: RetinaState                      # carries prev_coarse
    fixation_xy: Array                             # (2,) DTYPE in [0, 1]
    retina_cfg: RetinaConfig = eqx.field(static=True)
    size: int = eqx.field(static=True)
    tex_size: int = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    sensory_size: int = eqx.field(static=True, default=0)
    n_actions: int = eqx.field(static=True, default=4)

    # ----------------------------------------------------------------

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        *,
        size: int = 5,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] = (4, 4),
        tex_size: int = 64,
        step_cost: float = 0.0,
        goal_reward: float = 1.0,
        max_steps: int = 50,
        retina_cfg: RetinaConfig | None = None,
    ) -> "VisualGridBody":
        cfg = retina_cfg if retina_cfg is not None else RetinaConfig()
        bank = _make_cell_bank(key, size * size, tex_size)
        return cls(
            pos=jnp.asarray(start, jnp.int32),
            goal=jnp.asarray(goal, jnp.int32),
            step_idx=jnp.asarray(0, jnp.int32),
            step_cost=jnp.asarray(step_cost, DTYPE),
            goal_reward=jnp.asarray(goal_reward, DTYPE),
            tex_bank=bank,
            retina_state=init_retina_state(cfg),
            fixation_xy=jnp.asarray(FIXATION_CENTRE, DTYPE),
            retina_cfg=cfg,
            size=int(size),
            tex_size=int(tex_size),
            max_steps=int(max_steps),
            sensory_size=int(cfg.afferent_size),
            n_actions=4,
        )

    # ----------------------------------------------------------------

    def _observe(
        self,
        pos: Array,
        retina_state: RetinaState,
        fixation_xy: Array,
    ) -> tuple[RetinaState, Array]:
        """Fetch the current cell's texture and run the retina on it.

        Returns ``(new_retina_state, afferent_vector)``. ``fixation_xy``
        is supplied by the caller (updated by saccades from the
        oculomotor BG loop).
        """
        idx = pos[0] * self.size + pos[1]
        img = self.tex_bank[idx]                    # (tex, tex)
        new_state, sample = retina_step(
            retina_state, self.retina_cfg, img, fixation_xy,
        )
        # LGN contrast gain control: retina → thalamus must hit the
        # Poisson-rate operating point the cortex expects, independent
        # of scene contrast (Shapley & Enroth-Cugell 1984).
        aff = lgn_normalize(sample.as_afferent())
        return new_state, aff

    # --- BodyInterface ----------------------------------------------

    def reset(
        self, key: PRNGKey,
    ) -> tuple["VisualGridBody", SensorySample]:
        pos0 = jnp.zeros(2, jnp.int32)
        fix0 = jnp.asarray(FIXATION_CENTRE, DTYPE)
        # Fresh retina state (no motion across episode boundaries).
        new_retina = init_retina_state(self.retina_cfg)
        new_retina, aff = self._observe(pos0, new_retina, fix0)
        new_body = eqx.tree_at(
            lambda b: (b.pos, b.step_idx, b.retina_state, b.fixation_xy),
            self,
            (pos0, jnp.asarray(0, jnp.int32), new_retina, fix0),
        )
        return new_body, SensorySample(
            sensory=aff,
            reward=jnp.asarray(0.0, DTYPE),
            done=jnp.asarray(0.0, DTYPE),
            info={"pos": pos0, "fixation_xy": fix0},
        )

    def act(
        self,
        key: PRNGKey,
        body_action: Array,
        saccade_action: Array,
    ) -> tuple["VisualGridBody", SensorySample]:
        a = jnp.clip(jnp.asarray(body_action, jnp.int32), 0, 3)
        d = _DIRS[a]
        new_pos = jnp.clip(self.pos + d, 0, self.size - 1)
        on_goal = jnp.all(new_pos == self.goal)
        step_idx = self.step_idx + 1
        time_out = step_idx >= self.max_steps
        done = jnp.logical_or(on_goal, time_out).astype(DTYPE)
        reward = jnp.where(on_goal, self.goal_reward, -self.step_cost).astype(DTYPE)
        new_fix = _apply_saccade(self.fixation_xy, saccade_action)
        new_retina, aff = self._observe(new_pos, self.retina_state, new_fix)
        new_body = eqx.tree_at(
            lambda b: (b.pos, b.step_idx, b.retina_state, b.fixation_xy),
            self,
            (new_pos, step_idx, new_retina, new_fix),
        )
        return new_body, SensorySample(
            sensory=aff,
            reward=reward,
            done=done,
            info={
                "pos": new_pos,
                "on_goal": on_goal,
                "time_out": time_out,
                "fixation_xy": new_fix,
            },
        )
