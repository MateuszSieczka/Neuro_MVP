"""Abstract body interface — single boundary between brain and world.

Phase 3 design decisions
------------------------
The brain produces an **integer action id** plus a continuous readout
(cortex L5 rate). A ``BodyInterface`` interprets that action in the
world and returns a sensory vector, a scalar extrinsic reward, and a
``done`` flag. Everything else (curiosity, homeostasis, world-model
prediction error) is computed inside the brain, not the body.

The body owns **exactly one** piece of state: the physical configuration
of the agent in its world. No hidden reward machinery, no shaping,
no intrinsic motivation. Those belong to the brain.

Sensory encoding policy
-----------------------
A body returns a ``sensory`` ``jnp.ndarray`` of shape ``(sensory_size,)``
in the range ``[0, 1]``. The interpretation (one-hot position, Gaussian
population code, raw pixel intensity, …) is entirely up to the adapter.
The brain treats it as Poisson firing-rate input to the thalamus
(``relay.afferent``) — bodies should therefore return values that look
like rates, not raw spike counts.

References
----------
- Pfeifer & Bongard (2006) *How the Body Shapes the Way We Think*
  — body-agnostic brain / pluggable embodiment.
- Pouget, Dayan & Zemel (2000) — population coding of continuous
  variables into neural firing rates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import jax.numpy as jnp

from core.backend import DTYPE, Array, PRNGKey


class SensorySample(NamedTuple):
    """What a body hands back to the brain after ``reset`` or ``act``.

    Attributes
    ----------
    sensory:
        ``(sensory_size,)`` float32 in ``[0, 1]`` — Poisson-rate input
        for the thalamic afferent channel.
    reward:
        Scalar **extrinsic** reward in natural units (no shaping).
    done:
        ``1.0`` when the episode terminates, else ``0.0``.
    info:
        Optional diagnostic dict. Never read by the brain.
    """

    sensory: Array
    reward: Array
    done: Array
    info: dict


class BodyInterface(ABC):
    """ABC for any embodiment (bandit, gridworld, MuJoCo, Unity, ROS).

    The brain emits **two** discrete motor commands on every decision
    cycle, one per parallel cortico-BG-thalamo-cortical loop (Alexander,
    DeLong & Strick 1986):

      * ``body_action``    — skeletomotor command in ``[0, n_actions)``.
      * ``saccade_action`` — oculomotor command in
        ``[0, core.SACCADE_ACTION_DIM)``.

    Every body must accept both; bodies with no visual sensor simply
    ignore the saccade argument (bandit, plain gridworld). The
    ``VisualGridBody`` and other visually-sensed bodies interpret the
    saccade index as a relative shift of the retinal fixation point.
    """

    #: Static — brain needs this to size the thalamic afferent channel.
    sensory_size: int
    #: Static — number of discrete body actions. BG body-actor ``motor_dim``.
    n_actions: int

    @abstractmethod
    def reset(self, key: PRNGKey) -> tuple["BodyInterface", SensorySample]:
        """Return a fresh body at t=0 and the first sensory sample.

        Returns a *new* body instance (bodies are immutable pytrees /
        value objects; step functions do not mutate in place).
        """

    @abstractmethod
    def act(
        self,
        key: PRNGKey,
        body_action: Array,
        saccade_action: Array,
    ) -> tuple["BodyInterface", SensorySample]:
        """Advance the world by one decision cycle.

        Parameters
        ----------
        key:
            PRNG key for any stochastic world dynamics.
        body_action:
            Integer scalar in ``[0, n_actions)`` (``jnp.int32``).
        saccade_action:
            Integer scalar in ``[0, SACCADE_ACTION_DIM)``. Bodies
            without a visual sensor should ignore this argument.

        Returns
        -------
        new_body, sample:
            Body after the transition and the observation of the new
            state, including the reward emitted during the transition.
        """


# ---------------------------------------------------------------------
# Helper used by adapters: convert a scalar to an unclipped sensory
# vector via Gaussian population coding.
# ---------------------------------------------------------------------


def one_hot(idx: Array, n: int) -> Array:
    """One-hot vector of size ``n`` as float32 in ``[0, 1]``.

    Used by categorical-sensory bodies (GridWorld position, bandit
    context tag). Guaranteed sum == 1.0 so the brain receives a
    well-scaled Poisson rate.
    """
    i = jnp.asarray(idx, jnp.int32)
    return (jnp.arange(n) == i).astype(DTYPE)


def gauss_pop_encode(
    x: Array | float, n: int, *,
    x_min: float = 0.0, x_max: float = 1.0,
    sigma: float | None = None,
    peak: float = 1.0,
) -> Array:
    """Gaussian population code of scalar ``x`` across ``n`` tuning curves.

    Each of the ``n`` units has a preferred value ``c_i`` uniformly spaced
    on ``[x_min, x_max]``. Activation is a Gaussian bump
    ``peak · exp(-0.5 · ((x − c_i) / sigma)^2)``. Default ``sigma`` is
    the center spacing (``(x_max−x_min)/(n−1)``), giving half-overlap
    between adjacent tuning curves — the canonical Pouget-Dayan-Zemel
    (2000) cortical population code.

    Returns a dense ``(n,)`` float32 vector in ``[0, peak]``. Typical
    density (fraction of units > 0.5·peak) is ~2·sigma/(x_max−x_min),
    ~0.3 under defaults.
    """
    centers = jnp.linspace(x_min, x_max, n, dtype=DTYPE)
    if sigma is None:
        sigma_val = (x_max - x_min) / max(1, n - 1)
    else:
        sigma_val = float(sigma)
    xv = jnp.asarray(x, DTYPE)
    z = (xv - centers) / jnp.asarray(sigma_val, DTYPE)
    return (jnp.asarray(peak, DTYPE) * jnp.exp(-0.5 * z * z)).astype(DTYPE)


def uniform_dense(n: int, level: float | Array = 0.3) -> Array:
    """Constant dense sensory pattern of shape ``(n,)``.

    Used by stateless bodies (bandits) that have no context to encode
    but still need to drive the thalamo-cortical chain above rheobase
    (a single sparse afferent is insufficient; Sherman & Guillery 2006).
    """
    lvl = jnp.asarray(level, DTYPE)
    return jnp.broadcast_to(lvl, (int(n),)).astype(DTYPE)


def discretise_joint_command(
    joint_command: Array,
    n_actions: int,
) -> Array:
    """Convert a continuous ``(motor_dim,)`` tanh-bounded joint command
    into a discrete action id in ``[0, n_actions)``.

    Phase 6A transition adapter: until MJX is wired in (Phase 6B), the
    existing gridworld / bandit / visual_grid bodies only understand
    discrete action ids.  We use a **sign-split argmax** over joint
    channels (Georgopoulos 1986 directional tuning): each motor DoF
    contributes one positive and one negative "direction", and the
    dominant direction wins.  This doubles motor_dim channels into
    ``2 · motor_dim`` candidate actions; the final id is clipped into
    ``[0, n_actions)``.

    This is a **pure function** — no state, no key — so it composes
    cleanly inside JIT.

    Parameters
    ----------
    joint_command : Array shape (motor_dim,) in [-1, 1]
    n_actions : int
        Discrete action space of the body.

    Returns
    -------
    Array : int32 scalar in ``[0, n_actions)``.
    """
    jc = jnp.asarray(joint_command, DTYPE).reshape(-1)
    pos = jnp.maximum(jc, 0.0)
    neg = jnp.maximum(-jc, 0.0)
    stacked = jnp.concatenate([pos, neg], axis=0)   # (2 * motor_dim,)
    raw_idx = jnp.argmax(stacked)
    return jnp.clip(raw_idx, 0, int(n_actions) - 1).astype(jnp.int32)
