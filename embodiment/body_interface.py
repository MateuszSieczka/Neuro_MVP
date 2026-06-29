"""The brain↔world boundary — one continuous interface for any body.

The predictive-coding substrate speaks one language at its edge
(``core.pc_brain``): a flat **rate vector** clamps the sensory node and a
bounded **joint command** ``∈ [−1, 1]`` is read off the motor node.  A
``BodyInterface`` is the I/O adapter on the other side of that boundary
(integration contract §0.5): it consumes the joint command, advances the
world, and returns the reafferent sensory vector (plus an extrinsic reward
and a ``done`` flag).

Two design commitments make embodiment pluggable:

* **Continuous control only.**  The body receives a continuous
  ``joint_command`` — there is no discrete action id and no separate
  saccade channel.  Action is inference on the substrate (U.5), not a
  policy over a discrete set; a body is a plant, not an action enumerator.
* **A named sensory layout.**  The sensory vector is a concatenation of
  semantically distinct segments (proprioception, target-error, …).  The
  body publishes a :class:`SensoryLayout` so a goal can address channels by
  *name* (e.g. pin "target-error" to zero) instead of by magic index — the
  brain still sees only an undifferentiated rate vector.

References
----------
- Pfeifer & Bongard (2006) *How the Body Shapes the Way We Think* — a
  body-agnostic brain with pluggable embodiment.
- Pouget, Dayan & Zemel (2000) — population coding of continuous variables
  into firing rates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import jax.numpy as jnp

from core.backend import DTYPE, Array, PRNGKey


class SensorySample(NamedTuple):
    """What a body returns after ``reset`` or ``act``.

    Attributes
    ----------
    sensory:
        ``(sensory_size,)`` float32 rate vector — clamps the sensory node.
    reward:
        Scalar **extrinsic** reward in natural units (no shaping; shaping
        belongs to the driver, intrinsic drives to the brain).
    done:
        ``1.0`` when the episode terminates, else ``0.0``.
    info:
        Diagnostic pytree of arrays (tip / target / …).  Never clamped into
        the brain; safe to carry through ``jax.lax.scan``.
    """

    sensory: Array
    reward: Array
    done: Array
    info: dict


class SensorySegment(NamedTuple):
    """A named contiguous slice ``[start, stop)`` of the sensory vector."""

    name: str
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


class SensoryLayout(NamedTuple):
    """Named, contiguous partition of a body's sensory vector.

    Lets goals and probes address sensory channels by name rather than by
    raw index.  Hashable / static — safe as an ``eqx.field(static=True)``.
    """

    segments: tuple[SensorySegment, ...]

    @classmethod
    def from_sizes(cls, named_sizes: tuple[tuple[str, int], ...]) -> "SensoryLayout":
        """Build a layout by laying named segments back-to-back from 0."""
        segments: list[SensorySegment] = []
        cursor = 0
        for name, size in named_sizes:
            segments.append(SensorySegment(name, cursor, cursor + int(size)))
            cursor += int(size)
        return cls(tuple(segments))

    @property
    def total(self) -> int:
        return self.segments[-1].stop if self.segments else 0

    def segment(self, name: str) -> SensorySegment:
        for seg in self.segments:
            if seg.name == name:
                return seg
        raise KeyError(f"no sensory segment named {name!r}; have {self.names}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(seg.name for seg in self.segments)

    def mask(self, names: tuple[str, ...]) -> Array:
        """Boolean ``(total,)`` mask, ``True`` over the named segments."""
        m = jnp.zeros(self.total, dtype=bool)
        for name in names:
            seg = self.segment(name)
            m = m.at[seg.start:seg.stop].set(True)
        return m


class BodyInterface(ABC):
    """ABC for any continuous-control embodiment (MuJoCo, a synthetic plant, …).

    A body is an immutable value object: ``reset`` / ``act`` return a *new*
    body instance (no in-place mutation) so the whole loop is a pure
    functional transition that composes inside ``jax.lax.scan``.
    """

    #: Sensory vector length — sizes the substrate's sensory node.
    sensory_size: int
    #: Joint-command length — sizes the substrate's motor node.
    motor_dim: int

    @property
    @abstractmethod
    def layout(self) -> SensoryLayout:
        """Named partition of this body's ``(sensory_size,)`` vector."""

    @abstractmethod
    def reset(self, key: PRNGKey) -> tuple["BodyInterface", SensorySample]:
        """Return a fresh body at t=0 and its first sensory sample."""

    @abstractmethod
    def act(
        self, key: PRNGKey, joint_command: Array,
    ) -> tuple["BodyInterface", SensorySample]:
        """Advance the world by one cycle under ``joint_command``.

        Parameters
        ----------
        key:
            PRNG key for any stochastic world dynamics.
        joint_command:
            ``(motor_dim,)`` float32, tanh-bounded in ``[−1, 1]`` — the
            command read off the substrate's motor node.

        Returns
        -------
        new_body, sample:
            The body after the transition and the observation of the new
            state, including the reward emitted during the transition.
        """


def zero_value_code_in(layout: SensoryLayout, names: tuple[str, ...],
                       value_code: Array) -> Array:
    """Place ``value_code`` into the named segments of a zeroed sensory vector.

    Helper for building a partial preference: the returned ``(total,)``
    vector carries ``value_code`` on each named segment and 0 elsewhere; the
    matching :meth:`SensoryLayout.mask` selects which dimensions are pinned.
    """
    pref = jnp.zeros(layout.total, DTYPE)
    for name in names:
        seg = layout.segment(name)
        pref = pref.at[seg.start:seg.stop].set(value_code.astype(DTYPE))
    return pref
