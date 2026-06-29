"""Embodiment layer — interface contract (runs anywhere, no MuJoCo).

The pure interface pieces: sensory layout, population coding, and a
synthetic body proving the contract is body-agnostic.  The MJX arm + babble
→ reach drivers (which need MuJoCo) live in ``test_embodiment_mjx.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from core.backend import DTYPE, Array, PRNGKey
from embodiment.body_interface import (
    BodyInterface, SensorySample, SensoryLayout, zero_value_code_in,
)
from sensory import gaussian_population_encode


# ---------------------------------------------------------------------
# Sensory layout — addressing channels by name, never by magic index
# ---------------------------------------------------------------------


def test_layout_segments_and_mask():
    layout = SensoryLayout.from_sizes((("proprio", 4), ("target_x", 3), ("target_y", 3)))
    assert layout.total == 10
    assert layout.names == ("proprio", "target_x", "target_y")
    assert layout.segment("target_x").start == 4
    assert layout.segment("target_y").stop == 10

    mask = layout.mask(("target_x", "target_y"))
    assert mask.shape == (10,)
    assert int(jnp.sum(mask)) == 6
    assert not bool(mask[0]) and bool(mask[4]) and bool(mask[9])


def test_layout_unknown_segment_raises():
    layout = SensoryLayout.from_sizes((("a", 2),))
    with pytest.raises(KeyError):
        layout.segment("missing")


def test_zero_value_code_placement():
    layout = SensoryLayout.from_sizes((("proprio", 4), ("goal", 3)))
    code = jnp.array([1.0, 2.0, 3.0])
    pref = zero_value_code_in(layout, ("goal",), code)
    assert jnp.allclose(pref[:4], 0.0)
    assert jnp.allclose(pref[4:], code)


# ---------------------------------------------------------------------
# Population coding
# ---------------------------------------------------------------------


def test_population_encode_peaks_at_value():
    n = 21
    code = gaussian_population_encode(0.0, n, x_min=-1.0, x_max=1.0)
    assert code.shape == (n,)
    assert int(jnp.argmax(code)) == n // 2          # centre unit fires hardest
    assert jnp.all(code >= 0.0) and jnp.all(code <= 1.0)


# ---------------------------------------------------------------------
# A synthetic body proves the interface is body-agnostic (no MuJoCo)
# ---------------------------------------------------------------------


class _LinearPlantBody(eqx.Module, BodyInterface):
    """Minimal continuous plant: reafference = B @ command (a unit test body)."""

    weight: Array                                   # (sensory_size, motor_dim)
    sensory_layout: SensoryLayout = eqx.field(static=True)
    sensory_size: int = eqx.field(static=True)
    motor_dim: int = eqx.field(static=True)

    @classmethod
    def create(cls, key: PRNGKey, *, sensory_size: int, motor_dim: int):
        layout = SensoryLayout.from_sizes((("reafference", sensory_size),))
        return cls(
            weight=jax.random.normal(key, (sensory_size, motor_dim), DTYPE),
            sensory_layout=layout, sensory_size=sensory_size, motor_dim=motor_dim,
        )

    @property
    def layout(self) -> SensoryLayout:
        return self.sensory_layout

    def _sample(self, command: Array, done: float) -> SensorySample:
        sensory = (self.weight @ command).astype(DTYPE)
        return SensorySample(
            sensory=sensory, reward=jnp.asarray(0.0, DTYPE),
            done=jnp.asarray(done, DTYPE), info={},
        )

    def reset(self, key: PRNGKey):
        return self, self._sample(jnp.zeros(self.motor_dim, DTYPE), 0.0)

    def act(self, key: PRNGKey, joint_command: Array):
        return self, self._sample(jnp.asarray(joint_command, DTYPE), 0.0)


def test_synthetic_body_conforms_to_interface():
    body = _LinearPlantBody.create(jax.random.PRNGKey(0), sensory_size=5, motor_dim=2)
    assert isinstance(body, BodyInterface)
    body, sample = body.reset(jax.random.PRNGKey(1))
    assert sample.sensory.shape == (5,)
    cmd = jnp.array([1.0, -1.0])
    body, sample = body.act(jax.random.PRNGKey(2), cmd)
    assert sample.sensory.shape == (5,)
    assert jnp.allclose(sample.sensory, body.weight @ cmd)


def test_body_interface_is_abstract():
    with pytest.raises(TypeError):
        BodyInterface()           # cannot instantiate the ABC directly
