"""K-armed Gaussian bandit — simplest VTA test.

Semantics
---------
- Episode is non-terminal (``done`` always 0); the brain runs N steps.
- Sensory is a constant one-hot context tag (arm id is never revealed;
  the brain must learn arm values from its own actions).
- Reward for arm ``i`` is ``𝒩(μ_i, σ²)`` with ``μ_i`` fixed at
  initialisation. This is the textbook non-stationary-free MAB
  (Sutton & Barto 2018 §2.3).

Phase 3 rationale
-----------------
With a single context the critic gets zero signal from state; the
actor learns D1/D2 weights purely from per-action TD errors (VTA RPE
derived from reward alone, since ``γ·V(s') = γ·V(s) ≈ const``). This
isolates the BG+VTA loop from all sensory/cortical dynamics.

References
----------
- Sutton & Barto (2018) §2 — k-armed bandit.
- Tobler, Fiorillo & Schultz (2005) — adaptive coding of reward.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from core.backend import DTYPE, Array, PRNGKey, split_key

from .body_interface import BodyInterface, SensorySample, uniform_dense


class GaussianBanditBody(eqx.Module, BodyInterface):
    """Stationary k-armed Gaussian bandit."""

    arm_means: Array                          # (n_actions,)
    noise_sigma: Array                        # scalar
    sensory_level: Array                      # scalar
    sensory_size: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        *,
        n_actions: int = 3,
        mean_spread: float = 1.0,
        noise_sigma: float = 0.1,
        sensory_size: int = 32,
        sensory_level: float = 0.3,
    ) -> "GaussianBanditBody":
        """Sample arm means uniformly in ``[-mean_spread, mean_spread]``.

        The bandit is stateless (single context), so sensory is a
        **dense constant** pattern across ``sensory_size`` afferents at
        firing rate ``sensory_level``. The constant width (32) gives
        enough converging drive to exceed relay + L4 rheobase even
        though no positional information is carried (Sherman & Guillery
        2006 on subcortical sensory drive intensity).
        """
        means = jax.random.uniform(
            key, (n_actions,),
            minval=-mean_spread, maxval=mean_spread, dtype=DTYPE,
        )
        return cls(
            arm_means=means,
            noise_sigma=jnp.asarray(noise_sigma, DTYPE),
            sensory_level=jnp.asarray(sensory_level, DTYPE),
            sensory_size=int(sensory_size),
            n_actions=int(n_actions),
        )

    def _sensory(self) -> Array:
        """Constant dense context — drives the chain above rheobase."""
        return uniform_dense(self.sensory_size, self.sensory_level)

    def reset(self, key: PRNGKey) -> tuple["GaussianBanditBody", SensorySample]:
        sample = SensorySample(
            sensory=self._sensory(),
            reward=jnp.asarray(0.0, DTYPE),
            done=jnp.asarray(0.0, DTYPE),
            info={"arm_means": self.arm_means},
        )
        return self, sample

    def act(
        self, key: PRNGKey, action: Array,
    ) -> tuple["GaussianBanditBody", SensorySample]:
        a = jnp.asarray(action, jnp.int32)
        mu = self.arm_means[a]
        noise = jax.random.normal(key, (), dtype=DTYPE) * self.noise_sigma
        r = mu + noise
        sample = SensorySample(
            sensory=self._sensory(),
            reward=r.astype(DTYPE),
            done=jnp.asarray(0.0, DTYPE),
            info={"chosen": a, "mu": mu},
        )
        return self, sample
