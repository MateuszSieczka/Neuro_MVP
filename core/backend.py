"""
JAX backend — pure-function analogue of ``simulation_context``.

Everything downstream of Phase 1 builds on this module.  It exposes:

* type aliases for arrays and PRNG keys;
* pure, jit-friendly versions of decay / complement / φ₁ / exp-Euler;
* PRNG key management utilities (``split_key``, ``fold_in_step``);
* a frozen ``BackendContext`` (``eqx.Module``) that plays the same role
  as ``SimulationContext`` but is a pytree leaf — safe inside
  ``jax.jit`` / ``jax.lax.scan`` carries.

Design notes
------------
* All functions are *pure*: no mutation, no side effects.  State lives in
  the caller's pytree.
* ``float32`` is the default dtype; cast once at the edges and stay there.
  φ₁ uses float32 internally — the small-|z| Taylor branch bounds the
  relative error at |z²/6| ≤ 10⁻⁹ which is well below float32 ULP.
* No runtime branch on JAX availability.  This module *requires* JAX.
  The NumPy engine is kept alive in ``simulation_context`` for the
  transition period but new modules (backend, state, sparse …) are
  JAX-only.

Neuromorphic hardware portability
---------------------------------
AdEx is a strict superset of LIF: setting ``a = b = 0`` and ``delta_t → 0``
reduces the equations to pure LIF at the kernel level.  Deployment to
LIF-only chips (Akida, TrueNorth) is a *parameter* transformation, not a
code branch.  ``force_lif_params`` clamps a ``NeuronParams`` pytree for
such targets without touching the step function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array as _Array

    Array = _Array
    PRNGKey = _Array
else:
    Array = jax.Array
    PRNGKey = jax.Array

# Default precision for membrane potentials, weights, traces.
DTYPE = jnp.float32


# ------------------------------------------------------------------
# BackendContext — pytree-safe analogue of SimulationContext.
# ------------------------------------------------------------------


class BackendContext(eqx.Module):
    """Global simulation parameters as an ``eqx.Module`` pytree leaf.

    Mirrors ``core.simulation_context.SimulationContext`` but every
    method returns a JAX scalar / array, never a Python float, so the
    context can participate in jit-traced computations without
    triggering re-tracing.
    """

    dt: Array  # 0-d float32 array (ms)

    def __init__(self, dt: float = 1.0) -> None:
        # Store as a 0-d array so .dt is a traceable value.
        self.dt = jnp.asarray(dt, dtype=DTYPE)

    # -- time constants ------------------------------------------------

    def decay(self, tau: Array | float) -> Array:
        """``exp(-dt / tau)``; returns 0 where ``tau <= 0``."""
        tau_arr = jnp.asarray(tau, dtype=DTYPE)
        safe_tau = jnp.where(tau_arr > 0.0, tau_arr, 1.0)
        d = jnp.exp(-self.dt / safe_tau)
        return jnp.where(tau_arr > 0.0, d, 0.0)

    def complement(self, tau: Array | float) -> Array:
        """``1 - exp(-dt / tau)``; the gain companion of ``decay``."""
        return 1.0 - self.decay(tau)

    def ms_to_steps(self, ms: float) -> int:
        """Convert a duration (ms) to an integer timestep count (≥ 1).

        ``ms`` must be a Python float — this is a *static* conversion
        used at graph-construction time, not inside jit'd code.
        """
        return max(1, int(round(ms / float(self.dt))))

    def steps_to_ms(self, steps: int) -> float:
        return float(steps) * float(self.dt)

    def to_hz(self, rate_per_step: Array | float) -> Array:
        return jnp.asarray(rate_per_step, dtype=DTYPE) * (1000.0 / self.dt)

    def from_hz(self, hz: Array | float) -> Array:
        return jnp.asarray(hz, dtype=DTYPE) * (self.dt / 1000.0)

    # -- exponential-Euler integrator ---------------------------------

    def exp_euler_step(self, v: Array, F_v: Array, J_v: Array) -> Array:
        """One Exponential-Rosenbrock-1 step of ``dV/dt = F(V)``.

        ``V_{n+1} = V_n + φ₁(h·J)·h·F(V_n)``; A-stable, handles AdEx +
        NMDA stiffness without the O(N³) cost of an implicit solver.
        """
        h = self.dt
        hz = h * J_v
        return v + phi1(hz) * h * F_v


# Default 1 ms context — import as ``from core.backend import DEFAULT``.
DEFAULT = BackendContext(dt=1.0)


# ------------------------------------------------------------------
# φ₁ — scalar helper, jit-friendly, branchless.
# ------------------------------------------------------------------


@jax.jit
def phi1(z: Array) -> Array:
    """``φ₁(z) = (exp(z) - 1) / z`` — entire, numerically stable.

    Three regimes, all evaluated (GPU SIMT friendly) then selected:

    * ``|z| < 1e-4``  → Taylor ``1 + z/2`` (rel. error ≤ |z²/6| ≤ 10⁻⁹).
    * ``z > 50``      → asymptotic ``exp(min(z, 80)) / z`` (float32
      caps ``exp(88) ≈ 1.7e38``; anything above saturates cleanly).
    * otherwise       → ``expm1(z) / z`` (hardware-accurate).
    """
    z = jnp.asarray(z, dtype=DTYPE)
    small_branch = 1.0 + 0.5 * z
    # Guard the denominator so the mid branch never divides by 0.
    safe_z = jnp.where(jnp.abs(z) < 1e-4, 1.0, z)
    mid_branch = jnp.expm1(safe_z) / safe_z
    large_branch = jnp.exp(jnp.minimum(z, 80.0)) / jnp.where(z != 0.0, z, 1.0)

    out = jnp.where(z > 50.0, large_branch, mid_branch)
    out = jnp.where(jnp.abs(z) < 1e-4, small_branch, out)
    return out


# ------------------------------------------------------------------
# PRNG helpers.
# ------------------------------------------------------------------


def make_key(seed: int) -> PRNGKey:
    """Create a PRNG key from an integer seed."""
    return jax.random.PRNGKey(seed)


def split_key(key: PRNGKey, n: int = 2) -> Array:
    """Split ``key`` into ``n`` fresh keys.  ``n=2`` by default."""
    return jax.random.split(key, n)


def fold_in_step(key: PRNGKey, step: int | Array) -> PRNGKey:
    """Deterministically derive a subkey for timestep ``step``.

    Prefer this over carrying a running key through ``lax.scan`` when
    you need reproducibility per simulation step without state.
    """
    return jax.random.fold_in(key, jnp.asarray(step, dtype=jnp.uint32))


# ------------------------------------------------------------------
# Batch utility — wrap a pure step so it maps over a leading axis.
# ------------------------------------------------------------------


def vmap_step(step_fn):  # type: ignore[no-untyped-def]
    """``jax.vmap`` a pure ``(state, input) -> (state, output)`` step.

    Axes are ``0`` everywhere; use manually for anything more elaborate.
    """
    return jax.vmap(step_fn, in_axes=(0, 0), out_axes=(0, 0))
