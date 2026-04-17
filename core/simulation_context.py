"""
Simulation Context — explicit time management and unit system.

All components share a single SimulationContext to ensure consistent
dt, decay factor computation, and unit conversion. Eliminates the
implicit dt=1ms assumption scattered across the codebase.

Units:
  Time:       milliseconds (ms)
  Potential:  millivolts (mV)
  Current:    effective synaptic charge (mV-equivalent per timestep)
  Rate:       spikes per timestep (instantaneous), or Hz (via to_hz())
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, kw_only=True)
class SimulationContext:
    """Immutable global simulation parameters shared by all components.

    Every module that computes ``exp(-dt/tau)`` should call
    ``ctx.decay(tau)`` instead of inlining the formula, ensuring
    consistency and allowing dt changes to propagate automatically.
    """

    dt: float = 1.0  # Timestep in ms

    def decay(self, tau: float) -> float:
        """Exact exponential decay factor ``exp(-dt / tau)``."""
        if tau <= 0.0:
            return 0.0
        return float(np.exp(-self.dt / tau))

    def complement(self, tau: float) -> float:
        """``1 - exp(-dt / tau)``  — the 'gain' complement of decay."""
        return 1.0 - self.decay(tau)

    def ms_to_steps(self, ms: float) -> int:
        """Convert a duration in ms to integer timestep count (≥ 1)."""
        return max(1, int(round(ms / self.dt)))

    def steps_to_ms(self, steps: int) -> float:
        """Convert timestep count back to milliseconds."""
        return steps * self.dt

    def to_hz(self, rate_per_step: float) -> float:
        """Convert spikes-per-timestep to Hz (spikes per second)."""
        return rate_per_step * (1000.0 / self.dt)

    def from_hz(self, hz: float) -> float:
        """Convert Hz to spikes-per-timestep."""
        return hz * (self.dt / 1000.0)

    # ------------------------------------------------------------------
    # Exponential Euler integrator (Krok 1.2)
    # ------------------------------------------------------------------

    @staticmethod
    def phi1(z: np.ndarray) -> np.ndarray:
        """φ₁(z) = (exp(z) - 1) / z  with full float32 precision.

        Uses np.expm1 to avoid catastrophic cancellation near z ≈ 0.
        Branch for |z| < 1e-4 uses Taylor: 1 + z/2 (error < z²/6 ≈ 10⁻⁹).
        For |z| > 500 the asymptotic form exp(z)/z is used with a capped
        exponent to avoid overflow/NaN while still producing a value
        large enough to trigger AdEx spike detection.
        """
        z_safe = np.asarray(z, dtype=np.float64)
        small = np.abs(z_safe) < 1e-4
        large = z_safe > 500.0
        mid = ~small & ~large
        out = np.empty_like(z_safe)
        out[small] = 1.0 + z_safe[small] * 0.5
        out[mid] = np.expm1(z_safe[mid]) / z_safe[mid]
        # Asymptotic: phi1(z) ≈ exp(z)/z.  Cap exp arg to avoid inf.
        out[large] = np.exp(np.minimum(z_safe[large], 700.0)) / z_safe[large]
        return out

    def exp_euler_step(
        self,
        v: np.ndarray,
        F_v: np.ndarray,
        J_v: np.ndarray,
    ) -> np.ndarray:
        """Exponential Rosenbrock order-1 integration step.

        V_{n+1} = V_n + φ₁(h·J) · h · F(V_n)

        where:
          F(V) = full RHS of ODE (per neuron)
          J(V) = ∂F/∂V  (scalar Jacobian per neuron)
          h    = self.dt

        This is A-stable and handles stiffness from NMDA (τ=100ms)
        and AdEx exponential term without O(N³) implicit costs.

        Args:
            v:   (N,) membrane potentials.
            F_v: (N,) RHS evaluated at current V.
            J_v: (N,) Jacobian ∂F/∂V at current V.

        Returns:
            (N,) updated membrane potentials.
        """
        h = self.dt
        hz = h * J_v
        return v + self.phi1(hz) * h * F_v


# ── Default context (dt = 1 ms) ──────────────────────────────────────
DEFAULT_CONTEXT = SimulationContext(dt=1.0)
