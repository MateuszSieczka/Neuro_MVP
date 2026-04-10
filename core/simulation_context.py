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

from dataclasses import dataclass, field
from functools import lru_cache

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


# ── Default context (dt = 1 ms) ──────────────────────────────────────
DEFAULT_CONTEXT = SimulationContext(dt=1.0)
