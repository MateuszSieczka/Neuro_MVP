"""
Astrocyte Field — local precision estimation via Ca²⁺ dynamics.

Reference:
  De Pittà, Volman, Berry & Ben-Jacob (2011) "Computational quest for
      understanding the role of astrocyte signaling"
  Araque et al. (2014) Tripartite synapse

Changes from legacy:
  1. Uses AstrocyteConfig from config.py (τ_ca=5000ms, biological range)
  2. Sigmoid D-Serine release (not step-function threshold)
  3. Gap junction Ca²⁺ wave propagation (1D diffusion between zones)
  4. Derived decays from SimulationContext

Architecture:
  One AstrocyteField per neural region (encoder, decoder, BG, etc.).
  Each field covers n_zones, where each zone monitors a group of synapses.
  Ca²⁺ integrates local prediction error energy → precision estimate.
  D-Serine release modulates NMDA gain (graded, sigmoid).
  Gap junctions propagate Ca²⁺ waves across syncytium.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import AstrocyteConfig


class AstrocyteField:
    """Field of astrocytes providing local precision estimation.

    Ca²⁺ dynamics:
      dCa/dt = -Ca/τ_ca + accumulation × |PE|²
      τ_ca = 5000ms (biological: 2-10s)

    D-Serine release (sigmoid, De Pittà et al. 2011):
      release_rate = d_serine_max × σ((Ca - ca_threshold) / ca_release_k)
      NOT an all-or-nothing step function.

    Gap junction diffusion (1D):
      dCa_i += D × (Ca_{i-1} + Ca_{i+1} - 2×Ca_i)
    """

    def __init__(
        self,
        n_zones: int | None = None,
        config: AstrocyteConfig | None = None,
    ) -> None:
        self.config = config or AstrocyteConfig()
        cfg = self.config
        n = n_zones if n_zones is not None else cfg.n_zones
        self.n_zones: int = n

        # ── Ca²⁺ state per zone ───────────────────────────────────────
        self.calcium: NDArray[np.float32] = np.zeros(n, dtype=np.float32)

        # ── D-Serine (gliotransmitter) level per zone ─────────────────
        self.d_serine: NDArray[np.float32] = np.zeros(n, dtype=np.float32)

        # ── Precomputed decays from config ────────────────────────────
        self._ca_decay: float = cfg.ca_decay
        self._d_serine_decay: float = cfg.d_serine_decay

    def update(self, local_errors: NDArray[np.float32]) -> None:
        """Update Ca²⁺ from local prediction error energy.

        Args:
            local_errors: (n_zones,) or (n_synapses,) error vector.
        """
        errors = self._to_zones(local_errors)
        cfg = self.config

        # ── Ca²⁺ dynamics ─────────────────────────────────────────────
        self.calcium = (
            self.calcium * self._ca_decay
            + cfg.ca_accumulation * errors * (1.0 - self._ca_decay)
        )

        # ── Gap junction diffusion ────────────────────────────────────
        if self.n_zones > 2 and cfg.gap_junction_D > 0:
            laplacian = np.zeros_like(self.calcium)
            laplacian[1:-1] = (
                self.calcium[:-2] + self.calcium[2:] - 2.0 * self.calcium[1:-1]
            )
            # Boundary: zero-flux (Neumann)
            laplacian[0] = self.calcium[1] - self.calcium[0]
            laplacian[-1] = self.calcium[-2] - self.calcium[-1]
            self.calcium += cfg.gap_junction_D * laplacian
            np.maximum(self.calcium, 0.0, out=self.calcium)

        # ── Sigmoid D-Serine release (De Pittà et al. 2011) ──────────
        self.d_serine *= self._d_serine_decay
        sigmoid_arg = (self.calcium - cfg.ca_threshold) / max(cfg.ca_release_k, 1e-6)
        release_rate = cfg.d_serine_max / (1.0 + np.exp(-sigmoid_arg))
        self.d_serine += release_rate.astype(np.float32)
        np.clip(self.d_serine, 0.0, 1.0, out=self.d_serine)

    def _to_zones(self, values: NDArray[np.float32]) -> NDArray[np.float32]:
        """Map arbitrary-length array to n_zones by averaging groups."""
        if values.shape[0] == self.n_zones:
            return np.abs(values).astype(np.float32) ** 2
        n = values.shape[0]
        zone_size = max(1, n // self.n_zones)
        result = np.zeros(self.n_zones, dtype=np.float32)
        for i in range(self.n_zones):
            start = i * zone_size
            end = min(start + zone_size, n)
            if start < n:
                result[i] = float(np.mean(np.abs(values[start:end]) ** 2))
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def precision(self) -> NDArray[np.float32]:
        """Per-zone precision (inverse uncertainty).

        High Ca²⁺ = high error = low precision → learn more.
        Low Ca²⁺ = low error = high precision → consolidate.
        """
        return (1.0 / (1.0 + self.calcium)).astype(np.float32)

    @property
    def synaptic_gain(self) -> NDArray[np.float32]:
        """Per-zone NMDA gain from D-Serine (graded, not binary)."""
        cfg = self.config
        gain_range = cfg.gain_max - cfg.gain_baseline
        return (cfg.gain_baseline + gain_range * self.d_serine).astype(np.float32)

    @property
    def metabolic_lr(self) -> NDArray[np.float32]:
        """Per-zone learning rate multiplier (metabolic support ∝ Ca²⁺)."""
        return (1.0 + self.config.metabolic_scale * self.calcium).astype(np.float32)

    @property
    def mean_precision(self) -> float:
        """Scalar summary: mean precision across zones."""
        return float(np.mean(self.precision))

    @property
    def mean_calcium(self) -> float:
        """Scalar summary: mean Ca²⁺ level."""
        return float(np.mean(self.calcium))

    def reset_state(self) -> None:
        """Reset transient state between episodes."""
        self.calcium.fill(0.0)
        self.d_serine.fill(0.0)
