"""
ThetaGammaOscillator — nested oscillation pacemaker.

Reference: Lisman & Jensen (2013) "The theta-gamma neural code"

Theta (4-8 Hz): episodic encoding phase, gates memory storage/retrieval.
Gamma (30-100 Hz): local binding, paces k-WTA competition.

Phase-Amplitude Coupling (PAC):
  gamma_amplitude = base + pac_depth × (1 + cos(theta_phase)) / 2

NE shifts theta frequency UP (arousal → faster cycling).
5-HT shifts theta frequency DOWN (patience → longer cycles).

The oscillator drives:
  - k-WTA evaluation timing (gamma trough)
  - Episodic memory gating (theta phase)
  - STDP window modulation (phase-dependent plasticity)
"""

from __future__ import annotations

import numpy as np

from .config import OscillatorConfig
from .simulation_context import SimulationContext, DEFAULT_CONTEXT


class ThetaGammaOscillator:
    """Nested theta-gamma oscillator with phase-amplitude coupling.

    tick() advances both phases by dt and returns flags for gamma
    and theta cycle completions. The gamma amplitude is modulated
    by theta phase (PAC).
    """

    def __init__(
        self,
        config: OscillatorConfig | None = None,
        ctx: SimulationContext | None = None,
    ) -> None:
        self.config = config or OscillatorConfig()
        self.ctx = ctx or DEFAULT_CONTEXT

        # ── Phase accumulators (0 → 2π) ───────────────────────────────
        self.theta_phase: float = 0.0
        self.gamma_phase: float = 0.0

        # ── Effective frequencies (modulated by NE / 5-HT) ───────────
        self._theta_freq: float = self.config.theta_freq_hz
        self._gamma_freq: float = self.config.gamma_freq_hz

        # ── PAC amplitude (modulated by theta phase) ──────────────────
        self.gamma_amplitude: float = 1.0

    def tick(
        self,
        ne_level: float = 0.0,
        sero_level: float = 0.0,
    ) -> tuple[bool, bool]:
        """Advance oscillator by one dt step.

        Args:
            ne_level:   Noradrenaline [0, 1] → speeds up theta.
            sero_level: Serotonin [0, 1] → slows down theta.

        Returns:
            (gamma_reset, theta_reset): True if cycle completed.
        """
        cfg = self.config
        dt_s = self.ctx.dt / 1000.0  # ms → s

        # ── Modulated theta frequency ─────────────────────────────────
        theta_f = (
            cfg.theta_freq_hz
            + cfg.ne_theta_shift * ne_level
            + cfg.sero_theta_shift * sero_level
        )
        self._theta_freq = float(np.clip(theta_f, cfg.theta_min_hz, cfg.theta_max_hz))

        # ── Phase advance ─────────────────────────────────────────────
        TWO_PI = 2.0 * np.pi
        d_theta = TWO_PI * self._theta_freq * dt_s
        d_gamma = TWO_PI * self._gamma_freq * dt_s

        self.theta_phase += d_theta
        self.gamma_phase += d_gamma

        # ── PAC: gamma amplitude modulated by theta ───────────────────
        # Maximal gamma at theta trough (phase = π), minimal at peak (0)
        pac = cfg.pac_depth * (1.0 + np.cos(self.theta_phase)) / 2.0
        self.gamma_amplitude = 1.0 - cfg.pac_depth + pac

        # ── Cycle completion flags ────────────────────────────────────
        gamma_reset = False
        theta_reset = False

        if self.gamma_phase >= TWO_PI:
            self.gamma_phase -= TWO_PI
            gamma_reset = True

        if self.theta_phase >= TWO_PI:
            self.theta_phase -= TWO_PI
            theta_reset = True

        return gamma_reset, theta_reset

    @property
    def theta_encoding_phase(self) -> bool:
        """True during theta trough (encoding window: π/2 → 3π/2)."""
        return np.pi / 2 < self.theta_phase < 3 * np.pi / 2

    @property
    def theta_retrieval_phase(self) -> bool:
        """True during theta peak (retrieval window: 0 → π/2, 3π/2 → 2π)."""
        return not self.theta_encoding_phase

    @property
    def effective_theta_hz(self) -> float:
        return self._theta_freq

    @property
    def effective_gamma_hz(self) -> float:
        return self._gamma_freq

    def reset(self) -> None:
        self.theta_phase = 0.0
        self.gamma_phase = 0.0
        self.gamma_amplitude = 1.0
        self._theta_freq = self.config.theta_freq_hz
        self._gamma_freq = self.config.gamma_freq_hz
