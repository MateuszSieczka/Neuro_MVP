"""
ThetaGammaOscillator -- nested oscillation pacemaker.

Reference: Lisman & Jensen (2013) "The theta-gamma neural code"

Theta (4-8 Hz): episodic encoding phase, gates memory storage/retrieval.
Gamma (30-100 Hz): local binding, paces k-WTA competition.

Phase-Amplitude Coupling (PAC) -- Lisman & Jensen (2013):
  Gamma oscillation phase-resets at each theta trough (theta = pi),
  creating genuine cross-frequency coupling.  Gamma amplitude
  is additionally modulated by theta phase:
    gamma_amplitude = base + pac_depth * (1 - cos(theta_phase)) / 2

NE shifts theta frequency UP (arousal -> faster cycling).
5-HT shifts theta frequency DOWN (patience -> longer cycles).

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

        # ── SWS slow oscillation mode ─────────────────────────────────
        # Unified: during SWS, theta freq → sws_freq_hz (~1 Hz),
        # gamma suppressed.  Up/Down states derive from theta phase.
        self._sws_mode: bool = False
        self._in_up_state: bool = False

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
        if self._sws_mode:
            # SWS: slow oscillation ~1 Hz (Steriade et al. 1993).
            # NE/5-HT modulation ignored during sleep.
            self._theta_freq = cfg.sws_freq_hz
        else:
            theta_f = (
                cfg.theta_freq_hz
                + cfg.ne_theta_shift * ne_level
                + cfg.sero_theta_shift * sero_level
            )
            self._theta_freq = float(np.clip(theta_f, cfg.theta_min_hz, cfg.theta_max_hz))

        # ── Phase advance ─────────────────────────────────────────────
        TWO_PI = 2.0 * np.pi
        d_theta = TWO_PI * self._theta_freq * dt_s

        # During SWS, gamma is suppressed (cortical silence between
        # sharp-wave ripples; Steriade et al. 1993).
        if self._sws_mode:
            d_gamma = 0.0
        else:
            d_gamma = TWO_PI * self._gamma_freq * dt_s

        old_theta = self.theta_phase
        self.theta_phase += d_theta
        self.gamma_phase += d_gamma

        # ── PAC: gamma phase-reset at theta trough (Lisman & Jensen 2013)
        # True PAC = gamma bursts restart at each theta trough (θ=π).
        # This locks gamma oscillation onset to the theta cycle,
        # producing the nested "neural code" observed in hippocampus.
        theta_crossed_trough = (old_theta < np.pi) and (self.theta_phase >= np.pi)
        if theta_crossed_trough:
            self.gamma_phase = 0.0

        # Gamma amplitude envelope modulated by theta phase
        # (maximal gamma power at theta trough, minimal at peak)
        # Lisman & Jensen (2013): gamma peaks at theta trough because
        # pyramidal depolarisation is maximal, driving interneuron activity.
        # cos(theta)=-1 at trough, so (1-cos)/2 = 1 at trough.
        pac = cfg.pac_depth * (1.0 - np.cos(self.theta_phase)) / 2.0
        self.gamma_amplitude = 1.0 - cfg.pac_depth + pac

        # ── Cycle completion flags ────────────────────────────────────
        gamma_reset = theta_crossed_trough  # phase-reset counts as cycle
        theta_reset = False

        if self.gamma_phase >= TWO_PI:
            self.gamma_phase -= TWO_PI
            gamma_reset = True

        if self.theta_phase >= TWO_PI:
            self.theta_phase -= TWO_PI
            theta_reset = True

        # SWS Up/Down state derives from unified theta phase:
        # Up = 0→π (first half), Down = π→2π (second half).
        if self._sws_mode:
            self._in_up_state = self.theta_phase < np.pi
            self.gamma_amplitude = 0.0  # gamma suppressed

        return gamma_reset, theta_reset

    @property
    def theta_encoding_phase(self) -> bool:
        """True during theta peak (encoding window: 0 → π).

        Hasselmo (2005): encoding occurs at theta peak where ACh
        is high and external (sensory) input dominates over internal
        (CA3 autoassociative) recall.  cos(θ) > 0 in this window.
        """
        return self.theta_phase < np.pi

    @property
    def theta_retrieval_phase(self) -> bool:
        """True during theta trough (retrieval window: π → 2π).

        Hasselmo (2005): retrieval occurs at theta trough where ACh
        is low and internal CA3 recall dominates over external input.
        cos(θ) < 0 in this window.
        """
        return self.theta_phase >= np.pi

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
        self._sws_mode = False
        self._in_up_state = False

    # ------------------------------------------------------------------
    # SWS slow oscillation mode (unified with tick)
    # ------------------------------------------------------------------

    def enter_sws(self) -> None:
        """Switch to Slow-Wave Sleep oscillation.

        In SWS, tick() uses sws_freq_hz (~1 Hz) for theta and
        suppresses gamma.  Up/Down states derive from theta phase:
          Up  = θ < π (first half of slow oscillation)
          Down = θ ≥ π (second half)
        """
        self._sws_mode = True
        self.theta_phase = 0.0  # start fresh slow oscillation
        self.gamma_phase = 0.0
        self._in_up_state = False

    def exit_sws(self) -> None:
        """Return to normal theta-gamma oscillation."""
        self._sws_mode = False
        self._in_up_state = False

    def tick_sws(self) -> tuple[bool, bool]:
        """Advance slow oscillation by one dt step.

        Thin wrapper around tick() for backward compatibility with
        replay_buffer.  Returns Up/Down state transitions instead
        of gamma/theta resets.

        Returns:
            (up_onset, down_onset): True at Up/Down state transitions.
        """
        was_up = self._in_up_state
        self.tick(ne_level=0.0, sero_level=0.0)
        is_up = self._in_up_state
        up_onset = is_up and not was_up
        down_onset = not is_up and was_up
        return up_onset, down_onset

    @property
    def sws_mode(self) -> bool:
        return self._sws_mode

    @property
    def in_up_state(self) -> bool:
        return self._in_up_state

    # ------------------------------------------------------------------
    # Seizure brake
    # ------------------------------------------------------------------

    _SEIZURE_THRESHOLD_MULT: float = 3.0

    def check_seizure(
        self,
        mean_rate: float,
        baseline_rate: float = 0.05,
    ) -> bool:
        """Check if mean firing rate exceeds seizure threshold.

        If rate > 3× baseline → signals forced Down state for 200ms.
        """
        threshold = self._SEIZURE_THRESHOLD_MULT * max(baseline_rate, 0.01)
        return mean_rate > threshold
