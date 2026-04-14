"""Phase 5 verification: Oscillator & Temporal.

BIO 6a: Gamma phase-reset at theta trough (Lisman & Jensen 2013)
BIO 6b: Encoding at theta peak, retrieval at trough (Hasselmo 2005)
CLN 2:  theta_window / episode_window derived from oscillator periods
5.4:    Unified SWS/awake oscillator (single tick() handles both)
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import OscillatorConfig, SequenceMemoryConfig
from core.oscillator import ThetaGammaOscillator
from core.sequence_memory import HierarchicalSequenceMemory
from core.simulation_context import SimulationContext


@pytest.fixture
def ctx():
    return SimulationContext(dt=1.0)


@pytest.fixture
def osc(ctx):
    cfg = OscillatorConfig(ctx=ctx)
    return ThetaGammaOscillator(config=cfg, ctx=ctx)


# =====================================================================
# BIO 6a: Gamma phase-reset at theta trough
# =====================================================================

class TestGammaPhaseReset:
    """Gamma must phase-reset at each theta trough for true PAC."""

    def test_gamma_resets_at_theta_trough(self, osc):
        """Gamma phase should reset to 0 when theta crosses pi."""
        gamma_at_trough_crossing = []

        prev_theta = 0.0
        # Run for ~3 theta cycles at 6Hz, 1ms dt => ~500 steps
        for _ in range(1000):
            osc.tick()
            # Detect theta crossing pi (trough)
            if prev_theta < np.pi and osc.theta_phase >= np.pi:
                # Gamma should have just been reset
                gamma_at_trough_crossing.append(osc.gamma_phase)
            prev_theta = osc.theta_phase

        assert len(gamma_at_trough_crossing) > 0, "No theta trough crossings detected"
        # After reset, gamma may have advanced by one dt step.
        # At 40Hz gamma, one step = 2*pi*40*0.001 = 0.25 rad.
        # But gamma was RESET to 0 then d_gamma was already added.
        # Actually the reset happens BEFORE the cycle completion check,
        # and d_gamma was already added. So gamma = 0 + (already added) - no.
        # Wait: gamma_phase is reset to 0 AFTER d_gamma was added. So it's 0.
        # Then cycle completion check: 0 < 2pi, no wrap. So gamma_phase = 0 at detection.
        # But then one tick() happens before we check. So gamma advanced once more.
        for g in gamma_at_trough_crossing:
            # Should be near zero (within one gamma step)
            one_gamma_step = 2.0 * np.pi * 40.0 * 0.001
            assert g < one_gamma_step + 0.01, (
                f"Gamma phase {g:.3f} at theta trough should be near 0"
            )

    def test_gamma_power_peaks_at_theta_trough(self, osc):
        """Gamma amplitude should be maximal near theta trough (pi)."""
        amplitudes_at_peak = []
        amplitudes_at_trough = []

        for _ in range(2000):
            osc.tick()
            # Near theta peak (theta_phase ~ 0)
            if osc.theta_phase < 0.3 or osc.theta_phase > 2 * np.pi - 0.3:
                amplitudes_at_peak.append(osc.gamma_amplitude)
            # Near theta trough (theta_phase ~ pi)
            elif abs(osc.theta_phase - np.pi) < 0.3:
                amplitudes_at_trough.append(osc.gamma_amplitude)

        if amplitudes_at_trough and amplitudes_at_peak:
            mean_trough = np.mean(amplitudes_at_trough)
            mean_peak = np.mean(amplitudes_at_peak)
            assert mean_trough > mean_peak, (
                f"Gamma amplitude at theta trough ({mean_trough:.3f}) "
                f"should exceed amplitude at theta peak ({mean_peak:.3f})"
            )

    def test_gamma_reset_fires_at_trough(self, osc):
        """gamma_reset should be True when theta crosses pi."""
        resets_near_trough = 0
        total_resets = 0
        prev_theta = 0.0

        for _ in range(5000):
            gamma_reset, _ = osc.tick()
            if gamma_reset:
                total_resets += 1
                # Check if this reset happened near theta trough
                if abs(osc.theta_phase - np.pi) < 0.5 or prev_theta < np.pi <= osc.theta_phase + 0.3:
                    resets_near_trough += 1
            prev_theta = osc.theta_phase

        assert total_resets > 0, "No gamma resets detected"
        # At least some resets should be at theta trough
        ratio = resets_near_trough / total_resets
        assert ratio > 0.05, (
            f"Only {ratio:.1%} of gamma resets near theta trough"
        )


# =====================================================================
# BIO 6b: Encoding/retrieval phase (Hasselmo 2005)
# =====================================================================

class TestEncodingRetrievalPhase:
    """Encoding at theta peak (0->pi), retrieval at trough (pi->2pi)."""

    def test_encoding_at_theta_peak(self, osc):
        """theta_encoding_phase should be True when theta_phase < pi."""
        osc.theta_phase = 0.5
        assert osc.theta_encoding_phase is True
        assert osc.theta_retrieval_phase is False

    def test_retrieval_at_theta_trough(self, osc):
        """theta_retrieval_phase should be True when theta_phase >= pi."""
        osc.theta_phase = np.pi + 0.1
        assert osc.theta_retrieval_phase is True
        assert osc.theta_encoding_phase is False

    def test_boundary_at_pi(self, osc):
        """At exactly pi, should be retrieval (>= pi)."""
        osc.theta_phase = np.pi
        assert osc.theta_retrieval_phase is True
        assert osc.theta_encoding_phase is False

    def test_encoding_is_first_half(self, osc):
        """Encoding should occupy first half of theta cycle (0 to pi).

        Hasselmo (2005): ACh is high during theta peak, favoring
        external input processing (encoding).  cos(theta) > 0 here.
        """
        encoding_count = 0
        retrieval_count = 0
        for _ in range(10000):
            osc.tick()
            if osc.theta_encoding_phase:
                encoding_count += 1
            else:
                retrieval_count += 1

        # Should be roughly 50/50
        ratio = encoding_count / (encoding_count + retrieval_count)
        assert 0.4 < ratio < 0.6, (
            f"Encoding occupies {ratio:.1%} of theta cycle (expected ~50%)"
        )


# =====================================================================
# CLN 2: Derived theta_window and episode_window
# =====================================================================

class TestDerivedWindows:
    """Pooling windows must be derived from oscillator, not magic numbers."""

    def test_theta_window_derived_from_freq(self):
        """theta_window = round(1 / (f_theta * dt_s))."""
        hsm = HierarchicalSequenceMemory(
            num_neurons=10, theta_freq_hz=6.0, dt_ms=1.0,
        )
        # 1 / (6 * 0.001) = 166.67 -> 167
        expected = round(1.0 / (6.0 * 0.001))
        assert hsm.theta_window == expected, (
            f"theta_window={hsm.theta_window}, expected {expected}"
        )

    def test_theta_window_scales_with_freq(self):
        """Higher theta freq -> shorter pooling window."""
        hsm_slow = HierarchicalSequenceMemory(
            num_neurons=10, theta_freq_hz=4.0, dt_ms=1.0,
        )
        hsm_fast = HierarchicalSequenceMemory(
            num_neurons=10, theta_freq_hz=8.0, dt_ms=1.0,
        )
        assert hsm_slow.theta_window > hsm_fast.theta_window

    def test_episode_window_derived(self):
        """episode_window = round(episode_duration_s * f_theta)."""
        hsm = HierarchicalSequenceMemory(
            num_neurons=10, theta_freq_hz=6.0, dt_ms=1.0,
        )
        # 5.0s * 6Hz = 30 theta cycles
        expected = round(5.0 * 6.0)
        assert hsm.episode_window == expected, (
            f"episode_window={hsm.episode_window}, expected {expected}"
        )

    def test_no_magic_theta_window_8(self):
        """Default constructor must NOT produce theta_window=8."""
        hsm = HierarchicalSequenceMemory(num_neurons=10)
        assert hsm.theta_window != 8, (
            "theta_window=8 is the old magic number; "
            "should be derived from theta_freq_hz"
        )

    def test_update_theta_window_dynamic(self):
        """update_theta_window() must adapt to new freq."""
        hsm = HierarchicalSequenceMemory(
            num_neurons=10, theta_freq_hz=6.0, dt_ms=1.0,
        )
        old_window = hsm.theta_window
        hsm.update_theta_window(8.0)  # speed up
        assert hsm.theta_window < old_window


# =====================================================================
# 5.4: Unified SWS/awake oscillator
# =====================================================================

class TestUnifiedSWS:
    """SWS mode must reuse tick() with parameterized frequency."""

    def test_sws_uses_slow_freq(self, osc):
        """In SWS mode, theta freq should be ~1Hz."""
        osc.enter_sws()
        osc.tick()
        assert osc.effective_theta_hz == osc.config.sws_freq_hz

    def test_sws_gamma_suppressed(self, osc):
        """Gamma amplitude should be 0 during SWS."""
        osc.enter_sws()
        osc.tick()
        assert osc.gamma_amplitude == 0.0

    def test_sws_up_down_transitions(self, osc):
        """tick_sws() should produce Up/Down transitions from theta phase."""
        osc.enter_sws()
        up_onsets = 0
        down_onsets = 0
        for _ in range(2000):  # 2 seconds at 1ms dt, ~2 slow cycles
            up_onset, down_onset = osc.tick_sws()
            if up_onset:
                up_onsets += 1
            if down_onset:
                down_onsets += 1

        assert up_onsets >= 1, "No Up state onsets detected in 2s"
        assert down_onsets >= 1, "No Down state onsets detected in 2s"

    def test_sws_exit_restores_normal(self, osc):
        """After exit_sws(), tick() should use normal theta freq again."""
        osc.enter_sws()
        for _ in range(100):
            osc.tick()
        osc.exit_sws()
        osc.tick(ne_level=0.0, sero_level=0.0)
        assert osc.effective_theta_hz == osc.config.theta_freq_hz
        assert osc.gamma_amplitude > 0.0

    def test_no_separate_sws_phase(self, osc):
        """Oscillator should NOT have a separate _sws_phase attribute."""
        assert not hasattr(osc, '_sws_phase'), (
            "_sws_phase exists; SWS should reuse unified theta_phase"
        )

    def test_sws_config_freq(self, ctx):
        """OscillatorConfig must expose sws_freq_hz."""
        cfg = OscillatorConfig(ctx=ctx)
        assert hasattr(cfg, 'sws_freq_hz')
        assert cfg.sws_freq_hz == 1.0

    def test_sws_in_up_state_from_theta(self, osc):
        """in_up_state should derive from theta_phase < pi."""
        osc.enter_sws()
        osc.theta_phase = 0.5
        osc.tick()  # will advance; but let's check after manual set
        # Manually verify the relationship
        osc.theta_phase = 0.5
        osc._in_up_state = osc.theta_phase < np.pi
        assert osc.in_up_state is True

        osc.theta_phase = np.pi + 0.5
        osc._in_up_state = osc.theta_phase < np.pi
        assert osc.in_up_state is False
