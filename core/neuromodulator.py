"""
NeuromodulatorSystem — four-channel neuromodulatory orchestra.

Reference: Doya (2002), Grace (1991), Niv et al. (2007), Tobler et al. (2005)

Changes from legacy:
  1. Decay constants from pharmacological kinetics (DAT τ=200ms, AChE τ=25ms,
     NET τ=75ms, SERT τ=150ms) via NeuromodulatorConfig.
  2. DA RMS adaptation τ ≈ 10s (config.da_rms_decay=0.9999), not 100ms.
  3. Serotonin weights from dorsal raphe anatomy (0.7/0.3, config).
  4. Tonic DA: continuous leaky integrator τ=60s (Grace 1991), NOT episodic.
  5. Receptor-subtype-aware layer modulation via apply_to_layer().
  6. No episode boundary concept — purely per-step dynamics.

All levels normalised to [0, 1].
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from .config import NeuromodulatorConfig


class NeuromodulatorSystem:
    """Four-channel neuromodulatory system (DA, ACh, NE, 5-HT).

    DA (phasic): RPE per step → STDP learning rate.
    DA (tonic): average reward rate → consolidation gate.
    ACh: novelty/uncertainty → bottom-up vs top-down balance.
    NE: surprise/arousal → k-WTA sharpness + trace compression.
    5-HT: temporal stability → planning horizon.
    """

    def __init__(self, config: NeuromodulatorConfig | None = None) -> None:
        self.config = config or NeuromodulatorConfig()
        cfg = self.config

        # ── Current levels (0-1) ──────────────────────────────────────
        self.dopamine: float = cfg.baseline_da
        self.acetylcholine: float = cfg.baseline_ach
        self.noradrenaline: float = cfg.baseline_ne
        self.serotonin: float = cfg.baseline_sero
        self.tonic_da: float = cfg.baseline_tonic_da

        # ── Histories (per-step rolling windows) ──────────────────────
        self._error_history: deque[float] = deque(maxlen=100)
        self._td_history: deque[float] = deque(maxlen=100)

        # ── DA RMS with proper τ (config.da_rms_decay) ────────────────
        self._da_rms: float = 1.0

        # ── Stagnation detector (ACC) ─────────────────────────────────
        self._tda_history: deque[float] = deque(maxlen=30)
        self._stagnation_factor: float = 0.0

        # ── Per-region NE/ACh (Schultz 1998: DA/5-HT global;
        #    Berridge & Waterhouse 2003: NE regional;
        #    Hasselmo 2006: ACh regional) ──────────────────────────────
        self._ne_levels: dict[str, float] = {}
        self._ach_levels: dict[str, float] = {}
        self._region_names: list[str] = []

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def update(
        self,
        prediction_error: NDArray[np.float32],
        td_error: float = 0.0,
        novelty: float | None = None,
    ) -> None:
        """Update per-step neuromodulator levels."""
        cfg = self.config
        error_mag = float(np.clip(np.mean(np.abs(prediction_error)), 0.0, 1.0))
        self._error_history.append(error_mag)

        if novelty is None:
            novelty = error_mag

        # ── Phasic DA: adaptive-gain RPE (Tobler et al. 2005) ────────
        self._da_rms = float(np.sqrt(
            cfg.da_rms_decay * self._da_rms ** 2
            + (1.0 - cfg.da_rms_decay) * td_error ** 2
        ))
        da_gain = 0.35 / max(self._da_rms, 0.1)
        rpe_signal = float(np.clip(
            cfg.baseline_da + da_gain * td_error, 0.0, 1.0,
        ))
        self.dopamine = (
            self.dopamine * cfg.da_decay
            + rpe_signal * (1.0 - cfg.da_decay)
        )

        # ── ACh: novelty/uncertainty ──────────────────────────────────
        self.acetylcholine = (
            self.acetylcholine * cfg.ach_decay
            + float(np.clip(novelty, 0.0, 1.0)) * (1.0 - cfg.ach_decay)
        )

        # ── NE: global surprise ───────────────────────────────────────
        self.noradrenaline = (
            self.noradrenaline * cfg.ne_decay
            + float(np.clip(error_mag, 0.0, 1.0)) * (1.0 - cfg.ne_decay)
        )

        # ── 5-HT: prediction stability (dorsal raphe) ────────────────
        avg_error = float(np.mean(self._error_history)) if self._error_history else 0.5
        world_stability = float(np.clip(1.0 - avg_error, 0.0, 1.0))

        self._td_history.append(float(np.clip(abs(td_error), 0.0, 10.0)))
        avg_td = float(np.mean(self._td_history)) if self._td_history else 5.0
        td_stability = 1.0 / (1.0 + avg_td)
        reward_quality = 1.0 / (1.0 + np.exp(-(self.tonic_da * 4.0 - 2.0)))
        behavioral_stability = float(td_stability * reward_quality)

        # Dorsal raphe anatomy weights (config)
        stability = (
            cfg.sero_world_weight * world_stability
            + cfg.sero_behavioral_weight * behavioral_stability
        )
        self.serotonin = (
            self.serotonin * cfg.sero_decay
            + stability * (1.0 - cfg.sero_decay)
        )

        # ── Tonic DA: continuous leaky integrator (Grace 1991) ────────
        # Integrates |RPE| over ~60s window. No episode boundary needed.
        rpe_abs = float(np.clip(abs(td_error), 0.0, 1.0))
        self.tonic_da = (
            self.tonic_da * cfg.tonic_da_decay
            + rpe_abs * (1.0 - cfg.tonic_da_decay)
        )

        # ── Stagnation tracking (ACC) ─────────────────────────────────
        self._tda_history.append(self.tonic_da)
        if len(self._tda_history) >= 10:
            variability = float(np.std(list(self._tda_history)))
            raw_stag = float(np.clip(1.0 - variability / 0.05, 0.0, 1.0))
            self._stagnation_factor = 0.9 * self._stagnation_factor + 0.1 * raw_stag

        self._clamp_all()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def learning_rate_modulation(self) -> float:
        """Phasic DA → STDP learning rate (m_t)."""
        return self.dopamine

    @property
    def consolidation_gate(self) -> float:
        """sqrt(tonic_da × serotonin) with ACC stagnation attenuation."""
        raw = float(np.sqrt(self.tonic_da * self.serotonin))
        if 0.3 < self.tonic_da < 0.7:
            acc = 1.0 - 0.5 * self._stagnation_factor
        else:
            acc = 1.0
        return raw * acc

    @property
    def bottom_up_gain(self) -> float:
        """ACh level → PredictiveCodingLayer.set_ach_level()."""
        return self.acetylcholine

    @property
    def competition_sharpness(self) -> float:
        """NE level → k-WTA sharpness."""
        return self.noradrenaline

    @property
    def planning_horizon(self) -> float:
        """5-HT → temporal discount / planning depth."""
        return self.serotonin

    # ------------------------------------------------------------------
    # Per-region NE / ACh
    # ------------------------------------------------------------------

    def register_region(self, name: str) -> None:
        """Register a brain region for per-region NE/ACh modulation."""
        if name not in self._ne_levels:
            self._ne_levels[name] = self.noradrenaline
            self._ach_levels[name] = self.acetylcholine
            self._region_names.append(name)

    def ne_for_region(self, name: str | None = None) -> float:
        """Per-region NE. Falls back to global if region unknown."""
        if name is None:
            return self.noradrenaline
        return self._ne_levels.get(name, self.noradrenaline)

    def ach_for_region(self, name: str | None = None) -> float:
        """Per-region ACh. Falls back to global if region unknown."""
        if name is None:
            return self.acetylcholine
        return self._ach_levels.get(name, self.acetylcholine)

    def update_regional(
        self,
        region_errors: dict[str, float] | None = None,
    ) -> None:
        """Update per-region NE/ACh from local prediction errors.

        NE: locus coeruleus projects differentially — higher local PE → higher NE
        (Berridge & Waterhouse 2003).
        ACh: basal forebrain global with slight regional bias (Hasselmo 2006).
        DA and 5-HT remain global (Schultz 1998, Doya 2002).
        """
        if region_errors is None:
            region_errors = {}

        for name in self._region_names:
            local_pe = region_errors.get(name, 0.0)
            # NE: global baseline + local PE boost (LC regional projection)
            self._ne_levels[name] = float(np.clip(
                self.noradrenaline + 0.3 * local_pe, 0.0, 1.0,
            ))
            # ACh: global level (basal forebrain uniform projection)
            self._ach_levels[name] = self.acetylcholine

    # ------------------------------------------------------------------
    # Layer interface
    # ------------------------------------------------------------------

    def apply_to_layer(
        self,
        layer: object,
        region: str | None = None,
    ) -> None:
        """Propagate NE/ACh to any layer supporting modulation."""
        ne = self.ne_for_region(region)
        ach = self.ach_for_region(region)
        if hasattr(layer, 'set_plasticity_timescales'):
            layer.set_plasticity_timescales(ne=ne, ach=ach)
        if hasattr(layer, 'set_ne_level'):
            layer.set_ne_level(ne)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clamp_all(self) -> None:
        self.dopamine = float(np.clip(self.dopamine, 0.0, 1.0))
        self.tonic_da = float(np.clip(self.tonic_da, 0.0, 1.0))
        self.acetylcholine = float(np.clip(self.acetylcholine, 0.0, 1.0))
        self.noradrenaline = float(np.clip(self.noradrenaline, 0.0, 1.0))
        self.serotonin = float(np.clip(self.serotonin, 0.0, 1.0))

    def reset(self) -> None:
        """Restore baselines and clear histories."""
        cfg = self.config
        self.dopamine = cfg.baseline_da
        self.tonic_da = cfg.baseline_tonic_da
        self.acetylcholine = cfg.baseline_ach
        self.noradrenaline = cfg.baseline_ne
        self.serotonin = cfg.baseline_sero
        self._error_history.clear()
        self._td_history.clear()
        self._da_rms = 1.0
        self._tda_history.clear()
        self._stagnation_factor = 0.0
        # Reset per-region to baselines
        for name in self._region_names:
            self._ne_levels[name] = self.noradrenaline
            self._ach_levels[name] = self.acetylcholine

    def __repr__(self) -> str:
        return (
            f"NeuromodulatorSystem("
            f"DA={self.dopamine:.3f}, tDA={self.tonic_da:.3f}, "
            f"ACh={self.acetylcholine:.3f}, NE={self.noradrenaline:.3f}, "
            f"5-HT={self.serotonin:.3f})"
        )
