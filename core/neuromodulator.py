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
from .receptor import hill_response, RECEPTOR_PARAMS, ReceptorType


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
        self._da_rms: float = cfg.baseline_da

        # ── ACC error neuron persistence trace (Behrens et al. 2007) ──
        # Leaky integrator of PE magnitude. Persistent elevated PE →
        # high stagnation signal → attenuate consolidation.
        self._acc_pe_trace: float = 0.0
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
        reward: float = 0.0,
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
        # Weber-Fechner adaptive coding (Tobler et al. 2005):
        # gain = baseline_da / max(da_rms, baseline_da)
        # At da_rms=1: gain=0.5 → coding range [0, 1].
        # At da_rms→0: gain=1.0 → maximum sensitivity.
        da_gain = cfg.baseline_da / max(self._da_rms, cfg.baseline_da)
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
        # Hill equation for DRN 5-HT response to tonic DA
        # Using RECEPTOR_PARAMS for HT1A (ec50=0.4, hill_n=1.0)
        # to maintain consistency with pharmacological database.
        _ht1a = RECEPTOR_PARAMS[ReceptorType.HT1A]
        reward_quality = float(hill_response(
            self.tonic_da, _ht1a.ec50, _ht1a.hill_n,
        ))
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

        # ── Tonic DA: average reward rate (Niv et al. 2007; Grace 1991) ─
        # Tonic VTA DA firing rate tracks the average reward rate in
        # the environment, NOT |RPE| (which converges to 0 as V → V*).
        # Rich environment → high tonic DA → D1 bias → exploitation.
        # Poor/unknown environment → low tonic DA → D2 bias → caution.
        #
        # Raw reward is transformed through the DA neuron f–I curve
        # (logistic with unit gain; Dreyer et al. 2010 Fig 2).  The
        # logistic σ(r) = 1/(1+exp(−r)) is parameter-free and maps
        # reward in natural units to bounded firing rate (0, 1).
        # DA neurons fire at 2–8 Hz baseline with bursts to ~20 Hz
        # (Grace 1991), i.e. ~10× dynamic range, consistent with
        # σ mapping ±3 units to 5–95% of output.
        # τ_tonic ≈ 60 s (Grace 1991): minute-scale integration.
        _r_clipped = float(np.clip(reward, -20.0, 20.0))  # overflow guard
        reward_signal = float(1.0 / (1.0 + np.exp(-_r_clipped)))
        self.tonic_da = (
            self.tonic_da * cfg.tonic_da_decay
            + reward_signal * (1.0 - cfg.tonic_da_decay)
        )

        # ── ACC error neuron persistence (Behrens et al. 2007) ────────
        # Leaky integrator over PE magnitude. High sustained PE = agent
        # not learning = stagnation. τ from config (30s default).
        self._acc_pe_trace = (
            cfg.acc_pe_decay * self._acc_pe_trace
            + (1.0 - cfg.acc_pe_decay) * error_mag
        )
        self._stagnation_factor = float(np.clip(self._acc_pe_trace, 0.0, 1.0))

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
        """tonic_da × serotonin with ACC stagnation attenuation.

        Biological: consolidation requires both sustained performance
        (tonic DA) AND predictable environment (5-HT).  Multiplicative
        gating is the correct conjunction (Doya 2002).
        """
        raw = float(self.tonic_da * self.serotonin)
        # ACC stagnation only in exploitation band. Below 0.3: agent
        # already exploring (low DA). Above 0.7: agent performing well
        # (high DA). Range from D1/D2 balance crossover (Frank 2005).
        if 0.3 < self.tonic_da < 0.7:
            # ACC stagnation attenuates consolidation multiplicatively.
            # stagnation_factor ∈ [0,1] directly scales consolidation.
            acc = 1.0 - self._stagnation_factor
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
            # NE: LC regional gain follows Weber law (Berridge & Waterhouse 2003):
            # NE_region = NE_global × (1 + local_pe / (1 + local_pe))
            # Proportional boost scaled by local surprise without arbitrary coefficient.
            pe_boost = local_pe / (1.0 + abs(local_pe))
            self._ne_levels[name] = float(np.clip(
                self.noradrenaline * (1.0 + pe_boost), 0.0, 1.0,
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
        self._da_rms = cfg.baseline_da
        self._acc_pe_trace = 0.0
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
