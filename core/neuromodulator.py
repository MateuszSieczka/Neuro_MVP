from collections import deque

import numpy as np

from .config import NeuromodulatorConfig


class NeuromodulatorSystem:
    """
    Four-channel neuromodulatory system modelling the brain's "global orchestra conductor".

    Each channel is a first-order low-pass filter (exponential decay + phasic drive):

      level[t+1] = level[t] * decay + signal[t] * (1 - decay)

    ┌────────────────┬────────────────────────────────────────────────────────┐
    │ Modulator      │ Role                                                   │
    ├────────────────┼────────────────────────────────────────────────────────┤
    │ Dopamine (DA)  │ Phasic: RPE (reward prediction error) per step.        │
    │                │ Tonic: average reward rate (VTA background firing).     │
    │ Acetylcholine  │ Novelty / uncertainty → bottom-up vs top-down balance  │
    │ Noradrenaline  │ Surprise / arousal → sharpens k-WTA competition        │
    │ Serotonin      │ Temporal stability → longer vs shorter planning horizon│
    └────────────────┴────────────────────────────────────────────────────────┘

    Dopamine dual-mode (Grace 1991; Niv, Daw, Joel & Dayan 2007):
    - Phasic DA: burst firing on positive RPE, pause on negative RPE.
      Drives per-step synaptic plasticity (three-factor Hebbian rule).
    - Tonic DA: sustained VTA firing tracks average reward rate.
      High tonic DA → agent is consistently rewarded → consolidation.
      Low tonic DA → agent is struggling → full plasticity, exploration.
      The tonic signal is updated from raw reward per step, not TD error,
      because it represents ventral striatum's running average of
      experienced reward, not prediction error.

    All levels are normalised to [0, 1] at every step.
    """

    def __init__(self, config: NeuromodulatorConfig | None = None) -> None:
        self.config = config or NeuromodulatorConfig()

        # Current modulator levels (normalised 0–1)
        self.dopamine: float = self.config.baseline_da
        self.acetylcholine: float = self.config.baseline_ach
        self.noradrenaline: float = self.config.baseline_ne
        self.serotonin: float = self.config.baseline_sero

        # Tonic DA: slow VTA background firing tracking average reward rate
        self.tonic_da: float = self.config.baseline_tonic_da

        # Rolling histories for temporal averaging
        self._error_history: deque[float] = deque(maxlen=100)
        self._reward_history: deque[float] = deque(maxlen=100)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        prediction_error: np.ndarray,
        td_error: float = 0.0,
        novelty: float | None = None,
    ) -> None:
        """
        Update per-step neuromodulator levels based on incoming signals.

        Parameters
        ----------
        prediction_error : array
            Per-step prediction error from world model or normalized |TD|.
        td_error : float
            Raw TD error for phasic dopamine computation.
        novelty : float or None
            Novelty signal for acetylcholine. If None, uses error_magnitude.
        """
        error_magnitude = float(np.clip(np.mean(np.abs(prediction_error)), 0.0, 1.0))

        # Serotonina śledzi stabilność w czasie na podstawie ZNORMALIZOWANEGO błędu.
        # Przycinamy do [0,1] tutaj, aby historia była porównywalna niezależnie od skali środowiska.
        self._error_history.append(error_magnitude)

        if novelty is None:
            novelty = float(np.clip(error_magnitude, 0.0, 1.0))

        # ── Phasic Dopamine: RPE via logistic sigmoid ─────────────────
        # Logistic sigmoid: td=0 → 0.5 (neutral), td>0 → >0.5, td<0 → <0.5.
        rpe_signal = float(1.0 / (1.0 + np.exp(-td_error)))

        self.dopamine = (
                self.dopamine * self.config.da_decay
                + rpe_signal * (1.0 - self.config.da_decay)
        )

        # Tonic DA is NOT updated here — it's updated episodically
        # via update_tonic_da() called at episode boundaries.

        # ── Acetylcholine: novelty / uncertainty ──────────────────────
        self.acetylcholine = (
            self.acetylcholine * self.config.ach_decay
            + float(np.clip(novelty, 0.0, 1.0)) * (1.0 - self.config.ach_decay)
        )

        # ── Noradrenaline: global surprise (raw error magnitude) ───────
        self.noradrenaline = (
            self.noradrenaline * self.config.ne_decay
            + float(np.clip(error_magnitude, 0.0, 1.0)) * (1.0 - self.config.ne_decay)
        )

        # ── Serotonin: prediction accuracy (dorsal raphe) ─────────────
        # Tracks the mean absolute prediction error over a 100-step window.
        # High serotonin = accurate predictions = stable environment model.
        # This is ONE component of consolidation — the other is tonic DA
        # (are rewards actually good?). Both are needed.
        avg_error = float(np.mean(self._error_history)) if self._error_history else 0.5
        stability = float(np.clip(1.0 - avg_error, 0.0, 1.0))
        self.serotonin = (
            self.serotonin * self.config.sero_decay
            + stability * (1.0 - self.config.sero_decay)
        )

        self._clamp_all()

    def update_tonic_da(self, episode_return: float, episode_steps: int) -> None:
        """Update tonic DA at episode boundaries from return z-score.

        Biological basis (Niv, Daw, Joel & Dayan 2007):
        VTA tonic firing rate tracks how well the agent is performing
        relative to its own recent history. This is scale-invariant.

        - Return >> recent mean → z>0 → sigmoid>0.5 → tonic_da rises
        - Return << recent mean → z<0 → sigmoid<0.5 → tonic_da falls
        - Return ≈ mean → z≈0 → sigmoid≈0.5 → tonic_da drifts to 0.5

        When variance is very low (constant returns), the z-score is
        unreliable. We handle this by clamping std to a minimum of 1.0,
        which means small deviations from the mean produce small z-scores,
        keeping tonic_da near 0.5 (neutral). This correctly represents
        "no information about improvement" regardless of absolute level.
        """
        self._reward_history.append(episode_return)

        if len(self._reward_history) < 2:
            return

        mean_r = float(np.mean(self._reward_history))
        std_r = float(np.std(self._reward_history))
        # Clamp std to minimum 1.0 — not 1e-6.
        # When variance is near zero, small fluctuations shouldn't produce
        # extreme z-scores. A std floor of 1.0 means the agent needs to
        # improve by at least ~1 unit of return to register as positive.
        std_r = max(std_r, 1.0)

        z = (episode_return - mean_r) / std_r
        signal = float(1.0 / (1.0 + np.exp(-z)))

        self.tonic_da = (
            self.tonic_da * self.config.tonic_da_decay
            + signal * (1.0 - self.config.tonic_da_decay)
        )
        self.tonic_da = float(np.clip(self.tonic_da, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Properties — typed read-outs for other modules
    # ------------------------------------------------------------------

    @property
    def learning_rate_modulation(self) -> float:
        """Phasic dopamine level: scales the STDP learning rate (m_t in update_weights)."""
        return self.dopamine

    @property
    def consolidation_gate(self) -> float:
        """Gate for plasticity reduction based on tonic DA and serotonin.

        Biological basis (Niv et al. 2007; Doya 2002):
        Consolidation requires BOTH:
        - Tonic DA high → agent is consistently rewarded (VTA background)
        - Serotonin high → predictions are temporally stable (dorsal raphe)

        The geometric mean requires both signals to contribute:
        if either is low, the gate stays open (full plasticity).

        Critical: tonic DA at 0.5 is neutral (no evidence of improvement
        or decline). Only values above ~0.5 indicate genuine success.
        The geometric mean naturally handles this: sqrt(0.5 × 0.9) = 0.67,
        which gives moderate consolidation. Only when both are truly high
        (tda=0.8, sero=0.9 → gate=0.85) does strong consolidation occur.
        """
        return float(np.sqrt(self.tonic_da * self.serotonin))

    @property
    def bottom_up_gain(self) -> float:
        """
        Acetylcholine level: controls bottom-up vs top-down balance in
        PredictiveCodingLayer.  Pass to layer.set_ach_level().
        """
        return self.acetylcholine

    @property
    def competition_sharpness(self) -> float:
        """
        Noradrenaline level: can be used to dynamically tighten k-WTA
        (higher NE → fewer effective winners → sparser SDR).
        """
        return self.noradrenaline

    @property
    def planning_horizon(self) -> float:
        """
        Serotonin level: proxy for temporal discount factor γ.
        High 5-HT → long-horizon planning; Low 5-HT → myopic, reactive.
        """
        return self.serotonin

    @property
    def tau_compression(self) -> float:
        """
        Noradrenaline-driven compression signal for eligibility trace timescales.

        Pass directly to layer.set_plasticity_timescales(ne=...).
        High NE → tau_e compressed → old correlations fade faster →
        quicker adaptation when environment changes (phase B of a bandit task).
        Low NE → full tau_e → consolidation of familiar patterns.
        """
        return self.noradrenaline

    @property
    def membrane_reactivity(self) -> float:
        """
        Acetylcholine-driven compression of membrane time constant.

        Pass directly to layer.set_plasticity_timescales(ach=...).
        High ACh → tau_m compressed → faster membrane integration →
        stronger bottom-up signal influence vs. accumulated prediction.
        Low ACh → slow tau_m → top-down predictions dominate.
        """
        return self.acetylcholine

    def apply_to_layer(self, layer) -> None:
        """
        Convenience: propagate both NE and ACh to any layer that
        supports set_plasticity_timescales().

        Zamknięta pętla w jednym wywołaniu:
          neuromodulator.update(...) → neuromodulator.apply_to_layer(layer)
        Wystarczy umieścić to wywołanie w głównej pętli agenta po każdym kroku.
        """
        if hasattr(layer, 'set_plasticity_timescales'):
            layer.set_plasticity_timescales(
                ne=self.noradrenaline,
                ach=self.acetylcholine,
            )
        # Dark matter recruitment (NE threshold drop) — preserved as before
        if hasattr(layer, 'set_ne_level'):
            layer.set_ne_level(self.noradrenaline)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clamp_all(self) -> None:
        """Ensure all levels remain in [0, 1] after every update."""
        self.dopamine = float(np.clip(self.dopamine, 0.0, 1.0))
        self.tonic_da = float(np.clip(self.tonic_da, 0.0, 1.0))
        self.acetylcholine = float(np.clip(self.acetylcholine, 0.0, 1.0))
        self.noradrenaline = float(np.clip(self.noradrenaline, 0.0, 1.0))
        self.serotonin = float(np.clip(self.serotonin, 0.0, 1.0))

    def reset(self) -> None:
        """Restore all levels to configured baselines and clear histories."""
        self.dopamine = self.config.baseline_da
        self.tonic_da = self.config.baseline_tonic_da
        self.acetylcholine = self.config.baseline_ach
        self.noradrenaline = self.config.baseline_ne
        self.serotonin = self.config.baseline_sero
        self._error_history.clear()
        self._reward_history.clear()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NeuromodulatorSystem("
            f"DA={self.dopamine:.3f}, "
            f"tDA={self.tonic_da:.3f}, "
            f"ACh={self.acetylcholine:.3f}, "
            f"NE={self.noradrenaline:.3f}, "
            f"5-HT={self.serotonin:.3f})"
        )