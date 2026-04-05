from collections import deque

import numpy as np

from config import NeuromodulatorConfig


class NeuromodulatorSystem:
    """
    Four-channel neuromodulatory system modelling the brain's "global orchestra conductor".

    Each channel is a first-order low-pass filter (exponential decay + phasic drive):

      level[t+1] = level[t] * decay + signal[t] * (1 - decay)

    ┌────────────────┬────────────────────────────────────────────────────────┐
    │ Modulator      │ Role                                                   │
    ├────────────────┼────────────────────────────────────────────────────────┤
    │ Dopamine (DA)  │ Reward prediction error → scales STDP learning rate    │
    │ Acetylcholine  │ Novelty / uncertainty → bottom-up vs top-down balance  │
    │ Noradrenaline  │ Surprise / arousal → sharpens k-WTA competition        │
    │ Serotonin      │ Temporal stability → longer vs shorter planning horizon│
    └────────────────┴────────────────────────────────────────────────────────┘

    All levels are normalised to [0, 1] at every step.
    """

    def __init__(self, config: NeuromodulatorConfig | None = None) -> None:
        self.config = config or NeuromodulatorConfig()

        # Current modulator levels (normalised 0–1)
        self.dopamine: float = self.config.baseline_da
        self.acetylcholine: float = self.config.baseline_ach
        self.noradrenaline: float = self.config.baseline_ne
        self.serotonin: float = self.config.baseline_sero

        # Rolling histories for temporal averaging
        self._error_history: deque[float] = deque(maxlen=100)
        self._reward_history: deque[float] = deque(maxlen=100)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        prediction_error: np.ndarray,
        reward: float = 0.0,
        novelty: float | None = None,
    ) -> None:
        """
        Update all neuromodulator levels based on incoming signals.

        Args:
            prediction_error: Signed error vector from a PredictiveCodingLayer
                              (or any layer). Shape: (n,).
            reward:           External reward signal from the environment (any scale;
                              internally the *change* relative to history is used).
            novelty:          Optional explicit novelty signal in [0, 1].
                              When None, inferred as clipped mean |error|.
        """
        error_magnitude = float(np.mean(np.abs(prediction_error)))

        if novelty is None:
            novelty = float(np.clip(error_magnitude, 0.0, 1.0))

        # Reward prediction error: how much better/worse than expected?
        avg_reward = float(np.mean(self._reward_history)) if self._reward_history else 0.0
        rpe = float(np.clip(reward - avg_reward + 0.5, 0.0, 1.0))  # shift to [0,1]

        self._error_history.append(error_magnitude)
        self._reward_history.append(reward)

        # ── Dopamine: phasic reward prediction error ──────────────────
        self.dopamine = (
            self.dopamine * self.config.da_decay
            + rpe * (1.0 - self.config.da_decay)
        )

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

        # ── Serotonin: temporal stability (inversely tied to mean error)─
        avg_error = float(np.mean(self._error_history)) if self._error_history else 0.5
        stability = float(np.clip(1.0 - avg_error, 0.0, 1.0))
        self.serotonin = (
            self.serotonin * self.config.sero_decay
            + stability * (1.0 - self.config.sero_decay)
        )

        self._clamp_all()

    # ------------------------------------------------------------------
    # Properties — typed read-outs for other modules
    # ------------------------------------------------------------------

    @property
    def learning_rate_modulation(self) -> float:
        """Dopamine level: scales the STDP learning rate (m_t in update_weights)."""
        return self.dopamine

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clamp_all(self) -> None:
        """Ensure all levels remain in [0, 1] after every update."""
        self.dopamine = float(np.clip(self.dopamine, 0.0, 1.0))
        self.acetylcholine = float(np.clip(self.acetylcholine, 0.0, 1.0))
        self.noradrenaline = float(np.clip(self.noradrenaline, 0.0, 1.0))
        self.serotonin = float(np.clip(self.serotonin, 0.0, 1.0))

    def reset(self) -> None:
        """Restore all levels to configured baselines and clear histories."""
        self.dopamine = self.config.baseline_da
        self.acetylcholine = self.config.baseline_ach
        self.noradrenaline = self.config.baseline_ne
        self.serotonin = self.config.baseline_sero
        self._error_history.clear()
        self._reward_history.clear()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NeuromodulatorSystem("
            f"DA={self.dopamine:.3f}, "
            f"ACh={self.acetylcholine:.3f}, "
            f"NE={self.noradrenaline:.3f}, "
            f"5-HT={self.serotonin:.3f})"
        )