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

        # TD error magnitude history — behavioral stability signal for serotonin.
        # Biological basis (Doya 2002): dorsal raphe nucleus receives projections
        # from VTA and habenula, integrating reward prediction accuracy into the
        # serotonergic stability signal. High |TD| = critic surprised = unstable.
        self._td_history: deque[float] = deque(maxlen=100)

        # ── Stagnation detection (ACC/mPFC learning monitor) ──────────
        # Biological basis (Kolling et al. 2016; Shenhav et al. 2013):
        # Anterior cingulate cortex (ACC) tracks the rate of reward
        # improvement. When learning stagnates (no improvement despite
        # continued effort), ACC signals orbitofrontal cortex to increase
        # exploration and maintain synaptic plasticity.
        #
        # Implementation: track rolling variance of tonic DA changes.
        # When tDA barely moves for many episodes → stagnation_factor→1.
        # This attenuates the consolidation gate, preventing premature
        # plasticity reduction that traps the agent at a suboptimal plateau.
        self._tda_history: deque[float] = deque(maxlen=30)
        self._stagnation_factor: float = 0.0  # 0=improving, 1=fully stagnated

        # ── Intrinsic progress tracking (Lisman & Grace 2005) ─────────
        # VTA receives inputs from hippocampus/PFC about learning progress.
        # When extrinsic reward is uninformative, world model prediction
        # error improvement drives tonic DA, enabling gradual exploration
        # reduction as the agent masters environmental dynamics.
        self._episode_pred_errors: deque[float] = deque(maxlen=30)

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

        # ── Serotonin: combined prediction stability (dorsal raphe) ─────
        # Biological basis (Doya 2002; Nakamura et al. 2008):
        # Dorsal raphe nucleus integrates BOTH sensory prediction accuracy
        # (via cortical afferents) AND reward prediction accuracy (via VTA/
        # habenula). Serotonin should only be high when the agent genuinely
        # understands its environment AND its behavioral outcomes are predictable.
        #
        # World model stability alone caused premature consolidation:
        # agent could perfectly predict valley physics while having no idea
        # how to reach the goal. Combined signal prevents this.
        avg_error = float(np.mean(self._error_history)) if self._error_history else 0.5
        world_stability = float(np.clip(1.0 - avg_error, 0.0, 1.0))

        # Behavioral stability: how predictable are reward outcomes?
        # 1/(1+|δ|) is self-normalizing: works regardless of reward scale.
        self._td_history.append(float(np.clip(abs(td_error), 0.0, 10.0)))
        avg_td_mag = float(np.mean(self._td_history)) if self._td_history else 5.0
        behavioral_stability = float(1.0 / (1.0 + avg_td_mag))

        # Geometric mean: BOTH must be stable for consolidation.
        stability = float(np.sqrt(world_stability * behavioral_stability))
        self.serotonin = (
            self.serotonin * self.config.sero_decay
            + stability * (1.0 - self.config.sero_decay)
        )

        self._clamp_all()

    def update_tonic_da(self, episode_return: float, episode_steps: int,
                         prediction_error_avg: float = 0.0) -> None:
        """Update tonic DA at episode boundaries based on dynamic range adaptation.

        Biologiczna podstawa (Niv, Daw, Joel & Dayan 2007; Tobler et al. 2005):
        Toniczna aktywność neuronów VTA koduje długoterminowe tempo nagrody
        (average reward rate) i determinuje próg konsolidacji wyuczonych zachowań.
        Neurony adaptują swój zakres odpowiedzi do historycznych minimów i maksimów
        (dynamic range adaptation).

        Zamiast z-score (który zawodzi przy zerowej wariancji w wyuczonym zadaniu),
        mapujemy obecny wynik na ułamek historycznego okna [min, max].
        Dzięki temu stałe osiąganie historycznego maksimum (np. powtarzalne 500 pkt)
        generuje sygnał 1.0, utrzymując stan wysokiej konsolidacji i blokując
        katastroficzne zapominanie.
        """
        self._reward_history.append(episode_return)

        if len(self._reward_history) < 2:
            return

        min_r = float(min(self._reward_history))
        max_r = float(max(self._reward_history))

        # Zabezpieczenie przed brakiem zróżnicowania nagrody.
        # Biologiczna interpretacja (Tobler et al. 2005): when all experienced
        # rewards are identical, the VTA has no basis to distinguish good
        # from bad policy — its phasic response is zero and tonic firing
        # reflects absence of reward prediction signal.  signal=0.0 keeps
        # tonic DA at its baseline (0.0), maintaining full exploration.
        # The 0.5 fallback was causing premature exploration reduction in
        # sparse-reward environments where all episodes return the same
        # score until the first success.
        if max_r - min_r < 1e-6:
            reward_signal = 0.0
        else:
            # Liniowe mapowanie obecnego wyniku do przedziału [0.0, 1.0]
            reward_signal = (episode_return - min_r) / (max_r - min_r)

        # ── Intrinsic progress signal (Lisman & Grace 2005) ───────────
        # Track per-episode average prediction error and compare recent
        # episodes to older ones.  Decreasing error = learning progress.
        self._episode_pred_errors.append(prediction_error_avg)
        intrinsic_signal = 0.0
        if len(self._episode_pred_errors) >= 10:
            err_arr = np.array(self._episode_pred_errors)
            midpoint = len(err_arr) // 2
            avg_older = float(np.mean(err_arr[:midpoint]))
            avg_recent = float(np.mean(err_arr[midpoint:]))
            if avg_older > 1e-8:
                intrinsic_signal = float(
                    np.clip((avg_older - avg_recent) / avg_older, 0.0, 1.0)
                )

        signal = max(reward_signal, intrinsic_signal)

        # Asymmetric VTA adaptation (Koob & Le Moal 2001; Volkow et al. 2017):
        # Rising: standard adaptation — quickly recognize improvement.
        # Falling: D2 auto-receptor desensitization and VTA afferent
        #   plasticity create hysteresis — consolidated tonic DA resists
        #   decline.  Time constant ~3× longer for decrease than increase.
        #   This prevents a single bad episode from cascading into
        #   catastrophic forgetting by reopening plasticity prematurely.
        if float(signal) >= self.tonic_da:
            decay = self.config.tonic_da_decay                               # rise: tc ≈ 10 ep
        else:
            decay = 1.0 - (1.0 - self.config.tonic_da_decay) / 3.0          # fall: tc ≈ 30 ep

        self.tonic_da = (
            self.tonic_da * decay
            + float(signal) * (1.0 - decay)
        )
        self.tonic_da = float(np.clip(self.tonic_da, 0.0, 1.0))

        # ── Stagnation tracking (ACC learning-rate monitor) ───────────
        # Record tonic DA history and compute improvement rate.
        # If tDA has been flat (low variance) for many episodes,
        # the stagnation factor rises, attenuating the consolidation gate.
        self._tda_history.append(self.tonic_da)
        if len(self._tda_history) >= 10:
            tda_arr = np.array(self._tda_history)
            # Rate of change: std of recent tDA values
            tda_variability = float(np.std(tda_arr))
            # Map variability to stagnation: low variability → high stagnation
            # Threshold: tda_std < 0.02 means no meaningful improvement.
            # Smoothed with EMA to avoid jitter.
            raw_stagnation = float(np.clip(1.0 - tda_variability / 0.05, 0.0, 1.0))
            self._stagnation_factor = (
                0.9 * self._stagnation_factor + 0.1 * raw_stagnation
            )

    # ------------------------------------------------------------------
    # Properties — typed read-outs for other modules
    # ------------------------------------------------------------------

    @property
    def learning_rate_modulation(self) -> float:
        """Phasic dopamine level: scales the STDP learning rate (m_t in update_weights)."""
        return self.dopamine

    @property
    def consolidation_gate(self) -> float:
        """Gate for plasticity reduction based on tonic DA, serotonin, and learning progress.

        Biological basis (Niv et al. 2007; Doya 2002; Kolling et al. 2016):
        Consolidation requires BOTH:
        - Tonic DA high → agent is consistently rewarded (VTA background)
        - Serotonin high → predictions are temporally stable (dorsal raphe)

        ACC modulation (Kolling et al. 2016; Shenhav et al. 2013):
        When learning has stagnated AND tonic DA is in the ambiguous
        middle range (0.3–0.7), ACC signals that the current policy may
        be suboptimal and attenuates the consolidation gate. This prevents
        the agent from getting trapped at a local optimum.

        Crucially, if tDA > 0.7 (strong evidence of success), stagnation
        attenuation is NOT applied — the agent genuinely needs consolidation
        to protect its successful policy. The attenuator is targeted
        specifically at the "mediocre plateau" regime.
        """
        raw_gate = float(np.sqrt(self.tonic_da * self.serotonin))
        # ACC attenuation only in the "stuck in middle" regime.
        # High tDA (>0.7) = genuine success → full consolidation.
        # Low tDA (<0.3) = early learning → gate is already low.
        # Middle tDA (0.3–0.7) with stagnation → attenuate gate.
        if 0.3 < self.tonic_da < 0.7:
            acc_attenuation = 1.0 - 0.5 * self._stagnation_factor
        else:
            acc_attenuation = 1.0
        return raw_gate * acc_attenuation

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
        self._td_history.clear()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NeuromodulatorSystem("
            f"DA={self.dopamine:.3f}, "
            f"tDA={self.tonic_da:.3f}, "
            f"ACh={self.acetylcholine:.3f}, "
            f"NE={self.noradrenaline:.3f}, "
            f"5-HT={self.serotonin:.3f})"
        )