import numpy as np

from .config import PyramidalConfig
from .competitive_layer import CompetitiveLIFLayer


class PyramidalLayer(CompetitiveLIFLayer):
    """
    Multi-compartment pyramidal neuron layer with burst-dependent plasticity.

    Architecture
    ============
    Three functional compartments per neuron:

      Basal dendrites  ←  feedforward / bottom-up spikes  (w, inherited)
      Apical dendrites ←  top-down context predictions     (w_apical)
      Soma             ←  integrates both; generates spikes

    Key equations
    -------------
    Apical integration (passive, slow):
      v_apical[t+1] = v_apical[t] × exp(−dt/tau_apical)
                    + (top_down_prediction @ w_apical) × (1 − exp(−dt/tau_apical))

    Apical priming:
      apical_priming = (v_apical > apical_threshold)

    Effective somatic threshold (ACh-modulated):
      v_thresh_eff = v_thresh_adaptive
                   − apical_boost × apical_priming × ach_apical_scale

    Burst detection:
      is_burst = has_spiked AND apical_priming

    Burst STDP boost:
      e[:, burst_neurons] *= burst_stdp_factor

    Homeostatic adaptation (integrated — no parent update):
      avg_rate = avg_rate × α + has_spiked × (1 − α)
      v_thresh_adaptive += thresh_adapt_lr × (avg_rate − target_rate)

    Biological references
    ---------------------
    - Larkum, Zhu & Sachmann (1999): BAC firing via apical calcium spikes.
    - Sjöström & Häusser (2006): Dendritic coincidence detection.
    - Payeur et al. (2021): Burst-dependent synaptic plasticity.
    - Sacramento et al. (2018): Dendritic error learning.

    Top-down pathway
    ----------------
    Higher layers call receive_prediction(p) where p ∈ [0,1]^num_inputs.
    This is buffered as top_down_prediction and drives v_apical in the
    NEXT forward() call, matching the one-timestep signalling delay in
    the NetworkGraph feedback pass.

    Imagination / hallucination
    ---------------------------
    When no bottom-up input is present (silence) but top-down is strong
    (apical_boost ≫ 0) and apical_threshold is crossed, the effective
    threshold can be reduced enough to allow spontaneous soma firing — the
    equivalent of "imagining" a percept driven purely by predictions.
    This gate is partly controlled by ach_apical_scale: high ACh → small
    scale (trust bottom-up) → less imagination; low ACh → large scale
    → predictions dominate → hallucination / dreaming mode.

    Weight shapes
    -------------
    w        : (num_inputs, num_neurons)  — inherited basal feedforward weights
    w_apical : (num_inputs, num_neurons)  — apical context weights
               transpose: w_apical.T has shape (num_neurons, num_inputs)
               → generate_prediction() projects spikes → input space (tied weights).
    """

    def __init__(
        self,
        num_inputs: int,
        num_neurons: int = 20,
        config: PyramidalConfig | None = None,
    ) -> None:
        self.pyr_config = config or PyramidalConfig()

        # Disable parent's homeostatic update — we run our own combined
        # (homeostatic + apical modulation) version in _update_adaptive_threshold.
        super().__init__(num_inputs, num_neurons, self.pyr_config)
        self._homeostatic = False      # suppress LIFLayer._update_homeostatic()
        self._homeostatic_kwta = False  # POPRAWKA Bug D: suppress CompetitiveLIFLayer._update_kwta_homeostasis()
        # Bez tego v_thresh_adaptive jest modyfikowane z dwóch miejsc naraz:
        # 1) _update_adaptive_threshold() co krok (lokalny LR)
        # 2) _update_kwta_homeostasis() przy każdym resecie fazy (okno * LR)
        # Efekt: destabilizacja progu i schizofreniczne zachowanie homeostazy.

        # ── Apical compartment weights ────────────────────────────────
        # Tied to feedback direction: w_apical.T maps spikes → input space,
        # matching generate_prediction() output dimensionality.
        self.w_apical: np.ndarray = np.random.uniform(
            0.0, 0.1, (num_inputs, num_neurons)
        ).astype(np.float32)


        # ── Apical membrane state ─────────────────────────────────────
        self.v_apical: np.ndarray = np.zeros(num_neurons, dtype=np.float32)
        self._apical_decay: float = np.exp(
            -self.pyr_config.dt / self.pyr_config.tau_apical
        )

        # Top-down prediction received via receive_prediction()
        self.top_down_prediction: np.ndarray = np.zeros(num_neurons, dtype=np.float32)

        # ── Burst state ───────────────────────────────────────────────
        self.is_burst: np.ndarray = np.zeros(num_neurons, dtype=bool)

        # ── Homeostatic adaptive threshold (managed internally) ───────
        self.v_thresh_adaptive: np.ndarray = np.full(
            num_neurons, self.pyr_config.v_thresh, dtype=np.float32
        )
        self.avg_rate: np.ndarray = np.zeros(num_neurons, dtype=np.float32)
        self._homeo_decay: float = np.exp(
            -self.pyr_config.dt / self.pyr_config.homeostatic_tau
        )

        # ACh scaling: modulates how strongly apical boost applies.
        # Set via set_ach_level() by NeuromodulatorSystem / NetworkGraph.
        # High ACh → bottom-up trust → smaller apical influence.
        self._ach_apical_scale: float = 1.0

        # Prediction error exposed for NetworkGraph.update_weights()
        self.prediction_error: np.ndarray = np.zeros(num_inputs, dtype=np.float32)

        self.plateau_timer: np.ndarray = np.zeros(num_neurons, dtype=np.int32)
        self.in_plateau: np.ndarray = np.zeros(num_neurons, dtype=bool)


    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        pre_f32 = pre_spikes.astype(np.float32)

        # POPRAWKA Bug 1: Aktualizacja śladów STDP na początku kroku.
        # PyramidalLayer nie wywołuje super().forward(), więc LIFLayer nigdy nie aktualizuje
        # x_pre, x_post ani e. Bez tego self.e = 0 zawsze → update_weights() nic nie robi.
        self.x_pre *= self._pre_decay
        self.x_post *= self._post_decay
        pre_active = pre_spikes > 0
        self.x_pre[pre_active] += 1.0

        # ── 1. Integracja apikalna ─────────────────────────────────────
        apical_current = self.top_down_prediction.astype(np.float32)

        self.v_apical = (
                self.v_apical * self._apical_decay
                + apical_current * (1.0 - self._apical_decay)
        )

        # ── 2. Wyzwalanie i utrzymanie Calcium Plateau ──────────────────
        new_plateaus = (self.v_apical > self.pyr_config.apical_threshold) & (self.plateau_timer == 0)
        self.plateau_timer[new_plateaus] = getattr(self.pyr_config, 'plateau_duration_ms', 50)
        self.in_plateau = self.plateau_timer > 0
        self.plateau_timer[self.in_plateau] -= 1

        # ── 3. Relaksacja Bogacza (Faza Ingerencji) ────────────────────
        # POPRAWKA Bug A: Obliczamy drive feedforward z basal weights (self.w)
        # przed pętlą relaksacji. Pełni rolę "wejścia bazalnego" (thalamic drive),
        # uzgadnianego z predykcją apikalną (w_apical). Bez tego self.w było uczone
        # przez STDP, ale NIGDY nie wpływało na dynamikę v — sieć była "ślepa" na
        # wyuczone wzorce feedforward.
        ff_drive = pre_f32 @ self.w  # (num_neurons,) — stały przez całą pętlę

        # POPRAWKA Bug C: Proaktywna inhibicja k-WTA przed relaksacją i detekcją spike'ów
        self._apply_proactive_inhibition()

        relaxation_steps = getattr(self.pyr_config, 'relaxation_steps', 10)
        relaxation_rate = getattr(self.pyr_config, 'relaxation_rate', 0.1)

        for _ in range(relaxation_steps):
            r = np.clip((self.v - getattr(self.pyr_config, 'v_rest', -70.0)) / (
                        getattr(self.pyr_config, 'v_thresh', -55.0) - getattr(self.pyr_config, 'v_rest', -70.0)), 0.0,
                        1.0)
            my_prediction = r @ self.w_apical.T

            self.prediction_error = pre_f32 - my_prediction
            error_gradient = self.prediction_error @ self.w_apical

            # POPRAWKA Bug A: Gradient łączony = PC error + apical top-down + basal ff_drive
            combined_gradient = error_gradient + ff_drive

            self.v += relaxation_rate * combined_gradient
            np.clip(self.v, getattr(self.pyr_config, 'v_reset', -75.0),
                    getattr(self.pyr_config, 'v_thresh', -55.0) + 10.0, out=self.v)
        # ── 4. Nieliniowy efektywny próg ────────────────────────
        effective_thresh = (
                self.v_thresh_adaptive
                - self.pyr_config.apical_boost
                * self.in_plateau.astype(np.float32)
                * self._ach_apical_scale
        )

        if getattr(self.pyr_config, 'background_noise_std', 0.0) > 0:
            noise = np.random.normal(0.0, self.pyr_config.background_noise_std, self.num_neurons)
            self.v += noise.astype(np.float32)

        # ── 5. Instalacja progu i generacja spike'ów ──────────────────
        homeostatic_thresh = self.v_thresh_adaptive.copy()
        self.v_thresh_adaptive = effective_thresh

        # Omijamy integrację napięcia z rodzica (bo zrobiliśmy relaksację),
        # więc sami obsługujemy detekcję impulsów i ślady.
        in_refrac = self.refrac_count > 0
        self.refrac_count[in_refrac] -= 1

        self.has_spiked = (self.v >= self.v_thresh_adaptive) & ~in_refrac
        self.v[self.has_spiked] = getattr(self.pyr_config, 'v_reset', -75.0)
        self.refrac_count[self.has_spiked] = getattr(self.pyr_config, 'refrac_period', 2)

        self.x_post[self.has_spiked] += 1.0

        # POPRAWKA Bug 1 (cd.): Aktualizacja śladu kwalifikowalności.
        # Musi być tutaj, przed wzmocnieniem burst (krok 6), które mnożył będzie
        # już wypełniony ślad e zamiast zerowej macierzy.
        self.e *= self._trace_decay
        if np.any(self.has_spiked):
            self.e[:, self.has_spiked] += self.x_pre[:, np.newaxis]
        if np.any(pre_active):
            self.e[pre_active, :] += self.x_post[np.newaxis, :]

        # ZMODYFIKOWANE: Ręczna integracja k-WTA bez niszczenia self.v
        self.window_spike_counts += self.has_spiked.astype(np.int32)
        self._current_window_size += 1

        if getattr(self, '_phase_reset_pending', False):
            self._apply_lateral_inhibition()
            if getattr(self, '_homeostatic_kwta', False) and self._current_window_size > 0:
                self._update_kwta_homeostasis(self._current_window_size)
            self._reset_window()
        spikes = self.has_spiked.copy()

        self.v_thresh_adaptive = homeostatic_thresh

        # ── 6. Burst detection i STDP boost ──────────────────────────
        self.is_burst = self.has_spiked & self.in_plateau
        if np.any(self.is_burst):
            burst_mask = self.is_burst.astype(np.float32)
            boost = 1.0 + (self.pyr_config.burst_stdp_factor - 1.0) * burst_mask
            self.e *= boost[np.newaxis, :]

        # ── 7. Homeostatic adaptation ────────────────────────────────
        self._update_adaptive_threshold()

        return spikes

    # ------------------------------------------------------------------
    # Prediction interface (compatible with NetworkGraph)
    # ------------------------------------------------------------------

    def receive_prediction(self, prediction: np.ndarray) -> None:
        """
        Accept a top-down prediction from the layer above.
        Stored and used in the NEXT forward() call (apical integration).

        Args:
            prediction: Rate-coded prediction (num_inputs,), values in [0, 1].
        """
        self.top_down_prediction = prediction.astype(np.float32)

    # Override update_weights
    def update_weights(self, m_t: float, pred_error: np.ndarray) -> None:
        """
        Aktualizuje wagi feedforward oraz sprzężone wagi apikalne.
        Dendritic Error Learning jest teraz sterowane lokalną Wolną Energią, a nie w_backward.
        """
        # Trzy-czynnikowe STDP dla wag bazalnych (feedforward)
        super().update_weights(m_t, pred_error)

        # Uczenie wag apikalnych (Hebbian zależny od zrelaksowanego błędu i dopaminy BG)
        if np.any(self.has_spiked):
            positive_error = np.clip(self.prediction_error, 0.0, None)
            if np.any(positive_error > 0):
                # m_t (z Jąder Podstawnych) decyduje czy ta reprezentacja była "warta" zapamiętania
                dw_apical = self.pyr_config.apical_lr * np.outer(
                    positive_error, self.has_spiked.astype(np.float32)
                )
                self.w_apical += dw_apical * m_t
                np.clip(self.w_apical, 0.0, 1.0, out=self.w_apical)

    def generate_prediction(self) -> np.ndarray:
        """
        Generate a top-down prediction for the layer below.

        Uses tied weights: projects current spike pattern through w_apical.T,
        yielding a rate-coded prediction in the input space (num_inputs,).
        Tied weights enforce a generative/recognition symmetry consistent
        with the Helmholtz machine and predictive coding literature.

        Returns:
            Prediction vector (num_inputs,), values in [0, 1].
        """
        raw = self.has_spiked.astype(np.float32) @ self.w_apical.T
        return np.clip(raw * self.pyr_config.feedback_strength, 0.0, 1.0)

    def set_ach_level(self, ach: float) -> None:
        """
        Set acetylcholine level to modulate top-down vs. bottom-up balance.

        High ACh → strong bottom-up trust → apical boost is scaled DOWN.
        Low ACh  → internal predictions dominate → apical boost at full strength
                   → imagination / dreaming mode possible.

        ACh = 1.0 → _ach_apical_scale = 0.5  (apical boost halved)
        ACh = 0.0 → _ach_apical_scale = 1.0  (apical boost full)
        """
        self._ach_apical_scale = float(1.0 - 0.5 * np.clip(ach, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Internal: homeostatic threshold management
    # ------------------------------------------------------------------

    def _update_adaptive_threshold(self) -> None:
        """
        Combined homeostatic threshold adaptation.

        Updates v_thresh_adaptive based on the current firing rate error,
        independently from any apical modulation (which is applied transiently
        only during the forward pass and never stored in v_thresh_adaptive).
        """
        cfg = self.pyr_config

        self.avg_rate = (
            self.avg_rate * self._homeo_decay
            + self.has_spiked.astype(np.float32) * (1.0 - self._homeo_decay)
        )
        rate_error = self.avg_rate - cfg.target_rate
        self.v_thresh_adaptive += cfg.thresh_adapt_lr * rate_error
        np.clip(
            self.v_thresh_adaptive,
            cfg.thresh_min,
            cfg.thresh_max,
            out=self.v_thresh_adaptive,
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset transient state including apical compartment and burst flag."""
        super().reset_state()
        self.v_apical.fill(0.0)
        self.top_down_prediction.fill(0.0)
        self.is_burst.fill(False)
        self.prediction_error.fill(0.0)
        self.v_thresh_adaptive.fill(self.pyr_config.v_thresh)
        self.avg_rate.fill(0.0)