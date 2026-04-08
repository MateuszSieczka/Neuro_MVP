import numpy as np
from .config import PredictiveCodingConfig
from .competitive_layer import CompetitiveLIFLayer
from .spike_encoder import PoissonEncoder


class PredictiveCodingLayer(CompetitiveLIFLayer):
    """
    Extends CompetitiveLIFLayer with Predictive Coding mechanics (Friston, 2010).

    Each layer simultaneously:
      - Receives bottom-up sensory error signals from the layer below.
      - Receives top-down predictions from the layer above.
      - Computes a signed prediction error (actual − predicted).
      - Generates its own top-down prediction for the layer below via feedback_w.

    Acetylcholine (ACh) controls the bottom-up/top-down balance:
      - ACh → 1.0 : trust raw sensory input  (novel / uncertain environment)
      - ACh → 0.0 : trust internal predictions (familiar / confident state)

    Feedback weights are updated by a Hebbian rule: neurons that fire should
    predict the input patterns that caused them to fire.
    """

    def __init__(
        self,
        num_inputs: int,
        num_neurons: int = 20,
        config: PredictiveCodingConfig | None = None,
    ) -> None:
        self.pc_config = config or PredictiveCodingConfig()
        super().__init__(num_inputs, num_neurons, self.pc_config)

        self._encoder = PoissonEncoder()

        # Top-down feedback weights: this layer → layer below  (num_neurons × num_inputs)
        self.feedback_w: np.ndarray = np.random.uniform(
            0.0, 0.1, (num_neurons, num_inputs)
        ).astype(np.float32)

        relaxation_steps: int = 10
        relaxation_rate: float = 0.1

        # Top-down prediction currently received from the layer above
        self.top_down_prediction: np.ndarray = np.zeros(num_neurons, dtype=np.float32)

        # Signed prediction error computed on last forward pass
        self.prediction_error: np.ndarray = np.zeros(num_inputs, dtype=np.float32)

        # Acetylcholine level (set externally by NeuromodulatorSystem)
        # Controls bottom-up vs top-down weighting of effective input.
        self.ach_level: float = 0.8

        # Spatial attention gain (set externally by SpatialAttentionController)
        # Multiplicatively scales feedforward drive before relaxation.
        # >1.0 = attended (boosted), <1.0 = suppressed, 1.0 = neutral.
        self.attention_gain: float = 1.0

        self.error_spikes: np.ndarray = np.zeros(num_inputs, dtype=bool)

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        """
        One timestep of predictive coding integration.

        Steps:
          1. Aktualizacja śladów STDP (x_pre, x_post, e) — POPRAWKA Bug 1.
          2. Compute signed prediction error  (actual − top-down prediction).
          3. Build effective firing rate as an ACh-weighted blend of raw
             signal and top-down prediction.
          4. Convert the blended rate to binary Poisson spikes so that
             the LIF layer and its STDP traces receive proper discrete events.
          5. Delegate to CompetitiveLIFLayer (LIF + k-WTA inhibition).
          6. Update feedback weights: neurons that fired strengthen their
             predictions for the positive-error input components.

        Args:
            pre_spikes: Bottom-up input from the layer below (or raw sensory input).

        Returns:
            Boolean spike array (num_neurons,).
        """
        pre_f32 = pre_spikes.astype(np.float32)

        # POPRAWKA Bug 1: Aktualizacja śladów STDP zanim cokolwiek się wydarzy.
        # LIFLayer.forward() robi to wewnętrznie, ale my bypasujemy super().forward(),
        # więc musimy zrobić to ręcznie. Bez tego self.e = 0 zawsze → dw = 0 zawsze.
        self.x_pre *= self._pre_decay
        self.x_post *= self._post_decay
        pre_active = pre_spikes > 0
        self.x_pre[pre_active] += 1.0

        # POPRAWKA Bug A: Obliczamy drive feedforward z wyuczonych wag self.w
        # PRZED pętlą relaksacji (jest stały w całej pętli, więc liczymy raz).
        # Bez tego self.w są uczone przez STDP, ale nigdy nie wpływają na dynamikę v.
        # Teraz pełnią rolę "wejścia z kory pierwszorzędowej" (bottom-up drive),
        # a pętla relaksacji uzgadnia go z predykcją top-down (feedback_w).
        ff_drive = pre_f32 @ self.w  # (num_neurons,)

        # Spatial attention: multiplicatively modulate feedforward drive.
        # Attended columns (gain > 1) have stronger bottom-up signal,
        # giving their neurons higher priority in k-WTA competition.
        ff_drive *= self.attention_gain

        # POPRAWKA Bug C: Proaktywna inhibicja k-WTA przed relaxacją
        self._apply_proactive_inhibition()

        # Membrane leak between timesteps (LIF consistency).
        # Without this, voltages accumulated from the previous timestep's
        # relaxation persist indefinitely, violating LIF physics.
        # Biological basis: inter-spike membrane potential decays toward
        # v_rest with time constant tau_m between input events.
        self.v *= self._mem_decay

        # POPRAWKA Bug D: ff_drive injected ONCE into membrane potential
        # before relaxation begins (models thalamo-cortical volley arriving
        # at the start of the processing cycle). Previously ff_drive was
        # summed into combined_gradient inside every iteration of the
        # relaxation loop, causing 10×0.1 = 1.0× accumulation that drove
        # 22/32 neurons above threshold and defeated k-WTA (should be k=4).
        self.v += ff_drive

        # ZMODYFIKOWANE: Pętla relaksacji
        for i in range(self.pc_config.relaxation_steps):
            # 1. Przybliżenie obecnej aktywności na podstawie potencjału v
            r = np.clip((self.v - self.config.v_rest) / (self.config.v_thresh - self.config.v_rest), 0.0, 1.0)

            # 2. Przewidywanie wejścia na podstawie NASZEGO stanu
            my_prediction = r @ self.feedback_w

            # 3. Błąd predykcji (co dostajemy vs co przewidujemy)
            self.prediction_error = pre_f32 - my_prediction

            # 4. Gradient: ACh-weighted bottom-up error + top-down prediction.
            #    ACh → 1.0: trust sensory input (novel environment)
            #    ACh → 0.0: trust internal predictions (familiar state)
            #    Biological basis (Hasselmo 2006): ACh from basal forebrain
            #    suppresses top-down feedback in superficial cortical layers
            #    while enhancing bottom-up processing.
            error_gradient = self.prediction_error @ self.feedback_w.T
            combined_gradient = (self.ach_level * error_gradient
                                 + (1.0 - self.ach_level) * self.top_down_prediction)

            if np.linalg.norm(combined_gradient) < self.pc_config.relaxation_threshold:
                break

            # 5. Aktualizacja potencjału (gradient łączony: dół + góra)
            self.v += self.pc_config.relaxation_rate * combined_gradient
            np.clip(self.v, self.config.v_reset, self.config.v_thresh + 10.0, out=self.v)


        # Faza Generowania Impulsu po relaksacji (standardowy LIF / k-WTA)
        in_refrac = self.refrac_count > 0
        self.refrac_count[in_refrac] -= 1

        # Wykorzystujemy zrelaksowany potencjał v do oceny spike'ów
        if hasattr(self, 'v_thresh_adaptive'):
            ne_drop = getattr(self, '_ne_level', 0.0) * self.config.ne_thresh_drop
            thresh = self.v_thresh_adaptive - ne_drop
        else:
            thresh = self.config.v_thresh
        self.has_spiked = (self.v >= thresh) & ~in_refrac

        self.v[self.has_spiked] = self.config.v_reset
        self.refrac_count[self.has_spiked] = self.config.refrac_period
        self.x_post[self.has_spiked] += 1.0

        # Ręczna integracja okna czasowego dla k-WTA i homeostazy
        self.window_spike_counts += self.has_spiked.astype(np.int32)
        self._current_window_size += 1

        if getattr(self, '_phase_reset_pending', False):
            self._apply_lateral_inhibition()
            if getattr(self, '_homeostatic_kwta', False) and self._current_window_size > 0:
                self._update_kwta_homeostasis(self._current_window_size)
            self._reset_window()

        # Aktualizacja śladu kwalifikowalności po wykryciu spike'ów.
        self.e *= self._trace_decay

        if np.any(self.has_spiked):
            self.e[:, self.has_spiked] += self.x_pre[:, np.newaxis]
        if np.any(pre_active):
            self.e[pre_active, :] += self.x_post[np.newaxis, :]

        # Error spikes remain accessible as an attribute for consumers that need them,
        # but the layer's OUTPUT is has_spiked (num_neurons) — consistent with
        # LIFLayer.forward() and PyramidalLayer.forward().
        # Biologicznie: szlaki feedforward korowe transmitują wzorce impulsów
        # (firing rates neuronów piramidalnych), a nie surowe sygnały błędu.
        # Błąd jest wewnętrznym sygnałem księgowym warstwy.
        positive_error = np.clip(self.prediction_error, 0.0, 1.0) * self.ach_level
        self.error_spikes = self._encoder.encode(positive_error).astype(bool)

        return self.has_spiked.astype(np.float32)

    # ------------------------------------------------------------------
    # Prediction interface
    # ------------------------------------------------------------------

    def generate_prediction(self) -> np.ndarray:
        """
        Generate top-down prediction for the layer below.

        Projects the current firing pattern through feedback_w to produce
        an expected input pattern for the lower layer next timestep.

        Returns:
            Prediction vector of shape (num_inputs,), values in [0, 1].
        """
        raw = self.has_spiked.astype(np.float32) @ self.feedback_w
        return np.clip(raw * self.pc_config.feedback_strength, 0.0, 1.0)

    def receive_prediction(self, prediction: np.ndarray) -> None:
        """
        Accept a top-down prediction from the layer above.

        Args:
            prediction: Expected input pattern, shape (num_inputs,).
        """
        self.top_down_prediction = prediction.astype(np.float32)

    def set_ach_level(self, ach: float) -> None:
        """
        Update the acetylcholine modulation level.

        Args:
            ach: Float in [0, 1]. 1.0 = fully bottom-up; 0.0 = fully top-down.
        """
        self.ach_level = float(np.clip(ach, 0.0, 1.0))

    def set_attention_gain(self, gain: float) -> None:
        """
        Set spatial attention gain for this layer.

        Args:
            gain: Multiplicative gain for feedforward drive.
                  >1.0 = attended, <1.0 = suppressed, 1.0 = neutral.
        """
        self.attention_gain = float(max(gain, 0.1))

    # Override update_weights to train the backward matrix
    def update_weights(self, m_t: float, pred_error: np.ndarray) -> None:
        """
        Updates feedforward weights via STDP and backward weights via
        Dendritic Error Learning (alignment with forward activity).
        """
        super().update_weights(m_t, pred_error)

        if np.any(self.has_spiked):
            dw = self.pc_config.feedback_learning_rate * np.outer(
                self.has_spiked.astype(np.float32), self.prediction_error
            )
            self.feedback_w += dw * m_t

            # NORMALIZACJA (Zapobiega wybuchowi wag feedbacku)
            if self.pc_config.feedback_norm:
                norms = np.linalg.norm(self.feedback_w, axis=1, keepdims=True) + 1e-8
                self.feedback_w /= norms
            else:
                np.clip(self.feedback_w, -1.0, 2.0, out=self.feedback_w)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset transient state including prediction buffers. Weights are preserved."""
        super().reset_state()
        self.top_down_prediction.fill(0.0)
        self.prediction_error.fill(0.0)
        self.attention_gain = 1.0