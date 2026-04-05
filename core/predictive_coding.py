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
        self.top_down_prediction: np.ndarray = np.zeros(num_inputs, dtype=np.float32)

        # Signed prediction error computed on last forward pass
        self.prediction_error: np.ndarray = np.zeros(num_inputs, dtype=np.float32)

        # Acetylcholine level (set externally by NeuromodulatorSystem)
        # Controls bottom-up vs top-down weighting of effective input.
        self.ach_level: float = 0.8

        self.error_spikes: np.ndarray = np.zeros(num_inputs, dtype=bool)

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        """
        One timestep of predictive coding integration.

        Steps:
          1. Compute signed prediction error  (actual − top-down prediction).
          2. Build effective firing rate as an ACh-weighted blend of raw
             signal and top-down prediction.
          3. Convert the blended rate to binary Poisson spikes so that
             the LIF layer and its STDP traces receive proper discrete events.
          4. Delegate to CompetitiveLIFLayer (LIF + k-WTA inhibition).
          5. Update feedback weights: neurons that fired strengthen their
             predictions for the positive-error input components.

        Args:
            pre_spikes: Bottom-up input from the layer below (or raw sensory input).

        Returns:
            Boolean spike array (num_neurons,).
        """
        pre_f32 = pre_spikes.astype(np.float32)

        # Pętla relaksacji (minimalizacja Wolnej Energii wg Bogacza)
        for i in range(self.pc_config.relaxation_steps):
            # 1. Błąd predykcji obecnego stanu (co niższa warstwa mówi vs co myślimy)

            self.prediction_error = pre_f32 - self.top_down_prediction


            # 2. Transpozycja wag predykcyjnych
            # Błąd propagowany w górę wymusza zmianę potencjału v
            error_gradient = self.prediction_error @ self.feedback_w.T

            if np.linalg.norm(error_gradient) < self.pc_config.relaxation_threshold:
                break

            # 3. Aktualizacja potencjału (gradient descent po Wolnej Energii)
            self.v += self.pc_config.relaxation_rate * error_gradient
            np.clip(self.v, self.config.v_reset, self.config.v_thresh + 10.0, out=self.v)

        # Faza Generowania Impulsu po relaksacji (standardowy LIF / k-WTA)
        in_refrac = self.refrac_count > 0
        self.refrac_count[in_refrac] -= 1

        # Wykorzystujemy zrelaksowany potencjał v do oceny spike'ów
        thresh = self.v_thresh_adaptive if getattr(self, '_homeostatic', False) else self.config.v_thresh
        self.has_spiked = (self.v >= thresh) & ~in_refrac

        self.v[self.has_spiked] = self.config.v_reset
        self.refrac_count[self.has_spiked] = self.config.refrac_period

        # Zwracamy błąd po relaksacji jako twarde impulsy do nauki sieci
        positive_error = np.clip(self.prediction_error, 0.0, 1.0) * self.ach_level
        self.error_spikes = self._encoder.encode(positive_error).astype(bool)

        return self.error_spikes.astype(np.float32)

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
                np.clip(self.feedback_w, 0.0, 1.0, out=self.feedback_w)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset transient state including prediction buffers. Weights are preserved."""
        super().reset_state()
        self.top_down_prediction.fill(0.0)
        self.prediction_error.fill(0.0)

