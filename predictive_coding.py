import numpy as np
from config import PredictiveCodingConfig
from competitive_layer import CompetitiveLIFLayer


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

        # Top-down feedback weights: this layer → layer below  (num_neurons × num_inputs)
        self.feedback_w: np.ndarray = np.random.uniform(
            0.0, 0.1, (num_neurons, num_inputs)
        ).astype(np.float32)

        # Top-down prediction currently received from the layer above
        self.top_down_prediction: np.ndarray = np.zeros(num_inputs, dtype=np.float32)

        # Signed prediction error computed on last forward pass
        self.prediction_error: np.ndarray = np.zeros(num_inputs, dtype=np.float32)

        # Acetylcholine level (set externally by NeuromodulatorSystem)
        # Controls bottom-up vs top-down weighting of effective input.
        self.ach_level: float = 0.8

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        """
        One timestep of predictive coding integration.

        Steps:
          1. Compute signed prediction error  (actual − top-down prediction).
          2. Build effective input as an ACh-weighted blend of raw signal and prediction.
          3. Delegate to CompetitiveLIFLayer (LIF + k-WTA inhibition).
          4. Update feedback weights: neurons that fired strengthen their
             predictions for the positive-error input components.

        Args:
            pre_spikes: Bottom-up input from the layer below (or raw sensory input).

        Returns:
            Boolean spike array (num_neurons,).
        """
        pre_f32 = pre_spikes.astype(np.float32)

        # 1. Prediction error: what surprised us at this level?
        self.prediction_error = pre_f32 - self.top_down_prediction

        # 2. ACh-gated effective input
        #    High ACh  → lean on raw data;  Low ACh → lean on prior prediction
        effective_input = (
            self.ach_level * pre_f32
            + (1.0 - self.ach_level) * np.clip(self.top_down_prediction, 0.0, None)
        ).astype(np.float32)

        # 3. Standard LIF + k-WTA integration
        spikes = super().forward(effective_input)

        # 4. Feedback weight update (Hebbian predictive rule)
        #    Only the positive part of the error is learned — neurons should
        #    learn to predict inputs they underestimated, not suppress over-predictions.
        if np.any(self.has_spiked):
            positive_error = np.clip(self.prediction_error, 0.0, None)
            dw = self.pc_config.feedback_learning_rate * np.outer(
                self.has_spiked.astype(np.float32), positive_error
            )
            self.feedback_w += dw
            np.clip(self.feedback_w, 0.0, 1.0, out=self.feedback_w)

        return spikes

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

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset transient state including prediction buffers. Weights are preserved."""
        super().reset_state()
        self.top_down_prediction.fill(0.0)
        self.prediction_error.fill(0.0)