import numpy as np

from config import WorldModelConfig


class WorldModel:
    """
    Predictive internal model of environmental dynamics.

    Maps (current_state, action) → predicted_next_state using a bilinear
    linear layer with gradient-descent updates.  Supports:

      1. Online prediction and learning after each real transition.
      2. Mental Rehearsal: imagining action consequences without executing them.
      3. Curiosity signal: prediction error magnitude as intrinsic motivation.

    State convention:
      States are spike-rate vectors in [0, 1]^state_size, produced by e.g. the
      final CompetitiveLIFLayer / PredictiveCodingLayer in the hierarchy.
      state_size MUST match the num_neurons of the layer whose output is used as state.

    Ha & Schmidhuber (2018) "World Models" — lightweight SNN-compatible variant.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: WorldModelConfig | None = None,
    ) -> None:
        self.config = config or WorldModelConfig()
        self.state_size = state_size
        self.action_size = action_size
        self.combined_size = state_size + action_size

        # Prediction weights: [state ‖ action] → next_state
        self.w: np.ndarray = np.random.normal(
            0.0, 0.01, (self.combined_size, state_size)
        ).astype(np.float32)
        self.b: np.ndarray = np.zeros(state_size, dtype=np.float32)

        # Scalar MSE of the last update() call
        self.prediction_error: float = 0.0

        # Running log of MSE for novelty normalisation
        self._error_history: list[float] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_action(self, action: int) -> np.ndarray:
        """One-hot encode an integer action index."""
        vec = np.zeros(self.action_size, dtype=np.float32)
        vec[int(action)] = 1.0
        return vec

    def _action_vector(self, action: int | np.ndarray) -> np.ndarray:
        if isinstance(action, (int, np.integer)):
            return self._encode_action(int(action))
        return np.asarray(action, dtype=np.float32)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def predict(self, state: np.ndarray, action: int | np.ndarray) -> np.ndarray:
        """
        Predict the next state without updating weights.

        Args:
            state:  Current state vector (state_size,), values in [0, 1].
            action: Integer action index or pre-encoded one-hot vector.

        Returns:
            Predicted next state in [0, 1]^state_size.
        """
        action_vec = self._action_vector(action)
        combined = np.concatenate([state.astype(np.float32), action_vec])
        raw = combined @ self.w + self.b
        return np.clip(raw, 0.0, 1.0)

    def update(
        self,
        state: np.ndarray,
        action: int | np.ndarray,
        actual_next_state: np.ndarray,
    ) -> np.ndarray:
        """
        Update model weights from an observed real transition.

        Uses gradient descent on MSE loss:
            L = ½ ‖actual − predicted‖²
            ∇_w L = −outer(combined, error)

        Args:
            state:            State before action.
            action:           Executed action (int or one-hot).
            actual_next_state: Observed next state after action.

        Returns:
            Signed prediction error vector (actual − predicted), shape (state_size,).
        """
        action_vec = self._action_vector(action)
        actual = actual_next_state.astype(np.float32)

        predicted = self.predict(state, action_vec)
        error = actual - predicted  # signed error vector

        self.prediction_error = float(np.mean(error ** 2))
        self._error_history.append(self.prediction_error)

        combined = np.concatenate([state.astype(np.float32), action_vec])
        self.w += self.config.learning_rate * np.outer(combined, error)
        self.b += self.config.learning_rate * error

        return error

    def mental_rehearsal(
        self,
        current_state: np.ndarray,
        candidate_actions: list[int],
    ) -> dict[int, dict]:
        """
        Simulate candidate actions internally without affecting real state.

        For each action, computes:
          - predicted_state: expected next state if this action is taken.
          - novelty:         how surprising this transition would be (0–1).
          - familiarity:     1 − novelty.

        Curiosity-driven exploration: prefer actions with HIGH novelty
        (Free Energy / prediction-error minimisation via active inference).
        Exploitation: prefer actions with LOW novelty (known, safe transitions).

        Args:
            current_state:     Current state vector (state_size,).
            candidate_actions: List of integer action indices.

        Returns:
            Dict  action_index → {predicted_state, novelty, familiarity}
        """
        # Baseline for novelty normalisation: recent average MSE
        recent_errors = self._error_history[-20:] if self._error_history else [0.5]
        avg_baseline = float(np.mean(recent_errors)) + 1e-8

        results: dict[int, dict] = {}
        for action in candidate_actions:
            predicted_next = self.predict(current_state, action)

            # State-change magnitude as a novelty proxy
            state_change = float(np.mean(np.abs(predicted_next - current_state)))
            novelty = float(np.clip(state_change / avg_baseline, 0.0, 1.0))

            results[action] = {
                "predicted_state": predicted_next,
                "novelty": novelty,
                "familiarity": 1.0 - novelty,
            }
        return results

    def curiosity_signal(self, prediction_error: np.ndarray) -> float:
        """
        Convert a prediction error vector into a scalar intrinsic reward.

        High curiosity = large prediction error = novel situation encountered.
        This signal can be passed to NeuromodulatorSystem.update(novelty=...).

        Returns:
            Float in [0, 1].
        """
        return float(np.clip(np.mean(np.abs(prediction_error)), 0.0, 1.0))

    def reset_error_history(self) -> None:
        """Clear error history. Call between episodes to reset novelty baseline."""
        self._error_history.clear()
        self.prediction_error = 0.0