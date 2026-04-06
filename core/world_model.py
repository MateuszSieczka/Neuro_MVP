import numpy as np

from .config import SNNWorldModelConfig, PredictiveCodingConfig




# ======================================================================
# SNN-native world model — STDP + Hebbian, zero gradient descent
# ======================================================================

class SNNWorldModel:
    """
    SNN-native world model: replaces gradient descent with STDP + Hebbian learning.

    Architecture
    ============
    Two learned components:

      Encoder (PredictiveCodingLayer):
        Input:   [state_spikes ‖ action_spikes]  shape (state_size + action_size,)
        Output:  internal spike pattern           shape (hidden_size,)
        Learned: STDP three-factor rule (same as all other layers).

      Decoder (linear Hebbian layer):
        Input:   internal spikes                  shape (hidden_size,)
        Output:  predicted_next_state             shape (state_size,) ∈ [0,1]
        Learned: Δw_decode = decode_lr × outer(internal_spikes, state_error)
                 where state_error = actual_next_state − predicted_next_state.

    Why this fixes the dual-regime conflict
    ----------------------------------------
    In the original WorldModel, gradient descent converges in O(100) steps
    while STDP needs O(10 000) exposures. Curiosity from the fast-converging
    MSE model saturates and stops driving exploration while the SNN is still
    in early learning.

    Here both encoder and decoder learn on the same timescale (Hebbian/STDP)
    and both are modulated by the same neuromodulatory signals (dopamine →
    m_t → STDP gate; acetylcholine → ACh level → encoder's bottom-up/top-down
    balance).  There is no impedance mismatch.

    Mental rehearsal (imagination)
    --------------------------------
    mental_rehearsal() runs the encoder in "imagination mode":
      1. Save the current top-down prediction of the encoder layer.
      2. Force-feed the candidate [state, action] spikes.
      3. Read out the decoder prediction.
      4. Restore the top-down prediction.
    This is a forward pass without weight updates.  When the PyramidalLayer
    replaces PredictiveCodingLayer as the encoder, the apical compartment
    allows imagination even with silent basal input (see PyramidalLayer docs).

    Curiosity
    ---------
    curiosity_signal() returns the mean absolute state prediction error — a
    scalar in [0, 1] that can be passed directly to NeuromodulatorSystem.update(
    novelty=...).  Both components (encoder prediction_error and decoder state
    error) contribute; larger errors → more novelty → higher ACh/NE.

    Interface compatibility
    -----------------------
    SNNWorldModel exposes the same predict / mental_rehearsal /
    curiosity_signal / reset_error_history methods as the legacy WorldModel
    so existing agent loop code requires no changes.  The update() method
    signature differs: it accepts spike arrays for state/action rather than
    continuous vectors.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: SNNWorldModelConfig | None = None,
    ) -> None:
        # Deferred import avoids circular dependency at module load time
        from .predictive_coding import PredictiveCodingLayer

        self.config = config or SNNWorldModelConfig()
        self.state_size = state_size
        self.action_size = action_size
        self.input_size = state_size + action_size
        self.hidden_size = self.config.hidden_size

        # ── Encoder: PredictiveCodingLayer ────────────────────────────
        encoder_pc_config = PredictiveCodingConfig(
            feedback_strength=self.config.feedback_strength,
            feedback_learning_rate=self.config.feedback_learning_rate,
            k_winners=self.config.k_winners,
            window_ms=self.config.window_ms,
            i_inh=self.config.i_inh,
        )
        self._encoder = PredictiveCodingLayer(
            self.input_size, self.hidden_size, encoder_pc_config
        )

        # ── Decoder: Hebbian linear readout ───────────────────────────
        # Maps internal spike pattern → predicted next state (rate-coded).
        self.w_decode: np.ndarray = np.random.normal(
            0.0, 0.01, (self.hidden_size, state_size)
        ).astype(np.float32)

        # ── Running state ─────────────────────────────────────────────
        self._last_internal_spikes: np.ndarray = np.zeros(
            self.hidden_size, dtype=np.float32
        )
        self.last_prediction: np.ndarray = np.zeros(state_size, dtype=np.float32)
        self.prediction_error: np.ndarray = np.zeros(state_size, dtype=np.float32)
        self.prediction_error_scalar: float = 0.0
        self._error_history: list[float] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(
            self,
            state_spikes: np.ndarray,
            action: int | np.ndarray,
    ) -> np.ndarray:
        """
        Predict the next state from (state_spikes, action).
        """
        combined = self._build_input(state_spikes, action)

        # Ignorujemy oddolny błąd zwracany przez warstwę PC
        self._encoder.forward(combined)

        # Bezpośrednio pobieramy aktywność reprezentacji ukrytej (kształt: hidden_size)
        internal_spikes = self._encoder.has_spiked.astype(np.float32)
        self._last_internal_spikes = internal_spikes

        raw = self._last_internal_spikes @ self.w_decode
        self.last_prediction = np.clip(raw, 0.0, 1.0)
        return self.last_prediction

    def update(
            self,
            state_spikes: np.ndarray,
            action: int | np.ndarray,
            actual_next_state: np.ndarray,
            m_t: float = 1.0,
    ) -> np.ndarray:
        """
        Observe a real transition and update the decoder (Hebbian).
        """
        combined = self._build_input(state_spikes, action)

        # Wykonujemy krok forward i wyciągamy prawidłowy wektor aktywacji neuronów
        self._encoder.forward(combined)
        internal_spikes = self._encoder.has_spiked.astype(np.float32)
        self._last_internal_spikes = internal_spikes

        actual = actual_next_state.astype(np.float32)
        predicted = np.clip(internal_spikes @ self.w_decode, 0.0, 1.0)

        # Błąd dekodera
        self.prediction_error = actual - predicted
        self.prediction_error_scalar = float(np.mean(self.prediction_error ** 2))
        self._error_history.append(self.prediction_error_scalar)

        # Propagacja błędu przyszłości z dekodera do enkodera (przestrzeń neuronów, kształt: 64)
        decoder_gradient = self.prediction_error @ self.w_decode.T

        # NAPRAWA: Przekazujemy TYLKO błąd z dekodera (rozmiar 64) do uczenia feedforward (LIFLayer.w).
        # PredictiveCodingLayer i tak automatycznie użyje własnego self.prediction_error (rozmiar 10)
        # pod spodem do aktualizacji wag top-down (feedback_w).
        self._encoder.update_weights(m_t=m_t, pred_error=decoder_gradient)

        # Aktualizacja Hebbowska dekodera
        if np.any(internal_spikes > 0):
            dw = self.config.decode_lr * np.outer(
                internal_spikes, self.prediction_error
            )
            self.w_decode += dw * m_t
            np.clip(self.w_decode, -1.0, 1.0, out=self.w_decode)

        return self.prediction_error

    def mental_rehearsal(
            self,
            current_state_spikes: np.ndarray,
            candidate_actions: list[int],
    ) -> dict[int, dict]:
        """
        Internally simulate candidate actions without real-world interaction.

        For each candidate action the encoder runs a short micro-imagination
        loop (config.rehearsal_steps forward passes).  This lets the membrane
        accumulate action-specific signal so that k-WTA competition can
        differentiate subtly different inputs (e.g. one-hot action vectors
        that differ by a single element).

        The encoder state is saved before and restored after all candidates
        have been evaluated — imagination is side-effect-free.
        """
        recent_errors = self._error_history[-20:] if self._error_history else [0.5]
        avg_baseline = float(np.mean(recent_errors)) + 1e-8

        enc = self._encoder

        # Full snapshot of encoder state before imagination
        saved_state = self._snapshot_encoder()

        results: dict[int, dict] = {}
        for action in candidate_actions:
            combined = self._build_input(current_state_spikes, action)

            # Restore clean state before each candidate
            self._restore_encoder(saved_state)

            # Micro-imagination loop: multiple forward passes let the
            # membrane charge and k-WTA differentiate action-specific input.
            for _ in range(self.config.rehearsal_steps):
                enc.forward(combined)

            internal = enc.has_spiked.astype(np.float32)
            predicted_next = np.clip(internal @ self.w_decode, 0.0, 1.0)

            state_change = float(
                np.mean(np.abs(predicted_next - current_state_spikes[:self.state_size]))
            )
            novelty = float(np.clip(state_change / avg_baseline, 0.0, 1.0))
            results[action] = {
                "predicted_state": predicted_next,
                "novelty": novelty,
                "familiarity": 1.0 - novelty,
            }

        # Restore full encoder state after all imagination
        self._restore_encoder(saved_state)

        return results

    def curiosity_signal(
        self,
        prediction_error: np.ndarray | None = None,
    ) -> float:
        """
        Scalar intrinsic motivation from prediction error.

        Combines encoder-level error (prediction_error attribute of the
        PCLayer, in input space) and decoder-level state error to give a
        richer novelty signal than either alone.

        Returns:
            Float in [0, 1] — suitable for NeuromodulatorSystem.update(novelty=...).
        """
        if prediction_error is None:
            prediction_error = self.prediction_error
        encoder_error = float(
            np.clip(np.mean(np.abs(self._encoder.prediction_error)), 0.0, 1.0)
        )
        decoder_error = float(
            np.clip(np.mean(np.abs(prediction_error)), 0.0, 1.0)
        )
        # Geometric mean: both must be high for curiosity to be high.
        # This prevents spurious curiosity when only one level is surprised.
        return float(np.sqrt(encoder_error * decoder_error + 1e-8))

    def set_ach_level(self, ach: float) -> None:
        """
        Forward ACh level to the encoder layer.
        High ACh → trust raw input; Low ACh → trust internal predictions.
        """
        self._encoder.set_ach_level(ach)

    def reset_error_history(self) -> None:
        """Clear running error history. Call between episodes."""
        self._error_history.clear()
        self.prediction_error_scalar = 0.0

    def reset_state(self) -> None:
        """Reset transient state of encoder. Weights are preserved."""
        self._encoder.reset_state()
        self._last_internal_spikes.fill(0.0)
        self.last_prediction.fill(0.0)
        self.prediction_error.fill(0.0)
        self.prediction_error_scalar = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot_encoder(self) -> dict:
        """Capture full encoder state for side-effect-free simulation."""
        enc = self._encoder
        snap = {
            "v": enc.v.copy(),
            "has_spiked": enc.has_spiked.copy(),
            "refrac_count": enc.refrac_count.copy(),
            "x_pre": enc.x_pre.copy(),
            "x_post": enc.x_post.copy(),
            "e": enc.e.copy(),
            "top_down_prediction": enc.top_down_prediction.copy(),
            "prediction_error": enc.prediction_error.copy(),
        }
        if hasattr(enc, "window_spike_counts"):
            snap["window_spike_counts"] = enc.window_spike_counts.copy()
            snap["_current_window_size"] = enc._current_window_size
            snap["_phase_reset_pending"] = enc._phase_reset_pending
        if hasattr(enc, "avg_rate"):
            snap["avg_rate"] = enc.avg_rate.copy()
        if hasattr(enc, "v_thresh_adaptive"):
            snap["v_thresh_adaptive"] = enc.v_thresh_adaptive.copy()
        return snap

    def _restore_encoder(self, snap: dict) -> None:
        """Restore encoder state from a snapshot."""
        enc = self._encoder
        enc.v[:] = snap["v"]
        enc.has_spiked[:] = snap["has_spiked"]
        enc.refrac_count[:] = snap["refrac_count"]
        enc.x_pre[:] = snap["x_pre"]
        enc.x_post[:] = snap["x_post"]
        enc.e[:] = snap["e"]
        enc.top_down_prediction[:] = snap["top_down_prediction"]
        enc.prediction_error[:] = snap["prediction_error"]
        if "window_spike_counts" in snap:
            enc.window_spike_counts[:] = snap["window_spike_counts"]
            enc._current_window_size = snap["_current_window_size"]
            enc._phase_reset_pending = snap["_phase_reset_pending"]
        if "avg_rate" in snap:
            enc.avg_rate[:] = snap["avg_rate"]
        if "v_thresh_adaptive" in snap:
            enc.v_thresh_adaptive[:] = snap["v_thresh_adaptive"]

    def _encode_action(self, action: int) -> np.ndarray:
        """One-hot encode an integer action index."""
        vec = np.zeros(self.action_size, dtype=np.float32)
        vec[int(action)] = 1.0
        return vec

    def _build_input(
        self,
        state_spikes: np.ndarray,
        action: int | np.ndarray,
    ) -> np.ndarray:
        """Concatenate state spikes and action encoding into encoder input."""
        if isinstance(action, (int, np.integer)):
            action_vec = self._encode_action(int(action))
        else:
            action_vec = np.asarray(action, dtype=np.float32)
        return np.concatenate(
            [state_spikes.astype(np.float32), action_vec]
        )