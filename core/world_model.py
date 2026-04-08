import numpy as np

from .config import SNNWorldModelConfig, PredictiveCodingConfig
from .spike_encoder import GaussianPopulationEncoder




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
        self.hidden_size = self.config.hidden_size

        # ── Population coding (Pouget et al. 2000) ───────────────────
        # Converts low-dimensional continuous state into a distributed
        # population representation where every neuron fires at a
        # non-negative rate.  This fixes two issues:
        #   1. STDP traces (pre_active = pre_spikes > 0) can now see
        #      all state dimensions, including negative values.
        #   2. Representational capacity scales with n_neurons_per_dim,
        #      not raw state dimensionality.
        if self.config.n_neurons_per_dim > 0:
            self._pop_encoder = GaussianPopulationEncoder(
                n_dims=state_size,
                n_neurons_per_dim=self.config.n_neurons_per_dim,
            )
            encoded_state_size = self._pop_encoder.output_size
        else:
            self._pop_encoder = None
            encoded_state_size = state_size

        self.input_size = encoded_state_size + action_size

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
        # Running baseline for absolute epistemic novelty (EMA).
        # Tracks average prediction uncertainty so mental_rehearsal can
        # determine if a candidate action is MORE uncertain than usual
        # (absolute novelty) rather than just ranking candidates (relative).
        self._epistemic_baseline: float = 0.5
        self._epistemic_baseline_decay: float = 0.95

        # Running baseline for curiosity_signal() sensory adaptation.
        # Biological basis (Ulanovsky et al. 2003; Katz et al. 2006):
        # Cortical neurons adapt their response gain to the statistics
        # of recent input. After sustained exposure to a stimulus,
        # responses diminish (adaptation). But when a NOVEL stimulus
        # appears, the response relative to adapted baseline is LARGE.
        # This ensures novelty detection works even after overall PE drops.
        self._curiosity_baseline: float = 0.3
        self._curiosity_baseline_decay: float = 0.97  # τ ≈ 33 steps
    # ------------------------------------------------------------------
    # Encoder readout
    # ------------------------------------------------------------------

    def _encoder_rate(self) -> np.ndarray:
        """Graded rate-coded activity from encoder membrane potential.

        Biological basis: cortical population coding uses graded firing
        rates (Shadlen & Newsome 1998), not just binary spikes. The rate
        r = (v - v_rest) / (v_thresh - v_rest) maps subthreshold membrane
        potential to a normalised firing probability, providing a continuous
        signal for downstream readout even when the population is too small
        or the input too weak for reliable spiking.
        """
        return np.clip(
            (self._encoder.v - self._encoder.config.v_rest)
            / (self._encoder.config.v_thresh - self._encoder.config.v_rest),
            0.0, 1.0,
        )

    def _maybe_phase_reset(self) -> None:
        """Auto-trigger k-WTA phase reset when the encoder’s window is full.

        Unlike NetworkGraph (which has a global oscillator), the world model
        runs its encoder independently. Without explicit phase resets, k-WTA
        competition and homeostatic threshold adaptation never activate.
        """
        if self._encoder._current_window_size >= self.config.window_ms:
            self._encoder.trigger_phase_reset()

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

        self._maybe_phase_reset()
        self._encoder.forward(combined)

        # Graded rate-coded activity: provides continuous signal for decoder
        # even when binary spike count is zero or bursty.
        internal_activity = self._encoder_rate()
        self._last_internal_spikes = internal_activity

        raw = internal_activity @ self.w_decode
        self.last_prediction = raw
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

        # Forward pass with auto-triggered phase resets for k-WTA
        self._maybe_phase_reset()
        self._encoder.forward(combined)
        internal_activity = self._encoder_rate()
        self._last_internal_spikes = internal_activity

        actual = actual_next_state.astype(np.float32)
        predicted = internal_activity @ self.w_decode

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
        if np.max(internal_activity) > 0.01:
            dw = self.config.decode_lr * np.outer(
                internal_activity, self.prediction_error
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

        # ── Two-pass approach for relative novelty ────────────────────
        # Pass 1: collect raw encoder PE and decoder predictions per action.
        # Pass 2: normalize novelty across actions (relative ranking).
        #
        # Biological basis (Bromberg-Martin & Hikosaka 2009): dopaminergic
        # neurons in lateral VTA encode RELATIVE information value across
        # available options, not absolute prediction error.  An action is
        # "epistemically valuable" when it resolves MORE uncertainty than
        # the alternatives, not when uncertainty is high in absolute terms.
        raw_results: list[tuple[int, float, np.ndarray, np.ndarray]] = []

        for action in candidate_actions:
            combined = self._build_input(current_state_spikes, action)

            # Restore clean state before each candidate
            self._restore_encoder(saved_state)

            # Micro-imagination loop: multiple forward passes let the
            # membrane charge and k-WTA differentiate action-specific input.
            # Collect decoder predictions per step to measure prediction
            # stability (familiar transitions converge, novel ones oscillate).
            step_predictions = []
            for _ in range(self.config.rehearsal_steps):
                enc.forward(combined)
                step_rate = np.clip(
                    (enc.v - enc.config.v_rest)
                    / (enc.config.v_thresh - enc.config.v_rest),
                    0.0, 1.0,
                )
                step_predictions.append(step_rate @ self.w_decode)

            predicted_next = step_predictions[-1]

            # Encoder PE: top-down vs bottom-up mismatch
            encoder_pe = float(np.mean(np.abs(enc.prediction_error)))

            # Decoder instability: variance of predictions across rehearsal
            # steps.  Familiar (state,action) → encoder converges quickly →
            # low variance.  Novel → encoder oscillates → high variance.
            if len(step_predictions) > 1:
                decoder_var = float(np.mean(np.var(
                    np.stack(step_predictions), axis=0
                )))
            else:
                decoder_var = 0.0

            # Combined raw epistemic signal
            raw_epistemic = encoder_pe + decoder_var

            raw_results.append((action, raw_epistemic, predicted_next, step_predictions[-1]))

        # ── Pass 2: absolute novelty normalization ────────────────────
        # Biological basis (Bromberg-Martin & Hikosaka 2009): VTA novelty
        # responses reflect ABSOLUTE information gain, not relative ranking.
        # An action is epistemically valuable when it produces MORE
        # prediction uncertainty than the agent's running baseline, not
        # merely more than the other candidates.
        #
        # Previous relative normalization guaranteed one action always had
        # novelty=1.0 even in fully familiar territory, injecting noise
        # into action selection. Absolute novelty correctly reports "all
        # familiar" (low values) vs "all novel" (high values).
        raw_values = np.array([r[1] for r in raw_results], dtype=np.float32)

        # Update running epistemic baseline (EMA across rehearsal calls)
        batch_mean = float(np.mean(raw_values))
        d = self._epistemic_baseline_decay
        self._epistemic_baseline = d * self._epistemic_baseline + (1 - d) * batch_mean
        baseline = max(self._epistemic_baseline, 1e-6)

        results: dict[int, dict] = {}
        for action, raw_ep, predicted_next, _ in raw_results:
            # Absolute novelty: ratio to running baseline, clipped to [0, 2]
            # 0 = fully familiar, 1 = baseline surprise, 2 = extremely novel
            novelty = float(np.clip(raw_ep / baseline, 0.0, 2.0))

            results[action] = {
                "predicted_state": predicted_next,
                "novelty": novelty,
                "familiarity": max(0.0, 1.0 - novelty),
            }

        # Restore full encoder state after all imagination
        self._restore_encoder(saved_state)

        return results

    def curiosity_signal(
        self,
        prediction_error: np.ndarray | None = None,
    ) -> float:
        """
        Scalar intrinsic motivation from decoder prediction error.

        Uses ADAPTIVE BASELINE (sensory adaptation): the raw PE is compared
        to a running average.  This prevents curiosity collapse — even when
        absolute PE drops as the WM learns, a visit to a novel state produces
        PE above the adapted baseline, generating strong curiosity.

        Returns:
            Float in [0, 2] — 1.0 = baseline surprise, >1 = novel, <1 = familiar.
        """
        if prediction_error is None:
            prediction_error = self.prediction_error

        # Primary signal: decoder state prediction error
        decoder_error = float(
            np.clip(np.mean(np.abs(prediction_error)), 0.0, 1.0)
        )

        # Secondary: encoder prediction error (internal PC mismatch)
        encoder_error = float(
            np.clip(np.mean(np.abs(self._encoder.prediction_error)), 0.0, 1.0)
        )

        raw_curiosity = 0.7 * decoder_error + 0.3 * encoder_error

        # Sensory adaptation: EMA baseline tracks average PE.
        # When PE is at baseline → curiosity ≈ 1.0 (neutral).
        # When PE spikes (novel state) → curiosity > 1.0 (explore!).
        # When PE drops below baseline → curiosity < 1.0 (familiar).
        d = self._curiosity_baseline_decay
        self._curiosity_baseline = d * self._curiosity_baseline + (1 - d) * raw_curiosity
        baseline = max(self._curiosity_baseline, 1e-6)

        relative_curiosity = raw_curiosity / baseline
        return float(np.clip(relative_curiosity, 0.0, 2.0))

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
        self._epistemic_baseline = 0.5
        # NOTE: do NOT reset _curiosity_baseline here — it must persist
        # across episodes for sensory adaptation to work correctly.
        # Resetting it would cause curiosity spikes at episode boundaries.

    def reset_state(self) -> None:
        """Reset transient state of encoder. Weights are preserved."""
        self._encoder.reset_state()
        self._last_internal_spikes.fill(0.0)
        self.last_prediction.fill(0.0)
        self.prediction_error.fill(0.0)
        self.prediction_error_scalar = 0.0
        self._epistemic_baseline = 0.5

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
        """Concatenate (population-encoded) state and action into encoder input."""
        state_f32 = state_spikes.astype(np.float32)
        if self._pop_encoder is not None:
            state_encoded = self._pop_encoder.encode(state_f32)
        else:
            state_encoded = state_f32

        if isinstance(action, (int, np.integer)):
            action_vec = self._encode_action(int(action))
        else:
            action_vec = np.asarray(action, dtype=np.float32)
        return np.concatenate([state_encoded, action_vec])