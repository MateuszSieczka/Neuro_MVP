"""
SNN World Model — ErrorNeuronLayer encoder + ensemble Hebbian decoder.

Reference:
  Friston et al. (2015) "Active inference and epistemic value"
  Pouget, Dayan & Zemel (2000)  Population coding
  De Pittà et al. (2011) Astrocyte Ca²⁺ dynamics

Changes from legacy:
  1. Multi-step mental rehearsal (depth D modulated by 5-HT)
  2. Ensemble decoder (n_ensemble weight sets → posterior variance)
  3. Precision-weighted curiosity (no hardcoded 0.7/0.3)
  4. Uses WorldModelConfig from config.py (not SNNWorldModelConfig)
  5. Uses new ErrorNeuronLayer API (.belief, .prediction_error_rate)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import WorldModelConfig, ErrorNeuronConfig, init_weights
from .spike_encoder import GaussianPopulationEncoder
from .error_neuron import ErrorNeuronLayer
from .astrocyte import AstrocyteField, AstrocyteConfig


# =====================================================================
# Data containers
# =====================================================================

@dataclass
class EncoderSnapshot:
    """Full ErrorNeuronLayer transient state for side-effect-free imagination."""
    v_state: NDArray[np.float32]
    v_error: NDArray[np.float32]
    spikes_state: NDArray[np.bool_]
    spikes_error: NDArray[np.bool_]
    refrac_state: NDArray[np.int32]
    refrac_error: NDArray[np.int32]
    state_rate: NDArray[np.float32]
    error_rate: NDArray[np.float32]
    e_bu: NDArray[np.float32]
    e_td: NDArray[np.float32]

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if isinstance(val, np.ndarray):
                object.__setattr__(self, field_name, val.copy())


@dataclass
class RehearsalResult:
    """Output of mental_rehearsal for a single candidate action."""
    predicted_state: NDArray[np.float32]
    novelty: float
    familiarity: float
    ensemble_variance: float  # Posterior uncertainty from ensemble


# =====================================================================
# World Model
# =====================================================================

class SNNWorldModel:
    """SNN world model: ErrorNeuronLayer encoder + ensemble Hebbian decoder.

    Encoder (ErrorNeuronLayer):
      Input → Error Neurons (fast τ ~4ms, ε = input − g(μ))
              ↕ W_bu / W_td
           State Neurons (slow τ ~20ms, belief μ)

    Decoder (ensemble of n_ensemble Hebbian readouts):
      Each decoder: state_rates → w_decode[k] → predicted_next_state
      Variance across ensemble → epistemic uncertainty
      ΔW = lr × outer(belief, prediction_error). No backprop.

    AstrocyteField:
      Monitors decoder error → Ca²⁺ → precision estimate.
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
        self.hidden_size = self.config.hidden_size
        cfg = self.config

        # ── Population coding (Pouget et al. 2000) ───────────────────
        if cfg.n_neurons_per_dim > 0:
            self._pop_encoder = GaussianPopulationEncoder(
                n_dims=state_size,
                n_neurons_per_dim=cfg.n_neurons_per_dim,
            )
            encoded_state_size = self._pop_encoder.output_size
        else:
            self._pop_encoder = None
            encoded_state_size = state_size

        self.input_size = encoded_state_size + action_size

        # ── Encoder: ErrorNeuronLayer ─────────────────────────────────
        encoder_config = ErrorNeuronConfig(
            n_state=self.hidden_size,
            n_error=min(self.input_size, self.hidden_size),
            tau_state=20.0,
            tau_error=4.0,
            w_bu_lr=cfg.encoder_lr,
            w_td_lr=cfg.encoder_lr,
        )
        self.encoder = ErrorNeuronLayer(self.input_size, encoder_config)

        # ── AstrocyteField for decoder precision ─────────────────────
        n_zones = max(4, self.hidden_size // 16)
        self.astrocyte = AstrocyteField(
            n_zones=n_zones,
            config=AstrocyteConfig(tau_ca=500.0, ca_accumulation=0.15),
        )

        # ── Ensemble decoder (posterior uncertainty) ──────────────────
        self.n_ensemble: int = max(1, cfg.n_ensemble)
        self.w_decode: list[NDArray[np.float32]] = [
            np.random.normal(
                0.0, 0.01, (self.hidden_size, state_size),
            ).astype(np.float32)
            for _ in range(self.n_ensemble)
        ]

        # ── Running state ─────────────────────────────────────────────
        self.last_prediction: NDArray[np.float32] = np.zeros(
            state_size, dtype=np.float32,
        )
        self.prediction_error: NDArray[np.float32] = np.zeros(
            state_size, dtype=np.float32,
        )
        self.prediction_error_scalar: float = 0.0
        self.error_history: list[float] = []

        self._curiosity_history: list[float] = []
        self._curiosity_history_maxlen: int = 2000

        # ── Rehearsal depth (modulated by 5-HT at runtime) ────────────
        self._current_rehearsal_depth: int = cfg.max_rehearsal_depth

    # ------------------------------------------------------------------
    # Ensemble helpers
    # ------------------------------------------------------------------

    def _ensemble_predict(
        self,
        belief: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float]:
        """Predict via all ensemble members → mean + variance.

        Returns:
            (mean_prediction, mean_variance_per_dim)
        """
        preds = np.stack([belief @ w for w in self.w_decode])  # (M, state_size)
        mean_pred = np.mean(preds, axis=0).astype(np.float32)
        var_per_dim = np.mean(np.var(preds, axis=0))
        return mean_pred, float(var_per_dim)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(
        self,
        state_spikes: NDArray[np.float32],
        action: int | NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Predict next state. One dt step."""
        combined = self._build_input(state_spikes, action)
        self.encoder.forward(combined)
        belief = self.encoder.belief
        mean_pred, _ = self._ensemble_predict(belief)
        self.last_prediction = mean_pred
        return mean_pred

    def update(
        self,
        state_spikes: NDArray[np.float32],
        action: int | NDArray[np.float32],
        actual_next_state: NDArray[np.float32],
        m_t: float = 1.0,
    ) -> NDArray[np.float32]:
        """Observe real transition; update encoder (STDP) and decoder (Hebbian)."""
        combined = self._build_input(state_spikes, action)
        self.encoder.forward(combined)
        belief = self.encoder.belief
        actual = actual_next_state.astype(np.float32)

        # Ensemble mean prediction
        mean_pred, _ = self._ensemble_predict(belief)
        self.prediction_error = actual - mean_pred
        self.prediction_error_scalar = float(np.mean(self.prediction_error ** 2))
        self.error_history.append(self.prediction_error_scalar)

        # Astrocyte tracks decoder error → precision
        self.astrocyte.update(self.prediction_error)

        # Encoder: three-factor STDP × modulation × astrocyte precision
        self.encoder.update_weights(
            modulation=m_t,
            precision=self.astrocyte.precision,
        )

        # Decoder: Hebbian update per ensemble member (with slight noise)
        max_belief = np.max(np.abs(belief))
        if max_belief > 0.01:
            belief_norm = belief / (max_belief + 1e-6)
            for k in range(self.n_ensemble):
                pred_k = belief @ self.w_decode[k]
                error_k = actual - pred_k
                dw = self.config.decode_lr * np.outer(belief_norm, error_k)
                self.w_decode[k] += (dw * m_t).astype(np.float32)
                np.clip(self.w_decode[k], -1.0, 1.0, out=self.w_decode[k])

        return self.prediction_error

    def mental_rehearsal(
        self,
        current_state_spikes: NDArray[np.float32],
        candidate_actions: list[int],
    ) -> dict[int, RehearsalResult]:
        """Multi-step epistemic evaluation (Friston et al. 2015).

        For each action: simulate D forward steps through encoder,
        accumulate discounted epistemic value (prediction error + ensemble
        variance). Depth D = self._current_rehearsal_depth.
        """
        saved = self.snapshot_encoder()
        baseline_precision = self.astrocyte.mean_precision
        depth = max(1, self._current_rehearsal_depth)
        gamma = 0.99

        raw_results: list[tuple[int, float, float, NDArray[np.float32]]] = []

        for action in candidate_actions:
            self.restore_encoder(saved)
            combined = self._build_input(current_state_spikes, action)

            total_epistemic = 0.0
            total_variance = 0.0
            predicted_next = np.zeros(self.state_size, dtype=np.float32)

            for step in range(depth):
                self.encoder.forward(combined)
                belief = self.encoder.belief

                # Encoder prediction error
                encoder_pe = float(np.mean(self.encoder.prediction_error_rate))

                # Ensemble variance → posterior uncertainty
                mean_pred, ens_var = self._ensemble_predict(belief)
                predicted_next = mean_pred

                # Epistemic value: combined error + variance + (1 − precision)
                step_epistemic = encoder_pe + ens_var + (1.0 - baseline_precision)
                total_epistemic += (gamma ** step) * step_epistemic
                total_variance += (gamma ** step) * ens_var

                # For multi-step: feed predicted state back as input
                if step < depth - 1:
                    combined = self._build_input(mean_pred, action)

            raw_results.append((action, total_epistemic, total_variance, predicted_next))

        # Normalize across candidates
        max_epist = max((r[1] for r in raw_results), default=1e-8)
        max_epist = max(max_epist, 1e-6)

        results: dict[int, RehearsalResult] = {}
        for action, raw_ep, raw_var, pred in raw_results:
            novelty = float(np.clip(raw_ep / max_epist, 0.0, 2.0))
            results[action] = RehearsalResult(
                predicted_state=pred,
                novelty=novelty,
                familiarity=max(0.0, 1.0 - novelty),
                ensemble_variance=raw_var,
            )

        self.restore_encoder(saved)
        return results

    def curiosity_signal(
        self,
        prediction_error: NDArray[np.float32] | None = None,
    ) -> float:
        """Precision-weighted curiosity (replaces hardcoded 0.7/0.3).

        curiosity = precision_decoder × decoder_error + precision_encoder × encoder_error
        Precision from astrocyte field. Slow z-score normalization.
        """
        if prediction_error is None:
            prediction_error = self.prediction_error

        decoder_error = float(np.mean(prediction_error ** 2))
        encoder_error = float(np.mean(self.encoder.prediction_error_rate ** 2))

        # Precision-weighted combination (astrocyte provides decoder precision)
        decoder_precision = self.astrocyte.mean_precision
        # Encoder precision: inverse of encoder error rate magnitude
        encoder_precision = 1.0 / (1.0 + encoder_error)
        raw = decoder_precision * decoder_error + encoder_precision * encoder_error

        self._curiosity_history.append(raw)
        if len(self._curiosity_history) > self._curiosity_history_maxlen:
            self._curiosity_history = self._curiosity_history[-self._curiosity_history_maxlen:]

        if len(self._curiosity_history) < 10:
            return float(np.clip(raw * 2.0, 0.0, 2.0))

        hist = np.array(self._curiosity_history)
        mu = float(np.mean(hist))
        sigma = float(np.std(hist)) + 1e-8
        z = (raw - mu) / sigma
        return float(np.clip(1.0 + 0.5 * z, 0.0, 2.0))

    def set_rehearsal_depth(self, serotonin: float) -> None:
        """5-HT modulates planning horizon (Doya 2002).

        High 5-HT → more patience → deeper rehearsal.
        """
        sero = float(np.clip(serotonin, 0.0, 1.0))
        max_depth = self.config.max_rehearsal_depth
        self._current_rehearsal_depth = max(1, int(1 + sero * (max_depth - 1)))

    def set_ach_level(self, ach: float) -> None:
        """ACh → encoder error neuron gain. High ACh = bottom-up dominant."""
        self.encoder.set_ach_level(ach)

    def set_plasticity_timescales(self, ne: float, ach: float = 0.5) -> None:
        """Delegate NE/ACh to encoder."""
        self.encoder.set_ach_level(ach)

    def set_ne_level(self, ne: float) -> None:
        """No-op: ErrorNeuronLayer has no dark-matter recruitment."""
        pass

    def reset_error_history(self) -> None:
        self.error_history.clear()
        self.prediction_error_scalar = 0.0

    def reset_state(self) -> None:
        """Reset transient state. Weights preserved."""
        self.encoder.reset_state()
        self.last_prediction.fill(0.0)
        self.prediction_error.fill(0.0)
        self.prediction_error_scalar = 0.0
        self.astrocyte.reset_state()

    # ------------------------------------------------------------------
    # Encoder snapshot / restore (imagination & sleep replay)
    # ------------------------------------------------------------------

    def snapshot_encoder(self) -> EncoderSnapshot:
        enc = self.encoder
        return EncoderSnapshot(
            v_state=enc.v_state,
            v_error=enc.v_error,
            spikes_state=enc.spikes_state,
            spikes_error=enc.spikes_error,
            refrac_state=enc.refrac_state,
            refrac_error=enc.refrac_error,
            state_rate=enc.state_rate,
            error_rate=enc.error_rate,
            e_bu=enc.e_bu,
            e_td=enc.e_td,
        )

    def restore_encoder(self, snap: EncoderSnapshot) -> None:
        enc = self.encoder
        enc.v_state[:] = snap.v_state
        enc.v_error[:] = snap.v_error
        enc.spikes_state[:] = snap.spikes_state
        enc.spikes_error[:] = snap.spikes_error
        enc.refrac_state[:] = snap.refrac_state
        enc.refrac_error[:] = snap.refrac_error
        enc.state_rate[:] = snap.state_rate
        enc.error_rate[:] = snap.error_rate
        enc.e_bu[:] = snap.e_bu
        enc.e_td[:] = snap.e_td

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_input(
        self,
        state_spikes: NDArray[np.float32],
        action: int | NDArray[np.float32],
    ) -> NDArray[np.float32]:
        state_f32 = state_spikes.astype(np.float32)
        if self._pop_encoder is not None:
            state_encoded = self._pop_encoder.encode(state_f32)
        else:
            state_encoded = state_f32

        if isinstance(action, (int, np.integer)):
            action_vec = np.zeros(self.action_size, dtype=np.float32)
            action_vec[int(action)] = 1.0
        else:
            action_vec = np.asarray(action, dtype=np.float32)

        return np.concatenate([state_encoded, action_vec])
