"""
Sequence Memory — temporal transition learning with multi-scale hierarchy.

Reference:
  Lisman & Jensen (2013)  Theta-gamma neural code
  Hawkins & Ahmad (2016)  Temporal Memory

Changes from legacy:
  1. Multi-scale hierarchy: gamma (raw spikes), theta (phase-level),
     episode (seconds) timescales
  2. HierarchicalSequenceMemory with configurable salience threshold
  3. Temporal cluster discovery via bidirectional association
  4. Uses SequenceMemoryConfig from config.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import SequenceMemoryConfig


class SequenceMemory:
    """Temporal sequence learning via spike-timing transition weights.

    DG-like pattern separation (Rolls 2013): plastic projection into
    expanded space + competitive k-WTA thresholding before Hebbian
    outer product. Prevents attractor collapse from overlapping inputs.

    DG projection learns via competitive Hebbian learning (Rolls 2013;
    Treves & Rolls 1994): dw = lr × outer(sparse_output, input).
    This improves pattern separation compared to fixed random projection.

    W[i, j] = how strongly neuron j at time t predicts neuron i at t+1.
    Learning rule: dW = lr × outer(post_t, pre_{t-1})
    """

    def __init__(
        self,
        num_neurons: int,
        config: SequenceMemoryConfig | None = None,
    ) -> None:
        self.config = config or SequenceMemoryConfig()
        self.num_neurons = num_neurons

        # DG-like expansion (D5 pattern separation)
        cfg = self.config
        self._expanded_size = num_neurons * cfg.expansion_factor
        self._k = max(1, int(self._expanded_size * cfg.sparsity_k))

        # Fixed random projection matrix (DG granule cells)
        rng = np.random.RandomState(42)  # deterministic for reproducibility
        self._w_dg: NDArray[np.float32] = (
            rng.randn(num_neurons, self._expanded_size).astype(np.float32)
            / np.sqrt(num_neurons)
        )
        # Store initial column norm for Oja-like normalisation
        self._init_col_norm: NDArray[np.float32] = np.maximum(
            np.linalg.norm(self._w_dg, axis=0, keepdims=True),
            1e-8,
        ).astype(np.float32)

        self.transition_w: NDArray[np.float32] = np.zeros(
            (self._expanded_size, self._expanded_size), dtype=np.float32,
        )
        self.prev_pattern: NDArray[np.float32] = np.zeros(
            self._expanded_size, dtype=np.float32,
        )
        self.predicted_next: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )
        self.temporal_error: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )

    def _pattern_separate(self, pattern: NDArray[np.float32]) -> NDArray[np.float32]:
        """DG-like pattern separation: project + k-WTA competitive threshold."""
        projected = np.maximum(pattern @ self._w_dg, 0.0)  # ReLU
        if np.max(projected) < 1e-10:
            return np.zeros(self._expanded_size, dtype=np.float32)
        # k-WTA: keep top-k, zero rest
        threshold = np.partition(projected, -self._k)[-self._k]
        sparse = np.where(projected >= threshold, projected, 0.0).astype(
            np.float32,
        )
        # Competitive Hebbian learning (Rolls 2013; Treves & Rolls 1994):
        # Winners strengthen their input connections → better pattern
        # separation over time.  Column-normalised to prevent runaway.
        dg_lr = self.config.dg_learning_rate
        if dg_lr > 0 and np.any(sparse > 0):
            dw = dg_lr * np.outer(pattern, sparse)  # (n_in, expanded)
            self._w_dg += dw
            # Column normalisation: keep ||column|| constant (Oja-like)
            col_norms = np.linalg.norm(self._w_dg, axis=0, keepdims=True)
            col_norms = np.maximum(col_norms, 1e-8)
            self._w_dg *= (self._init_col_norm / col_norms)
        return sparse

    def observe(self, current_pattern: NDArray[np.float32]) -> NDArray[np.float32]:
        """Record pattern and learn transition from previous."""
        pattern = current_pattern.astype(np.float32)
        separated = self._pattern_separate(pattern)

        # Temporal prediction error (in original space)
        self.temporal_error = pattern - self.predicted_next

        if np.any(self.prev_pattern > 0) and np.any(separated > 0):
            dw = self.config.learning_rate * np.outer(separated, self.prev_pattern)
            self.transition_w += dw
            self.transition_w *= self.config.decay
            np.clip(
                self.transition_w, 0.0, self.config.max_weight,
                out=self.transition_w,
            )

        self.prev_pattern = separated.copy()
        self.predicted_next = self._predict_from(separated)
        return self.temporal_error

    def predict_next(self) -> NDArray[np.float32]:
        """Predict next activation pattern."""
        return self.predicted_next.copy()

    def novelty_signal(self) -> float:
        """Scalar novelty from temporal prediction error magnitude."""
        return float(np.clip(np.mean(np.abs(self.temporal_error)), 0.0, 1.0))

    def _predict_from(self, separated: NDArray[np.float32]) -> NDArray[np.float32]:
        """Project separated pattern through transition_w, project back."""
        if not np.any(separated > 0):
            return np.zeros(self.num_neurons, dtype=np.float32)
        # Forward in expanded space then project back via pseudo-inverse
        raw_expanded = separated @ self.transition_w.T
        # Approximate inverse projection: _w_dg.T maps expanded → original
        raw = np.clip(raw_expanded, 0.0, 1.0) @ self._w_dg.T
        return np.clip(raw, 0.0, 1.0).astype(np.float32)

    def reset_state(self) -> None:
        """Reset transient state. Transition weights preserved."""
        self.prev_pattern.fill(0.0)
        self.predicted_next.fill(0.0)
        self.temporal_error.fill(0.0)

    def reset_all(self) -> None:
        """Full reset including learned weights."""
        self.reset_state()
        self.transition_w.fill(0.0)


class HierarchicalSequenceMemory:
    """Multi-scale sequence memory with oscillator-coupled timescales.

    Level 0 (gamma ~30ms): Raw spike transitions per timestep
    Level 1 (theta ~125ms): Phase-level pooled transitions
    Level 2 (episode ~seconds): Episode-level macro-transitions

    Theta-level learning gated by oscillator encoding phase
    (0 < φ_theta < π, Hasselmo 2005).
    Pooling windows derived from oscillator frequency and dt.
    """

    # Episode-scale pooling spans ~5 seconds of theta cycles
    # (Buzsáki & Moser 2013 — hippocampal episode ~3-10s)
    _EPISODE_DURATION_S: float = 5.0

    def __init__(
        self,
        num_neurons: int,
        config: SequenceMemoryConfig | None = None,
        salience_threshold: float = 0.5,
        theta_freq_hz: float = 6.0,
        dt_ms: float = 1.0,
    ) -> None:
        cfg = config or SequenceMemoryConfig()
        self.num_neurons = num_neurons
        self.salience_threshold = salience_threshold
        self._dt_ms = dt_ms

        # Derive pooling windows from oscillator periods (CLN 2):
        # theta_window = ticks per theta cycle = 1 / (f_theta × dt_s)
        # episode_window = theta cycles per episode = duration_s × f_theta
        dt_s = dt_ms / 1000.0
        self.theta_window: int = max(
            1, round(1.0 / (theta_freq_hz * dt_s)),
        )
        self.episode_window: int = max(
            1, round(self._EPISODE_DURATION_S * theta_freq_hz),
        )

        # Level 0: gamma-scale raw transitions
        self.level0 = SequenceMemory(num_neurons, cfg)

        # Level 1: theta-scale pooled
        self.level1 = SequenceMemory(num_neurons, cfg)
        self._theta_buffer: list[NDArray[np.float32]] = []

        # Level 2: episode-scale
        self.level2 = SequenceMemory(num_neurons, cfg)
        self._episode_buffer: list[NDArray[np.float32]] = []

        self._step_count: int = 0

    def update_theta_window(self, theta_freq_hz: float) -> None:
        """Dynamically update pooling window from oscillator theta frequency.

        theta_ticks = round(1 / (f_theta × dt_s))
        """
        dt_s = self._dt_ms / 1000.0
        if theta_freq_hz > 0.0 and dt_s > 0.0:
            self.theta_window = max(1, round(1.0 / (theta_freq_hz * dt_s)))

    def observe(
        self,
        current_pattern: NDArray[np.float32],
        salience: float = 0.0,
        theta_phase: float | None = None,
        theta_reset: bool = False,
    ) -> NDArray[np.float32]:
        """Multi-scale observation with oscillator phase gating.

        Args:
            current_pattern: Spike activity from the attached layer.
            salience: NE-driven salience signal for level-1 gating.
            theta_phase: Current theta oscillator phase (radians).
                If provided, theta-level learning is gated to encoding
                window (0 ≤ φ < π, Hasselmo 2005).
            theta_reset: True if theta cycle just completed.
                Forces theta pooling flush regardless of buffer length.
        """
        self._step_count += 1
        pattern = current_pattern.astype(np.float32)

        # Level 0: every step
        error0 = self.level0.observe(pattern)

        # Accumulate for theta pooling
        self._theta_buffer.append(pattern)

        # Flush on theta_reset or when buffer reaches dynamic window
        should_flush = theta_reset or (
            len(self._theta_buffer) >= self.theta_window
        )

        if should_flush and len(self._theta_buffer) > 0:
            pooled_theta = np.mean(
                np.stack(self._theta_buffer), axis=0,
            ).astype(np.float32)
            self._theta_buffer.clear()

            # Gate theta-level learning by encoding phase
            # Hasselmo (2005): encoding at theta peak (0 → π),
            # retrieval at theta trough (π → 2π).
            in_encoding_phase = True
            if theta_phase is not None:
                in_encoding_phase = theta_phase < np.pi

            if in_encoding_phase and salience >= self.salience_threshold:
                self.level1.observe(pooled_theta)

            # Accumulate for episode pooling
            self._episode_buffer.append(pooled_theta)
            if len(self._episode_buffer) >= self.episode_window:
                pooled_episode = np.mean(
                    np.stack(self._episode_buffer), axis=0,
                ).astype(np.float32)
                self._episode_buffer.clear()

                # Level 2: episode boundary
                self.level2.observe(pooled_episode)

        return error0

    def predict_next(self) -> NDArray[np.float32]:
        """Level 0 prediction (fastest timescale)."""
        return self.level0.predict_next()

    def novelty_signal(self) -> float:
        """Combined multi-scale novelty."""
        n0 = self.level0.novelty_signal()
        n1 = self.level1.novelty_signal()
        n2 = self.level2.novelty_signal()
        # Multi-scale weighting: faster oscillations carry more
        # immediate novelty (Lisman & Jensen 2013, θ-γ nesting).
        # Gamma (∼40Hz): 60%, theta (∼8Hz): 30%, episode (∼0.2Hz): 10%.
        return float(np.clip(0.6 * n0 + 0.3 * n1 + 0.1 * n2, 0.0, 1.0))

    def reset_state(self) -> None:
        """Reset transient state across all levels."""
        self.level0.reset_state()
        self.level1.reset_state()
        self.level2.reset_state()
        self._theta_buffer.clear()
        self._episode_buffer.clear()
        self._step_count = 0

    def reset_all(self) -> None:
        """Full reset including all learned weights."""
        self.level0.reset_all()
        self.level1.reset_all()
        self.level2.reset_all()
        self._theta_buffer.clear()
        self._episode_buffer.clear()
        self._step_count = 0
