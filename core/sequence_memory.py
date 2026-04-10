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

        self.transition_w: NDArray[np.float32] = np.zeros(
            (num_neurons, num_neurons), dtype=np.float32,
        )
        self.prev_pattern: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )
        self.predicted_next: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )
        self.temporal_error: NDArray[np.float32] = np.zeros(
            num_neurons, dtype=np.float32,
        )

    def observe(self, current_pattern: NDArray[np.float32]) -> NDArray[np.float32]:
        """Record pattern and learn transition from previous."""
        pattern = current_pattern.astype(np.float32)

        self.temporal_error = pattern - self.predicted_next

        if np.any(self.prev_pattern > 0) and np.any(pattern > 0):
            dw = self.config.learning_rate * np.outer(pattern, self.prev_pattern)
            self.transition_w += dw
            self.transition_w *= self.config.decay
            np.clip(
                self.transition_w, 0.0, self.config.max_weight,
                out=self.transition_w,
            )

        self.prev_pattern = pattern.copy()
        self.predicted_next = self._predict_from(self.prev_pattern)
        return self.temporal_error

    def predict_next(self) -> NDArray[np.float32]:
        """Predict next activation pattern."""
        return self.predicted_next.copy()

    def get_associated_neurons(
        self,
        neuron_index: int,
        threshold: float = 0.1,
    ) -> NDArray[np.int64]:
        """Neurons whose future activation is predicted by neuron_index."""
        weights_from = self.transition_w[:, neuron_index]
        return np.where(weights_from > threshold)[0]

    def get_temporal_clusters(
        self,
        threshold: float = 0.1,
    ) -> list[set[int]]:
        """Discover emergent concept clusters via bidirectional association."""
        n = self.num_neurons
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        symmetric = np.minimum(self.transition_w, self.transition_w.T)
        for i in range(n):
            for j in range(i + 1, n):
                if symmetric[i, j] > threshold:
                    union(i, j)

        clusters: dict[int, set[int]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, set()).add(i)

        return [c for c in clusters.values() if len(c) > 1]

    def novelty_signal(self) -> float:
        """Scalar novelty from temporal prediction error magnitude."""
        return float(np.clip(np.mean(np.abs(self.temporal_error)), 0.0, 1.0))

    def _predict_from(self, pattern: NDArray[np.float32]) -> NDArray[np.float32]:
        """Project pattern through transition_w to predict successor."""
        if not np.any(pattern > 0):
            return np.zeros(self.num_neurons, dtype=np.float32)
        raw = pattern @ self.transition_w.T
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
    """Multi-scale sequence memory with gamma/theta/episode timescales.

    Level 0 (gamma ~30ms): Raw spike transitions per timestep
    Level 1 (theta ~125ms): Phase-level pooled transitions
    Level 2 (episode ~seconds): Episode-level macro-transitions

    Each level pools transitions from the level below. Higher levels
    capture longer temporal structure. Salience-gated to avoid noise.
    """

    def __init__(
        self,
        num_neurons: int,
        config: SequenceMemoryConfig | None = None,
        salience_threshold: float = 0.5,
        theta_window: int = 8,
        episode_window: int = 50,
    ) -> None:
        cfg = config or SequenceMemoryConfig()
        self.num_neurons = num_neurons
        self.salience_threshold = salience_threshold
        self.theta_window = theta_window
        self.episode_window = episode_window

        # Level 0: gamma-scale raw transitions
        self.level0 = SequenceMemory(num_neurons, cfg)

        # Level 1: theta-scale pooled
        self.level1 = SequenceMemory(num_neurons, cfg)
        self._theta_buffer: list[NDArray[np.float32]] = []

        # Level 2: episode-scale
        self.level2 = SequenceMemory(num_neurons, cfg)
        self._episode_buffer: list[NDArray[np.float32]] = []

        self._step_count: int = 0

    def observe(
        self,
        current_pattern: NDArray[np.float32],
        salience: float = 0.0,
    ) -> NDArray[np.float32]:
        """Multi-scale observation with salience gating."""
        self._step_count += 1
        pattern = current_pattern.astype(np.float32)

        # Level 0: every step
        error0 = self.level0.observe(pattern)

        # Accumulate for theta pooling
        self._theta_buffer.append(pattern)
        if len(self._theta_buffer) >= self.theta_window:
            pooled_theta = np.mean(
                np.stack(self._theta_buffer), axis=0,
            ).astype(np.float32)
            self._theta_buffer.clear()

            # Level 1: theta boundary
            if salience >= self.salience_threshold:
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
