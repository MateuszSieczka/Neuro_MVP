import numpy as np

from .config import SequenceMemoryConfig


class SequenceMemory:
    """
    Temporal sequence learning via spike-timing-dependent transition weights.

    Learns which activation patterns tend to follow which other patterns,
    enabling:
      - Prediction of the next pattern in a learned sequence.
      - Discovery of temporal co-occurrences (e.g., "key" pattern precedes
        "door opens" pattern → emergent concept association).
      - Anomaly detection when an unexpected successor appears.

    The transition matrix W[i, j] represents how strongly neuron j's
    activation at time t predicts neuron i's activation at time t+1.

    Learning rule (temporal Hebbian):
        dW[i, j] = lr * post_i(t) * pre_j(t-1)

    This is the missing component identified in the architecture review:
    k-WTA alone creates Sparse Distributed Representations but cannot
    discover *temporal* structure (e.g., "key always precedes door-open").
    SequenceMemory fills this gap by learning transition probabilities
    between SDR patterns, allowing unnamed concepts to emerge as
    tightly-coupled temporal clusters.
    """

    def __init__(
        self,
        num_neurons: int,
        config: SequenceMemoryConfig | None = None,
    ) -> None:
        self.config = config or SequenceMemoryConfig()
        self.num_neurons = num_neurons

        # Transition weights: prev_pattern → predicted next_pattern
        self.transition_w: np.ndarray = np.zeros(
            (num_neurons, num_neurons), dtype=np.float32
        )

        # Previous activation pattern (for temporal Hebbian update)
        self.prev_pattern: np.ndarray = np.zeros(num_neurons, dtype=np.float32)

        # Predicted next pattern (generated before observing the actual next)
        self.predicted_next: np.ndarray = np.zeros(num_neurons, dtype=np.float32)

        # Temporal prediction error
        self.temporal_error: np.ndarray = np.zeros(num_neurons, dtype=np.float32)

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def observe(self, current_pattern: np.ndarray) -> np.ndarray:
        """
        Record an activation pattern and learn the transition from previous.

        Args:
            current_pattern: Binary spike pattern or float activation (num_neurons,).

        Returns:
            Temporal prediction error (current − predicted).
        """
        pattern = current_pattern.astype(np.float32)

        # 1. Prediction error: what was unexpected in this transition?
        self.temporal_error = pattern - self.predicted_next

        # 2. Temporal Hebbian learning: dW = lr * outer(post_t, pre_{t-1})
        if np.any(self.prev_pattern > 0) and np.any(pattern > 0):
            dw = self.config.learning_rate * np.outer(pattern, self.prev_pattern)
            self.transition_w += dw
            self.transition_w *= self.config.decay
            np.clip(
                self.transition_w, 0.0, self.config.max_weight,
                out=self.transition_w,
            )

        # 3. Advance state for next timestep
        self.prev_pattern = pattern.copy()
        self.predicted_next = self._predict_from(self.prev_pattern)

        return self.temporal_error

    def predict_next(self) -> np.ndarray:
        """
        Predict the next activation pattern based on current state.

        Returns:
            Predicted activation rates (num_neurons,), values in [0, 1].
        """
        return self.predicted_next.copy()

    # ------------------------------------------------------------------
    # Concept / cluster discovery
    # ------------------------------------------------------------------

    def get_associated_neurons(
        self, neuron_index: int, threshold: float = 0.1,
    ) -> np.ndarray:
        """
        Find neurons whose future activation is predicted by *neuron_index*.

        A neuron j is associated with neuron i if transition_w[j, i] > threshold
        (neuron i active at t → neuron j likely active at t+1).

        Args:
            neuron_index: Index of the query neuron.
            threshold: Minimum transition weight to count as associated.

        Returns:
            Array of associated neuron indices.
        """
        weights_from = self.transition_w[:, neuron_index]
        return np.where(weights_from > threshold)[0]

    def get_temporal_clusters(self, threshold: float = 0.1) -> list[set[int]]:
        """
        Discover emergent concept clusters via bidirectional temporal association.

        Two neurons belong to the same cluster if they *mutually* predict each
        other (transition_w[i,j] > threshold AND transition_w[j,i] > threshold).
        Transitive associations are merged via union-find.

        This is the mechanism for concept emergence: neurons that consistently
        co-occur in temporal proximity form a cluster — an unnamed but real
        internal concept (e.g., the "key-door" association in MiniGrid).

        Returns:
            List of sets, each containing neuron indices in a temporal cluster.
            Only non-trivial clusters (size ≥ 2) are returned.
        """
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

    # ------------------------------------------------------------------
    # Novelty
    # ------------------------------------------------------------------

    def novelty_signal(self) -> float:
        """
        Scalar novelty based on temporal prediction error magnitude.

        Returns:
            Float in [0, 1].
        """
        return float(np.clip(np.mean(np.abs(self.temporal_error)), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_from(self, pattern: np.ndarray) -> np.ndarray:
        """Project a pattern through transition_w to predict the successor."""
        if not np.any(pattern > 0):
            return np.zeros(self.num_neurons, dtype=np.float32)
        raw = pattern @ self.transition_w.T
        return np.clip(raw, 0.0, 1.0)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset transient state. Learned transition weights are preserved."""
        self.prev_pattern.fill(0.0)
        self.predicted_next.fill(0.0)
        self.temporal_error.fill(0.0)

    def reset_all(self) -> None:
        """Full reset including learned weights."""
        self.reset_state()
        self.transition_w.fill(0.0)


class HierarchicalSequenceMemory(SequenceMemory):
    """
    Ignoruje mikro-kroki; rejestruje przejścia stanów tylko w momentach
    o wysokim natężeniu istotności (np. szczyty błędu / uderzenia Noradrenaliny).
    """

    def __init__(
            self,
            num_neurons: int,
            config: SequenceMemoryConfig | None = None,
            salience_threshold: float = 0.5
    ) -> None:
        super().__init__(num_neurons, config)
        self.salience_threshold = salience_threshold

    def observe(self, current_pattern: np.ndarray, salience: float = 0.0) -> np.ndarray:
        """
        Ocenia krok pod kątem wagi informacyjnej przed wykonaniem Hebbian Update.
        """
        if salience >= self.salience_threshold:
            return super().observe(current_pattern)

        # Jeśli szum: brak błędu temporalnego, stan wewnętrzny bez zmian.
        return np.zeros(self.num_neurons, dtype=np.float32)