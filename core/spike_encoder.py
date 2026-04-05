import numpy as np


class PoissonEncoder:
    """
    Converts continuous rate values in [0, 1] to binary Poisson spike trains.

    Each element in the rate vector is interpreted as a firing probability
    per timestep (dt).  A uniform random sample determines whether each
    element produces a spike (1) or not (0).

    This is the standard bridge between rate-coded signals (e.g., top-down
    predictions, sensor readings) and spike-based SNN processing.  Without
    this conversion, continuous values corrupt the STDP trace bookkeeping
    inside LIFLayer (which expects binary pre-synaptic events).
    """

    def encode(self, rates: np.ndarray) -> np.ndarray:
        """
        Generate one timestep of Poisson spikes from a rate vector.

        Boundary guarantees:
          - rate = 0.0  → always returns 0  (deterministic silence)
          - rate = 1.0  → always returns 1  (deterministic spike)
          - 0 < rate < 1 → stochastic spike with P(spike) = rate

        Args:
            rates: Array of firing probabilities, values will be clipped to [0, 1].

        Returns:
            Binary spike array of same shape, dtype float32.
        """
        rates_clipped = np.clip(rates, 0.0, 1.0)
        return (np.random.rand(*rates_clipped.shape) < rates_clipped).astype(np.float32)

    def encode_value(self, values: np.ndarray, max_rate: float = 1.0) -> np.ndarray:
        """
        Normalize arbitrary values to [0, max_rate] then generate Poisson spikes.

        Useful for raw sensor data that is not pre-normalized.

        Args:
            values: Raw sensor/state values (any range).
            max_rate: Maximum firing probability after normalization (default 1.0).

        Returns:
            Binary spike array, dtype float32.
        """
        if values.size == 0:
            return values.astype(np.float32)

        v_min, v_max = float(values.min()), float(values.max())
        if np.isclose(v_min, v_max):
            rates = np.full_like(values, max_rate * 0.5, dtype=np.float32)
        else:
            rates = ((values - v_min) / (v_max - v_min) * max_rate).astype(np.float32)

        return self.encode(rates)
