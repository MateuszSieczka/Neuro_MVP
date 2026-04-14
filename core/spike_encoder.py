import numpy as np


class GaussianPopulationEncoder:
    """
    Population coding via Gaussian receptive fields (place cells).

    Biological grounding (Pouget, Dayan & Zemel 2000; O'Keefe & Dostrovsky 1971):
      Cortical and hippocampal neurons encode continuous variables as
      distributed population activity. Each neuron has a preferred stimulus
      value (center) and a tuning width (sigma). The population firing
      pattern uniquely identifies the input value with much higher
      resolution than any single neuron.

    For each input dimension, `n_neurons_per_dim` Gaussian receptive fields
    are tiled uniformly across `[value_min, value_max]`. Given an input value
    x_i, neuron j fires at rate:

        r_j = exp(-(x_i - c_j)^2 / (2 * sigma^2))

    where c_j is the receptive field center and sigma is chosen so that
    adjacent fields overlap at ~60% peak (sigma = 0.8 * spacing).

    Output is a flat vector of shape (n_dims * n_neurons_per_dim,) with
    all values in [0, 1]. This is a deterministic, stateless mapping —
    no learned parameters, no internal state.
    """

    def __init__(
        self,
        n_dims: int,
        n_neurons_per_dim: int = 15,
        value_min: float = -1.0,
        value_max: float = 1.0,
    ) -> None:
        self.n_dims = n_dims
        self.n_neurons_per_dim = n_neurons_per_dim
        self.output_size = n_dims * n_neurons_per_dim

        # Receptive field centers: uniform tiling of the value range.
        # We extend slightly beyond [value_min, value_max] to ensure
        # boundary values still activate at least one neuron strongly.
        margin = 0.1 * (value_max - value_min)
        self._centers = np.linspace(
            value_min - margin, value_max + margin, n_neurons_per_dim
        ).astype(np.float32)  # (n_neurons_per_dim,)

        # Sigma: inter-center spacing × 0.5 maximises Fisher information
        # in small populations (Pouget et al. 2000).  0.8× was optimal
        # for 1000+ neurons; 15 neurons/dim needs sharper curves for
        # adequate state discrimination in RL tasks.
        spacing = self._centers[1] - self._centers[0] if n_neurons_per_dim > 1 else 1.0
        self._inv_2sigma2 = 1.0 / (2.0 * (0.5 * spacing) ** 2)

    def encode(self, values: np.ndarray) -> np.ndarray:
        """
        Encode a continuous vector into population firing rates.

        Args:
            values: 1D array of shape (n_dims,), arbitrary range
                    (values outside [value_min, value_max] gracefully
                    activate edge neurons at reduced rate).

        Returns:
            1D array of shape (n_dims * n_neurons_per_dim,), values in [0, 1].
        """
        # values: (n_dims,) → (n_dims, 1), centers: (n_neurons_per_dim,)
        v = values.astype(np.float32).reshape(-1, 1)
        # Gaussian tuning curves: (n_dims, n_neurons_per_dim)
        rates = np.exp(-((v - self._centers) ** 2) * self._inv_2sigma2)
        return rates.ravel().astype(np.float32)


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
