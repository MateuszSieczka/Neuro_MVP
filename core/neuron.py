import numpy as np
from .config import LIFConfig


class LIFLayer:
    """
    Vectorized Leaky Integrate-and-Fire layer supporting asynchronous Eligibility Traces,
    Three-Factor STDP, exact exponential integration, and absolute refractory periods.
    """

    def __init__(self, num_inputs: int, num_neurons: int = 1, config: LIFConfig | None = None) -> None:
        self.config = config or LIFConfig()
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons

        # State variables
        self.v: np.ndarray = np.full(num_neurons, self.config.v_rest, dtype=np.float32)
        self.has_spiked: np.ndarray = np.zeros(num_neurons, dtype=bool)
        self.refrac_count: np.ndarray = np.zeros(num_neurons, dtype=np.int32)

        # Synaptic states and traces
        self.w: np.ndarray = np.random.uniform(0.1, 0.5, (num_inputs, num_neurons)).astype(np.float32)
        self.e: np.ndarray = np.zeros((num_inputs, num_neurons), dtype=np.float32)

        # Asynchronous pre/post traces for STDP correlation
        self.x_pre: np.ndarray = np.zeros(num_inputs, dtype=np.float32)
        self.x_post: np.ndarray = np.zeros(num_neurons, dtype=np.float32)

        # Pre-computed exact exponential decay factors
        self._mem_decay: float = np.exp(-self.config.dt / self.config.tau_m)
        self._trace_decay: float = np.exp(-self.config.dt / self.config.tau_e)
        self._pre_decay: float = np.exp(-self.config.dt / self.config.tau_pre)
        self._post_decay: float = np.exp(-self.config.dt / self.config.tau_post)

    def forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        """
        Advances the layer state by one timestep, resolving exact membrane
        integration, refractory periods, and trace correlations.

        Args:
            pre_spikes: 1D array of presynaptic spikes.

        Returns:
            1D boolean array indicating postsynaptic spikes.
        """
        # 1. Update transient pre/post traces
        self.x_pre *= self._pre_decay
        self.x_post *= self._post_decay

        pre_active = pre_spikes > 0
        self.x_pre[pre_active] += 1.0

        # 2. Refractory state management
        in_refrac = self.refrac_count > 0
        self.refrac_count[in_refrac] -= 1

        # 3. Exact exponential membrane integration
        injected_current = pre_spikes @ self.w
        integrated_v = self.v * self._mem_decay + (self.config.v_rest + injected_current) * (1.0 - self._mem_decay)

        # Apply integration only to neurons not in refractory period
        self.v = np.where(in_refrac, self.config.v_reset, integrated_v)

        # 4. Action potential generation
        self.has_spiked = (self.v >= self.config.v_thresh) & ~in_refrac

        # Reset and lock refractory period for spiked neurons
        self.v[self.has_spiked] = self.config.v_reset
        self.refrac_count[self.has_spiked] = self.config.refrac_period
        self.x_post[self.has_spiked] += 1.0

        # 5. Eligibility trace correlation (Asynchronous STDP)
        self.e *= self._trace_decay

        if np.any(self.has_spiked):
            self.e[:, self.has_spiked] += self.x_pre[:, np.newaxis]

        if np.any(pre_active):
            self.e[pre_active, :] += self.x_post[np.newaxis, :]

        return self.has_spiked

    def update_weights(self, m_t: float, pred_error: np.ndarray) -> None:
        """
        Executes weight updates based on the three-factor plasticity rule.

        Args:
            m_t: Modulatory precision signal.
            pred_error: 1D array representing prediction error.
        """
        if np.isclose(m_t, 0.0):
            return

        dw = self.config.learning_rate * m_t * self.e * pred_error
        self.w += dw

    def reset_state(self) -> None:
        """
        Resets transient neuron states between episodes.
        Synaptic weights are explicitly preserved.
        """
        self.v.fill(self.config.v_rest)
        self.e.fill(0.0)
        self.x_pre.fill(0.0)
        self.x_post.fill(0.0)
        self.refrac_count.fill(0)
        self.has_spiked.fill(False)