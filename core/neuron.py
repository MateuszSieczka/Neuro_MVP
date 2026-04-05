import numpy as np
from .config import LIFConfig, HomeostaticLIFConfig


class LIFLayer:
    """
    Vectorized Leaky Integrate-and-Fire layer.

    Supports:
      - Exact exponential membrane integration.
      - Absolute refractory periods.
      - Asynchronous eligibility traces (STDP correlation).
      - Three-factor STDP weight update (e × m_t × pred_error).
      - Optional homeostatic plasticity (HomeostaticLIFConfig):
          Per-neuron adaptive threshold tracks a slow exponential moving
          average of firing rate and nudges v_thresh toward config.target_rate.

    Homeostatic threshold:
      v_thresh_adaptive is the true threshold used for spike detection.
      It is initialized to config.v_thresh and evolves as:

        avg_rate[t+1]        = avg_rate[t] × α  +  has_spiked[t] × (1 − α)
        v_thresh_adaptive[t] += thresh_adapt_lr × (avg_rate[t] − target_rate)
        v_thresh_adaptive    = clip(v_thresh_adaptive, thresh_min, thresh_max)

      where α = exp(−dt / homeostatic_tau).

    Note: subclasses that manage their own threshold adaptation (e.g.
    PyramidalLayer) set self._homeostatic = False to suppress the parent's
    update and avoid double-correction.
    """

    def __init__(
        self,
        num_inputs: int,
        num_neurons: int = 1,
        config: LIFConfig | None = None,
    ) -> None:
        self.config = config or LIFConfig()
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons

        # ── Membrane state ────────────────────────────────────────────
        self.v: np.ndarray = np.full(num_neurons, self.config.v_rest, dtype=np.float32)
        self.has_spiked: np.ndarray = np.zeros(num_neurons, dtype=bool)
        self.refrac_count: np.ndarray = np.zeros(num_neurons, dtype=np.int32)

        # ── Synaptic weights and traces ───────────────────────────────
        self.w: np.ndarray = np.random.uniform(
            0.1, 0.5, (num_inputs, num_neurons)
        ).astype(np.float32)
        self.e: np.ndarray = np.zeros((num_inputs, num_neurons), dtype=np.float32)
        self.x_pre: np.ndarray = np.zeros(num_inputs, dtype=np.float32)
        self.x_post: np.ndarray = np.zeros(num_neurons, dtype=np.float32)

        # ── Pre-computed decay factors ────────────────────────────────
        self._mem_decay: float = np.exp(-self.config.dt / self.config.tau_m)
        self._trace_decay: float = np.exp(-self.config.dt / self.config.tau_e)
        self._pre_decay: float = np.exp(-self.config.dt / self.config.tau_pre)
        self._post_decay: float = np.exp(-self.config.dt / self.config.tau_post)

        # ── Homeostatic plasticity ─────────────────────────────────────
        # Enabled whenever config carries the HomeostaticLIFConfig fields.
        # Subclasses may set _homeostatic = False to suppress this layer's
        # update and manage v_thresh_adaptive themselves.
        self._homeostatic: bool = isinstance(self.config, HomeostaticLIFConfig)
        if self._homeostatic:
            self.v_thresh_adaptive: np.ndarray = np.full(
                num_neurons, self.config.v_thresh, dtype=np.float32
            )
            self.avg_rate: np.ndarray = np.zeros(num_neurons, dtype=np.float32)
            self._homeo_decay: float = np.exp(
                -self.config.dt / self.config.homeostatic_tau
            )

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        """
        One timestep: integrate, fire, update traces.

        Args:
            pre_spikes: 1D array of presynaptic spikes (num_inputs,).

        Returns:
            Boolean spike array (num_neurons,).
        """
        # 1. Trace decay
        self.x_pre *= self._pre_decay
        self.x_post *= self._post_decay

        pre_active = pre_spikes > 0
        self.x_pre[pre_active] += 1.0

        # 2. Refractory management
        in_refrac = self.refrac_count > 0
        self.refrac_count[in_refrac] -= 1

        # 3. Exact exponential membrane integration
        injected_current = pre_spikes.astype(np.float32) @ self.w
        integrated_v = (
            self.v * self._mem_decay
            + (self.config.v_rest + injected_current) * (1.0 - self._mem_decay)
        )
        self.v = np.where(in_refrac, self.config.v_reset, integrated_v)

        # 4. Spike detection — uses adaptive threshold if available
        thresh = (
            self.v_thresh_adaptive
            if self._homeostatic
            else np.float32(self.config.v_thresh)
        )
        self.has_spiked = (self.v >= thresh) & ~in_refrac

        # 5. Reset spiked neurons
        self.v[self.has_spiked] = self.config.v_reset
        self.refrac_count[self.has_spiked] = self.config.refrac_period
        self.x_post[self.has_spiked] += 1.0

        # 6. Eligibility trace correlation (asynchronous STDP)
        self.e *= self._trace_decay
        if np.any(self.has_spiked):
            self.e[:, self.has_spiked] += self.x_pre[:, np.newaxis]
        if np.any(pre_active):
            self.e[pre_active, :] += self.x_post[np.newaxis, :]

        # 7. Homeostatic threshold adaptation (if enabled and not externally managed)
        if self._homeostatic:
            self._update_homeostatic()

        return self.has_spiked

    # ------------------------------------------------------------------
    # Homeostatic plasticity
    # ------------------------------------------------------------------

    def _update_homeostatic(self) -> None:
        """
        Slow threshold adaptation toward target_rate.

        Called at the end of each forward pass when homeostatic mode is active.
        Should NOT be called if a subclass manages v_thresh_adaptive itself
        (guard: set self._homeostatic = False in the subclass __init__).
        """
        cfg = self.config  # type: HomeostaticLIFConfig

        # Exponential moving average of per-neuron firing rate
        self.avg_rate = (
            self.avg_rate * self._homeo_decay
            + self.has_spiked.astype(np.float32) * (1.0 - self._homeo_decay)
        )

        # Threshold correction: positive error → too active → raise threshold
        rate_error = self.avg_rate - cfg.target_rate
        self.v_thresh_adaptive += cfg.thresh_adapt_lr * rate_error
        np.clip(
            self.v_thresh_adaptive,
            cfg.thresh_min,
            cfg.thresh_max,
            out=self.v_thresh_adaptive,
        )

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def update_weights(self, m_t: float, pred_error: np.ndarray) -> None:
        """
        Three-factor STDP: Δw = lr × m_t × e × pred_error.

        Args:
            m_t:        Dopaminergic modulation (scalar).
            pred_error: Prediction error vector (num_neurons,).
        """
        if np.isclose(m_t, 0.0):
            return
        dw = self.config.learning_rate * m_t * self.e * pred_error
        self.w += dw

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset transient state between episodes. Weights are preserved."""
        self.v.fill(self.config.v_rest)
        self.e.fill(0.0)
        self.x_pre.fill(0.0)
        self.x_post.fill(0.0)
        self.refrac_count.fill(0)
        self.has_spiked.fill(False)

        if self._homeostatic:
            self.v_thresh_adaptive.fill(self.config.v_thresh)
            self.avg_rate.fill(0.0)