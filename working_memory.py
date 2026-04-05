import numpy as np
from config import WorkingMemoryConfig


class WorkingMemoryModule:
    """
    Persistent working memory via recurrent attractor dynamics.

    Biological grounding:
      - Prefrontal cortex maintains task-relevant representations through
        re-entrant loops among pyramidal neurons.
      - Slow membrane time constant (tau_m ≈ 300 ms) allows voltage to persist
        between sparse input events.
      - Hebbian lateral weights form "cliques" of co-active neurons; once a
        clique is seeded it sustains itself via recurrent excitation.

    Gating:
      - Acetylcholine (ACh) opens/closes the gate.
      - Gate OPEN  (ACh ≥ threshold): layer accepts external sensory input,
        updates content, and learns lateral associations.
      - Gate CLOSED (ACh < threshold): layer ignores external input and
        sustains current content through w_lateral alone.

    Weight convention:
      - w_ff:      (num_external_inputs × num_neurons) — feedforward from senses.
      - w_lateral: (num_neurons × num_neurons)         — recurrent attractor loop.
                   Diagonal is always zero (no self-excitation).
    """

    def __init__(
        self,
        num_external_inputs: int,
        num_neurons: int,
        config: WorkingMemoryConfig | None = None,
    ) -> None:
        self.config = config or WorkingMemoryConfig()
        self.num_neurons = num_neurons
        self.num_external_inputs = num_external_inputs

        # ── Membrane state ────────────────────────────────────────────
        self.v: np.ndarray = np.full(num_neurons, self.config.v_rest, dtype=np.float32)
        self.has_spiked: np.ndarray = np.zeros(num_neurons, dtype=bool)
        self.refrac_count: np.ndarray = np.zeros(num_neurons, dtype=np.int32)

        # ── Synaptic weights ──────────────────────────────────────────
        self.w_ff: np.ndarray = np.random.uniform(
            0.1, 0.5, (num_external_inputs, num_neurons)
        ).astype(np.float32)

        self.w_lateral: np.ndarray = np.zeros(
            (num_neurons, num_neurons), dtype=np.float32
        )  # Diagonal stays zero throughout learning

        # ── Eligibility traces (for feedforward path only) ────────────
        self.e: np.ndarray = np.zeros(
            (num_external_inputs, num_neurons), dtype=np.float32
        )
        self.x_pre: np.ndarray = np.zeros(num_external_inputs, dtype=np.float32)
        self.x_post: np.ndarray = np.zeros(num_neurons, dtype=np.float32)

        # ── Pre-computed exact exponential decay factors ───────────────
        self._mem_decay: float = np.exp(-self.config.dt / self.config.tau_m)
        self._trace_decay: float = np.exp(-self.config.dt / self.config.tau_e)
        self._pre_decay: float = np.exp(-self.config.dt / self.config.tau_pre)
        self._post_decay: float = np.exp(-self.config.dt / self.config.tau_post)

        # ── Gate and content ──────────────────────────────────────────
        self.gate_open: bool = False
        # Current held representation (float copy of last has_spiked)
        self.content: np.ndarray = np.zeros(num_neurons, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gating interface
    # ------------------------------------------------------------------

    def gate(self, ach_level: float) -> None:
        """
        Open or close the working memory gate.

        Args:
            ach_level: Acetylcholine level from NeuromodulatorSystem (0–1).
                       Gate opens when ach_level ≥ config.gate_threshold.
        """
        self.gate_open = float(ach_level) >= self.config.gate_threshold

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def forward(self, external_input: np.ndarray) -> np.ndarray:
        """
        One timestep of working memory dynamics.

        When OPEN:  integrates external_input through w_ff + recurrent via w_lateral.
        When CLOSED: ignores external_input; sustains content through w_lateral only.

        Args:
            external_input: Sensory or layer-below spike pattern (num_external_inputs,).

        Returns:
            Boolean spike array (num_neurons,).
        """
        # ── Trace decay ───────────────────────────────────────────────
        self.x_pre *= self._pre_decay
        self.x_post *= self._post_decay

        # ── Input current ─────────────────────────────────────────────
        if self.gate_open:
            external_current = external_input.astype(np.float32) @ self.w_ff
            pre_active = external_input > 0
            self.x_pre[pre_active] += 1.0
        else:
            # Gate closed: external path is suppressed
            external_current = np.zeros(self.num_neurons, dtype=np.float32)
            pre_active = np.zeros(self.num_external_inputs, dtype=bool)

        # Recurrent contribution is always active (attractor maintenance)
        recurrent_current = (
            self.content @ self.w_lateral * self.config.lateral_strength
        )
        total_current = external_current + recurrent_current

        # ── LIF integration ───────────────────────────────────────────
        in_refrac = self.refrac_count > 0
        self.refrac_count[in_refrac] -= 1

        integrated_v = self.v * self._mem_decay + (
            self.config.v_rest + total_current
        ) * (1.0 - self._mem_decay)

        self.v = np.where(in_refrac, self.config.v_reset, integrated_v)
        self.has_spiked = (self.v >= self.config.v_thresh) & ~in_refrac

        self.v[self.has_spiked] = self.config.v_reset
        self.refrac_count[self.has_spiked] = self.config.refrac_period
        self.x_post[self.has_spiked] += 1.0

        # ── Eligibility traces (feedforward path, gate-gated) ─────────
        if self.gate_open:
            self.e *= self._trace_decay
            if np.any(self.has_spiked):
                self.e[:, self.has_spiked] += self.x_pre[:, np.newaxis]
            if np.any(pre_active):
                self.e[pre_active, :] += self.x_post[np.newaxis, :]

        # ── Update content, then learn lateral connections ────────────
        self.content = self.has_spiked.astype(np.float32)
        self._update_lateral_weights()

        return self.has_spiked

    # ------------------------------------------------------------------
    # Lateral Hebbian learning
    # ------------------------------------------------------------------

    def _update_lateral_weights(self) -> None:
        """
        Hebbian co-activation rule: neurons that fire together wire together.

        Row-wise L∞ normalization prevents runaway lateral excitation
        (equivalent to Oja's rule in the limit).
        Self-connections (diagonal) are always kept at zero.
        """
        active = self.has_spiked.astype(np.float32)
        if np.sum(active) < 2:
            return

        dw = self.config.lateral_lr * np.outer(active, active)
        np.fill_diagonal(dw, 0.0)  # No self-excitation
        self.w_lateral += dw

        # Soft normalization: clip rows that exceed 1.0 maximum weight
        row_max = np.max(self.w_lateral, axis=1, keepdims=True)
        scale = np.where(row_max > 1.0, row_max, 1.0)
        self.w_lateral /= scale

        # Ensure diagonal stays zero after normalization
        np.fill_diagonal(self.w_lateral, 0.0)

    # ------------------------------------------------------------------
    # Feedforward weight update (three-factor rule, same as LIFLayer)
    # ------------------------------------------------------------------

    def update_weights(self, m_t: float, pred_error: np.ndarray) -> None:
        """
        Three-factor STDP update for feedforward weights.

        Args:
            m_t:        Dopaminergic modulation signal (scalar).
            pred_error: Prediction error vector (num_neurons,).
        """
        if np.isclose(m_t, 0.0):
            return
        dw = self.config.learning_rate * m_t * self.e * pred_error
        self.w_ff += dw

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """
        Reset transient neuron state between episodes.
        Learned weights (w_ff and w_lateral) are preserved.
        """
        self.v.fill(self.config.v_rest)
        self.e.fill(0.0)
        self.x_pre.fill(0.0)
        self.x_post.fill(0.0)
        self.refrac_count.fill(0)
        self.has_spiked.fill(False)
        self.content.fill(0.0)