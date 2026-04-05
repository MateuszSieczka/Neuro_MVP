from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .spike_encoder import PoissonEncoder

if TYPE_CHECKING:
    from .neuromodulator import NeuromodulatorSystem
    from .predictive_coding import PredictiveCodingLayer
    from .sequence_memory import SequenceMemory


@dataclass
class LayerConnection:
    """Describes a directed connection between two named layers."""
    source: str
    target: str
    connection_type: str  # 'feedforward' or 'feedback'


class NetworkGraph:
    """
    Global orchestrator for a hierarchical SNN.

    This is the missing "glue" component identified in the architecture review.
    Without a central manager, there is no way to:
      - Synchronize timesteps across layers with different latencies.
      - Route spikes between layers with proper rate-to-spike conversion.
      - Distribute neuromodulatory signals globally.
      - Propagate local prediction errors for credit assignment.

    Responsibilities:
      1. Maintain a registry of named layers.
      2. Manage feedforward (bottom-up) and feedback (top-down) connections.
      3. Synchronize a global timestep (tick) across all layers.
      4. Route spikes between layers, using Poisson encoding on feedback paths
         where rate-coded predictions must be converted to spikes.
      5. Distribute neuromodulatory signals (ACh, DA) to all layers.
      6. Collect per-layer prediction errors for local credit assignment.

    Credit assignment strategy:
      Each PredictiveCodingLayer computes its own local prediction error.
      The NetworkGraph collects these errors and uses them (together with
      the dopaminergic signal from NeuromodulatorSystem) to drive three-factor
      STDP updates at each layer independently.  This avoids the need for
      backpropagation while still assigning credit in deep hierarchies.
    """

    def __init__(self, dt: float = 1.0) -> None:
        self.dt = dt
        self.timestep: int = 0

        self._layers: dict[str, PredictiveCodingLayer] = {}
        self._order: list[str] = []  # bottom-up processing order
        self._connections: list[LayerConnection] = []
        self._encoder = PoissonEncoder()

        # Optional SequenceMemory per layer (for temporal pattern tracking)
        self._sequence_memories: dict[str, SequenceMemory] = {}

    # ------------------------------------------------------------------
    # Layer registration
    # ------------------------------------------------------------------

    def add_layer(self, name: str, layer: PredictiveCodingLayer) -> None:
        """
        Register a layer under a unique name.

        Layers are processed in registration order during the feedforward pass,
        so register bottom layers (closer to sensory input) first.
        """
        if name in self._layers:
            raise ValueError(f"Layer '{name}' already registered.")
        self._layers[name] = layer
        self._order.append(name)

    def connect(
        self,
        source: str,
        target: str,
        connection_type: str = "feedforward",
    ) -> None:
        """
        Add a directional connection between two registered layers.

        Args:
            source: Name of the source layer.
            target: Name of the target layer.
            connection_type: 'feedforward' (bottom-up) or 'feedback' (top-down).
        """
        for name in (source, target):
            if name not in self._layers:
                raise ValueError(f"Layer '{name}' not registered.")
        if connection_type not in ("feedforward", "feedback"):
            raise ValueError(f"Unknown connection type: '{connection_type}'.")
        self._connections.append(
            LayerConnection(source, target, connection_type)
        )

    def attach_sequence_memory(
        self, layer_name: str, seq_mem: SequenceMemory,
    ) -> None:
        """Attach a SequenceMemory to a named layer for temporal tracking."""
        if layer_name not in self._layers:
            raise ValueError(f"Layer '{layer_name}' not registered.")
        self._sequence_memories[layer_name] = seq_mem

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_layer(self, name: str) -> PredictiveCodingLayer:
        """Retrieve a registered layer by name."""
        return self._layers[name]

    @property
    def layer_names(self) -> list[str]:
        """Layer names in bottom-up processing order."""
        return list(self._order)

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def step(
        self,
        sensory_inputs: dict[str, np.ndarray],
        neuromodulator: NeuromodulatorSystem | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Advance the entire network by one timestep.

        Pipeline:
          1. Distribute neuromodulator signals (ACh) to all PC layers.
          2. Feedback pass: higher layers send predictions to lower layers
             (predictions are rate-coded; receiving layers handle conversion).
          3. Feedforward pass: each layer receives sensory input or spikes
             from the layer below, in registration order.
          4. Sequence memory observation (if attached).
          5. Increment global timestep.

        Args:
            sensory_inputs: Dict mapping input layer name → spike array.
            neuromodulator: Optional NeuromodulatorSystem for global modulation.

        Returns:
            Dict mapping layer name → output spike array from this timestep.
        """
        outputs: dict[str, np.ndarray] = {}

        # 1. Distribute neuromodulation
        if neuromodulator is not None:
            for layer in self._layers.values():
                if hasattr(layer, "set_ach_level"):
                    layer.set_ach_level(neuromodulator.bottom_up_gain)

        # 2. Feedback pass (top-down predictions)
        #    Shape guard: only deliver if the prediction dimensionality
        #    matches the target layer's input size.
        for conn in self._connections:
            if conn.connection_type == "feedback":
                src = self._layers[conn.source]
                tgt = self._layers[conn.target]
                if hasattr(src, "generate_prediction") and hasattr(
                    tgt, "receive_prediction"
                ):
                    prediction = src.generate_prediction()
                    if prediction.shape[0] == tgt.num_inputs:
                        tgt.receive_prediction(prediction)

        # 3. Feedforward pass (bottom-up processing)
        for name in self._order:
            layer = self._layers[name]

            if name in sensory_inputs:
                input_spikes = sensory_inputs[name]
            else:
                input_spikes = self._resolve_feedforward_input(name, outputs)

            outputs[name] = layer.forward(input_spikes)

            # 4. Sequence memory observation
            if name in self._sequence_memories:
                self._sequence_memories[name].observe(outputs[name])

        self.timestep += 1
        return outputs

    def update_weights(
        self,
        neuromodulator: NeuromodulatorSystem,
    ) -> None:
        """
        Apply three-factor STDP updates to all layers using local prediction errors.

        Each layer's input-space prediction_error is projected through the
        forward weight matrix to produce an output-space (num_neurons,)
        error signal suitable for the three-factor STDP rule.
        This is a form of error feedback alignment that preserves the
        impulsive nature of the network (no backpropagation).

        Args:
            neuromodulator: Source of the dopaminergic modulation signal (m_t).
        """
        m_t = neuromodulator.learning_rate_modulation
        for layer in self._layers.values():
            if hasattr(layer, "prediction_error") and hasattr(
                layer, "update_weights"
            ):
                # Project input-space error → output-space error
                # (num_inputs,) @ (num_inputs, num_neurons) → (num_neurons,)
                output_error = layer.prediction_error @ layer.w
                layer.update_weights(m_t=m_t, pred_error=output_error)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_feedforward_input(
        self, target_name: str, outputs: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Find the feedforward source for a layer and return its output spikes."""
        for conn in self._connections:
            if (
                conn.target == target_name
                and conn.connection_type == "feedforward"
                and conn.source in outputs
            ):
                return outputs[conn.source]

        # No feedforward source found — return silence
        layer = self._layers[target_name]
        return np.zeros(layer.num_inputs, dtype=np.float32)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset all layers, sequence memories, and the global timestep."""
        for layer in self._layers.values():
            if hasattr(layer, "reset_state"):
                layer.reset_state()
        for sm in self._sequence_memories.values():
            sm.reset_state()
        self.timestep = 0
