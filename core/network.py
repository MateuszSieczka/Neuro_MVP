"""
NetworkGraph — hierarchical multimodal SNN orchestrator.

Provides:
  1. Parallel hierarchies with topological sort (Kahn's algorithm).
  2. Multi-source feedforward aggregation (sum or concat).
  3. Precision-weighted top-down feedback (Friston 2010).
  4. Theta-gamma oscillator integration — gamma paces k-WTA,
     theta gates episodic encoding, phase resets propagate globally.
  5. Bottom-up prediction-error collection for spatial attention.
  6. Axonal delay lines for cross-modal temporal binding.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from .config import OscillatorConfig
from .oscillator import ThetaGammaOscillator
from .simulation_context import SimulationContext, DEFAULT_CONTEXT
from .spike_encoder import PoissonEncoder

if TYPE_CHECKING:
    from .attention import SpatialAttentionController
    from .neuromodulator import NeuromodulatorSystem
    from .sequence_memory import HierarchicalSequenceMemory, SequenceMemory

# Duck-typed layer protocol: must expose forward(input_spikes) -> NDArray,
# num_inputs: int, num_neurons: int.  Optional: set_ach_level, set_ne_level,
# prediction_error, receive_prediction, generate_prediction, update_weights,
# trigger_phase_reset, reset_state.


# ------------------------------------------------------------------
# Connection descriptor
# ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LayerConnection:
    """Directed connection between two named layers.

    Attributes:
        source, target:        Registered layer names.
        connection_type:       'feedforward' (bottom-up) or 'feedback' (top-down).
        aggregation_mode:      'sum' (element-wise) or 'concat' (feature axis).
        weight:                Per-connection scalar scaling.
        delay:                 Axonal delay in timesteps (0 = same step).
    """

    source: str
    target: str
    connection_type: str
    aggregation_mode: Literal["sum", "concat"] = "sum"
    weight: float = 1.0
    delay: int = 0


# ------------------------------------------------------------------
# NetworkGraph
# ------------------------------------------------------------------

class NetworkGraph:
    """Global orchestrator for a hierarchical, multimodal SNN.

    Key features:
      - Parallel hierarchies with topological sort.
      - Multi-source feedforward aggregation (sum / concat).
      - Precision-weighted feedback (inverse PE variance).
      - Theta-gamma oscillator drives phase resets and gamma amplitude.
      - Bottom-up prediction errors forwarded to spatial attention.
      - Axonal delay lines for synchronized multimodal binding.
    """

    def __init__(
        self,
        osc_config: OscillatorConfig | None = None,
        ctx: SimulationContext | None = None,
    ) -> None:
        self.ctx = ctx or DEFAULT_CONTEXT

        self._layers: dict[str, Any] = {}
        self._order: list[str] = []
        self._order_dirty: bool = True
        self._connections: list[LayerConnection] = []
        self._encoder = PoissonEncoder()

        self._sequence_memories: dict[
            str, SequenceMemory | HierarchicalSequenceMemory
        ] = {}

        # Layers updated via TD error directly (skip in update_weights)
        self._td_updated_layers: set[str] = set()

        self.oscillator = ThetaGammaOscillator(
            config=osc_config or OscillatorConfig(), ctx=self.ctx,
        )

        self.timestep: int = 0

        # Delay buffer for axonal propagation
        self._output_history: dict[str, deque[NDArray[np.float32]]] = {}
        self._max_delay: int = 0

        # Precomputed concat offsets per (source, target) pair
        self._concat_offsets: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    # Layer registration
    # ------------------------------------------------------------------

    def add_layer(self, name: str, layer: Any) -> None:
        """Register a layer under a unique name.

        Layer must implement:
          - forward(input_spikes: NDArray) -> NDArray
          - num_inputs: int
          - num_neurons: int
        """
        if name in self._layers:
            raise ValueError(f"Layer '{name}' already registered.")
        self._layers[name] = layer
        self._order.append(name)
        self._order_dirty = True

    def mark_td_updated(self, *layer_names: str) -> None:
        """Mark layers that receive TD-based updates directly.

        These layers are skipped during update_weights() to avoid
        double-updating (BG critic/actor get DA-modulated STDP separately).
        """
        for name in layer_names:
            if name not in self._layers:
                raise ValueError(f"Layer '{name}' not registered.")
            self._td_updated_layers.add(name)

    def connect(
        self,
        source: str,
        target: str,
        connection_type: str = "feedforward",
        aggregation_mode: Literal["sum", "concat"] = "sum",
        weight: float = 1.0,
        delay: int = 0,
    ) -> None:
        """Add a directional connection between two registered layers."""
        for name in (source, target):
            if name not in self._layers:
                raise ValueError(f"Layer '{name}' not registered.")
        if connection_type not in ("feedforward", "feedback"):
            raise ValueError(f"Unknown connection type: '{connection_type}'.")
        if aggregation_mode not in ("sum", "concat"):
            raise ValueError(f"Unknown aggregation mode: '{aggregation_mode}'.")
        if delay < 0:
            raise ValueError("Delay cannot be negative.")

        self._connections.append(
            LayerConnection(source, target, connection_type, aggregation_mode, weight, delay),
        )
        self._max_delay = max(self._max_delay, delay)
        self._order_dirty = True

    def attach_sequence_memory(
        self,
        layer_name: str,
        seq_mem: SequenceMemory | HierarchicalSequenceMemory,
    ) -> None:
        """Attach a SequenceMemory to a named layer for temporal tracking."""
        if layer_name not in self._layers:
            raise ValueError(f"Layer '{layer_name}' not registered.")
        self._sequence_memories[layer_name] = seq_mem

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_layer(self, name: str) -> Any:
        return self._layers[name]

    @property
    def layer_names(self) -> list[str]:
        if self._order_dirty:
            self._refresh_order()
        return list(self._order)

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def step(
        self,
        sensory_inputs: dict[str, NDArray[np.float32]],
        neuromodulator: NeuromodulatorSystem | None = None,
        attention: SpatialAttentionController | None = None,
    ) -> dict[str, NDArray[np.float32]]:
        """Advance the entire network by one timestep.

        Pipeline:
          1. Topological sort (if connections changed).
          2. Oscillator tick — gamma/theta phase advance.
          3. Neuromodulator distribution (ACh, NE to layers).
          4. Spatial attention gain application.
          5. Global phase-reset broadcast (gamma cycle boundary).
          6. Feedback pass — precision-weighted top-down predictions.
          7. Feedforward pass — bottom-up spike propagation.
          8. Sequence memory observation (salience-gated).
          9. Attention update (bottom-up PE + Hebbian reinforcement).
         10. Increment timestep.

        Returns:
            Dict mapping layer name -> output spike array.
        """
        if self._order_dirty:
            self._refresh_order()

        outputs: dict[str, NDArray[np.float32]] = {}
        salience_signal: float = 0.0
        ne_level: float = 0.0
        sero_level: float = 0.0

        # ── 2. Oscillator tick ────────────────────────────────────────
        if neuromodulator is not None:
            ne_level = neuromodulator.competition_sharpness
            sero_level = neuromodulator.planning_horizon
            salience_signal = neuromodulator.noradrenaline

        gamma_reset, theta_reset = self.oscillator.tick(
            ne_level=ne_level,
            sero_level=sero_level,
        )

        # ── 3. Distribute neuromodulator signals ──────────────────────
        if neuromodulator is not None:
            for layer in self._layers.values():
                if hasattr(layer, "set_ach_level"):
                    layer.set_ach_level(neuromodulator.bottom_up_gain)
                if hasattr(layer, "set_ne_level"):
                    layer.set_ne_level(neuromodulator.competition_sharpness)
                if hasattr(layer, "set_neuromodulators"):
                    layer.set_neuromodulators(
                        ne=neuromodulator.competition_sharpness,
                        sero=neuromodulator.planning_horizon,
                    )

        # ── 4. Spatial attention gains ────────────────────────────────
        if attention is not None:
            for name in attention.column_names:
                layer = self._layers.get(name)
                if layer is not None and hasattr(layer, "set_attention_gain"):
                    gain = attention.column_gains.get(name, 1.0)
                    layer.set_attention_gain(gain)

        # ── 5. Phase-reset broadcast ──────────────────────────────────
        if gamma_reset:
            for layer in self._layers.values():
                if hasattr(layer, "trigger_phase_reset"):
                    layer.trigger_phase_reset()

        # ── 6. Feedback pass — precision-weighted top-down predictions ─
        for conn in self._connections:
            if conn.connection_type != "feedback":
                continue
            src = self._layers[conn.source]
            tgt = self._layers[conn.target]

            if not (
                hasattr(src, "generate_prediction")
                and hasattr(tgt, "receive_prediction")
            ):
                continue

            prediction: NDArray[np.float32] = src.generate_prediction()

            # Precision weighting: scale prediction by inverse PE variance
            if hasattr(src, "prediction_error"):
                pe = src.prediction_error
                pe_var = float(np.var(pe)) + 1e-8
                precision = min(1.0 / pe_var, 10.0)
                prediction = prediction * np.float32(np.clip(precision, 0.1, 10.0))

            # Size matching for concat targets
            if prediction.shape[0] == tgt.num_neurons:
                tgt.receive_prediction(prediction)
            elif prediction.shape[0] > tgt.num_neurons:
                offset = self._concat_offsets.get(
                    (conn.target, conn.source), 0,
                )
                sliced = prediction[offset : offset + tgt.num_neurons]
                tgt.receive_prediction(sliced)

        # ── 7. Feedforward pass (bottom-up) ───────────────────────────
        per_column_pe: dict[str, float] = {}

        for name in self._order:
            layer = self._layers[name]

            ff_spikes = self._aggregate_feedforward_inputs(name, outputs)

            if name in sensory_inputs:
                sensory = sensory_inputs[name].astype(np.float32)
                input_spikes = np.clip(ff_spikes + sensory, 0.0, 1.0)
            else:
                input_spikes = ff_spikes

            outputs[name] = layer.forward(input_spikes)

            # Collect per-column prediction error for bottom-up attention
            if hasattr(layer, "prediction_error"):
                pe_mag = float(np.mean(np.abs(layer.prediction_error)))
                per_column_pe[name] = pe_mag

            # 8. Sequence memory observation
            if name in self._sequence_memories:
                seq_mem = self._sequence_memories[name]
                if hasattr(seq_mem, "salience_threshold"):
                    seq_mem.observe(outputs[name], salience=salience_signal)
                else:
                    seq_mem.observe(outputs[name])

        # Buffer outputs for delay lines
        for name, spike_array in outputs.items():
            if name not in self._output_history:
                self._output_history[name] = deque(
                    maxlen=max(1, self._max_delay + 1),
                )
            self._output_history[name].append(spike_array.copy())

        # ── 9. Attention update ───────────────────────────────────────
        if attention is not None:
            # Find association layer output
            assoc_out: NDArray[np.float32] | None = None
            for conn in self._connections:
                if (
                    conn.connection_type == "feedforward"
                    and conn.source in attention.column_names
                ):
                    assoc_out = outputs.get(conn.target)
                    break

            if assoc_out is None:
                assoc_out = outputs.get(
                    attention.column_names[0],
                    np.zeros(1, dtype=np.float32),
                )

            # Build bottom-up PE vector aligned with column order
            bu_errors: NDArray[np.float32] | None = None
            if per_column_pe:
                bu_errors = np.array(
                    [per_column_pe.get(n, 0.0) for n in attention.column_names],
                    dtype=np.float32,
                )

            global_ach = 0.5
            if neuromodulator is not None:
                global_ach = neuromodulator.bottom_up_gain

            attention.compute(
                assoc_out,
                global_ach=global_ach,
                ne_level=ne_level,
                bottom_up_errors=bu_errors,
            )

            col_acts = {
                n: outputs[n] for n in attention.column_names if n in outputs
            }
            attention.update(assoc_out, col_acts)

        # ── 10. Advance timestep ──────────────────────────────────────
        self.timestep += 1
        return outputs

    def update_weights(self, neuromodulator: NeuromodulatorSystem) -> None:
        """Apply three-factor plasticity to all layers.

        Plasticity signal combines DA direction with NE gating window.
        Layers marked via mark_td_updated() are skipped (they receive
        TD-error-based updates directly from the agent).
        """
        plasticity_signal: float = neuromodulator.learning_rate_modulation

        for name, layer in self._layers.items():
            if name in self._td_updated_layers:
                continue
            if hasattr(layer, "update_weights"):
                pe = (
                    layer.prediction_error
                    if hasattr(layer, "prediction_error")
                    else np.zeros(1, dtype=np.float32)
                )
                layer.update_weights(m_t=plasticity_signal, pred_error=pe)

    # ------------------------------------------------------------------
    # Delay buffer
    # ------------------------------------------------------------------

    def _get_delayed_output(
        self,
        source_name: str,
        delay: int,
        current_outputs: dict[str, NDArray[np.float32]],
    ) -> NDArray[np.float32] | None:
        """Retrieve delayed output for synchronized multimodal binding."""
        if delay == 0:
            return current_outputs.get(source_name)

        history = self._output_history.get(source_name)
        if not history or len(history) <= delay:
            return None

        return history[-(delay + 1)]

    # ------------------------------------------------------------------
    # Feedforward input aggregation
    # ------------------------------------------------------------------

    def _aggregate_feedforward_inputs(
        self,
        target_name: str,
        outputs: dict[str, NDArray[np.float32]],
    ) -> NDArray[np.float32]:
        """Collect and aggregate all feedforward inputs for *target_name*.

        Aggregation modes:
          'sum':    Weighted element-wise sum (all sources same width as target).
          'concat': Concatenation along feature axis.
        """
        target_layer = self._layers[target_name]
        num_inputs: int = target_layer.num_inputs

        active_sources: list[LayerConnection] = []
        delayed_outputs: dict[str, NDArray[np.float32]] = {}

        for conn in self._connections:
            if conn.target == target_name and conn.connection_type == "feedforward":
                delayed_out = self._get_delayed_output(
                    conn.source, conn.delay, outputs,
                )
                if delayed_out is not None:
                    active_sources.append(conn)
                    delayed_outputs[conn.source] = delayed_out

        if not active_sources:
            return np.zeros(num_inputs, dtype=np.float32)

        if len(active_sources) == 1:
            conn = active_sources[0]
            src_out = delayed_outputs[conn.source].astype(np.float32)
            return self._fit_to_size(src_out * conn.weight, num_inputs)

        mode = active_sources[0].aggregation_mode
        if mode == "concat":
            parts = [
                delayed_outputs[c.source].astype(np.float32) * c.weight
                for c in active_sources
            ]
            return self._fit_to_size(np.concatenate(parts), num_inputs)
        else:
            result = np.zeros(num_inputs, dtype=np.float32)
            for conn in active_sources:
                src_out = (
                    delayed_outputs[conn.source].astype(np.float32) * conn.weight
                )
                size = min(len(src_out), num_inputs)
                result[:size] += src_out[:size]
            return result

    @staticmethod
    def _fit_to_size(arr: NDArray[np.float32], target_size: int) -> NDArray[np.float32]:
        """Pad or truncate *arr* to exactly *target_size* elements."""
        n = len(arr)
        if n == target_size:
            return arr
        if n > target_size:
            return arr[:target_size]
        out = np.zeros(target_size, dtype=arr.dtype)
        out[:n] = arr
        return out

    # ------------------------------------------------------------------
    # Topological ordering (Kahn's BFS)
    # ------------------------------------------------------------------

    def _refresh_order(self) -> None:
        """Topologically sort layers based on feedforward connections."""
        in_degree: dict[str, int] = {name: 0 for name in self._layers}
        children: dict[str, list[str]] = {name: [] for name in self._layers}

        for conn in self._connections:
            if conn.connection_type == "feedforward":
                children[conn.source].append(conn.target)
                in_degree[conn.target] += 1

        queue: deque[str] = deque(
            name for name, deg in in_degree.items() if deg == 0
        )
        sorted_order: list[str] = []

        while queue:
            node = queue.popleft()
            sorted_order.append(node)
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(sorted_order) != len(self._layers):
            sorted_order = list(self._layers.keys())

        self._order = sorted_order
        self._order_dirty = False

        # Precompute concat offsets
        self._concat_offsets.clear()
        for target_name in self._layers:
            current_offset = 0
            for conn in self._connections:
                if (
                    conn.target == target_name
                    and conn.connection_type == "feedforward"
                    and conn.aggregation_mode == "concat"
                ):
                    src_layer = self._layers[conn.source]
                    self._concat_offsets[
                        (conn.source, conn.target)
                    ] = current_offset
                    current_offset += src_layer.num_neurons

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset all layers, sequence memories, timestep, and delay buffers."""
        for layer in self._layers.values():
            if hasattr(layer, "reset_state"):
                layer.reset_state()
        for sm in self._sequence_memories.values():
            sm.reset_state()
        self.timestep = 0
        self._output_history.clear()
        self.oscillator.reset()
