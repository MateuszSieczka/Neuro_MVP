from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from .oscillator import GlobalOscillator
from .spike_encoder import PoissonEncoder

if TYPE_CHECKING:
    from .neuromodulator import NeuromodulatorSystem
    from .predictive_coding import PredictiveCodingLayer
    from .sequence_memory import SequenceMemory, HierarchicalSequenceMemory


# ------------------------------------------------------------------
# Connection descriptor
# ------------------------------------------------------------------

@dataclass
class LayerConnection:
    """
    Directed connection between two named layers.

    Fields
    ------
    source, target      : registered layer names
    connection_type     : 'feedforward' (bottom-up) or 'feedback' (top-down)
    aggregation_mode    : how to combine when multiple sources feed one target
                            'sum'    — element-wise addition; all sources must
                                       have the same width as target.num_inputs.
                            'concat' — concatenation along the feature axis;
                                       target.num_inputs must equal the sum of
                                       all contributing source widths.
    weight              : scalar scaling applied to this source's output before
                          aggregation.  Useful for asymmetric multimodal fusion
                          (e.g. vision 0.7, audio 0.3) or cross-modal inhibition
                          (negative weight).
    """
    source: str
    target: str
    connection_type: str                                 # 'feedforward' | 'feedback'
    aggregation_mode: Literal["sum", "concat"] = "sum"  # aggregation strategy
    weight: float = 1.0                                  # per-connection scaling
    delay: int = 0

# ------------------------------------------------------------------
# NetworkGraph
# ------------------------------------------------------------------

class NetworkGraph:
    """
    Global orchestrator for a hierarchical, multimodal SNN.

    Key improvements over the original single-hierarchy version
    ===========================================================

    1. Parallel hierarchies
       Layers are stored in a name → layer dict; the processing order is a
       topological sort over registered feedforward connections rather than a
       flat registration order.  Registering vision/audio/text hierarchies
       independently and connecting them to a shared association layer "just
       works" — the topological sort ensures sources are always computed before
       their targets within the same timestep.

    2. Multi-source feedforward aggregation  (_aggregate_feedforward_inputs)
       A target layer may receive feedforward connections from multiple sources
       (e.g. V1 + A1 → association cortex).  Two aggregation modes:

         'sum'    — element-wise sum with optional per-connection weights.
                    All contributing sources must emit arrays of size
                    target.num_inputs.  Good for lateral/recurrent connections
                    or when source and target share the same representational
                    space.

         'concat' — concatenation.  target.num_inputs must equal the sum of
                    all contributing sources' output widths.  The canonical
                    choice when fusing genuinely different modalities
                    (e.g. a 30-neuron vision stream + 20-neuron audio stream
                    feeding a 50-input multimodal layer).

       If a target has a single feedforward source the aggregation mode is
       irrelevant (single-source fast path is used).

    3. Topological ordering
       add_layer() appends to the registry; on the first step() call the graph
       is topologically sorted so that layers are processed in an order
       consistent with their feedforward connections.  Cycles are not supported
       (feedback connections are handled in a separate prior pass, not counted
       in the sort).

    4. Backward-compatible API
       Existing code that uses add_layer() / connect() / step() / update_weights()
       / reset_state() continues to work without changes.  New features are
       opt-in via the aggregation_mode and weight parameters of connect().
    """

    def __init__(self, dt: float = 1.0) -> None:
        self.dt = dt
        self.timestep: int = 0

        self._layers: dict[str, "PredictiveCodingLayer"] = {}
        self._order: list[str] = []           # will be replaced by topo-sort on first step
        self._order_dirty: bool = True        # flag: re-sort before next step()
        self._connections: list[LayerConnection] = []
        self._encoder = PoissonEncoder()

        self._sequence_memories: dict[str, "SequenceMemory | HierarchicalSequenceMemory"] = {}


        self.oscillator = GlobalOscillator()

        # Bufor historii do synchronizacji Modalności
        self._output_history: dict[str, deque[np.ndarray]] = {}
        self._max_delay: int = 0

        self._concat_offsets: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    # Layer registration
    # ------------------------------------------------------------------

    def add_layer(self, name: str, layer: "PredictiveCodingLayer") -> None:
        """
        Register a layer under a unique name.

        Layers may be registered in any order — the graph topologically sorts
        them before the first step().
        """
        if name in self._layers:
            raise ValueError(f"Layer '{name}' already registered.")
        self._layers[name] = layer
        self._order.append(name)
        self._order_dirty = True

    def connect(
        self,
        source: str,
        target: str,
        connection_type: str = "feedforward",
        aggregation_mode: Literal["sum", "concat"] = "sum",
        weight: float = 1.0,
        delay: int = 0,
    ) -> None:
        """
        Add a directional connection between two registered layers.

        Args:
            source:           Name of the source layer.
            target:           Name of the target layer.
            connection_type:  'feedforward' or 'feedback'.
            aggregation_mode: 'sum' or 'concat' (feedforward only).
            weight:           Scaling factor for this source's contribution.
        """
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
            LayerConnection(source, target, connection_type, aggregation_mode, weight, delay)
        )
        self._max_delay = max(self._max_delay, delay)
        self._order_dirty = True

    def attach_sequence_memory(
        self, layer_name: str, seq_mem: "SequenceMemory | HierarchicalSequenceMemory",
    ) -> None:
        """Attach a SequenceMemory to a named layer for temporal tracking."""
        if layer_name not in self._layers:
            raise ValueError(f"Layer '{layer_name}' not registered.")
        self._sequence_memories[layer_name] = seq_mem

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_layer(self, name: str) -> "PredictiveCodingLayer":
        """Retrieve a registered layer by name."""
        return self._layers[name]

    @property
    def layer_names(self) -> list[str]:
        """Layer names in current processing order."""
        if self._order_dirty:
            self._refresh_order()
        return list(self._order)

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def step(
        self,
        sensory_inputs: dict[str, np.ndarray],
        neuromodulator: "NeuromodulatorSystem | None" = None,
    ) -> dict[str, np.ndarray]:
        """
        Advance the entire network by one timestep.

        Pipeline:
          1. (Re-)sort layers topologically if connections changed.
          2. Distribute neuromodulator signals (ACh) to all PC/pyramidal layers.
          3. Feedback pass: higher layers send predictions to lower layers.
          4. Feedforward pass: each layer processes its aggregated inputs.
          5. Sequence memory observation.
          6. Increment global timestep.

        Args:
            sensory_inputs: Dict mapping input layer name → spike array.
                            Layers not in this dict receive aggregated
                            feedforward spikes from their registered sources.
            neuromodulator: Optional NeuromodulatorSystem for global modulation.

        Returns:
            Dict mapping layer name → output spike array from this timestep.
        """
        # 1. Topological sort if needed
        if self._order_dirty:
            self._refresh_order()

        outputs: dict[str, np.ndarray] = {}
        salience_signal = 0.0
        phase_reset = False

        #Distribute neuromodulator
        if neuromodulator is not None:
            # Rejestrujemy ogólny poziom zaskoczenia na potrzeby filtru hierarchicznego
            salience_signal = neuromodulator.noradrenaline
            phase_reset = self.oscillator.tick(
                ne_level=neuromodulator.competition_sharpness,
                sero_level=neuromodulator.planning_horizon
            )
            for layer in self._layers.values():
                if hasattr(layer, "set_ach_level"):
                    layer.set_ach_level(neuromodulator.bottom_up_gain)
                # DODANE: Przepływ NE i 5-HT w dół, by sterować uwagą
                if hasattr(layer, "set_neuromodulators"):
                    layer.set_neuromodulators(
                        ne=neuromodulator.competition_sharpness,
                        sero=neuromodulator.planning_horizon
                    )
        # Broadcast global phase reset to ALL applicable layers BEFORE forward pass
        for layer in self._layers.values():
            if hasattr(layer, "trigger_phase_reset") and phase_reset:
                layer.trigger_phase_reset()

                # 3. Feedback pass (top-down predictions)
                for conn in self._connections:
                    if conn.connection_type == "feedback":
                        src = self._layers[conn.source]
                        tgt = self._layers[conn.target]

                        if hasattr(src, "generate_prediction") and hasattr(tgt, "receive_prediction"):
                            prediction = src.generate_prediction()

                            # Predykcja top-down docelowo mapuje się na NEURONY warstwy niższej
                            if prediction.shape[0] == tgt.num_neurons:
                                tgt.receive_prediction(prediction)
                            elif prediction.shape[0] > tgt.num_neurons:
                                # Bezpieczne cięcie dla złączonych wejść
                                offset = self._concat_offsets.get((conn.target, conn.source), 0)
                                sliced_pred = prediction[offset: offset + tgt.num_neurons]
                                tgt.receive_prediction(sliced_pred)

        # 4. Feedforward pass (bottom-up)
        for name in self._order:
            layer = self._layers[name]

            if name in sensory_inputs:
                input_spikes = sensory_inputs[name]
            else:
                input_spikes = self._aggregate_feedforward_inputs(name, outputs)

            outputs[name] = layer.forward(input_spikes)

            # 5. Sequence memory observation
            # ZAKTUALIZOWANE: Sequence memory observation z mechanizmem markerów
            if name in self._sequence_memories:
                seq_mem = self._sequence_memories[name]
                if hasattr(seq_mem, "salience_threshold"):
                    seq_mem.observe(outputs[name], salience=salience_signal)
                else:
                    seq_mem.observe(outputs[name])



        for name, spike_array in outputs.items():
            if name not in self._output_history:
                self._output_history[name] = deque(maxlen=max(1, self._max_delay + 1))
            self._output_history[name].append(spike_array.copy())


        self.timestep += 1
        return outputs

    def update_weights(self, neuromodulator: "NeuromodulatorSystem") -> None:
        """
        Zaktualizowana logika neuromodulacji:
        - Dopamina (DA) + Noradrenalina (NE) sterują plastycznością (m_t).
        - ACh jest używane lokalnie w warstwach do balansu BU/TD (już ustawione w step()).
        """
        # Globalny sygnał plastyczności: połączenie nagrody i nowości poznawczej
        # Biologicznie: NE otwiera okno plastyczności, DA nadaje kierunek (LTP/LTD)
        plasticity_signal = neuromodulator.learning_rate_modulation

        for name, layer in self._layers.items():
            if hasattr(layer, "update_weights"):
                # World Model powinien uczyć się zawsze (m_t = NE lub stałe),
                # ale warstwy decyzyjne potrzebują Dopaminy.
                # Tutaj stosujemy bezpieczny kompromis:
                layer.update_weights(m_t=plasticity_signal, pred_error=layer.prediction_error)


    # ------------------------------------------------------------------
    # Feedforward input aggregation  (core of the multimodal upgrade)
    # ------------------------------------------------------------------
        # DODANE: Metoda pomocnicza do pobierania opóźnionego sygnału
    def _get_delayed_output(self, source_name: str, delay: int,
                            current_outputs: dict[str, np.ndarray]) -> np.ndarray | None:
        """Retrieves delayed output for synchronized multimodal binding."""
        if delay == 0:
            return current_outputs.get(source_name)

        history = self._output_history.get(source_name)
        # Jeśli nie ma wystarczającej historii, oznacza to, że sygnał "aksonalny" jeszcze nie dotarł
        if not history or len(history) <= delay:
            return None

        # Pobieramy stan sprzed 'delay' kroków czasowych
        return history[-(delay + 1)]

    def _aggregate_feedforward_inputs(
            self,
            target_name: str,
            outputs: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Collect and aggregate all feedforward inputs for *target_name*.

        Logic
        -----
        1. Find all feedforward connections whose source has already produced
           output this step.
        2. If none → return silence (zeros matching target.num_inputs).
        3. If one  → return that source's output (scaled by connection weight).
        4. If many → aggregate according to the connection's aggregation_mode.

        Aggregation modes (all connections to the same target should use the
        same mode; if they differ, the mode of the first connection is used
        and a warning is printed):

          'sum'    : weighted element-wise sum.
                     Each source's contribution is clipped to target.num_inputs
                     so a partial-overlap sum is still well-defined even when
                     source output sizes differ slightly.

          'concat' : concatenate all sources.
                     The caller is responsible for ensuring that
                     sum(source.num_neurons for all sources) == target.num_inputs.
                     If the total mismatches, the result is zero-padded or
                     truncated to target.num_inputs (with a warning).
        """
        target_layer = self._layers[target_name]
        num_inputs = target_layer.num_inputs

        active_sources = []
        delayed_outputs = {}

        # 1. Filtruj źródła i pobieraj opóźnione sygnały
        for conn in self._connections:
            if conn.target == target_name and conn.connection_type == "feedforward":
                delayed_out = self._get_delayed_output(conn.source, conn.delay, outputs)
                if delayed_out is not None:
                    active_sources.append(conn)
                    delayed_outputs[conn.source] = delayed_out

        if not active_sources:
            return np.zeros(num_inputs, dtype=np.float32)

        # 2. Pula jedynego źródła
        if len(active_sources) == 1:
            conn = active_sources[0]
            src_out = delayed_outputs[conn.source].astype(np.float32)
            scaled = src_out * conn.weight
            return self._fit_to_size(scaled, num_inputs)

        # 3. Wielorakie agregacje opóźnionych źródeł
        mode = active_sources[0].aggregation_mode
        if mode == "concat":
            parts = [
                delayed_outputs[c.source].astype(np.float32) * c.weight
                for c in active_sources
            ]
            concatenated = np.concatenate(parts)
            return self._fit_to_size(concatenated, num_inputs)
        else:  # "sum"
            result = np.zeros(num_inputs, dtype=np.float32)
            for conn in active_sources:
                src_out = delayed_outputs[conn.source].astype(np.float32) * conn.weight
                size = min(len(src_out), num_inputs)
                result[:size] += src_out[:size]
            return result

    @staticmethod
    def _fit_to_size(arr: np.ndarray, target_size: int) -> np.ndarray:
        """
        Pad or truncate *arr* to exactly *target_size* elements.
        Pads with zeros on the right; truncates from the right.
        Used to handle slight size mismatches gracefully.
        """
        n = len(arr)
        if n == target_size:
            return arr
        if n > target_size:
            return arr[:target_size]
        # n < target_size → zero-pad
        out = np.zeros(target_size, dtype=arr.dtype)
        out[:n] = arr
        return out

    # ------------------------------------------------------------------
    # Topological ordering
    # ------------------------------------------------------------------

    def _refresh_order(self) -> None:
        """
        Topologically sort registered layers based on feedforward connections.

        Uses Kahn's algorithm (BFS-based).  Layers with no feedforward
        predecessors (sensory input layers) are processed first.
        Layers that are only reachable via feedback connections do not affect
        the topological order.

        If the feedforward graph has a cycle (architecturally invalid), the
        sort falls back to the original registration order and prints a warning.
        """
        # Build in-degree and adjacency for feedforward edges only
        in_degree: dict[str, int] = {name: 0 for name in self._layers}
        children: dict[str, list[str]] = {name: [] for name in self._layers}

        for conn in self._connections:
            if conn.connection_type == "feedforward":
                children[conn.source].append(conn.target)
                in_degree[conn.target] += 1

        # Kahn's BFS
        from collections import deque
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
            print(
                "[NetworkGraph] Warning: feedforward connections contain a cycle. "
                "Falling back to registration order."
            )
            sorted_order = list(self._layers.keys())

        self._order = sorted_order
        self._order_dirty = False

        # Pre-kalkulacja twardych mapowań offsetów dla concat
        self._concat_offsets.clear()
        for target_name in self._layers:
            current_offset = 0
            for conn in self._connections:
                if conn.target == target_name and conn.connection_type == "feedforward":
                    src_layer = self._layers[conn.source]
                    if conn.aggregation_mode == "concat":
                        # Zapisujemy dokładny początek wyjścia danego źródła w wektorze złączonym
                        self._concat_offsets[(conn.source, conn.target)] = current_offset
                        current_offset += src_layer.num_neurons

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset all layers, sequence memories, global timestep, and delay buffers."""
        for layer in self._layers.values():
            if hasattr(layer, "reset_state"):
                layer.reset_state()
        for sm in self._sequence_memories.values():
            sm.reset_state()
        self.timestep = 0
        # POPRAWKA Bug 2: Czyścimy historię opóźnień, by duchy z poprzedniego
        # epizodu nie przenikały do następnego przez połączenia delayed.
        self._output_history.clear()
        # Note: do NOT reset _order_dirty; the topology hasn't changed.