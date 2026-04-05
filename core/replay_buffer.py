from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# TYPE_CHECKING guard keeps runtime imports clean while allowing type hints
from typing import TYPE_CHECKING

from . import SequenceMemory

if TYPE_CHECKING:
    from .neuron import LIFLayer
    from .world_model import WorldModel
    from .neuromodulator import NeuromodulatorSystem


@dataclass
class Experience:
    """
    Atomic unit of episodic memory.

    Stores a full transition snapshot including per-layer eligibility traces,
    allowing replay to restore the learning state of an entire layer hierarchy
    (not just a single isolated layer).

    All arrays are deep-copied on construction so stored experiences are
    never aliased to the live network state.
    """
    state: np.ndarray             # Spike-rate representation before action
    action: int                   # Integer action index executed
    reward: float                 # Extrinsic reward received
    next_state: np.ndarray        # Spike-rate representation after action
    layer_traces: dict[str, np.ndarray]  # layer_name → eligibility trace snapshot
    prediction_error: np.ndarray  # World model error at this timestep

    def __post_init__(self) -> None:
        # Defensive copies — caller should not mutate stored experiences
        self.state = self.state.copy()
        self.next_state = self.next_state.copy()
        self.layer_traces = {
            name: trace.copy() for name, trace in self.layer_traces.items()
        }
        self.prediction_error = self.prediction_error.copy()


class ReplayBuffer:
    """
    Episodic replay buffer with hippocampally-inspired consolidation.

    Two replay modes:
      1. Online sampling  (sample):      Random mini-batch for continual learning.
      2. Offline consolidation (sleep_phase): Reverse-chronological replay of
         recent experiences, mirroring hippocampal sharp-wave ripple (SWR) replay
         observed in rodents after maze exploration.

    Reverse replay is biologically motivated: replaying backwards in time
    causally aligns eligibility traces with the reward signal, allowing the
    three-factor STDP rule to correctly assign credit to the synapses that
    led to the outcome (and not merely followed it).

    Multi-layer support:
      Experiences store eligibility traces per named layer (layer_traces dict),
      allowing sleep_phase to restore the full hierarchy's learning state during
      consolidation — not just a single isolated layer.

    Design constraints:
      - Buffer is a fixed-capacity FIFO (deque with maxlen). Oldest entries are
        silently dropped when capacity is exceeded.
    """

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self._buffer: deque[Experience] = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        layer_traces: dict[str, np.ndarray],
        prediction_error: np.ndarray,
    ) -> None:
        """
        Store a single timestep transition.

        All arrays are copied internally — the caller may safely modify them
        after this call without corrupting stored data.

        Args:
            state:            State before action.
            action:           Integer action index.
            reward:           Extrinsic reward received.
            next_state:       State after action.
            layer_traces:     Dict mapping layer name → eligibility trace snapshot.
            prediction_error: World model error at this timestep.
        """
        self._buffer.append(
            Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                layer_traces=layer_traces,
                prediction_error=prediction_error,
            )
        )

    # ------------------------------------------------------------------
    # Offline consolidation (sleep / SWR replay)
    # ------------------------------------------------------------------

    def sleep_phase(
        self,
        layers: dict[str, LIFLayer],
        world_model: WorldModel,
        neuromodulator: NeuromodulatorSystem,
        n_experiences: int | None = None,
        sequence_memories: dict[str, SequenceMemory] | None = None,
    ) -> list[float]:
        """
        Consolidate recent experience through reverse-order replay.

        For each replayed experience (latest → earliest):
          1. Restore the stored eligibility traces into ALL matching layers
             in the hierarchy (not just one isolated layer).
          2. Update the world model with the observed transition.
          3. Apply a dopamine-weighted STDP update to each layer, using
             the world model error as the third factor.

        Args:
            layers:         Dict mapping layer name → LIFLayer (or subclass).
                            Names must match those used in store().
            world_model:    WorldModel to refine during consolidation.
            neuromodulator: Source of dopaminergic modulation signal.
            n_experiences:  How many of the most recent experiences to replay.
                            None → replay entire buffer.

        Returns:
            List of per-experience world model MSE values (in replay order,
            i.e. most recent first).
        """
        if len(self._buffer) == 0:
            return []

        experiences = list(self._buffer)
        if n_experiences is not None:
            experiences = experiences[-n_experiences:] #Take most recent n

        # DODANE: Odtwarzanie w przód (Forward replay) dla pamięci sekwencyjnej
        # Zapewnia naukę poprawnych związków A -> B (przyczynowych)
        if sequence_memories is not None:
            for exp in experiences:
                for name, seq_mem in sequence_memories.items():
                    # Upraszczamy: przesyłamy zrekonstruowany stan układu
                    seq_mem.observe(exp.state)

            # Zapobiega przenikaniu ostatniego stanu ze snu do przebudzenia
            for seq_mem in sequence_memories.values():
                seq_mem.reset_state()

        errors: list[float] = []

        # Hippocampal reverse replay: most recent experience first
        for exp in reversed(experiences):
            # 1. Restore eligibility traces for ALL layers in the hierarchy
            for name, layer in layers.items():
                if name in exp.layer_traces:
                    layer.e = exp.layer_traces[name].copy()

            # 2. Update world model; get refined prediction error
            world_error = world_model.update(exp.state, exp.action, exp.next_state)

            # 3. Three-factor STDP: dopamine × eligibility × prediction error
            #    Use signed reward (not max(0, reward)) so that negative rewards
            #    can drive synaptic depression for aversive learning.
            m_t = neuromodulator.learning_rate_modulation * exp.reward
            for layer in layers.values():
                layer.update_weights(m_t=m_t, pred_error=world_error)

            errors.append(world_model.prediction_error)

        return errors

    # ------------------------------------------------------------------
    # Online sampling
    # ------------------------------------------------------------------

    def sample(self, n: int) -> list[Experience]:
        """
        Draw a random mini-batch without replacement.

        If n > len(buffer), returns the entire buffer in random order.
        """
        n = min(n, len(self._buffer))
        indices = np.random.choice(len(self._buffer), size=n, replace=False)
        buf_list = list(self._buffer)
        return [buf_list[i] for i in indices]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_ready(self, min_size: int = 1) -> bool:
        """True when the buffer contains at least min_size experiences."""
        return len(self._buffer) >= min_size

    def clear(self) -> None:
        """Empty the buffer (e.g. between training runs)."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ReplayBuffer(size={len(self._buffer)}/{self.capacity})"