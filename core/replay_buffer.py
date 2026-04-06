from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# TYPE_CHECKING guard keeps runtime imports clean while allowing type hints
from typing import TYPE_CHECKING

from .sequence_memory import SequenceMemory

if TYPE_CHECKING:
    from .neuron import LIFLayer
    from .world_model import WorldModel
    from .neuromodulator import NeuromodulatorSystem


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    layer_traces: dict[str, np.ndarray]
    layer_outputs: dict[str, np.ndarray]
    prediction_error: np.ndarray

    # DODANE: Przechowywanie lokalnych błędów warstw w momencie wystąpienia zdarzenia
    layer_errors: dict[str, np.ndarray] = field(default_factory=dict)

    # Neuromodulacyjna salience — ciągły sygnał istotności.
    # Biologicznie: DA burst przy nagrodzie, spadek NE po osiągnięciu celu.
    # Wartość 0.0 = rutyna, 1.0 = krytyczny moment (nagroda / zagrożenie).
    # W ciągłym uczeniu salience wygładza granice epizodów:
    #   effective_γ = γ × (1 − salience)
    # więc salience=1.0 zeruje dyskont (odpowiednik twardego końca epizodu).
    salience: float = 0.0

    def __post_init__(self) -> None:
        self.state = self.state.copy()
        self.next_state = self.next_state.copy()
        self.layer_traces = {name: trace.copy() for name, trace in self.layer_traces.items()}
        self.layer_outputs = {name: out.copy() for name, out in self.layer_outputs.items()}
        self.prediction_error = self.prediction_error.copy()
        # Kopiowanie zapisanych błędów
        self.layer_errors = {name: err.copy() for name, err in self.layer_errors.items()}


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
        state,
        action,
        reward,
        next_state,
        layer_traces,
        layer_outputs,
        prediction_error,
        layer_errors: dict[str, np.ndarray] | None = None,
        salience: float = 0.0,
    ) -> None:
        if layer_errors is None:
            layer_errors = {}
        self._buffer.append(
            Experience(
                state=state, action=action, reward=reward, next_state=next_state,
                layer_traces=layer_traces, layer_outputs=layer_outputs,
                prediction_error=prediction_error, layer_errors=layer_errors,
                salience=salience,
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
        gamma: float = 0.99,
    ) -> list[float]:
        """
        Consolidate recent experience through reverse-order replay.

        For each replayed experience (latest → earliest):
          1. Restore the stored eligibility traces into ALL matching layers
             in the hierarchy (not just one isolated layer).
          2. Update the world model with the observed transition.
          3. Apply a dopamine-weighted STDP update to each layer, using
             the world model error as the third factor.

        POPRAWKA Bug C: Zamiast używać exp.reward (które wynosi 0.0 dla większości
        kroków), obliczamy skumulowany zwrot G_t w tył (Monte Carlo return):
            G_t = r_t + γ * G_{t+1}
        Dzięki temu kroki POPRZEDZAJĄCE nagrodę otrzymują kredyt,
        realizując prawdziwe odwrotne odtwarzanie w stylu hipokampalnym.

        Args:
            layers:         Dict mapping layer name → LIFLayer (or subclass).
            world_model:    WorldModel to refine during consolidation.
            neuromodulator: Source of dopaminergic modulation signal.
            n_experiences:  How many of the most recent experiences to replay.
                            None → replay entire buffer.
            gamma:          Czynnik dyskontowy dla obliczania skumulowanego zwrotu.

        Returns:
            List of per-experience world model MSE values (in replay order,
            i.e. most recent first).
        """
        if len(self._buffer) == 0:
            return []

        experiences = list(self._buffer)
        if n_experiences is not None:
            experiences = experiences[-n_experiences:]

        # DODANE: Odtwarzanie w przód (Forward replay) dla pamięci sekwencyjnej
        if sequence_memories is not None:
            for exp in experiences:
                for name, seq_mem in sequence_memories.items():
                    # POPRAWKA Błędu 2A: Przekazujemy lokalne wyjście warstwy, nie globalny stan
                    if name in exp.layer_outputs:
                        seq_mem.observe(exp.layer_outputs[name])
            for seq_mem in sequence_memories.values():
                seq_mem.reset_state()

        errors: list[float] = []

        #  Wstępne obliczenie skumulowanych zwrotów G_t = r_t + γ*G_{t+1}
        # Iterujemy od najnowszego do najstarszego (tak jak potem robimy replay),
        # kumulując zwrot do tyłu.
        #
        # Salience-weighted discount: wysoka salience (DA burst) tłumi
        # propagację przyszłego zwrotu — biologicznie odpowiada „przecięciu"
        # śladu hipokampalnego w momencie silnego sygnału neuromodulacyjnego.
        # salience=1.0 zeruje γ (twarda granica epizodu);
        # salience ∈ (0,1) płynnie redukuje γ.
        cumulative_returns: list[float] = []
        G = 0.0
        for exp in reversed(experiences):
            effective_gamma = gamma * (1.0 - exp.salience)
            G = exp.reward + effective_gamma * G
            cumulative_returns.append(G)
        # cumulative_returns[0] = G dla najnowszego exp; odwrócimy dostęp poniżej.

        wm_saved_state = None
        if hasattr(world_model, '_snapshot_encoder'):
            wm_saved_state = world_model._snapshot_encoder()
        # Hippocampal reverse replay: most recent experience first
        for i, exp in enumerate(reversed(experiences)):
            # 1. Restore eligibility traces for ALL layers in the hierarchy
            for name, layer in layers.items():
                if name in exp.layer_traces:
                    layer.e = exp.layer_traces[name].copy()

            # 2. Reset stanu transient przed każdym krokiem (izolacja przejść SNN)
            # Zapobiega to "wyciekowi" czasu do tyłu między niezależnymi próbkami
            if hasattr(world_model, 'reset_state'):
                world_model.reset_state()
            # Update world model; get refined prediction error
            world_error = world_model.update(exp.state, exp.action, exp.next_state)

            # 3. Three-factor STDP: dopamina × skumulowany_zwrot × ślad × błąd_predykcji
            # Używamy lokalnego błędu predykcji każdej warstwy (num_neurons,),
            # NIE globalnego world_error (state_size,). LIFLayer.update_weights robi:
            #   dw = lr * m_t * e * pred_error
            # gdzie e ma kształt (num_inputs, num_neurons). Broadcast NumPy wymaga
            # pred_error o kształcie (num_neurons,) — globalny błąd świata o innym
            # rozmiarze powoduje ValueError (crash).
            # 3. Three-factor STDP: dopamina × skumulowany_zwrot × ślad × błąd_predykcji
            G_t = cumulative_returns[i]
            m_t = neuromodulator.learning_rate_modulation * G_t

            for name, layer in layers.items():
                # NAPRAWA BUG 1: Pobieramy zamrożony błąd z historii (exp), a nie nadpisany stan z końca epizodu
                local_error = exp.layer_errors.get(name)
                num_neurons = getattr(layer, 'num_neurons', None)
                num_inputs = getattr(layer, 'num_inputs', None)

                if local_error is not None and num_neurons is not None:
                    # NAPRAWA BUG 2: Warstwy PC mają błąd w wymiarze (num_inputs,), a klasyczne LIF (num_neurons,).
                    # Dopuszczamy oba kształty - LIFLayer.update_weights ułoży wymiary przez broadcasting.
                    if local_error.shape[0] in (num_neurons, num_inputs):
                        layer.update_weights(m_t=m_t, pred_error=local_error)
                    else:
                        # Fallback jeśli kształt jest całkowicie zepsuty
                        layer.update_weights(m_t=m_t, pred_error=np.ones(num_neurons, dtype=np.float32))
                elif num_neurons is not None:
                    # Prosta warstwa LIF bez wbudowanego błędu: otrzymuje mnożnik neutralny (1.0)
                    layer.update_weights(
                        m_t=m_t,
                        pred_error=np.ones(num_neurons, dtype=np.float32),
                    )

            errors.append(float(np.mean(world_error ** 2)))

        # Przywrócenie stanu sieci po przebudzeniu
        if wm_saved_state is not None:
            world_model._restore_encoder(wm_saved_state)
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