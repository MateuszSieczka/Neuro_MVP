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
    from .basal_ganglia import BasalGangliaAGISystem


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
    recorded_da: float = 0.0

    # BG traces for sleep-phase consolidation (hippocampal → striatal transfer).
    # Stores the eligibility traces of the critic and actor at the time of
    # the experience, enabling offline credit assignment via G_t.
    # Biological basis: hippocampal SWR replay reactivates cortico-striatal
    # eligibility traces that were active during the original experience
    # (Lansink et al., 2009; Pennartz et al., 2004).
    bg_traces: dict[str, np.ndarray] = field(default_factory=dict)

    # Augmented state for BG critic (state + trace or state + WM signal).
    # When BG input size > raw state size, critic.peek() needs the full
    # augmented vector. None means raw state == BG input (no augmentation).
    aug_state: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.state = self.state.copy()
        self.next_state = self.next_state.copy()
        self.layer_traces = {name: trace.copy() for name, trace in self.layer_traces.items()}
        self.layer_outputs = {name: out.copy() for name, out in self.layer_outputs.items()}
        self.prediction_error = self.prediction_error.copy()
        # Kopiowanie zapisanych błędów
        self.layer_errors = {name: err.copy() for name, err in self.layer_errors.items()}
        self.bg_traces = {name: trace.copy() for name, trace in self.bg_traces.items()}
        if self.aug_state is not None:
            self.aug_state = self.aug_state.copy()


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
        recorded_da: float = 0.0,
        bg_traces: dict[str, np.ndarray] | None = None,
        aug_state: np.ndarray | None = None,
    ) -> None:
        if layer_errors is None:
            layer_errors = {}
        if bg_traces is None:
            bg_traces = {}
        self._buffer.append(
            Experience(
                state=state, action=action, reward=reward, next_state=next_state,
                layer_traces=layer_traces, layer_outputs=layer_outputs,
                prediction_error=prediction_error, layer_errors=layer_errors,
                salience=salience, recorded_da=recorded_da,
                bg_traces=bg_traces,
                aug_state=aug_state,
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
        bg: "BasalGangliaAGISystem | None" = None,
        sleep_gain: float = 1.0,
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
            # Only genuinely salient events (NE > 0.5) create temporal
            # boundaries in hippocampal replay. Routine NE (arousal during
            # normal exploration) should NOT attenuate the discount factor
            # — otherwise G is systematically compressed (e.g., -6 instead
            # of -87 in MountainCar), causing the critic to target the
            # wrong value. Biological basis: low NE reflects tonic LC
            # firing (routine arousal), while high NE bursts mark genuine
            # surprise events that segment the temporal stream (Bouret &
            # Sara 2005).
            sal_effective = max(0.0, exp.salience - 0.5) * 2.0
            effective_gamma = gamma * (1.0 - sal_effective)
            G = exp.reward + effective_gamma * G
            cumulative_returns.append(G)
        # cumulative_returns[0] = G dla najnowszego exp; odwrócimy dostęp poniżej.

        wm_saved_state = None
        if hasattr(world_model, '_snapshot_encoder'):
            wm_saved_state = world_model._snapshot_encoder()

            # ── BG sleep: batch-normalize advantages (fixes Proposals 1+3) ──
            # Pre-compute and standardize advantages within the episode.
            # Two bugs fixed:
            #   1. ±10 clip made success/failure advantages identical → no gradient
            #   2. normalize_td() polluted running RMS → unstable online learning
            # Biological basis: hippocampal SWR replay assigns credit relative
            # to the episode's own distribution (Ambrose et al. 2016).
            _sleep_adv_normalized = None
            if bg is not None:
                _saved_td_rms = bg._td_rms
                _all_advantages = []
                for _i, _exp in enumerate(reversed(experiences)):
                    _G = cumulative_returns[_i]
                    _vs = bg.critic.peek(
                        _exp.aug_state if _exp.aug_state is not None else _exp.state
                    )
                    _all_advantages.append(_G - _vs)
                _adv_arr = np.array(_all_advantages, dtype=np.float32)
                _adv_mean = float(np.mean(_adv_arr))
                _adv_std = float(np.std(_adv_arr)) + 1e-8
                _max_abs = float(np.max(np.abs(_adv_arr))) + 1e-8

                # Variance guard (Ambrose et al. 2016): when all advantages
                # are near-zero noise (agent stuck, V ≈ G everywhere), SWR
                # replay has no meaningful signal to consolidate. Normalizing
                # this noise to [-1, 1] would teach the actor random
                # preferences. Skip BG sleep entirely in this case.
                if _adv_std < 0.5:
                    _sleep_adv_normalized = None
                else:
                    # Normalize by max absolute value: bounds signal to [-1, 1]
                    # per step, preventing any single advantage from dominating.
                    _sleep_adv_normalized = _adv_arr / _max_abs

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

                # 3. Aktualizacja: dopamina × bounded advantage
                # NAPRAWA: m_t musi być ograniczone do biologicznie
                # realistycznego zakresu. Poprzednio:
                #   m_t = recorded_da * |G_t|  →  0.5 * 86 = 43 (!!)
                # To powodowało 270× silniejsze uczenie w śnie niż online.
                # Teraz: używamy znormalizowanej przewagi (jeśli dostępna,
                # bounded [-1,1]) lub samego recorded_da. m_t ∈ [0, ~1].
                # Biologicznie: neuromodulacja STDP zależy od phasic DA
                # (prediction error), nie od surowego skumulowanego zwrotu.
                G_t = cumulative_returns[i]
                if _sleep_adv_normalized is not None:
                    m_t = exp.recorded_da * max(abs(float(_sleep_adv_normalized[i])), 0.1)
                else:
                    m_t = max(exp.recorded_da, 0.1)

                # Przekazujemy m_t wprost do modelu. World Model zadba o resztę samodzielnie!
                world_error = world_model.update(exp.state, exp.action, exp.next_state, m_t=m_t)

                errors.append(float(np.mean(world_error ** 2)))

                # 4. BG consolidation (Lansink et al. 2009):
                #    During hippocampal SWR replay, cortico-striatal synapses
                #    that were eligible during the original experience are
                #    reactivated. The Monte Carlo return G_t provides the
                #    teaching signal that was unavailable online (because
                #    the reward only arrived at the end of the episode).
                #
                #    The key insight: online TD learning in MountainCar gives
                #    δ ≈ 0 everywhere (reward is -1 everywhere, V converges
                #    to ≈-200). But G_t computed backward from the goal
                #    differentiates states that led to success from those
                #    that did not. This is exactly what the hippocampus
                #    provides during sleep (Diekelmann & Born 2010).
                #
                #    Signal: advantage A_t = G_t - V(s_t).
                #    Positive advantage → this trajectory was better than
                #    the current value estimate → reinforce.
                #    Negative advantage → trajectory worse than expected →
                #    attenuate (only weakly, to avoid unlearning useful weights).
                #
                #    Biological basis for asymmetry (Diekelmann & Born 2010):
                #    SWR replay preferentially replays sequences associated
                #    with reward (place cells for rewarded locations are
                #    replayed more). Negative experiences are replayed less
                #    and with lower gain. This is modeled by the asymmetric
                #    scaling: positive advantage gets full weight, negative
                #    gets 20% (modeling the reduced replay probability).
                if bg is not None and exp.bg_traces and _sleep_adv_normalized is not None:
                    # Restore BG eligibility traces from snapshot
                    if 'critic_e_h' in exp.bg_traces:
                        bg.critic.e_h = exp.bg_traces['critic_e_h'].copy()
                    if 'critic_e_v' in exp.bg_traces:
                        bg.critic.e_v = exp.bg_traces['critic_e_v'].copy()
                    if 'critic_e_bv' in exp.bg_traces:
                        bg.critic.e_bv = float(exp.bg_traces['critic_e_bv'].flat[0])
                    if 'actor_e' in exp.bg_traces:
                        bg.actor.e_actor = exp.bg_traces['actor_e'].copy()

                    # Batch-normalized advantage (within-episode standardization).
                    # Bypasses normalize_td to avoid polluting online RMS.
                    norm_adv = float(_sleep_adv_normalized[i])

                    # Asymmetric sleep plasticity (Ambrose et al. 2016):
                    # Positive advantages (states better than V predicts)
                    # get stronger consolidation. Over many episodes, this
                    # makes success-adjacent states accumulate relatively
                    # higher V while all V drifts toward correct scale.
                    SLEEP_POS_RATIO = 0.5
                    SLEEP_NEG_RATIO = 0.1
                    ratio = SLEEP_POS_RATIO if norm_adv >= 0 else SLEEP_NEG_RATIO
                    sleep_signal = norm_adv * ratio * sleep_gain

                    bg.critic.update(sleep_signal)
                    bg.actor.update(sleep_signal)

            # Restore BG running-RMS after sleep (prevent online pollution)
            if bg is not None and _sleep_adv_normalized is not None:
                bg._td_rms = _saved_td_rms

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