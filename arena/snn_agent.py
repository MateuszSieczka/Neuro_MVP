"""
arena.snn_agent — Adapter connecting the Neuro_MVP SNN to the arena Agent protocol.
"""

from __future__ import annotations

import numpy as np
import dataclasses
from typing import Any

from arena.core import Agent
from core.basal_ganglia import BasalGangliaAGISystem, ContinuousBGConfig
from core.neuromodulator import NeuromodulatorSystem
from core.world_model import SNNWorldModel
from core.config import (
    NeuromodulatorConfig,
    SNNWorldModelConfig,
)


class SNNAgent(Agent):
    def __init__(
        self,
        state_size: int,
        n_actions: int,
        bg_config: ContinuousBGConfig | None = None,
        wm_config: SNNWorldModelConfig | None = None,
        nm_config: NeuromodulatorConfig | None = None,
        use_world_model: bool = True,
        trace_decay: float = 0.0,
    ) -> None:
        self.state_size = state_size
        self.n_actions = n_actions
        self._use_wm = use_world_model
        self._trace_decay = trace_decay
        self._use_trace = trace_decay > 0.0

        bg_input_size = state_size * 2 if self._use_trace else state_size

        self._bg_config = bg_config or ContinuousBGConfig(
            gamma=0.95, critic_lr=0.01, actor_lr=0.005,
            exploration_noise=0.3, hidden_size=64,
        )
        
        # Zapisujemy wartości bazowe dla homeostazy neuromodulacyjnej
        self._base_actor_lr = self._bg_config.actor_lr
        self._base_critic_lr = self._bg_config.critic_lr
        self._base_noise = self._bg_config.exploration_noise

        self.bg = BasalGangliaAGISystem(
            state_size=bg_input_size, motor_dim=n_actions,
            internal_dim=1, config=self._bg_config,
        )

        # Zawsze włączamy Neuromodulator (nawet bez modelu świata)
        self.neuromod = NeuromodulatorSystem(nm_config)

        if self._use_wm:
            self._wm_config = wm_config or SNNWorldModelConfig(
                hidden_size=32, k_winners=4, rehearsal_steps=5,
            )
            self.world_model = SNNWorldModel(
                state_size=state_size, action_size=n_actions, config=self._wm_config,
            )

        self._trace = np.zeros(state_size, dtype=np.float32)
        self._last_td_error: float = 0.0
        self._step_count: int = 0
        self._episode_reward: float = 0.0     # Running reward for current episode
        self._avg_episode_reward: float = 0.0  # EMA of episode rewards (ventral striatum)

    def _augment_state(self, state: np.ndarray) -> np.ndarray:
        if not self._use_trace:
            return state.astype(np.float32)
        return np.concatenate([state.astype(np.float32), self._trace])

    def _update_trace(self, state: np.ndarray) -> None:
        if self._use_trace:
            self._trace = self._trace * self._trace_decay + state.astype(np.float32)

    def act(self, state: np.ndarray) -> int:
        aug = self._augment_state(state)
        self._update_trace(state)

        # 1. Forward Aktora (softmax policy + próbkowanie akcji + ślad)
        motor, _internal = self.bg.actor.forward(aug)

        # 2. Forward Krytyka
        self.bg.last_v = self.bg.critic.forward(aug)

        return self.bg.actor.get_action()

    def _peek_value(self, state_spikes: np.ndarray) -> float:
        """Bezpieczny podgląd V(s') bez niszczenia śladów membrany Krytyka."""
        return self.bg.critic.peek(state_spikes)

    def observe(
        self, state: np.ndarray, action: int, reward: float,
        next_state: np.ndarray, done: bool, info: dict[str, Any] | None = None,
    ) -> None:
        next_aug = self._augment_state(next_state)
        is_truncated = info.get("truncated", False) if info else False
        is_terminal = done and not is_truncated

        # 1. Czyste wyliczenie błędu TD
        if is_terminal:
            td_error = reward - self.bg.last_v
        else:
            next_v = self._peek_value(next_aug)
            td_error = reward + self.bg.config.gamma * next_v - self.bg.last_v

        # --- ZSYNCHRONIZOWANE NASYCENIE BŁĘDU (TD CLIPPING) ---
        # Clip [-10, 10] w update() Krytyka i Aktora chroni wagi.
        # Tu przekazujemy surowy δ — moduły same przycinają.

        clipped_td = float(np.clip(td_error, -10.0, 10.0))
        # 2. Aktualizacja obu modułów tym samym sygnałem
        self.bg.critic.update(clipped_td)
        self.bg.actor.update(clipped_td)

        self._last_td_error = td_error

        # 3. Aktualizacja Neuromodulatora
        # NM musi otrzymać SUROWY sygnał, by prawidłowo wyzwolić skok Noradrenaliny (NE=1.0)
        # co skompresuje ślady (tau) po każdym upadku.
        if self._use_wm:
            state_f32 = state.astype(np.float32)
            next_f32 = next_state.astype(np.float32)
            pred_error = self.world_model.update(
                state_f32, action, next_f32, m_t=max(abs(clipped_td), 0.1)
            )
            self.neuromod.update(
                prediction_error=pred_error, td_error=td_error,
                novelty=self.world_model.curiosity_signal(),
            )
        else:
            # Soft-normalize TD error to [0,1] for neuromodulator.
            # sigmoid(|δ|) smoothly maps small errors→~0 and large errors→~1,
            # unlike hard clip which saturates at |δ|>1.
            norm_td = float(abs(td_error) / (1.0 + abs(td_error)))
            self.neuromod.update(
                prediction_error=np.array([norm_td], dtype=np.float32),
                td_error=td_error, novelty=0.0
            )

        # 4. Neuromodulacyjna Plastyczność (Zamknięcie pętli autotuningu)
        self.bg.set_plasticity_timescales(ne=self.neuromod.tau_compression)

        if self._use_wm:
            self.neuromod.apply_to_layer(self.world_model)

        # 5. Adaptacja eksploracji — sygnał konwergencji oparty na nagrodach.
        # Biologicznie: brzuszne prążkowie (ventral striatum) śledzi średnią nagrodę.
        # Wysoka nagroda → aktywacja VTA → dopamina tonowa → redukcja szumu.
        # Niska nagroda → spadek tonu → wzrost eksploracji.
        #
        # Używamy EMA nagrody epizodowej zamiast serotoniny (która wymaga
        # skonwergowanego krytyka). To jest bardziej bezpośredni sygnał sukcesu.
        self._episode_reward += reward
        if done:
            # EMA z α=0.1 — powolne śledzenie (ok. 10 epizodów pamięci)
            self._avg_episode_reward = (
                0.9 * self._avg_episode_reward + 0.1 * self._episode_reward
            )
            self._episode_reward = 0.0

            # Sygmoidalna mapa nagrody na szum: im wyższa nagroda, tym mniej szumu.
            # reward_signal ∈ (0,1): 0 = brak nagrody, 1 = duża nagroda.
            # Próg ~100 kroków: at R=100 → signal=0.5; R=300→0.75; R=500→0.83
            reward_signal = self._avg_episode_reward / (self._avg_episode_reward + 100.0)

            if reward_signal > 0.5:  # Agent zaczyna się uczyć
                self.bg.actor.noise_scale = max(0.05, self.bg.actor.noise_scale * 0.85)
            else:
                self.bg.actor.noise_scale = min(1.0, self.bg.actor.noise_scale * 1.05)

        self._step_count += 1

    def reset(self) -> None:
        self.bg.reset_state()
        self._trace = np.zeros(self.state_size, dtype=np.float32)
        self._episode_reward = 0.0
        if self._use_wm:
            self.world_model.reset_state()
            self.world_model.reset_error_history()