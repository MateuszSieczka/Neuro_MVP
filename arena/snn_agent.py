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

        # 1. Forward Aktora (generuje JEDEN czysty wektor szumu i śladu!)
        motor, _internal = self.bg.actor.forward(aug)

        # 2. Forward Krytyka
        self.bg.last_v = self.bg.critic.forward(aug)

        return int(np.argmax(motor))

    def _peek_value(self, state_spikes: np.ndarray) -> float:
        """Bezpieczny podgląd V(s') bez niszczenia śladów membrany Krytyka."""
        state_f32 = state_spikes.astype(np.float32)
        v_hid = self.bg.critic.v_hidden * self.bg.critic._mem_decay + np.dot(state_f32, self.bg.critic.w_h)
        spikes = (v_hid > 0.5).astype(np.float32)
        fr = self.bg.critic.hidden_firing_rate * self.bg.critic.fr_decay + spikes * (1.0 - self.bg.critic.fr_decay)
        return float(np.dot(self.bg.critic.w_v, fr))

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
        # Błąd musi być przycięty DLA OBU modułów identycznie!
        # Clip [-10.0, 10.0] chroni przed eksplozją wag (-49 przy V=50),
        # zachowując silny bodziec awersyjny, który jest zgodny z tempem nauki Krytyka.

        clipped_td = float(np.clip(td_error, -100.0, 100.0))
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
            self.neuromod.update(
                prediction_error=np.array([abs(td_error)], dtype=np.float32),
                td_error=td_error, novelty=0.0
            )

        # 4. Neuromodulacyjna Plastyczność (Zamknięcie pętli autotuningu)
        self.bg.set_plasticity_timescales(ne=self.neuromod.tau_compression)

        if self._use_wm:
            self.neuromod.apply_to_layer(self.world_model)

        self._step_count += 1

    def reset(self) -> None:
        self.bg.reset_state()
        self._trace = np.zeros(self.state_size, dtype=np.float32)
        if self._use_wm:
            self.world_model.reset_state()
            self.world_model.reset_error_history()