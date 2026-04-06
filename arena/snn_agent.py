"""
arena.snn_agent — Adapter connecting the Neuro_MVP SNN to the arena Agent protocol.

This is the ONLY file that imports from ``core``.  The arena framework itself
is completely SNN-agnostic.
"""

from __future__ import annotations

import numpy as np

from arena.core import Agent
from core.basal_ganglia import BasalGangliaAGISystem, ContinuousBGConfig
from core.neuromodulator import NeuromodulatorSystem
from core.world_model import SNNWorldModel
from core.config import (
    NeuromodulatorConfig,
    SNNWorldModelConfig,
)


class SNNAgent(Agent):
    """
    Wraps BasalGangliaAGI + WorldModel + Neuromodulator into an arena Agent.

    Architecture:
      act()     — uses actor.forward() ONLY to pick an action (no weight updates).
      observe() — runs ONE bg.step() with actual reward to compute TD error
                  and update both critic and actor weights.

    State trace (working memory):
      When ``trace_decay > 0``, the agent augments the raw state with a
      decaying trace of previous states.  This gives the actor+critic
      access to temporal context without requiring recurrence in the BG.
      Biologically: persistent prefrontal activity / sustained firing.
      Effective input size = state_size * 2 (raw + trace).
    """

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

        # With trace, BG sees [state ‖ trace] = 2× state_size
        bg_input_size = state_size * 2 if self._use_trace else state_size

        self._bg_config = bg_config or ContinuousBGConfig(
            gamma=0.95,
            critic_lr=0.01,
            actor_lr=0.005,
            exploration_noise=0.3,
            hidden_size=64,
        )

        self.bg = BasalGangliaAGISystem(
            state_size=bg_input_size,
            motor_dim=n_actions,
            internal_dim=1,
            config=self._bg_config,
        )

        if self._use_wm:
            self._wm_config = wm_config or SNNWorldModelConfig(
                hidden_size=32,
                k_winners=4,
                rehearsal_steps=5,
            )
            self.world_model = SNNWorldModel(
                state_size=state_size,
                action_size=n_actions,
                config=self._wm_config,
            )
            self.neuromod = NeuromodulatorSystem(nm_config)

        # State trace (working memory)
        self._trace = np.zeros(state_size, dtype=np.float32)

        # Tracking
        self._last_td_error: float = 0.0
        self._step_count: int = 0

    def _augment_state(self, state: np.ndarray) -> np.ndarray:
        """Concatenate raw state with decaying trace if enabled."""
        if not self._use_trace:
            return state.astype(np.float32)
        return np.concatenate([state.astype(np.float32), self._trace])

    def _update_trace(self, state: np.ndarray) -> None:
        """Decay trace and add current state."""
        if self._use_trace:
            self._trace = self._trace * self._trace_decay + state.astype(np.float32)

    def act(self, state: np.ndarray) -> int:
        aug = self._augment_state(state)
        self._update_trace(state)

        # Actor forward ONLY — no learning, no critic update.
        motor, _internal = self.bg.actor.forward(aug)

        # Critic forward to set last_v for the upcoming TD calc.
        v = self.bg.critic.forward(aug)
        self.bg.last_v = v

        # Discretise: argmax over motor channels
        action = int(np.argmax(motor))
        return action

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        next_aug = self._augment_state(next_state)

        # BG learning step — ONE step with actual reward.
        _, _, td_error = self.bg.step(next_aug, reward, is_terminal=done)
        self._last_td_error = td_error

        # World model + neuromodulator (optional)
        if self._use_wm:
            state_f32 = state.astype(np.float32)
            next_f32 = next_state.astype(np.float32)
            pred_error = self.world_model.update(
                state_f32, action, next_f32, m_t=max(abs(td_error), 0.1)
            )
            self.neuromod.update(
                prediction_error=pred_error,
                td_error=td_error,
                novelty=self.world_model.curiosity_signal(),
            )

        self._step_count += 1

    def reset(self) -> None:
        self.bg.reset_state()
        self._trace = np.zeros(self.state_size, dtype=np.float32)
        if self._use_wm:
            self.world_model.reset_state()
            self.world_model.reset_error_history()
