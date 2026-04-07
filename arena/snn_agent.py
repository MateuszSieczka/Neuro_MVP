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
from core.replay_buffer import ReplayBuffer
from core.active_inference import ActiveInferenceModule

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
            self.active_inference = ActiveInferenceModule(self.world_model)
            self.replay_buffer = ReplayBuffer(capacity=1000)

        self._trace = np.zeros(state_size, dtype=np.float32)
        self._last_td_error: float = 0.0
        self._step_count: int = 0
        self._episode_return: float = 0.0   # Accumulated return for tonic DA
        self._episode_steps: int = 0

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

        if self._use_wm:
            # Active Inference: Oblicz pragmatyczne preferencje Aktora (logity MSN)
            logits = np.dot(aug, self.bg.actor.w_mu)
            pragmatic_values = {a: float(logits[a]) for a in range(self.n_actions)}

            candidate_actions = list(range(self.n_actions))
            selected_action = self.active_inference.select_action(
                state_spikes=state.astype(np.float32),
                # <--- POPRAWKA: Przekazujemy czysty 'state', a nie rozszerzony 'aug'!
                candidate_actions=candidate_actions,
                pragmatic_values=pragmatic_values,
                ne_level=self.neuromod.noradrenaline
            )
            # Zmuszamy Aktora do wykonania tej akcji, by policy gradient i ślad E zapisały się poprawnie
            self.bg.actor.forward(aug, forced_action=selected_action)
        else:
            # Fallback dla środowisk gęstych bez modelu świata (standardowa eksploracja z szumem)
            self.bg.actor.forward(aug)

        # Forward Krytyka (V(s))
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

        clipped_td = float(np.clip(td_error, -10.0, 10.0))

        # 2. Adaptive DA gain normalization (Tobler, Fiorillo & Schultz 2005)
        #    VTA dopamine neurons scale their phasic burst magnitude inversely
        #    with the variance of recent reward prediction errors. This is a
        #    fundamental property of midbrain DA signaling:
        #    - High variance → low gain → prevents weight explosions
        #    - Low variance → high gain → fine-tuning during consolidation
        #    The BG system maintains a running RMS of TD error for this.
        norm_td = self.bg.normalize_td(clipped_td)

        # 3. Consolidation-gated plasticity modulation
        #    Biological basis (Niv et al. 2007; Doya 2002):
        #    Plasticity decreases when the agent is BOTH:
        #    (a) consistently rewarded (high tonic DA from VTA), AND
        #    (b) making stable predictions (high serotonin from raphe).
        #    Gate = sqrt(tonic_DA × 5-HT), requiring both signals.
        #    Linear scaling with low floor models the transition from
        #    early-phase LTP (labile, high plasticity) to late-phase L-LTP
        #    (protein-synthesis-dependent, protected). The floor at 0.05
        #    corresponds to the minimal synaptic modification that occurs
        #    even during deep consolidation (Frey & Morris 1997).
        #    gate=0 → 1.0, gate=0.5 → 0.50, gate=0.9 → 0.10, gate=0.95 → 0.05
        gate = self.neuromod.consolidation_gate
        plasticity_scale = max(0.05, 1.0 - gate)

        self.bg.critic.update(norm_td * plasticity_scale)
        self.bg.actor.update(norm_td * plasticity_scale)

        self._last_td_error = td_error

        # 4. Neuromodulator update (raw TD for proper NE/DA/5-HT dynamics)
        #    Note: neuromod receives RAW td_error, not normalized — the phasic
        #    DA sigmoid and NE channels need the actual error magnitude to
        #    properly track surprise and RPE direction.
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
            norm_td = float(abs(td_error) / (1.0 + abs(td_error)))
            self.neuromod.update(
                prediction_error=np.array([norm_td], dtype=np.float32),
                td_error=td_error, novelty=0.0,
            )

        # 4b. Episode-level tonic DA update (ventral striatum → VTA)
        self._episode_return += reward
        self._episode_steps += 1
        if done:
            self.neuromod.update_tonic_da(self._episode_return, self._episode_steps)
            self._episode_return = 0.0
            self._episode_steps = 0

        # 5. NE-driven trace compression (closed loop: TD → NM → BG timescales)
        self.bg.set_plasticity_timescales(ne=self.neuromod.tau_compression)

        if self._use_wm:
            self.neuromod.apply_to_layer(self.world_model)

        # 6. Exploration control: serotonin + tonic DA performance floor
        #    Primary signal: serotonin — prediction accuracy.
        #    (1-sero)² drives noise. Works well for dense reward.
        #
        #    Performance floor from tonic DA (Niv et al. 2007):
        #    Low tonic DA means the agent is NOT being consistently rewarded.
        #    In that regime, exploration must remain high regardless of how
        #    "stable" predictions are (serotonin). An agent that consistently
        #    picks the wrong action has low TD error → high serotonin, but
        #    low tonic DA reveals it hasn't found reward yet.
        #
        #    da_floor = 0.5 × (1 − tonic_da):
        #      tda=0.0 (no reward experience) → floor=0.5 (strong exploration)
        #      tda=0.5 (neutral/stagnating)   → floor=0.25 (moderate exploration)
        #      tda=1.0 (consistently rewarded) → floor=0.0 (serotonin takes over)
        #
        #    This addresses the sparse-reward trap (MountainCar / corridor):
        #    critic converges to V≈const, TD→0, sero→1, but tonic_da stays
        #    low → da_floor keeps noise alive → agent eventually discovers reward.
        MIN_EXPLORATION = 0.01
        sero_noise = (1.0 - self.neuromod.serotonin) ** 2
        tda = self.neuromod.tonic_da
        da_floor = 0.5 * (1.0 - tda)
        self.bg.actor.noise_scale = max(MIN_EXPLORATION, sero_noise, da_floor)

        # 7. Zapis do Replay Buffer (tylko z World Modelem)
        if self._use_wm:
            # SNNWorldModel posiada wewnątrz warstwę PredictiveCodingLayer jako _encoder
            layer_traces = {'encoder': self.world_model._encoder.e}
            layer_outputs = {'encoder': self.world_model._encoder.has_spiked.astype(np.float32)}
            layer_errors = {'encoder': self.world_model._encoder.prediction_error}

            self.replay_buffer.store(
                state=state.astype(np.float32),
                action=action,
                reward=reward,
                next_state=next_state.astype(np.float32),
                layer_traces=layer_traces,
                layer_outputs=layer_outputs,
                prediction_error=pred_error,
                layer_errors=layer_errors,
                salience=self.neuromod.noradrenaline,  # Salience = NE (arousal/zaskoczenie)
                recorded_da=self.neuromod.learning_rate_modulation  # Zamrożony sygnał DA!
            )

        # 8. Faza snu (Offline Consolidation) na koniec epizodu
        if done and self._use_wm and len(self.replay_buffer) > 0:
            layers_dict = {'encoder': self.world_model._encoder}
            # Odtwarzamy wstecznie epizod, używając zamrożonego recorded_da
            self.replay_buffer.sleep_phase(
                layers=layers_dict,
                world_model=self.world_model,
                neuromodulator=self.neuromod
            )
            # Po śnie czyścimy bufor, aby nie mieszać epizodów
            # (hipokamp przekazał wiedzę do kory)
            self.replay_buffer.clear()

        self._step_count += 1

    def reset(self) -> None:
        self.bg.reset_state()
        self._trace = np.zeros(self.state_size, dtype=np.float32)
        if self._use_wm:
            self.world_model.reset_state()
            self.world_model.reset_error_history()