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
    WorkingMemoryConfig,
    EpisodicMemoryConfig,
)
from core.replay_buffer import ReplayBuffer
from core.active_inference import ActiveInferenceModule
from core.episodic_memory import EpisodicMemory
from core.working_memory import WorkingMemoryModule

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

        # Working Memory: activated when both world model and trace are used.
        # Biological grounding: prefrontal working memory operates regardless
        # of input dimensionality. Even low-dimensional inputs are expanded
        # into a higher-dimensional persistent attractor representation
        # (Goldman-Rakic 1995). The minimum population size (8 neurons)
        # ensures enough capacity for stable attractor dynamics.
        self._use_working_memory = self._use_wm and self._use_trace
        if self._use_working_memory:
            self._wm_num_neurons = max(8, state_size)
            self.working_memory = WorkingMemoryModule(
                num_external_inputs=state_size,
                num_neurons=self._wm_num_neurons,
            )
            bg_input_size = state_size + self._wm_num_neurons
        elif self._use_trace:
            bg_input_size = state_size * 2
        else:
            bg_input_size = state_size

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

            # Episodic Memory (hippocampal CA3 one-shot binding):
            # Stores rare, high-NE transitions for injection into replay
            # buffer during sleep. Critical for sparse-reward environments
            # where a single success must be remembered despite hundreds
            # of failures (O'Neill et al. 2010; Cheng & Frank 2008).
            self.episodic_memory = EpisodicMemory(state_dim=state_size)

        self._trace = np.zeros(state_size, dtype=np.float32)
        self._last_td_error: float = 0.0
        self._step_count: int = 0
        self._episode_return: float = 0.0   # Accumulated return for tonic DA
        self._episode_steps: int = 0
        self._last_aug_state: np.ndarray | None = None  # Cached augmented state from act()

        # ── Exploration noise smoothing (D1/D2 receptor inertia) ─────
        # Biological basis: striatal receptor density changes over
        # hours/days, not per-step. A single bad episode should NOT
        # spike exploration — the noise "policy" must be tonic,
        # integrating over many episodes to avoid vicious cycles
        # (bad episode → high noise → bad episode → ...).
        # τ_noise ≈ 200 steps (one full episode) ensures episode-level
        # smoothing while still tracking genuine environmental changes.
        self._smooth_noise: float = 1.0  # Starts at max exploration
        self._noise_filter_decay: float = 0.98  # τ ≈ 50 steps (~1/3 episode)

        # ── Best-episode priority replay (hippocampal "golden trace") ─
        # Biology: hippocampus preferentially replays reward-associated
        # trajectories during SWR bursts (Dupret et al. 2010), with
        # frequency proportional to reward magnitude. This ensures
        # rare but significant experiences are consolidated across
        # multiple sleep cycles, not just once.
        self._best_episode_buffer: list = []  # Stores Experience objects from best episode
        self._best_episode_return: float = -np.inf

    def _augment_state(self, state: np.ndarray) -> np.ndarray:
        """Augment raw state with temporal context.

        When WorkingMemory is active (WM mode):
          State is concatenated with WM's persistent content signal.
          The WM module sustains activity through recurrent attractor
          dynamics (tau_m=300ms), providing a biologically grounded
          temporal context that outlasts the original stimulus.
          ACh gates WM update: high ACh (novel situation) → accept new
          input; low ACh (familiar) → sustain existing content.

        Fallback (trace mode):
          Simple exponential moving average of past states.
        """
        state_f32 = state.astype(np.float32)
        if self._use_working_memory:
            # Gate WM by ACh level from neuromodulator
            self.working_memory.gate(self.neuromod.acetylcholine)
            # Forward pass: integrate input if gate open, sustain if closed
            self.working_memory.forward(state_f32)
            # Content is the low-pass filtered activation (rate-coded)
            wm_signal = self.working_memory.content.copy()
            return np.concatenate([state_f32, wm_signal])
        elif self._use_trace:
            return np.concatenate([state_f32, self._trace])
        else:
            return state_f32

    def _update_trace(self, state: np.ndarray) -> None:
        """Update simple EMA trace (only used in non-WM trace mode)."""
        if self._use_trace and not self._use_working_memory:
            self._trace = self._trace * self._trace_decay + state.astype(np.float32)

    def act(self, state: np.ndarray) -> int:
        aug = self._augment_state(state)
        self._update_trace(state)
        self._last_aug_state = aug.copy()  # Cache for replay buffer in observe()

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
            # Save return for best-episode tracking (needed in sleep phase below)
            self._last_episode_return = self._episode_return
            # Compute episode-average prediction error for intrinsic progress
            ep_pe = 0.0
            if hasattr(self, 'world_model') and hasattr(self.world_model, '_error_history') and self.world_model._error_history:
                ep_pe = float(np.mean(self.world_model._error_history))
            self.neuromod.update_tonic_da(self._episode_return, self._episode_steps,
                                          prediction_error_avg=ep_pe)
            self._episode_return = 0.0
            self._episode_steps = 0

        # 5. NE-driven trace compression (closed loop: TD → NM → BG timescales)
        self.bg.set_plasticity_timescales(ne=self.neuromod.tau_compression)

        if self._use_wm:
            self.neuromod.apply_to_layer(self.world_model)

        # 6. Exploration control: delegate to BG which combines serotonin,
        #    tonic DA floor, and action entropy (Proposal 2).
        #
        #    Smoothing (D1/D2 receptor inertia): the raw target noise is
        #    filtered through a slow EMA to prevent individual bad episodes
        #    from spiking exploration into a vicious failure cascade.
        #    Biologically, receptor density changes on timescales of
        #    hours/days, not seconds (Seamans & Yang 2004).
        _target_noise = self.bg.compute_exploration_noise(
            self.neuromod.serotonin, self.neuromod.tonic_da
        )
        d = self._noise_filter_decay
        self._smooth_noise = self._smooth_noise * d + _target_noise * (1.0 - d)
        self.bg.actor.noise_scale = self._smooth_noise

        # 7. Zapis do Replay Buffer (tylko z World Modelem)
        if self._use_wm:
            # SNNWorldModel posiada wewnątrz warstwę PredictiveCodingLayer jako _encoder
            layer_traces = {'encoder': self.world_model._encoder.e}
            layer_outputs = {'encoder': self.world_model._encoder.has_spiked.astype(np.float32)}
            layer_errors = {'encoder': self.world_model._encoder.prediction_error}

            # BG eligibility trace snapshot (Lansink et al. 2009):
            # Hippocampal place cells co-fire with ventral striatal neurons
            # during behavior. During subsequent sleep SWR, these co-activation
            # patterns are replayed, reactivating the striatal eligibility
            # traces that were present during the original experience.
            bg_traces = {
                'critic_e_h': self.bg.critic.e_h.copy(),
                'critic_e_v': self.bg.critic.e_v.copy(),
                'critic_e_bv': np.array([self.bg.critic.e_bv], dtype=np.float32),
                'actor_e': self.bg.actor.e_actor.copy(),
            }

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
                recorded_da=self.neuromod.learning_rate_modulation,  # Zamrożony sygnał DA!
                bg_traces=bg_traces,
                aug_state=self._last_aug_state,
            )

            # 7b. Episodic Memory: NE-gated one-shot storage (hippocampal CA3).
            #     When NE is high (surprise/arousal), store this transition
            #     in the episodic memory for later injection during sleep.
            #     Critical for sparse-reward environments where successful
            #     episodes are rare and would be diluted in the main buffer
            #     (O'Neill et al. 2010; Cheng & Frank 2008).
            self.episodic_memory.try_store(
                state=state.astype(np.float32),
                action=action,
                reward=reward,
                next_state=next_state.astype(np.float32),
                ne_level=self.neuromod.noradrenaline,
                bg_traces={
                    'critic_e_h': self.bg.critic.e_h.copy(),
                    'critic_e_v': self.bg.critic.e_v.copy(),
                    'critic_e_bv': np.array([self.bg.critic.e_bv], dtype=np.float32),
                    'actor_e': self.bg.actor.e_actor.copy(),
                },
                aug_state=self._last_aug_state,
                layer_traces={'encoder': self.world_model._encoder.e.copy()},
                layer_outputs={'encoder': self.world_model._encoder.has_spiked.astype(np.float32)},
                prediction_error=pred_error.copy(),
            )

        # 8. Faza snu (Offline Consolidation) na koniec epizodu
        if done and self._use_wm and len(self.replay_buffer) > 0:
            # 8a. Inject episodic memories into replay buffer before sleep.
            for ep in self.episodic_memory.recall_all():
                self.replay_buffer.store(
                    state=ep.state,
                    action=ep.action,
                    reward=ep.reward,
                    next_state=ep.next_state,
                    layer_traces=ep.layer_traces,
                    layer_outputs=ep.layer_outputs,
                    prediction_error=(
                        ep.prediction_error
                        if ep.prediction_error is not None
                        else np.zeros(self.state_size, dtype=np.float32)
                    ),
                    salience=ep.salience,
                    recorded_da=self.neuromod.learning_rate_modulation,
                    bg_traces=ep.bg_traces,
                    aug_state=ep.aug_state,
                )

            # 8b. Proportional sleep gain (VTA DA modulation of replay).
            #     Biology (Bethus et al. 2010; McNamara et al. 2014):
            #     VTA phasic dopamine during hippocampal SWR determines
            #     the gain of hippocampal-to-cortical plasticity. Better
            #     episodes trigger stronger DA burst, which amplifies the
            #     replay consolidation signal. This ensures that rare but
            #     significantly better episodes receive proportionally
            #     STRONGER consolidation, while mediocre or bad episodes
            #     receive weaker consolidation.
            #
            #     Gain is computed as a z-score of the current episode's
            #     return relative to recent history, clipped to [0.3, 2.5].
            #     This provides up to 2.5× amplification for +3σ episodes
            #     and 0.3× attenuation for very bad episodes.
            ep_return = getattr(self, '_last_episode_return', 0.0)
            # Track best return for diagnostics
            if ep_return > self._best_episode_return:
                self._best_episode_return = ep_return

            sleep_gain = 1.0
            if len(self.neuromod._reward_history) >= 5:
                _r_arr = np.array(self.neuromod._reward_history)
                _r_mean = float(np.mean(_r_arr))
                _r_std = float(np.std(_r_arr)) + 1e-8
                _quality = (ep_return - _r_mean) / _r_std
                # Only AMPLIFY good episodes, never attenuate bad ones.
                # Bad episodes still contain useful learning signal (state
                # dynamics, non-goal value landscape). Attenuating them
                # slows down critic learning during early training.
                # Amplification range: [1.0, 2.5] — a +3σ episode gets
                # 2.5× consolidation, average episode gets 1.0×.
                sleep_gain = max(1.0, float(np.clip(1.0 + 0.5 * _quality, 1.0, 2.5)))

            layers_dict = {'encoder': self.world_model._encoder}
            self.replay_buffer.sleep_phase(
                layers=layers_dict,
                world_model=self.world_model,
                neuromodulator=self.neuromod,
                bg=self.bg,
                sleep_gain=sleep_gain,
            )
            # Biological basis (McClelland et al. 1995, Complementary Learning
            # Systems): hippocampus maintains recent experience traces across
            # multiple sleep cycles, not just one. The buffer's FIFO capacity
            # limit naturally drops old experiences, modelling the days-to-weeks
            # decay of hippocampal traces as they transfer to neocortex.
            # Removing clear() allows successful trajectories to be replayed
            # across multiple sleep phases, matching SWR multi-night replay.

        self._step_count += 1

    def reset(self) -> None:
        self.bg.reset_state()
        self._trace = np.zeros(self.state_size, dtype=np.float32)
        if self._use_wm:
            self.world_model.reset_state()
            self.world_model.reset_error_history()
        if self._use_working_memory:
            self.working_memory.reset_state()