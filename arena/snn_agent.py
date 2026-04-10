"""
arena.snn_agent — Adapter connecting the SNN to the arena Agent protocol.

Clean pipeline with explicit data flow (Phase 9 rewrite):
  1. Encode state → population spikes
  2. WM gate (ACh + DA conjunction) → augmented state
  3. BG: critic evaluates, actor D1/D2 proposes actions
  4. Active inference: combines pragmatic + epistemic → final action
  5. Observe outcome: update WM, world model, BG, neuromodulator
  6. Episodic storage (NE-gated)
  7. End-of-episode: SWS consolidation, then REM refinement

No task-specific tuning. Agent discovers parameters through:
  - Exploration noise: derived from epistemic uncertainty + NE
  - reward_scale: adaptive Welford normalization (in neuromodulator)
  - hidden_size: based on input dimensionality
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from arena.core import Agent
from core.basal_ganglia import (
    ActiveInferenceModule,
    BasalGangliaAGISystem,
    BGSnapshot,
)
from core.config import (
    BasalGangliaConfig,
    EpisodicMemoryConfig,
    NeuromodulatorConfig,
    ReplayBufferConfig,
    WorkingMemoryConfig,
    WorldModelConfig,
)
from core.episodic_memory import EpisodicMemory
from core.neuromodulator import NeuromodulatorSystem
from core.replay_buffer import Experience, ReplayBuffer
from core.working_memory import WorkingMemoryModule
from core.world_model import SNNWorldModel


class SNNAgent(Agent):
    """SNN-based RL agent using Basal Ganglia + Active Inference.

    All learning arises from biologically grounded mechanisms:
      - D1/D2 MSN pathways for action selection (Frank 2005)
      - DA TD-error for critic/actor plasticity
      - ACh/DA conjunction gating for working memory (O'Reilly & Frank 2006)
      - NE-gated episodic storage (O'Neill et al. 2010)
      - SWS/REM two-phase sleep consolidation
      - Expected Free Energy for exploration/exploitation (Friston 2010)
    """

    def __init__(
        self,
        state_size: int,
        n_actions: int,
        bg_config: BasalGangliaConfig | None = None,
        wm_config: WorldModelConfig | None = None,
        nm_config: NeuromodulatorConfig | None = None,
        ep_config: EpisodicMemoryConfig | None = None,
        rb_config: ReplayBufferConfig | None = None,
        wmem_config: WorkingMemoryConfig | None = None,
        use_world_model: bool = True,
        use_working_memory: bool = True,
    ) -> None:
        self.state_size = state_size
        self.n_actions = n_actions
        self._use_wm = use_world_model
        self._use_working_memory = use_working_memory and use_world_model

        # ── Working Memory ────────────────────────────────────────────
        self._wm_num_neurons = max(8, state_size)
        if self._use_working_memory:
            self.working_memory = WorkingMemoryModule(
                num_external_inputs=state_size,
                num_neurons=self._wm_num_neurons,
                config=wmem_config,
            )
            bg_input_size = state_size + self._wm_num_neurons
        else:
            bg_input_size = state_size

        # ── Basal Ganglia (D1/D2 Actor + LIF Critic) ─────────────────
        self._bg_config = bg_config or BasalGangliaConfig()
        self.bg = BasalGangliaAGISystem(
            state_size=bg_input_size,
            motor_dim=n_actions,
            internal_dim=1,
            config=self._bg_config,
        )

        # ── Neuromodulator ────────────────────────────────────────────
        self.neuromod = NeuromodulatorSystem(nm_config)

        # ── World Model + Active Inference ────────────────────────────
        if self._use_wm:
            self.world_model = SNNWorldModel(
                state_size=state_size,
                action_size=n_actions,
                config=wm_config,
            )
            self.active_inference = ActiveInferenceModule(self.world_model)
            self.replay_buffer = ReplayBuffer(config=rb_config)
            self.episodic_memory = EpisodicMemory(
                state_dim=state_size, config=ep_config,
            )

        # ── Transient state ───────────────────────────────────────────
        self._last_td_error: float = 0.0
        self._step_count: int = 0
        self._episode_return: float = 0.0
        self._episode_steps: int = 0
        self._last_aug_state: NDArray[np.float32] | None = None
        self._last_curiosity: float = 0.0

        # ── Exploration noise smoothing (D1/D2 receptor inertia) ──────
        self._smooth_noise: float = 1.0

        # ── Best-episode tracking for sleep gain ──────────────────────
        self._best_episode_return: float = -np.inf

    # ------------------------------------------------------------------
    # State augmentation
    # ------------------------------------------------------------------

    def _augment_state(self, state: NDArray[np.float32]) -> NDArray[np.float32]:
        """Concatenate raw state with WM content (if enabled)."""
        state_f32 = state.astype(np.float32)
        if self._use_working_memory:
            da_level = max(self.neuromod.learning_rate_modulation, 0.0)
            self.working_memory.gate(
                ach_level=self.neuromod.bottom_up_gain,
                da_level=da_level,
            )
            self.working_memory.forward(state_f32)
            wm_signal = self.working_memory.content.copy()
            return np.concatenate([state_f32, wm_signal])
        return state_f32

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray) -> int:
        aug = self._augment_state(state)
        self._last_aug_state = aug.copy()

        # Pre-compute critic value for TD error in observe()
        self.bg.last_v = self.bg.critic.forward(aug)

        # Set DA level on actor for D1/D2 modulation
        da_level = float(np.clip(
            self.neuromod.learning_rate_modulation + 0.5, 0.0, 1.0,
        ))
        self.bg.actor.set_da_level(da_level)

        if self._use_wm:
            # Active Inference: expected free energy selection
            selected = self.active_inference.select_action(
                state_spikes=state.astype(np.float32),
                candidate_actions=list(range(self.n_actions)),
                actor=self.bg.actor,
                ne_level=self.neuromod.competition_sharpness,
            )
            self.bg.actor.forward(aug, forced_action=selected)
        else:
            self.bg.actor.forward(aug)

        return self.bg.actor.get_action()

    # ------------------------------------------------------------------
    # Observation & learning
    # ------------------------------------------------------------------

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        next_aug = self._augment_state(next_state)
        is_truncated = info.get("truncated", False) if info else False
        is_terminal = done and not is_truncated

        # ── 1. Pre-update curiosity (Lisman & Grace 2005) ─────────────
        if self._use_wm:
            self._last_curiosity = self.world_model.curiosity_signal()
        else:
            self._last_curiosity = 0.0

        # ── 2. Intrinsic reward (inverse tonic DA weighting) ──────────
        intrinsic_weight = 0.1 * (1.0 - self.neuromod.learning_rate_modulation)
        intrinsic_r = max(intrinsic_weight, 0.0) * self._last_curiosity
        effective_reward = reward + intrinsic_r

        # ── 3. TD error ───────────────────────────────────────────────
        if is_terminal:
            td_error = effective_reward - self.bg.last_v
        else:
            next_v = self.bg.critic.peek(next_aug)
            td_error = (
                effective_reward
                + self._bg_config.gamma * next_v
                - self.bg.last_v
            )
        td_error = float(np.clip(td_error, -50.0, 50.0))

        # ── 4. Consolidation-gated plasticity ─────────────────────────
        gate = self.neuromod.consolidation_gate
        plasticity_scale = 0.8 + 0.2 / (1.0 + np.exp(8.0 * (gate - 0.7)))

        self.bg.critic.update(td_error * plasticity_scale)
        self.bg.actor.update(td_error * plasticity_scale)
        self._last_td_error = td_error

        # ── 5. World model + neuromodulator update ────────────────────
        if self._use_wm:
            state_f32 = state.astype(np.float32)
            next_f32 = next_state.astype(np.float32)
            wm_m_t = max(self.neuromod.learning_rate_modulation, 0.1)

            pred_error = self.world_model.update(
                state_f32, action, next_f32, m_t=wm_m_t,
            )
            self.neuromod.update(
                prediction_error=pred_error,
                td_error=td_error,
                novelty=self._last_curiosity,
            )

            # Working Memory plasticity
            if self._use_working_memory:
                wm_pe = np.zeros(self.working_memory.num_neurons, dtype=np.float32)
                pe_len = min(len(pred_error), self.working_memory.num_neurons)
                wm_pe[:pe_len] = pred_error[:pe_len]
                self.working_memory.prediction_error = wm_pe
                self.working_memory.update_weights(m_t=wm_m_t, pred_error=wm_pe)
        else:
            norm_pe = float(abs(td_error) / (1.0 + abs(td_error)))
            self.neuromod.update(
                prediction_error=np.array([norm_pe], dtype=np.float32),
                td_error=td_error,
                novelty=0.0,
            )

        # ── 6. Tonic DA (episode-level) ───────────────────────────────
        self._episode_return += reward
        self._episode_steps += 1
        if done:
            ep_pe = 0.0
            if self._use_wm and self.world_model.error_history:
                ep_pe = float(np.mean(self.world_model.error_history))
            self.neuromod.update_tonic_da(
                self._episode_return,
                self._episode_steps,
                prediction_error_avg=ep_pe,
            )
            self._episode_return = 0.0
            self._episode_steps = 0

        # ── 7. NE-driven trace compression ────────────────────────────
        self.bg.set_plasticity_timescales(ne=self.neuromod.tau_compression)

        if self._use_wm:
            self.neuromod.apply_to_layer(self.world_model)
            self.world_model.set_rehearsal_depth(self.neuromod.planning_horizon)

        # ── 8. Exploration noise (per-episode smoothing) ──────────────
        if done:
            target_noise = self.bg.compute_exploration_noise(
                self.neuromod.planning_horizon,
                getattr(self.neuromod, 'tonic_da', 0.0),
            )
            self._smooth_noise = self._smooth_noise * 0.8 + target_noise * 0.2

        # ── 9. Store experience ───────────────────────────────────────
        if self._use_wm:
            bg_snap = self.bg.snapshot_traces()
            pred_error_local = pred_error  # noqa: F841 — from step 5 above

            exp = Experience(
                state=state.astype(np.float32),
                action=action,
                reward=reward,
                next_state=next_state.astype(np.float32),
                prediction_error=pred_error_local,
                encoder_e_bu=self.world_model.encoder.e_bu.copy(),
                encoder_spikes=self.world_model.encoder.spikes_state.astype(
                    np.float32,
                ),
                bg_snapshot=bg_snap,
                aug_state=self._last_aug_state,
                salience=self.neuromod.competition_sharpness,
                recorded_da=self.neuromod.learning_rate_modulation,
                curiosity=self._last_curiosity,
                done=done,
            )
            self.replay_buffer.store(exp)

            # NE-gated episodic memory (one-shot, O'Neill et al. 2010)
            self.episodic_memory.try_store(
                state=state.astype(np.float32),
                action=action,
                reward=reward,
                next_state=next_state.astype(np.float32),
                ne_level=self.neuromod.competition_sharpness,
                prediction_error=pred_error_local.copy(),
                encoder_e_bu=self.world_model.encoder.e_bu.copy(),
                encoder_spikes=self.world_model.encoder.spikes_state.astype(
                    np.float32,
                ),
                bg_snapshot=bg_snap,
                aug_state=self._last_aug_state,
            )

        # ── 10. Sleep phase (end of episode) ──────────────────────────
        if done and self._use_wm and len(self.replay_buffer) > 0:
            # Sleep gain from episode quality (Bethus et al. 2010)
            if self._episode_return > self._best_episode_return:
                self._best_episode_return = self._episode_return

            sleep_gain = 1.0
            if hasattr(self.neuromod, '_reward_history') and len(
                self.neuromod._reward_history,
            ) >= 5:
                r_arr = np.array(self.neuromod._reward_history)
                r_mean = float(np.mean(r_arr))
                r_std = float(np.std(r_arr)) + 1e-8
                quality = (self._episode_return - r_mean) / r_std
                sleep_gain = float(np.clip(1.0 + 0.5 * quality, 1.0, 2.5))

            self.replay_buffer.sleep_phase(
                world_model=self.world_model,
                neuromodulator=self.neuromod,
                bg=self.bg,
                sleep_gain=sleep_gain,
            )

        self._step_count += 1

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.bg.reset_state()
        if self._use_wm:
            self.world_model.reset_state()
            self.world_model.reset_error_history()
        if self._use_working_memory:
            self.working_memory.reset_state()
