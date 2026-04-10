"""
arena.snn_agent — SNN agent wired through NetworkGraph.

All modules (encoder, WM, critic, actor) are registered as layers in the
NetworkGraph and connected via spike-based feedforward/feedback edges.
The agent only encodes sensory input, steps the graph, and reads outputs.

Pipeline:
  act():     Poisson-encode state → network.step() → actor.get_action()
  observe(): encode next_state → critic step → TD error → BG updates
             → world model update → neuromodulator → sleep consolidation
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from arena.core import Agent
from core.basal_ganglia import (
    ActiveInferenceModule,
    D1D2Actor,
    SNNDeepCritic,
)
from core.config import (
    BasalGangliaConfig,
    EpisodicMemoryConfig,
    NeuromodulatorConfig,
    OscillatorConfig,
    ReplayBufferConfig,
    WorkingMemoryConfig,
    WorldModelConfig,
)
from core.episodic_memory import EpisodicMemory
from core.neuromodulator import NeuromodulatorSystem
from core.network import NetworkGraph
from core.replay_buffer import Experience, ReplayBuffer
from core.spike_encoder import PoissonEncoder
from core.working_memory import WorkingMemoryModule
from core.world_model import SNNWorldModel


class SNNAgent(Agent):
    """SNN-based RL agent wired through NetworkGraph.

    All modules are NetworkGraph layers connected by spike-based edges.
    WM content reaches BG via graph connections (no manual concatenation).

    Biological grounding:
      - D1/D2 MSN action selection (Frank 2005)
      - DA TD-error for critic/actor STDP
      - ACh/DA soft gating for working memory (O'Reilly & Frank 2006)
      - NE-gated episodic storage (O'Neill et al. 2010)
      - SWS/REM two-phase sleep consolidation
      - Expected Free Energy for exploration (Friston 2010)
      - Fast epistemic path: error neurons → D1 excitability
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

        self._bg_config = bg_config or BasalGangliaConfig()
        self._poisson = PoissonEncoder()

        # ── Compute layer sizes ───────────────────────────────────────
        self._wm_num_neurons = max(8, state_size)
        if self._use_working_memory:
            bg_input_size = state_size + self._wm_num_neurons
        else:
            bg_input_size = state_size

        # ── Create modules ────────────────────────────────────────────
        self.critic = SNNDeepCritic(bg_input_size, self._bg_config)
        self.actor = D1D2Actor(
            bg_input_size, n_actions, 1, self._bg_config,
        )

        if self._use_working_memory:
            self.working_memory = WorkingMemoryModule(
                num_external_inputs=state_size,
                num_neurons=self._wm_num_neurons,
                config=wmem_config,
            )

        self.neuromod = NeuromodulatorSystem(nm_config)

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

        # ── Build NetworkGraph ────────────────────────────────────────
        self.network = NetworkGraph(
            osc_config=OscillatorConfig(ctx=self._bg_config.ctx),
            ctx=self._bg_config.ctx,
        )
        self._build_graph()

        # ── Backward compat: BG facade for replay_buffer/sleep ────────
        self.bg = _BGFacade(self.critic, self.actor, self._bg_config)

        # ── Transient state ───────────────────────────────────────────
        self._last_td_error: float = 0.0
        self._step_count: int = 0
        self._episode_return: float = 0.0
        self._episode_steps: int = 0
        self._last_curiosity: float = 0.0
        self._smooth_noise: float = 1.0
        self._best_episode_return: float = -np.inf
        self._last_encoded_state: NDArray[np.float32] | None = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> None:
        """Register all modules as layers and wire connections."""
        net = self.network

        # Critic and Actor get their own TD-based updates
        net.add_layer("critic", self.critic)
        net.add_layer("actor", self.actor)
        net.mark_td_updated("critic", "actor")

        if self._use_working_memory:
            net.add_layer("working_memory", self.working_memory)
            # WM receives sensory input via sensory_inputs dict
            # WM output → critic, actor (concat with sensory)
            net.connect("working_memory", "critic",
                        aggregation_mode="concat")
            net.connect("working_memory", "actor",
                        aggregation_mode="concat")

        if self._use_wm:
            net.add_layer("encoder", self.world_model.encoder)
            # Encoder gets sensory input via sensory_inputs dict
            # No feedforward connection to BG (epistemic path is via
            # set_epistemic_drive, not graph edge)

    # ------------------------------------------------------------------
    # Sensory input builder
    # ------------------------------------------------------------------

    def _build_sensory_inputs(
        self,
        encoded: NDArray[np.float32],
        raw_state: NDArray[np.float32],
    ) -> dict[str, NDArray[np.float32]]:
        """Build sensory_inputs dict for network.step().

        Graph feedforward from WM occupies positions [0:wm_num_neurons]
        via concat aggregation.  Sensory input is padded so Poisson-
        encoded state occupies [wm_num_neurons:], avoiding overlap.
        """
        sensory: dict[str, NDArray[np.float32]] = {}

        if self._use_working_memory:
            sensory["working_memory"] = encoded
            padded = np.zeros(
                self.state_size + self._wm_num_neurons, dtype=np.float32,
            )
            padded[self._wm_num_neurons:] = encoded
            sensory["critic"] = padded
            sensory["actor"] = padded
        else:
            sensory["critic"] = encoded
            sensory["actor"] = encoded

        if self._use_wm:
            sensory["encoder"] = self.world_model._build_input(
                raw_state, np.zeros(self.n_actions, dtype=np.float32),
            )

        return sensory

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray) -> int:
        state_f32 = state.astype(np.float32)
        encoded = self._poisson.encode_value(state_f32)
        self._last_encoded_state = encoded.copy()

        # ── Set DA / epistemic drive BEFORE graph step ────────────────
        da_level = float(np.clip(
            self.neuromod.learning_rate_modulation + 0.5, 0.0, 1.0,
        ))
        self.actor.set_da_level(da_level)

        if self._use_wm:
            self.actor.set_epistemic_drive(
                self.world_model.encoder.prediction_error_rate,
            )

        # ── WM gating (soft sigmoid) ─────────────────────────────────
        if self._use_working_memory:
            wm_da = max(self.neuromod.learning_rate_modulation, 0.0)
            self.working_memory.gate(
                ach_level=self.neuromod.bottom_up_gain,
                da_level=wm_da,
            )

        # ── Step the graph ────────────────────────────────────────────
        sensory = self._build_sensory_inputs(encoded, state_f32)
        self.network.step(
            sensory_inputs=sensory,
            neuromodulator=self.neuromod,
        )

        # ── Active Inference selection ────────────────────────────────
        if self._use_wm:
            selected = self.active_inference.select_action(
                state_spikes=state_f32,
                candidate_actions=list(range(self.n_actions)),
                actor=self.actor,
                ne_level=self.neuromod.competition_sharpness,
            )
            self.actor._last_action = selected

        return self.actor.get_action()

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
        state_f32 = state.astype(np.float32)
        next_f32 = next_state.astype(np.float32)
        next_encoded = self._poisson.encode_value(next_f32)

        is_truncated = info.get("truncated", False) if info else False
        is_terminal = done and not is_truncated

        # ── 1. Pre-update curiosity ───────────────────────────────────
        if self._use_wm:
            self._last_curiosity = self.world_model.curiosity_signal()
        else:
            self._last_curiosity = 0.0

        # ── 2. Intrinsic reward ───────────────────────────────────────
        intrinsic_weight = 0.1 * (1.0 - self.neuromod.learning_rate_modulation)
        intrinsic_r = max(intrinsic_weight, 0.0) * self._last_curiosity
        effective_reward = reward + intrinsic_r

        # ── 3. Critic step on next_state → TD error ──────────────────
        # Save V_trace baseline before processing next_state.
        prev_v_trace = self.critic.v_trace

        # Route through NetworkGraph (oscillator tick, neuromodulator
        # distribution, WM→critic/actor concat handled by graph).
        sensory = self._build_sensory_inputs(next_encoded, next_f32)
        outputs = self.network.step(
            sensory_inputs=sensory,
            neuromodulator=self.neuromod,
        )

        current_v = self.critic.last_value

        if is_terminal:
            td_error = effective_reward - prev_v_trace
        else:
            td_error = (
                effective_reward
                + self._bg_config.gamma * current_v
                - prev_v_trace
            )
        td_error = float(np.clip(td_error, -50.0, 50.0))
        self._last_td_error = td_error

        # ── 4. Consolidation-gated plasticity ─────────────────────────
        gate = self.neuromod.consolidation_gate
        plasticity_scale = 0.8 + 0.2 / (1.0 + np.exp(8.0 * (gate - 0.7)))

        self.critic.update(td_error * plasticity_scale)
        self.actor.update(td_error * plasticity_scale)

        # ── 5. World model + neuromodulator update ────────────────────
        if self._use_wm:
            wm_m_t = max(self.neuromod.learning_rate_modulation, 0.1)
            pred_error = self.world_model.update(
                state_f32, action, next_f32, m_t=wm_m_t,
            )
            self.neuromod.update(
                prediction_error=pred_error,
                td_error=td_error,
                novelty=self._last_curiosity,
            )

            # WM plasticity
            if self._use_working_memory:
                wm_pe = np.zeros(
                    self.working_memory.num_neurons, dtype=np.float32,
                )
                pe_len = min(len(pred_error), self.working_memory.num_neurons)
                wm_pe[:pe_len] = pred_error[:pe_len]
                self.working_memory.prediction_error = wm_pe
                self.working_memory.update_weights(
                    m_t=wm_m_t, pred_error=wm_pe,
                )
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
        self.critic.set_plasticity_timescales(self.neuromod.tau_compression)
        self.actor.set_plasticity_timescales(self.neuromod.tau_compression)

        if self._use_wm:
            self.neuromod.apply_to_layer(self.world_model)
            self.world_model.set_rehearsal_depth(
                self.neuromod.planning_horizon,
            )

        # ── 8. Exploration noise (per-episode smoothing) ──────────────
        if done:
            target_noise = self.bg.compute_exploration_noise(
                self.neuromod.planning_horizon,
                getattr(self.neuromod, 'tonic_da', 0.0),
            )
            self._smooth_noise = (
                self._smooth_noise * 0.8 + target_noise * 0.2
            )

        # ── 9. Store experience ───────────────────────────────────────
        if self._use_wm:
            # Reconstruct critic's effective input from graph outputs.
            if self._use_working_memory:
                wm_spikes = outputs.get(
                    "working_memory",
                    np.zeros(self._wm_num_neurons, dtype=np.float32),
                )
                aug_state = np.concatenate([wm_spikes, next_encoded])
            else:
                aug_state = next_encoded.copy()

            exp = Experience(
                state=state_f32,
                action=action,
                reward=reward,
                next_state=next_f32,
                prediction_error=pred_error,
                encoder_e_bu=self.world_model.encoder.e_bu.copy(),
                encoder_spikes=self.world_model.encoder.spikes_state.astype(
                    np.float32,
                ),
                aug_state=aug_state,
                salience=self.neuromod.competition_sharpness,
                recorded_da=self.neuromod.learning_rate_modulation,
                curiosity=self._last_curiosity,
                done=done,
            )
            self.replay_buffer.store(exp)

            self.episodic_memory.try_store(
                state=state_f32,
                action=action,
                reward=reward,
                next_state=next_f32,
                ne_level=self.neuromod.competition_sharpness,
                prediction_error=pred_error.copy(),
                encoder_e_bu=self.world_model.encoder.e_bu.copy(),
                encoder_spikes=self.world_model.encoder.spikes_state.astype(
                    np.float32,
                ),
                aug_state=aug_state,
            )

        # ── 10. Sleep phase (end of episode) ──────────────────────────
        if done and self._use_wm and len(self.replay_buffer) > 0:
            if self._episode_return > self._best_episode_return:
                self._best_episode_return = self._episode_return

            sleep_gain = 1.0
            if (
                hasattr(self.neuromod, '_reward_history')
                and len(self.neuromod._reward_history) >= 5
            ):
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
        self.network.reset_state()
        if self._use_wm:
            self.world_model.reset_state()
            self.world_model.reset_error_history()


# =====================================================================
# BG Facade for backward compatibility (replay buffer, sleep phase)
# =====================================================================

class _BGFacade:
    """Lightweight facade exposing .critic, .actor, .config interface.

    Replay buffer and sleep phase call bg.critic.forward(),
    bg.critic.update(), etc. This facade routes those calls to the
    real Critic/Actor instances owned by SNNAgent.
    """

    def __init__(
        self,
        critic: SNNDeepCritic,
        actor: D1D2Actor,
        config: BasalGangliaConfig,
    ) -> None:
        self.critic = critic
        self.actor = actor
        self.config = config

    @property
    def last_v(self) -> float:
        """Proxy for critic's last estimated value."""
        return self.critic.last_value

    def reset_state(self) -> None:
        self.critic.reset_state()
        self.actor.reset_state()

    def set_plasticity_timescales(self, ne: float) -> None:
        self.critic.set_plasticity_timescales(ne)
        self.actor.set_plasticity_timescales(ne)

    def compute_exploration_noise(
        self,
        serotonin: float,
        tonic_da: float,
    ) -> float:
        min_exploration = 0.15
        da_noise = max(0.3, 1.0 - tonic_da)
        sero_noise = max(0.3, 1.0 - serotonin)
        return max(min_exploration, da_noise * sero_noise)
