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
from core.columnar import build_columnar_network, split_input
from core.config import (
    AgentConfig,
    AttentionConfig,
    BasalGangliaConfig,
    EpisodicMemoryConfig,
    NeuromodulatorConfig,
    OscillatorConfig,
    ReplayBufferConfig,
    WorkingMemoryConfig,
    WorldModelConfig,
    CORTICAL_L4_RECEPTORS,
    PFC_RECEPTORS,
    STRIATUM_D1_RECEPTORS,
    STRIATUM_D2_RECEPTORS,
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
        agent_cfg: AgentConfig | None = None,
        use_world_model: bool = True,
        use_working_memory: bool = True,
        use_columnar: bool = False,
        receptive_field_size: int | None = None,
        neurons_per_column: int | None = None,
        assoc_neurons: int | None = None,
    ) -> None:
        self.state_size = state_size
        self.n_actions = n_actions
        self._agent_cfg = agent_cfg or AgentConfig()
        self._use_wm = use_world_model
        self._use_working_memory = use_working_memory and use_world_model
        self._use_columnar = use_columnar

        self._bg_config = bg_config or BasalGangliaConfig()
        self._poisson = PoissonEncoder()

        # ── Columnar-mode derived dimensions ──────────────────────────
        if self._use_columnar:
            self._rf_size = receptive_field_size or 4
            self._neurons_per_col = neurons_per_column or max(8, self._rf_size)
            self._assoc_neurons = assoc_neurons or max(32, state_size // 2)
            bg_base_size = self._assoc_neurons
        else:
            self._rf_size = 0
            self._neurons_per_col = 0
            self._assoc_neurons = 0
            bg_base_size = state_size

        # ── Compute layer sizes ───────────────────────────────────────
        self._wm_num_neurons = max(8, state_size)
        if self._use_working_memory:
            bg_input_size = bg_base_size + self._wm_num_neurons
        else:
            bg_input_size = bg_base_size

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
        self._attention = None  # SpatialAttentionController (columnar only)
        self._column_names: list[str] = []
        self._kwta_names: list[str] = []
        self._assoc_name: str = ""
        self._build_graph()

        # ── Backward compat: BG facade for replay_buffer/sleep ────────
        self.bg = _BGFacade(self.critic, self.actor, self._bg_config, self._agent_cfg)

        # ── Transient state ───────────────────────────────────────────
        self._last_td_error: float = 0.0
        self._step_count: int = 0
        self._last_curiosity: float = 0.0
        self._last_encoded_state: NDArray[np.float32] | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def use_world_model(self) -> bool:
        """Whether the world model is active."""
        return self._use_wm

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> None:
        """Register all modules as layers and wire connections."""
        net = self.network

        # ── Columnar layers (PC → k-WTA → assoc) ─────────────────────
        if self._use_columnar:
            _, col_names, kwta_names, assoc_name, attn = build_columnar_network(
                input_dim=self.state_size,
                receptive_field_size=self._rf_size,
                neurons_per_column=self._neurons_per_col,
                assoc_neurons=self._assoc_neurons,
                net=net,
            )
            self._column_names = col_names
            self._kwta_names = kwta_names
            self._assoc_name = assoc_name
            self._attention = attn

        # Critic and Actor get their own TD-based updates
        net.add_layer("critic", self.critic,
                       receptor_profile=STRIATUM_D1_RECEPTORS)
        net.add_layer("actor", self.actor,
                       receptor_profile=STRIATUM_D2_RECEPTORS)
        net.mark_td_updated("critic", "actor")

        # ── Columnar: assoc → BG via feedforward ─────────────────────
        if self._use_columnar:
            agg = "concat" if self._use_working_memory else "sum"
            net.connect(self._assoc_name, "critic",
                        aggregation_mode=agg)
            net.connect(self._assoc_name, "actor",
                        aggregation_mode=agg)

        if self._use_working_memory:
            net.add_layer("working_memory", self.working_memory,
                          receptor_profile=PFC_RECEPTORS)
            net.connect("working_memory", "critic",
                        aggregation_mode="concat")
            net.connect("working_memory", "actor",
                        aggregation_mode="concat")

        if self._use_wm:
            net.add_layer("encoder", self.world_model.encoder,
                          receptor_profile=CORTICAL_L4_RECEPTORS)

    # ------------------------------------------------------------------
    # Sensory input builder
    # ------------------------------------------------------------------

    def _build_sensory_inputs(
        self,
        encoded: NDArray[np.float32],
        raw_state: NDArray[np.float32],
    ) -> dict[str, NDArray[np.float32]]:
        """Build sensory_inputs dict for network.step().

        In flat mode: critic/actor receive Poisson-encoded state directly.
        In columnar mode: columns receive receptive field slices; BG
        receives from association layer via graph feedforward.
        """
        sensory: dict[str, NDArray[np.float32]] = {}

        if self._use_columnar:
            # Split encoded state across column receptive fields
            col_sensory = split_input(
                encoded, self._column_names, self._rf_size,
            )
            sensory.update(col_sensory)

            # WM still receives full encoded state
            if self._use_working_memory:
                sensory["working_memory"] = encoded
        else:
            # Flat mode: BG receives encoded state directly
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
            self.neuromod.learning_rate_modulation + self._agent_cfg.da_offset, 0.0, 1.0,
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
            attention=self._attention,
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
        acfg = self._agent_cfg
        intrinsic_weight = acfg.intrinsic_reward_weight * (1.0 - self.neuromod.learning_rate_modulation)
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
            attention=self._attention,
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
        td_error = float(np.clip(td_error, -self._agent_cfg.td_clip, self._agent_cfg.td_clip))
        self._last_td_error = td_error

        # ── 4. Consolidation-gated plasticity ─────────────────────────
        gate = self.neuromod.consolidation_gate
        acfg = self._agent_cfg
        plasticity_scale = acfg.consolidation_floor + (1.0 - acfg.consolidation_floor) / (
            1.0 + np.exp(acfg.consolidation_steepness * (gate - acfg.consolidation_midpoint))
        )

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

        # ── 6. Tonic DA now updated per-step inside neuromod.update() ──

        # ── 7. NE-driven trace compression ────────────────────────────
        self.critic.set_plasticity_timescales(
            self.neuromod.ne_for_region("critic"),
        )
        self.actor.set_plasticity_timescales(
            self.neuromod.ne_for_region("actor"),
        )

        if self._use_wm:
            self.neuromod.apply_to_layer(self.world_model, region="encoder")
            self.world_model.set_rehearsal_depth(
                self.neuromod.planning_horizon,
            )

        # ── 8. (Exploration noise is now continuous via tonic_da) ──────

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
                spike_trains=[
                    self.world_model.encoder.spikes_state.astype(np.float32),
                    self.world_model.encoder.spikes_error.astype(np.float32),
                ],
                synaptic_fingerprint={
                    "encoder_e_bu": self.world_model.encoder.e_bu.copy(),
                    "encoder_e_td": self.world_model.encoder.e_td.copy(),
                },
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
            # sleep_gain derived from tonic_da: high tonic_da → good recent
            # performance → more vigorous consolidation.
            acfg = self._agent_cfg
            sleep_gain = float(np.clip(
                1.0 + acfg.sleep_gain_scale * (self.neuromod.tonic_da * 2.0 - 1.0),
                1.0,
                acfg.sleep_gain_max,
            ))

            self.replay_buffer.sleep_phase(
                world_model=self.world_model,
                neuromodulator=self.neuromod,
                bg=self.bg,
                sleep_gain=sleep_gain,
                oscillator=self.network.oscillator,
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
        agent_cfg: AgentConfig | None = None,
    ) -> None:
        self.critic = critic
        self.actor = actor
        self.config = config
        self._agent_cfg = agent_cfg or AgentConfig()

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
        min_exploration = self._agent_cfg.min_exploration
        da_noise = max(0.3, 1.0 - tonic_da)
        sero_noise = max(0.3, 1.0 - serotonin)
        return max(min_exploration, da_noise * sero_noise)
