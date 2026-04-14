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
from core.astrocyte import AstrocyteField
from core.config import (
    AgentConfig,
    AstrocyteConfig,
    AttentionConfig,
    BasalGangliaConfig,
    EpisodicMemoryConfig,
    NeuromodulatorConfig,
    OscillatorConfig,
    ReplayBufferConfig,
    VTAConfig,
    WorkingMemoryConfig,
    WorldModelConfig,
    CORTICAL_L4_RECEPTORS,
    PFC_RECEPTORS,
    STRIATUM_D1_RECEPTORS,
    STRIATUM_D2_RECEPTORS,
    STRIATUM_ACTOR_RECEPTORS,
)
from core.episodic_memory import EpisodicMemory
from core.neuromodulator import NeuromodulatorSystem
from core.network import NetworkGraph
from core.replay_buffer import Experience, ReplayBuffer
from core.spike_encoder import GaussianPopulationEncoder, PoissonEncoder
from core.vta import VTACircuit
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

        # ── Population encoding (Pouget et al. 2000) ──────────────────
        # Gaussian receptive fields provide rich distributed
        # representation: each continuous state dimension is encoded by
        # n_neurons_per_dim tuning curves, giving the downstream BG
        # enough input diversity to form meaningful spike patterns.
        wm_cfg = wm_config or WorldModelConfig()
        self._pop_encoder = GaussianPopulationEncoder(
            n_dims=state_size,
            n_neurons_per_dim=wm_cfg.n_neurons_per_dim,
            value_min=-1.0,  # GymEnv fixed_bounds normalizes to ~[-1, 1]
            value_max=1.0,
        )
        self._encoded_size: int = self._pop_encoder.output_size

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
            bg_base_size = self._encoded_size

        # ── Compute layer sizes ───────────────────────────────────────
        self._wm_num_neurons = max(8, state_size)
        # Biologically: all afferent currents sum at the dendrite
        # (Alexander et al. 1986; Haber 2003).  WM→BG and cortex→BG
        # converge on the same MSN population via separate synapses.
        # WM contribution arrives through graph sum — no concat.
        bg_input_size = bg_base_size

        # ── Analytical mean input rate from population encoder ─────────
        # GaussianPopulationEncoder uses N tuning curves per dimension
        # with sigma = sigma_factor × spacing (sigma_factor = 0.5).
        # At any input value, the sum of all N Gaussian activations is
        # approximately σ×√(2π)/spacing (integral of overlapping curves).
        # Mean per neuron = sigma_factor × √(2π) / N.
        # With N=15: mean_rate ≈ 0.5 × 2.507 / 15 ≈ 0.084.
        # This is the expected fraction of active inputs after Poisson
        # encoding, used to calibrate synaptic gain to AdEx rheobase.
        _n_per_dim = wm_cfg.n_neurons_per_dim
        _sigma_factor = 0.5  # matches GaussianPopulationEncoder._inv_2sigma2
        _mean_input_rate = _sigma_factor * np.sqrt(2.0 * np.pi) / _n_per_dim

        # ── Create modules ────────────────────────────────────────────
        self.critic = SNNDeepCritic(bg_input_size, self._bg_config,
                                     mean_input_rate=_mean_input_rate)
        # internal_dim > 0 only when WM is active and needs a gate neuron.
        # Without WM, the internal neuron wastes capacity and
        # dilutes the motor population signal.
        _internal_dim = 1 if self._use_working_memory else 0
        self.actor = D1D2Actor(
            bg_input_size, n_actions, _internal_dim, self._bg_config,
            mean_input_rate=_mean_input_rate,
        )

        if self._use_working_memory:
            self.working_memory = WorkingMemoryModule(
                num_external_inputs=state_size,
                num_neurons=self._wm_num_neurons,
                config=wmem_config,
            )

        self.neuromod = NeuromodulatorSystem(nm_config)

        # ── Astrocyte fields (De Pittà et al. 2011) ────────────────────
        # One AstrocyteField per neural region provides continuous ATP
        # budget, Ca²⁺ precision estimation, and D-Serine NMDA modulation.
        # Zone count derived from neuron count (1 zone per ~8 neurons,
        # matching astrocyte territorial domain ~50 µm / Bushong 2002).
        _astro_cfg = AstrocyteConfig(ctx=self._bg_config.ctx)
        _critic_zones = max(1, self._bg_config.hidden_size // 8)
        _actor_zones = max(1, n_actions // 2)  # 1 zone per 2 action channels
        self._critic_astro = AstrocyteField(
            n_zones=_critic_zones, config=_astro_cfg,
        )
        self._actor_astro = AstrocyteField(
            n_zones=_actor_zones, config=_astro_cfg,
        )
        self.critic.set_astrocyte(self._critic_astro)
        self.actor.set_astrocyte(self._actor_astro)

        if self._use_working_memory:
            _wm_zones = max(1, self._wm_num_neurons // 8)
            self._wm_astro = AstrocyteField(
                n_zones=_wm_zones, config=_astro_cfg,
            )
        # Encoder astrocyte is managed by SNNWorldModel (already wired).

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

        # ── VTA dopaminergic circuit (Eshel et al. 2015) ──────────────
        # Replaces algebraic TD error and Welford normalisation with a
        # biophysical VTA circuit.  VP pathway reads V(s) (inhibitory),
        # PPTg pathway reads γ×V(s') (excitatory), reward feeds direct.
        # D2 autoreceptor provides intrinsic gain adaptation (Tobler 2005).
        self._vta_config = VTAConfig(ctx=self._bg_config.ctx)
        self.vta = VTACircuit(
            critic_hidden_size=self._bg_config.hidden_size,
            config=self._vta_config,
        )

        # ── Backward compat: BG facade for replay_buffer/sleep ────────
        # Facade created after VTA so it can provide value readout.
        self.bg = _BGFacade(
            self.critic, self.actor, self._bg_config, self._agent_cfg,
            vta=self.vta,
        )

        # ── Transient state ───────────────────────────────────────────
        self._last_td_error: float = 0.0
        self._last_curiosity: float = 0.0
        self._last_encoded_state: NDArray[np.float32] | None = None

        # ── Integration substeps (Wang 2002: cortical decisions ~20-50 ms) ─
        # The SNN needs multiple dt steps to integrate synaptic input
        # and develop meaningful spike patterns.  Derived from the slowest
        # BG membrane time constant to ensure at least one full τ_m of
        # integration per environmental decision.
        dt = self._bg_config.ctx.dt
        tau_max = max(self._bg_config.tau_m_msn_up,
                      self._bg_config.tau_m_critic)
        # Use one full τ_m of integration (biophysical minimum for
        # the membrane to reach ~63% of steady-state).  Clamping
        # below τ/dt prevents MSN neurons from depolarising to
        # spike cutoff, silencing the actor pathway.
        self._n_substeps: int = max(1, round(tau_max / dt))
        # Critic-only integration for V(s') in observe():
        # use critic's own τ_m (faster than MSN τ_m), saving ~40% compute.
        self._n_substeps_critic: int = max(1, round(
            self._bg_config.tau_m_critic / dt,
        ))

        # ── Dynamic headroom (Brette & Gerstner 2005) ─────────────────
        # headroom = 1/(1-exp(-n*dt/τ)) accounts for finite integration
        # window.  At n*dt = τ: 1/(1-e^{-1}) ≈ 1.58.  Derived from
        # actual n_substeps instead of assuming 1τ integration.
        _critic_n = self._n_substeps_critic
        _actor_n = self._n_substeps
        _tau_c = self._bg_config.tau_m_critic
        _tau_a = self._bg_config.tau_m_msn_up
        # Integration window must span at least ~10% of τ for meaningful
        # membrane depolarisation.  If this fires, n_substeps or τ_m
        # are misconfigured.
        _denom_c = 1.0 - np.exp(-_critic_n * dt / _tau_c)
        _denom_a = 1.0 - np.exp(-_actor_n * dt / _tau_a)
        assert _denom_c > 0.05, (
            f"Critic integration too short: {_critic_n}×{dt}ms vs τ={_tau_c}ms"
        )
        assert _denom_a > 0.05, (
            f"Actor integration too short: {_actor_n}×{dt}ms vs τ={_tau_a}ms"
        )
        _headroom_c = 1.0 / _denom_c
        _headroom_a = 1.0 / _denom_a
        # Recompute conductance scale with dynamic headroom
        from core.basal_ganglia import _derive_conductance_scale
        self.critic._cond_scale = _derive_conductance_scale(
            bg_input_size, self.critic._ncfg,
            self._bg_config.w_clip_critic,
            mean_input_rate=_mean_input_rate,
            headroom=_headroom_c,
            actual_mean_w=self.critic._init_mean_w,
        )
        self.actor._cond_scale = _derive_conductance_scale(
            bg_input_size, self.actor._ncfg,
            self._bg_config.w_clip,
            mean_input_rate=_mean_input_rate,
            headroom=_headroom_a,
            actual_mean_w=self.actor._init_mean_w,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def use_world_model(self) -> bool:
        """Whether the world model is active."""
        return self._use_wm

    # ------------------------------------------------------------------
    # ATP-triggered sleep (Kann & Kovács 2007)
    # ------------------------------------------------------------------

    def _needs_sleep(self) -> bool:
        """Check if mean ATP across all layer astrocytes is below threshold.

        When continuous activity depletes metabolic reserves, the network
        triggers consolidation independently of episode boundaries.
        This enables continuous learning (P2) with thermodynamic
        constraints (P3) driving the sleep-wake cycle.
        """
        atp_values: list[float] = []
        for astro in (self._critic_astro, self._actor_astro):
            atp_values.append(float(np.mean(astro.atp)))
        if self._use_working_memory:
            atp_values.append(float(np.mean(self._wm_astro.atp)))
        if not atp_values:
            return False
        mean_atp = sum(atp_values) / len(atp_values)
        return mean_atp < self._agent_cfg.sleep_atp_threshold

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
                       receptor_profile=STRIATUM_ACTOR_RECEPTORS)
        net.mark_td_updated("critic", "actor")

        # ── Columnar: assoc → BG via feedforward ─────────────────────
        if self._use_columnar:
            net.connect(self._assoc_name, "critic",
                        aggregation_mode="sum")
            net.connect(self._assoc_name, "actor",
                        aggregation_mode="sum")

        if self._use_working_memory:
            self.working_memory.set_astrocyte(self._wm_astro)
            net.add_layer("working_memory", self.working_memory,
                          receptor_profile=PFC_RECEPTORS)
            # Dendritic summation: PFC afferents add to cortical input
            # via separate synapses on the same MSN population.
            net.connect("working_memory", "critic",
                        aggregation_mode="sum")
            net.connect("working_memory", "actor",
                        aggregation_mode="sum")

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

            # WM receives Poisson-encoded raw state (matching state_size)
            if self._use_working_memory:
                sensory["working_memory"] = self._poisson.encode(raw_state)
        else:
            # Flat mode: BG receives encoded state directly
            # WM contribution arrives through graph sum (no padding).
            sensory["critic"] = encoded
            sensory["actor"] = encoded
            if self._use_working_memory:
                sensory["working_memory"] = self._poisson.encode(raw_state)

        if self._use_wm:
            sensory["encoder"] = self.world_model._build_input(
                raw_state, np.zeros(self.n_actions, dtype=np.float32),
            )

        return sensory

    def _build_critic_only_sensory(
        self,
        encoded: NDArray[np.float32],
        raw_state: NDArray[np.float32],
    ) -> dict[str, NDArray[np.float32]]:
        """Build sensory_inputs with only critic (+ encoder if WM).

        Used during observe()'s V(s') computation so the actor never
        processes next_state.  The actor receives no sensory input;
        network.step() will give it a zero-input forward pass where
        membrane decays toward rest with no spikes and no eligibility
        corruption.  Biologically correct: BG does not preview the
        next state before movement (Schultz 1997).
        """
        sensory: dict[str, NDArray[np.float32]] = {}

        if self._use_columnar:
            col_sensory = split_input(
                encoded, self._column_names, self._rf_size,
            )
            sensory.update(col_sensory)
            if self._use_working_memory:
                sensory["working_memory"] = self._poisson.encode(raw_state)
        else:
            sensory["critic"] = encoded
            if self._use_working_memory:
                sensory["working_memory"] = self._poisson.encode(raw_state)
            # Actor explicitly receives zeros — no next-state processing
            actor_size = self.actor.num_inputs
            sensory["actor"] = np.zeros(actor_size, dtype=np.float32)

        if self._use_wm:
            sensory["encoder"] = self.world_model._build_input(
                raw_state, np.zeros(self.n_actions, dtype=np.float32),
            )

        return sensory

    # _set_actor_policy_gradient REMOVED (was non-biological REINFORCE
    # hack that destroyed STDP eligibility by 216×).  WTA dynamics now
    # naturally gate eligibility to the winning action channel via
    # membrane-voltage separation (Wang 2002).

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray) -> int:
        state_f32 = state.astype(np.float32)
        pop_rates = self._pop_encoder.encode(state_f32)

        # ── Set DA / epistemic drive BEFORE graph step ────────────────
        da_level = float(np.clip(self.neuromod.dopamine, 0.0, 1.0))
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

        # ── Pass NE level for temperature-modulated exploration ──────
        self.actor.set_ne_level(self.neuromod.competition_sharpness)

        # ── Integrate over substeps (Wang 2002) ──────────────────────
        # Present the same sensory rate code for multiple SNN timesteps.
        # Each substep draws fresh Poisson spikes from the same population
        # rates — the rate is the signal, Poisson jitter provides
        # biologically realistic trial-to-trial variability.
        self.actor.reset_spike_counts()  # New decision cycle (Lo & Wang 2006)
        for _sub in range(self._n_substeps):
            encoded = self._poisson.encode(pop_rates)
            sensory = self._build_sensory_inputs(encoded, state_f32)
            self.network.step(
                sensory_inputs=sensory,
                neuromodulator=self.neuromod,
                attention=self._attention,
            )
        self._last_encoded_state = encoded.copy()

        # VTA: capture V(s) in VP pathway after critic integration.
        # The VP trace stores the critic's population activity snapshot
        # at decision time — this represents "what I expected" and will
        # inhibit VTA DA neurons during observe() (Eshel et al. 2015).
        self.vta.store_prediction(self.critic.activation)

        # WTA dynamics naturally gate eligibility: the winning action's
        # MSNs are most depolarised → highest voltage-based eligibility.
        # Losing actions are suppressed by the InhibitoryPool → near-rest
        # membrane → near-zero eligibility.  No explicit zeroing needed
        # (Wang 2002; Wickens et al. 2003).
        action = self.actor.get_action()

        # Exploration is emergent from STN-GPe pathway (Frank 2006):
        # low DA → STN tonic activity → global inhibition → margin
        # shrinks → membrane noise breaks symmetry.  No ε-greedy.

        return action

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
        next_pop_rates = self._pop_encoder.encode(next_f32)

        is_truncated = info.get("truncated", False) if info else False
        is_terminal = done and not is_truncated

        # ── 1. Pre-update curiosity ───────────────────────────────────
        # Curiosity drives exploration via NE/ACh pathways (Hasselmo
        # 2006), NOT as additive reward shaping (Pathak et al. 2017).
        # The world model prediction error flows through NE (surprise)
        # and ACh (novelty) pathways in neuromod.update() below.
        if self._use_wm:
            self._last_curiosity = self.world_model.curiosity_signal()
        else:
            self._last_curiosity = 0.0

        # ── 2. Critic step on next_state → VTA RPE ──────────────────
        # V(s) was captured in VTA VP trace during act() via
        # vta.store_prediction().  Now integrate critic on s' to get
        # population activity for V(s'), then let VTA compute RPE
        # from E/I balance (Eshel et al. 2015).

        # Freeze eligibility: the critic's and actor's eligibility
        # traces were accumulated during act(state) and must be
        # preserved for the weight update below.  inference_mode
        # makes forward() skip all plasticity-related state updates
        # (pre/post traces, eligibility) while still running membrane
        # dynamics and activation EMA normally.
        self.critic._inference_mode = True
        self.actor._inference_mode = True

        # Integrate over substeps to let the critic develop a
        # meaningful V(s') population activity.  Uses critic's own
        # τ_m (15ms) rather than the full MSN τ (25ms).
        for _sub in range(self._n_substeps_critic):
            next_encoded = self._poisson.encode(next_pop_rates)
            # Critic-only sensory: actor receives zeros so it does not
            # process next_state (Schultz 1997: BG does not preview the
            # next state before movement).
            sensory = self._build_critic_only_sensory(next_encoded, next_f32)
            outputs = self.network.step(
                sensory_inputs=sensory,
                neuromodulator=self.neuromod,
                attention=self._attention,
            )

        # ── VTA RPE computation (replaces algebraic TD error) ─────
        # VTA DA neuron output ∝ reward + γ_eff×V(s') − V(s)
        # where γ_eff emerges from PPTg pathway τ (serotonin-modulated)
        # and gain adaptation from D2 autoreceptors (Tobler 2005).
        # Raw environmental reward only — curiosity modulates exploration
        # through NE/ACh pathways, not reward shaping.
        td_error_normed = self.vta.compute_rpe(
            critic_activation=self.critic.activation,
            reward=reward,
            is_terminal=is_terminal,
            serotonin=self.neuromod.serotonin,
            n_substeps=self._n_substeps,
        )

        # Restore plasticity mode
        self.critic._inference_mode = False
        self.actor._inference_mode = False

        self._last_td_error = td_error_normed

        # ── 3b. VTA value weight update ───────────────────────────
        # Three-factor Hebbian: dw_value = lr × RPE × critic_activation(s).
        # Uses eligibility accumulated during store_prediction() (act phase).
        self.vta.update(td_error_normed)

        # ── 4. TD-modulated plasticity ─────────────────────────────
        # Schultz (1998): DA phasic signal is a SINGLE broadcast RPE
        # from VTA to both ventral (critic) and dorsal (actor) striatum.
        # Both receive the same Tobler-normalized signal.
        self.critic.update(td_error_normed)

        # Voltage-based eligibility (Clopath et al. 2010) naturally
        # gates credit to the winning action: InhibitoryPool suppresses
        # losing channels' membrane → near-rest voltage → near-zero
        # eligibility.  No explicit zeroing needed (Phase 2 HACK B).
        self.actor.update(td_error_normed)

        # ── 5. World model + neuromodulator update ────────────────────
        if self._use_wm:
            wm_m_t = max(self.neuromod.learning_rate_modulation,
                         self.neuromod.config.baseline_da * 0.2)
            pred_error = self.world_model.update(
                state_f32, action, next_f32, m_t=wm_m_t,
            )
            self.neuromod.update(
                prediction_error=pred_error,
                td_error=td_error_normed,
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
            # No world model: use raw TD error as prediction error signal.
            # The previous |td|/(1+|td|) squashing destroyed magnitude
            # information, producing a near-constant ~0.5 signal.
            # Raw |td_error| IS the reward prediction error (Schultz 1997).
            norm_pe = float(np.clip(abs(td_error_normed), 0.0, 10.0))
            # State change magnitude as sensory novelty proxy
            # (sensory cortex habituates to static input — Hasselmo 2006).
            # ACh responds to novelty, NE to surprise (|TD|).
            state_change = float(np.mean(np.abs(next_f32 - state_f32)))
            sensory_novelty = float(np.clip(state_change, 0.0, 1.0))
            self.neuromod.update(
                prediction_error=np.array([norm_pe], dtype=np.float32),
                td_error=td_error_normed,
                novelty=sensory_novelty,
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
            # Store population-encoded states for sleep replay.
            # WM contribution is not included — sleep replay runs
            # the critic directly without the graph, and WM is
            # inactive during SWS.
            aug_state = self._last_encoded_state.copy()
            aug_next_state = next_encoded.copy()

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
                aug_next_state=aug_next_state,
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

        # ── 10. Sleep phase (end of episode OR ATP depletion) ──────────
        # Consolidation vigor modulated by neuromodulatory state:
        # VTA D2 autoreceptor adapts RPE gain, serotonin modulates
        # temporal discount, oscillator gates Up/Down states.
        # No arbitrary sleep_gain formula — the VTA's intrinsic
        # dynamics handle adaptation (Tobler et al. 2005).
        # ATP-triggered sleep enables continuous learning without
        # episode boundaries (Kann & Kovács 2007; plan step 4.5).
        _should_sleep = done or self._needs_sleep()
        if _should_sleep and self._use_wm and len(self.replay_buffer) > 0:
            self.replay_buffer.sleep_phase(
                world_model=self.world_model,
                neuromodulator=self.neuromod,
                bg=self.bg,
                oscillator=self.network.oscillator,
            )

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.network.reset_state()
        self.vta.reset_state()
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

    After Phase 3, the value readout (V(s)) is computed by the VTA
    circuit, not the critic.  The facade provides a ``last_value``
    property that reads critic.activation via VTA's w_value weight,
    keeping the sleep-phase code functional.
    """

    def __init__(
        self,
        critic: SNNDeepCritic,
        actor: D1D2Actor,
        config: BasalGangliaConfig,
        agent_cfg: AgentConfig | None = None,
        vta: VTACircuit | None = None,
    ) -> None:
        self.critic = critic
        self.actor = actor
        self.config = config
        self._agent_cfg = agent_cfg or AgentConfig()
        self._vta = vta

    @property
    def vta(self) -> VTACircuit | None:
        """VTA circuit for sleep-phase RPE computation."""
        return self._vta

    @property
    def last_v(self) -> float:
        """Proxy for VTA-based value estimate from critic activation."""
        if self._vta is not None:
            return float(np.dot(self.critic.activation, self._vta.w_value))
        return 0.0

    def reset_state(self) -> None:
        self.critic.reset_state()
        self.actor.reset_state()

    def set_plasticity_timescales(self, ne: float) -> None:
        self.critic.set_plasticity_timescales(ne)
        self.actor.set_plasticity_timescales(ne)

    # compute_exploration_noise() removed — Phase 2 HACK A.
    # Exploration is now emergent from STN-GPe pathway in D1D2Actor.
