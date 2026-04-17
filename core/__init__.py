"""core — biologically grounded SNN primitives (JAX/Equinox).

Legacy NumPy modules are being incrementally replaced. Only ported
modules are re-exported here; subsystems still pending migration must
be imported by their fully qualified path.
"""

from .simulation_context import SimulationContext, DEFAULT_CONTEXT
from .config import (
    NeuronConfig,
    STDPConfig,
    HomeostaticConfig,
    ErrorNeuronConfig,
    InhibitoryPoolConfig,
    WorkingMemoryConfig,
    NeuromodulatorConfig,
    OscillatorConfig,
    AstrocyteConfig,
    AttentionConfig,
    SequenceMemoryConfig,
    WorldModelConfig,
    EpisodicMemoryConfig,
    ReplayBufferConfig,
    ActiveInferenceConfig,
    BasalGangliaConfig,
    VTAConfig,
    AgentConfig,
    ReceptorType,
    ReceptorProfile,
    CORTICAL_L4_RECEPTORS,
    CORTICAL_L5_RECEPTORS,
    PFC_RECEPTORS,
    STRIATUM_D1_RECEPTORS,
    STRIATUM_D2_RECEPTORS,
    STRIATUM_ACTOR_RECEPTORS,
    init_weights,
    compute_weight_std,
)

# -- JAX backend + state containers --
from .backend import (
    BackendContext, DEFAULT, DTYPE, Array, PRNGKey,
    phi1, make_key, split_key, fold_in_step, vmap_step,
)
from .state import (
    NeuronParams, NeuronState, init_neuron_state,
    SynapticParams, SynapticState, init_synaptic_state,
    OscillatorState, init_oscillator_state,
    HomeostaticState, init_homeostatic_state,
    EligibilityState, init_eligibility_state,
    AstrocyteState, init_astrocyte_state,
)

# -- Ported JAX primitives --
from .sparse import (
    SparseConnectivity, init_random_sparse, init_distance_dependent,
    matvec, prune_below, unmask, synaptogenesis, to_bcoo,
    active_count, density,
)
from .synapse import (
    init_synaptic_params, synapse_step, nmda_mg_block,
    receive_excitatory, receive_inhibitory, decay_channels,
    effective_conductances, compute_current,
)
from .neuron import (
    init_neuron_params, neuron_step, AstroMod, decay_pre_trace,
)
from .oscillator import (
    init_oscillator_params, oscillator_step, OscillatorOutput,
)
from .astrocyte import (
    init_astrocyte_params, astrocyte_step, aggregate_to_zones,
)
from .plasticity import (
    STDPParams, init_stdp_params,
    update_pre_trace, update_post_trace,
    stdp_pair_update, weight_update_three_factor, stdp_step,
    PlasticityOutput,
)

# -- Free energy / precision --
from .free_energy import (
    variational_free_energy, expected_free_energy,
    precision_weighted_update, broadcast_precision,
)

# -- Receptor pharmacology --
from .receptor import (
    ReceptorParams, init_receptor_params, hill_response,
    receptor_effects, aggregate_effects, compute_layer_modulation,
    LayerModulation, RECEPTOR_ORDER,
)

# -- Attention --
from .attention import (
    AttentionParams, AttentionState, AttentionOutput,
    init_attention_params, init_attention_state,
    attention_step, attention_learn,
)

# -- Encoding --
from .spike_encoder import (
    PopulationEncoderParams, init_population_encoder,
    gaussian_population_encode, poisson_spike, poisson_step,
    PoissonOutput,
)

# -- Neuromodulation --
from .neuromodulator import (
    NeuromodulatorParams, NeuromodulatorState,
    init_neuromodulator_params, init_neuromodulator_state,
    neuromodulator_step,
    learning_rate_modulation, consolidation_gate,
    bottom_up_gain, competition_sharpness, planning_horizon,
    transmitter_vector,
)
from .vta import (
    VTAParams, VTAState, VTAOutput,
    init_vta_params, init_vta_state,
    vta_store_prediction, vta_compute_rpe, vta_update, vta_reset_transient,
)

# -- Working / sequence / episodic memory --
from .working_memory import (
    WMParams, WMState, WMOutput,
    init_wm_params, init_wm_state,
    wm_step, wm_update_ff, wm_update_lateral, wm_reset_transient,
)
from .sequence_memory import (
    SeqMemParams, SeqMemState, SeqMemOutput,
    init_seqmem_params, init_seqmem_state,
    seqmem_step, seqmem_novelty, seqmem_reset_transient,
)
from .episodic_memory import (
    EpisodicParams, EpisodicState, StoreOutput, RecallOutput,
    init_episodic_params, init_episodic_state,
    dg_encode, try_store, recall, mark_replayed,
    episodic_size, episodic_clear,
)

# -- Inhibitory pool / error neuron / world model / replay --
from .interneuron import (
    IPoolParams, IPoolState, IPoolOutput,
    init_ipool_params, init_ipool_state, ipool_step,
    ipool_da_gain, ipool_reset_transient,
)
from .error_neuron import (
    ErrorNeuronParams, ErrorNeuronState, ErrorNeuronOutput,
    init_error_neuron_params, init_error_neuron_state,
    en_step, en_update_weights, en_generate_prediction,
    en_receive_prediction, en_belief, en_prediction_error_rate,
    en_reset_transient,
)
from .world_model import (
    WorldModelParams, WorldModelState, WorldModelOutput, RehearsalResult,
    init_world_model_params, init_world_model_state,
    wm_predict, wm_update, wm_mental_rehearsal,
    wm_curiosity_signal, wm_rehearsal_depth_from_serotonin,
    wm_reset_transient,
)
from .replay_buffer import (
    ReplayParams, ReplayState, Experience,
    init_replay_params, init_replay_state,
    replay_store, replay_sample_indices, replay_gather,
    replay_mark_replayed, replay_size, replay_recent_indices,
    replay_clear,
)

# -- Basal ganglia (critic + D1/D2 actor) --
from .basal_ganglia import (
    CriticParams, CriticState, CriticOutput,
    init_critic_params, init_critic_state,
    critic_step, critic_update, critic_reset_transient,
    ActorParams, ActorState, ActorOutput, ActorInputs,
    init_actor_params, init_actor_state,
    actor_step, actor_select_action, actor_update,
    actor_reset_evidence, actor_reset_transient, action_entropy,
)
