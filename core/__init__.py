"""core — biologically grounded SNN primitives (JAX/Equinox)."""

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
# NOTE: free_energy.py kept on disk for Phase 9 (EFE); not re-exported yet.
from .precision_bus import (
    PrecisionChannel,
    init_precision_channel,
    precision_update, precision_value, precision_mean,
    precision_standardize,
    precision_compose, precision_weight,
)

# -- Receptor pharmacology --
from .receptor import (
    ReceptorType, ReceptorParams, init_receptor_params, hill_response,
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

# -- Working memory --
from .working_memory import (
    WMParams, WMState, WMOutput,
    init_wm_params, init_wm_state,
    wm_step, wm_update_ff, wm_update_lateral, wm_reset_transient,
)
# NOTE: sequence_memory.py, episodic_memory.py kept on disk for Phase 5+; not re-exported yet.

# -- Inhibitory pool / error neuron / world model / replay --
from .interneuron import (
    IPoolParams, IPoolState, IPoolOutput,
    init_ipool_params, init_ipool_state, ipool_step,
    ipool_da_gain, ipool_reset_transient,
)
from .error_neuron import (
    ErrorNeuronParams, ErrorNeuronState, ErrorNeuronOutput,
    init_error_neuron_params, init_error_neuron_state,
    en_step, en_update_weights,
    en_receive_prediction, en_belief, en_prediction_error_rate,
    en_reset_transient,
)
from .world_model import (
    WorldModelParams, WorldModelState, WorldModelOutput, RehearsalResult,
    init_world_model_params, init_world_model_state,
    wm_predict, wm_update, wm_mental_rehearsal,
    wm_curiosity_signal, wm_boredom_signal, wm_learning_progress,
    wm_rehearsal_depth_from_serotonin,
    wm_reset_transient,
)
# NOTE: replay_buffer.py kept on disk for Phase 5 (sleep); not re-exported yet.

# -- Basal ganglia (critic + D1/D2 actor) --
from .basal_ganglia import (
    CriticParams, CriticState, CriticOutput,
    init_critic_params, init_critic_state,
    critic_step, critic_update, critic_commit_eligibility,
    critic_reset_transient,
    ActorParams, ActorState, ActorOutput, ActorInputs,
    init_actor_params, init_actor_state,
    actor_step, actor_select_action, actor_update,
    actor_commit_eligibility,
    actor_reset_evidence, actor_reset_transient,
)

# -- Cortical microcircuit (L4 / L2-3 / L5) --
from .cortex import (
    CorticalAreaParams, CorticalAreaState, CorticalInputs, CorticalOutput,
    init_cortical_area_params, init_cortical_area_state,
    cortical_area_step, cortical_area_update, cortical_area_reset_transient,
)

# -- Cerebellum (generic forward model, Marr-Albus-Ito) --
from .cerebellum import (
    CerebellumParams, CerebellumState, CerebellumOutput,
    init_cerebellum_params, init_cerebellum_state,
    cerebellum_step, cerebellum_update, cerebellum_reset_transient,
)

# -- Thalamus (relay nuclei + TRN) --
from .thalamus import (
    RelayParams, RelayState, TRNParams, TRNState, ThalamicOutput,
    init_relay_params, init_relay_state,
    init_trn_params, init_trn_state,
    thalamic_step, relay_reset_transient, trn_reset_transient,
)

# -- Brain graph (wiring primitives + ActionBrain) --
from .brain_graph import (
    DelayBuffer, init_delay_buffer, delay_push_pop,
    ActionBrainParams, ActionBrainState, ActionBrainOutput,
    init_action_brain_params, init_action_brain_state,
    action_brain_step, action_brain_cognitive_step,
    SACCADE_ACTION_DIM,
)
