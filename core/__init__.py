"""core — the predictive-coding substrate for neuro-AGI (JAX/Equinox).

One microcircuit, one learning rule, one objective (free-energy
minimisation) on an arbitrary, self-wiring graph.  The whole public API:

* :mod:`core.backend`      — JAX backend + PRNG helpers (the only base dep)
* :mod:`core.free_energy`  — the variational / expected free-energy objectives
* :mod:`core.pc_module`    — canonical PC module: relaxation + one rule (U.1)
* :mod:`core.pc_graph`     — arbitrary-topology PC graph + region assembly (U.2/U.3)
* :mod:`core.pc_brain`     — cognitive cycle as graph relaxation (U.3)
* :mod:`core.pc_active`    — action as inference / EFE / neuromodulation (U.5)
* :mod:`core.pc_structural`— self-wiring structural plasticity by free energy (U.4b)
* :mod:`core.pc_memory`    — replay buffer + episodic store primitives (§3)
* :mod:`core.pc_hippocampus`— hippocampal node group: pattern sep/completion (§3)
* :mod:`core.pc_sleep`     — offline mode: SWS replay + REM rollout + FSM (§3)
* :mod:`core.pc_precision` — Welford EMA + multi-channel precision tracking (§4)
* :mod:`core.pc_neuromod`  — neuromodulators as precision controllers + curiosity (§4)
* :mod:`core.pc_attention` — spatial attention as per-slice sensory Π gain (§4)
* :mod:`core.pc_oscillator`— theta/gamma + PAC timing; HC encode/retrieve gate (§4)
"""

# -- JAX backend --
from .backend import (
    DTYPE, Array, PRNGKey, make_key, split_key, fold_in_step,
)

# -- Free-energy objectives --
from .free_energy import variational_free_energy, expected_free_energy

# -- Canonical PC module (U.1): relaxation + one rule + backprop equivalence --
from .pc_module import (
    PCNetParams, PCNetState, PCStepOutput,
    init_pc_net_params, init_pc_net_state,
    pc_predictions, pc_errors, pc_free_energy,
    pc_feedforward, pc_relax, pc_weight_grads, pc_learn,
    pc_clamp_bottom, pc_supervised_step, pc_fixed_prediction_grads,
)

# -- Arbitrary-topology PC graph + region assembly (U.2/U.3) --
from .pc_graph import (
    PCGraphParams, PCGraphState, PCGraphStepOutput,
    init_pc_graph_params, init_pc_graph_state,
    pc_graph_predictions, pc_graph_errors, graph_free_energy,
    pc_graph_relax, pc_graph_clamp, pc_graph_learn, pc_graph_step, pc_graph_roll,
    init_region_graph, REGION_NODES, REGION_INDEX,
    PRECISION_EMA, PRECISION_WELFORD,
    FovealGaborInit, apply_foveal_gabor_init,
    apply_wm_persistence_init,
)

# -- Active inference: action / EFE / neuromodulation (U.5) --
from .pc_active import (
    scale_node_precision, set_action_prior,
    ActInferOutput, pc_act_infer, pc_act_learn_forward,
    pc_efe, PolicyChoice, efe_select, epistemic_value,
    DEFAULT_FORWARD_SETTLE_STEPS,
)

# -- Graph-driven brain: cognitive cycle as relaxation (U.3) --
from .pc_brain import (
    PCBrainParams, PCBrainState, PCBrainOutput, PCBrainActOutput,
    init_pc_brain, pc_brain_cognitive_step,
    pc_brain_act, pc_brain_learn_forward,
)

# -- Self-wiring structural plasticity by free energy (U.4b) --
from .pc_structural import (
    StructuralStepOutput,
    init_sparse_masks, active_fraction, active_count,
    pc_structural_learn, pc_structural_update, pc_structural_step,
)

# -- Memory primitives: replay buffer + episodic store (§3) --
from .pc_memory import (
    ReplayParams, ReplayState, Experience,
    init_replay_params, init_replay_state,
    replay_store, replay_size, replay_sample_indices,
    replay_recent_indices, replay_gather, replay_clear,
    EpisodicParams, EpisodicState, StoreOutput, RecallOutput,
    init_episodic_params, init_episodic_state,
    dg_encode, episodic_store, episodic_recall, episodic_size,
)

# -- Hippocampal node group: pattern separation / completion (§3) --
from .pc_hippocampus import (
    HippocampusParams, CompletionOutput,
    init_hippocampus, hippocampus_mismatch, hippocampus_surprise,
    hippocampus_encode, hippocampus_complete,
)

# -- Offline mode: sleep FSM + SWS replay + REM rollout (§3) --
from .pc_sleep import (
    SleepPhase, SleepParams, SleepState,
    init_sleep_params, init_sleep_state, sleep_step,
    is_wake, is_sws, is_rem,
    sws_replay, rem_rollout,
)

# -- Online precision tracking: Welford EMA + multi-channel (§4) --
from .pc_precision import (
    step_alpha, welford_precision_update,
    PrecisionChannel, init_precision_channel,
    precision_update, precision_value, precision_mean,
    precision_standardize, precision_compose,
)

# -- Neuromodulators as precision controllers + curiosity (§4) --
from .pc_neuromod import (
    NeuromodParams, NeuromodState,
    init_neuromod_params, init_neuromod_state, neuromod_step,
    neuromod_precision_gains, neuromod_beta, neuromod_curiosity,
    neuromod_horizon, neuromod_levels,
)

# -- Spatial attention as per-slice sensory precision gain (§4) --
from .pc_attention import (
    AttentionParams, AttentionState, AttentionOutput,
    init_attention_params, init_attention_state,
    attention_step, sensory_error_saliency, attention_precision_gains,
)

# -- Theta/gamma oscillator + PAC; HC encode/retrieve timing (§4) --
from .pc_oscillator import (
    OscillatorParams, OscillatorState, OscillatorOutput,
    init_oscillator_params, init_oscillator_state, oscillator_step,
)
