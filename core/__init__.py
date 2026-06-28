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
    pc_graph_relax, pc_graph_clamp, pc_graph_learn, pc_graph_step,
    init_region_graph, REGION_NODES, REGION_INDEX,
)

# -- Active inference: action / EFE / neuromodulation (U.5) --
from .pc_active import (
    scale_node_precision, set_action_prior,
    ActInferOutput, pc_act_infer, pc_act_learn_forward,
    pc_efe, PolicyChoice, efe_select, epistemic_value,
)

# -- Graph-driven brain: cognitive cycle as relaxation (U.3) --
from .pc_brain import (
    PCBrainParams, PCBrainState, PCBrainOutput,
    init_pc_brain, pc_brain_cognitive_step,
    pc_brain_act, pc_brain_learn_forward,
)

# -- Self-wiring structural plasticity by free energy (U.4b) --
from .pc_structural import (
    StructuralStepOutput,
    init_sparse_masks, active_fraction, active_count,
    pc_structural_learn, pc_structural_update, pc_structural_step,
)
