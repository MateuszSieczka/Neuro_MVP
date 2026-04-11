"""core -- Biologically grounded SNN components for AGI foundation."""

# -- Configuration --
from .config import (
    NeuronConfig,
    STDPConfig,
    HomeostaticConfig,
    CompetitiveConfig,
    PredictiveCodingConfig,
    PyramidalConfig,
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
    AgentConfig,
    ReceptorType,
    SynapseType,
    ReceptorProfile,
    CORTICAL_L4_RECEPTORS,
    CORTICAL_L5_RECEPTORS,
    PFC_RECEPTORS,
    STRIATUM_D1_RECEPTORS,
    STRIATUM_D2_RECEPTORS,
    init_weights,
    compute_weight_std,
)

# -- Simulation context --
from .simulation_context import SimulationContext, DEFAULT_CONTEXT
from .free_energy import (
    variational_free_energy,
    expected_free_energy,
    precision_weighted_update,
)

# -- Receptor pharmacology --
from .receptor import receptor_effect, compute_layer_modulation, aggregate_receptor_effects

# -- Synapse models --
from .synapse import SynapticChannels

# -- Neuron layers --
from .neuron import LIFLayer, HomeostaticState
from .competitive_layer import CompetitiveLIFLayer
from .predictive_coding import PredictiveCodingLayer
from .pyramidal_neuron import PyramidalLayer
from .error_neuron import ErrorNeuronLayer
from .interneuron import InhibitoryPool

# -- Neuromodulation --
from .neuromodulator import NeuromodulatorSystem
from .oscillator import ThetaGammaOscillator
from .astrocyte import AstrocyteField

# -- Attention --
from .attention import SpatialAttentionController

# -- Memory systems --
from .working_memory import WorkingMemoryModule
from .episodic_memory import EpisodicMemory, Episode
from .replay_buffer import ReplayBuffer, Experience
from .sequence_memory import SequenceMemory, HierarchicalSequenceMemory

# -- World model --
from .world_model import SNNWorldModel, EncoderSnapshot, RehearsalResult

# -- Basal ganglia --
from .basal_ganglia import (
    BasalGangliaAGISystem,
    ActiveInferenceModule,
    D1D2Actor,
    SNNDeepCritic,
)

# -- Network orchestration --
from .network import NetworkGraph, LayerConnection
from .columnar import build_columnar_network, split_input  # returns 5-tuple now

# -- Encoding --
from .spike_encoder import PoissonEncoder, GaussianPopulationEncoder
