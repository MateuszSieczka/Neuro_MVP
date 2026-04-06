from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class LIFConfig:
    """
    Hyperparameters for the Leaky Integrate-and-Fire (LIF) neuron.
    Uses frozen dataclass to prevent unintended runtime mutations.
    """
    v_rest: float = -70.0
    v_thresh: float = -55.0
    v_reset: float = -75.0
    tau_m: float = 20.0
    tau_e: float = 500.0
    tau_pre: float = 20.0    # Presynaptic trace decay
    tau_post: float = 20.0   # Postsynaptic trace decay
    refrac_period: int = 2   # Absolute refractory period in timesteps (dt)
    dt: float = 1.0
    learning_rate: float = 0.01

    # ── Dynamic timescale modulation ──────────────────────────────────
    # NE compresses eligibility trace windows (tau_e, tau_pre, tau_post).
    # At NE=1.0 the effective tau is divided by tau_ne_compression,
    # making old correlations fade faster → quicker context switching.
    # At NE=0.0 no compression is applied (full tau_e retained).
    tau_ne_compression: float = 4.0

    # ACh compresses the membrane time constant.
    # High ACh → faster integration → stronger bottom-up signal influence.
    # At ACh=1.0 effective tau_m is divided by tau_ach_compression.
    tau_ach_compression: float = 2.0


@dataclass(frozen=True, kw_only=True)
class HomeostaticLIFConfig(LIFConfig):
    """
    Extends LIFConfig with homeostatic plasticity — synaptic scaling.

    Dark Matter Neurons
    -------------------
    dark_matter_ratio controls the fraction of neurons initialized with an
    inflated threshold (dark_matter_thresh_offset mV above v_thresh).  These
    neurons are silent under normal conditions but can be recruited when
    global noradrenaline spikes, which temporarily lowers ALL thresholds
    by up to ne_thresh_drop mV.  This achieves capacity expansion for
    continual learning without dynamic matrix allocation (neurogenesis).

    homeostatic_tau:  Time constant for the sliding average of firing rate (ms).
                      Intentionally long (>>tau_m) so threshold adapts slowly.
    target_rate:      Desired spikes-per-timestep. 0.05 → 5% firing probability
                      per step, consistent with cortical sparse coding.
    thresh_adapt_lr:  Step size of threshold correction per timestep.
    thresh_min/max:   Hard physiological bounds on adaptive threshold.
    """
    target_rate: float = 0.05        # Target spikes / timestep
    homeostatic_tau: float = 1000.0  # Slow time constant (ms)
    thresh_adapt_lr: float = 0.01    # Threshold adaptation step
    thresh_min: float = -68.0        # Minimum allowed threshold (mV)
    thresh_max: float = -45.0        # Maximum allowed threshold (mV)

    # Dark Matter Neurons
    dark_matter_ratio: float = 0.0        # Fraction of neurons born as dark matter (0–1)
    dark_matter_thresh_offset: float = 20.0  # Extra mV added to threshold for dark neurons
    ne_thresh_drop: float = 15.0          # Max mV threshold drop at NE=1.0


@dataclass(frozen=True, kw_only=True)
class KWTAConfig(LIFConfig):
    """
    Hyperparameters for k-Winners-Take-All lateral inhibition.
    """
    k_winners: int = 3       # Liczba zwycięzców w populacji
    i_inh: float = 50.0      # Siła sygnału hamującego (mV odejmowane od V)
    window_ms: int = 100     # Okno czasowe integracji przed ewaluacją k-WTA


@dataclass(frozen=True, kw_only=True)
class HomeostaticKWTAConfig(HomeostaticLIFConfig):
    """
    KWTAConfig with homeostatic plasticity.
    Adds k-WTA fields on top of HomeostaticLIFConfig.
    """
    k_winners: int = 3
    i_inh: float = 50.0
    window_ms: int = 100


@dataclass(frozen=True, kw_only=True)
class PredictiveCodingConfig(HomeostaticKWTAConfig):
    """
    Hyperparameters for Predictive Coding layer.
    Extends KWTAConfig with feedback (top-down) dynamics.

    feedback_strength:       Skala, z jaką top-down prediction moduluje wejście.
    feedback_learning_rate:  Szybkość uczenia się wag feedback (oddzielna od STDP).
    """
    feedback_strength: float = 0.5
    feedback_learning_rate: float = 0.005
    relaxation_steps: int = 10
    relaxation_rate: float = 0.1
    relaxation_threshold: float = 0.01  # Wcześniejsze wyjście z pętli relaksacji (oszczędność CPU)
    feedback_norm: bool = True  # Czy normalizować wagi wsteczne

@dataclass(frozen=True, kw_only=True)
class PyramidalConfig(PredictiveCodingConfig):
    """
    Hyperparameters for multi-compartment pyramidal neuron layer.

    Biological grounding:
      Cortical pyramidal neurons have two anatomically distinct integration zones:
        - Basal dendrites (~100–200 µm from soma): receive thalamic/feedforward input.
        - Apical dendrites (~400–1000 µm from soma): receive feedback from
          higher cortical areas and other long-range projections.
      The apical compartment is electrically passive at low input levels but
      triggers a dendritic calcium spike (BAC firing) when strongly activated,
      which dramatically lowers the threshold for somatic spiking.

      Burst-dependent plasticity (Payeur et al., 2021):
        A spike that coincides with apical activation (burst) drives 3–5× stronger
        STDP than a singleton spike. This provides a top-down teaching signal that
        is separate from the feedforward credit-assignment pathway.

    Compartment parameters:
      tau_apical:        Time constant of apical membrane (ms). Must be >> tau_m
                         because apical dendrites are electrotonically remote.
      apical_threshold:  Normalised apical potential level above which apical
                         priming activates (triggers BAC-like gain modulation).
      apical_boost:      mV subtracted from somatic threshold when apical is primed.
                         Calibrate so that apical-alone never fires soma
                         (boost < |v_thresh − v_rest|) but meaningfully reduces
                         the required basal drive.
      burst_stdp_factor: Multiplier applied to eligibility traces when is_burst=True.
                         Biologically 3–5× based on Payeur et al.
      apical_lr:         Learning rate for Hebbian update of apical weights.
    """
    # Apical compartment
    tau_apical: float = 50.0          # Apical membrane time constant (ms)
    apical_threshold: float = 0.3     # Normalised apical potential for priming
    apical_boost: float = 10.0        # Somatic threshold reduction (mV) when primed
    burst_stdp_factor: float = 3.0    # STDP multiplier during burst
    apical_lr: float = 0.005          # Apical weight Hebbian learning rate
    plateau_duration_ms: int = 50   # Czas trwania nieliniowego plateau
    background_noise_std: float = 2.0  # DODANE: Odchylenie standardowe szumu błonowego (mV)


@dataclass(frozen=True, kw_only=True)
class WorkingMemoryConfig(LIFConfig):
    """
    Hyperparameters for Working Memory Module.
    Overrides tau_m to ~300 ms for sustained attractor dynamics.
    """
    tau_m: float = 300.0
    gate_threshold: float = 0.5
    lateral_strength: float = 0.5
    lateral_lr: float = 0.01


@dataclass(frozen=True, kw_only=True)
class NeuromodulatorConfig:
    """
    Hyperparameters for the four-channel neuromodulatory system.
    """
    da_decay: float = 0.95
    ach_decay: float = 0.90
    ne_decay: float = 0.93
    sero_decay: float = 0.97

    baseline_da: float = 0.5
    baseline_ach: float = 0.5
    baseline_ne: float = 0.3
    baseline_sero: float = 0.6


@dataclass(frozen=True, kw_only=True)
class SequenceMemoryConfig:
    """Hyperparameters for temporal sequence learning."""
    learning_rate: float = 0.01
    decay: float = 0.999
    max_weight: float = 1.0




@dataclass(frozen=True, kw_only=True)
class SNNWorldModelConfig:
    """
    Hyperparameters for the SNN-native world model.

    rehearsal_steps:        Number of encoder forward passes per candidate
                            action during mental_rehearsal().  Multiple steps
                            let the membrane accumulate action-specific signal
                            so k-WTA can differentiate subtly different inputs.
    """
    hidden_size: int = 64
    decode_lr: float = 0.005
    feedback_strength: float = 0.5
    feedback_learning_rate: float = 0.005
    k_winners: int = 5
    window_ms: int = 50
    i_inh: float = 50.0
    rehearsal_steps: int = 5


@dataclass(frozen=True, kw_only=True)
class EpisodicMemoryConfig:
    """
    Hyperparameters for one-shot episodic memory (hippocampal fast binding).

    Biological grounding:
      Hippocampal CA3 performs rapid pattern completion via auto-associative
      Hebbian learning with a very short eligibility trace (tau_e ≈ 0).
      A single exposure at high noradrenaline is sufficient to form a
      retrievable memory trace.

    ne_threshold:       Minimum noradrenaline level to trigger storage.
    similarity_thresh:  Cosine similarity above which a new pattern is
                        considered a duplicate (not stored again).
    capacity:           Maximum number of stored episodes.
    """
    ne_threshold: float = 0.7
    similarity_thresh: float = 0.85
    capacity: int = 500


@dataclass(frozen=True, kw_only=True)
class AttentionConfig:
    """
    Hyperparameters for spatial / object-based attention.

    Biological grounding:
      Attentional modulation in cortex operates through cholinergic
      projections from basal forebrain.  Unlike global tonic ACh,
      spatial attention is column-specific: higher areas project back
      and selectively amplify columns whose receptive fields overlap
      the attended location (Reynolds & Heeger, 2009).

    gain_strength:   Maximum multiplicative boost for the most-attended column.
    temperature:     Softmax temperature for attention distribution.
    learning_rate:   Hebbian update rate for attention projection weights.
    decay:           Temporal smoothing of attention weights.
    """
    gain_strength: float = 2.0
    temperature: float = 1.0
    learning_rate: float = 0.01
    decay: float = 0.9


@dataclass(frozen=True, kw_only=True)
class ActiveInferenceConfig:
    """
    Hyperparameters for Active Inference / Epistemic Foraging.

    Biological grounding:
      Under Active Inference (Friston, 2010), agents minimize expected
      free energy which combines pragmatic value (reward) and epistemic
      value (information gain).  The anterior cingulate cortex encodes
      expected uncertainty; locus coeruleus NE modulates the
      explore/exploit tradeoff.

    epistemic_weight:      Base weight for epistemic relative to pragmatic value.
    ne_epistemic_boost:    NE-driven amplification of epistemic drive.
    uncertainty_method:    'novelty' or 'variance'.
    n_candidates:          Candidate actions to evaluate (discrete spaces).
    pragmatic_temperature: Softmax temperature for action selection.
    """
    epistemic_weight: float = 0.5
    ne_epistemic_boost: float = 1.0
    uncertainty_method: str = "novelty"
    n_candidates: int = 8
    pragmatic_temperature: float = 1.0