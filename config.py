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


@dataclass(frozen=True, kw_only=True)
class KWTAConfig(LIFConfig):
    """
    Hyperparameters for k-Winners-Take-All lateral inhibition.
    """
    k_winners: int = 3       # Liczba zwycięzców w populacji
    i_inh: float = 50.0      # Siła sygnału hamującego (mV odejmowane od V)
    window_ms: int = 100     # Okno czasowe integracji przed ewaluacją k-WTA


@dataclass(frozen=True, kw_only=True)
class PredictiveCodingConfig(KWTAConfig):
    """
    Hyperparameters for Predictive Coding layer.
    Extends KWTAConfig with feedback (top-down) dynamics.

    feedback_strength:       Skala, z jaką top-down prediction moduluje wejście.
    feedback_learning_rate:  Szybkość uczenia się wag feedback (oddzielna od STDP).
    """
    feedback_strength: float = 0.5
    feedback_learning_rate: float = 0.005


@dataclass(frozen=True, kw_only=True)
class WorkingMemoryConfig(LIFConfig):
    """
    Hyperparameters for Working Memory Module.
    Overrides tau_m to ~300 ms for sustained attractor dynamics.

    gate_threshold:   Poziom ACh powyżej którego brama WM otwiera się (akceptuje wejście).
    lateral_strength: Skala prądu rekurencyjnego (content @ w_lateral * lateral_strength).
    lateral_lr:       Szybkość hebbowskiego uczenia się połączeń lateralnych.
    """
    tau_m: float = 300.0       # Wolna dynamika błony — podtrzymuje aktywność przez setki ms
    gate_threshold: float = 0.5
    lateral_strength: float = 0.5
    lateral_lr: float = 0.01


@dataclass(frozen=True, kw_only=True)
class NeuromodulatorConfig:
    """
    Hyperparameters for the four-channel neuromodulatory system.

    Decay factors control how fast each signal returns to baseline.
    Baseline values define the resting level of each modulator.
    """
    # Exponential decay per timestep (dt = 1 ms)
    da_decay: float = 0.95    # Dopamina: średnia długość "fali" ~20 ms
    ach_decay: float = 0.90   # Acetylocholina: szybka odpowiedź na nowość
    ne_decay: float = 0.93    # Noradrenalina: modulacja czujności
    sero_decay: float = 0.97  # Serotonina: wolna, stabilizująca

    # Resting baselines (normalized 0–1)
    baseline_da: float = 0.5
    baseline_ach: float = 0.5
    baseline_ne: float = 0.3
    baseline_sero: float = 0.6


@dataclass(frozen=True, kw_only=True)
class WorldModelConfig:
    """
    Hyperparameters for the World Model (internal environment simulator).

    learning_rate: Gradient descent step for transition prediction weights.
    """
    learning_rate: float = 0.005