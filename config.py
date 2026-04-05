from dataclasses import dataclass

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