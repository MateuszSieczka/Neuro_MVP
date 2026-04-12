"""
Configuration — self-tuning dataclass configs deriving dependent parameters
from a minimal set of biological priors.

Design principles:
  1. Every magic number is either:
     (a) a biological constant with a literature reference, or
     (b) derived from other biological constants via a named equation.
  2. Frozen dataclasses prevent runtime mutation.
  3. ``__post_init__`` computes dependent values so they are available
     immediately and consistently.
  4. All time constants in ms, potentials in mV, rates in spikes/timestep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple

import numpy as np

from .simulation_context import SimulationContext, DEFAULT_CONTEXT


# =====================================================================
# Enums
# =====================================================================

class ReceptorType(Enum):
    """Neurotransmitter receptor subtypes."""
    # Dopamine
    D1 = auto()   # Excitatory, cAMP↑, Go pathway (Surmeier et al. 2007)
    D2 = auto()   # Inhibitory, cAMP↓, NoGo pathway

    # Acetylcholine
    M1 = auto()   # Cortical excitatory (muscarinic)
    M4 = auto()   # Striatal inhibitory (muscarinic)
    NACHR = auto() # Fast nicotinic (thalamic input gating)

    # Noradrenaline
    ALPHA1 = auto()  # Excitatory (cortical arousal)
    ALPHA2 = auto()  # Presynaptic inhibition (autoreceptor)
    BETA = auto()    # Slow modulatory (β-adrenergic)

    # Serotonin
    HT1A = auto()  # Inhibitory (raphe autoreceptor, hippocampal)
    HT2A = auto()  # Excitatory (cortical, hallucinogenic at excess)

    # GABA
    GABA_A = auto()  # Fast inhibitory (Cl⁻ channel)
    GABA_B = auto()  # Slow inhibitory (G-protein coupled)

    # Glutamate
    AMPA = auto()   # Fast excitatory
    NMDA = auto()   # Slow, voltage-dependent (Mg²⁺ block)


class SynapseType(Enum):
    """Synapse conductance models."""
    AMPA = auto()
    NMDA = auto()
    GABA_A = auto()
    GABA_B = auto()


# =====================================================================
# SimulationContext-aware base
# =====================================================================

@dataclass(frozen=True, kw_only=True)
class BaseConfig:
    """Base for all configs — carries a reference to the simulation context."""
    ctx: SimulationContext = field(default_factory=lambda: DEFAULT_CONTEXT)


# =====================================================================
# Phase 0: Neuron & Synapse Configs (derived from biophysics)
# =====================================================================

@dataclass(frozen=True, kw_only=True)
class NeuronConfig(BaseConfig):
    """Adaptive Exponential Integrate-and-Fire (AdEx) parameters.

    Reference: Brette & Gerstner (2005) "Adaptive Exponential
    Integrate-and-Fire Model as an Effective Description of Neuronal Activity"

    Membrane equation:
      C_m dV/dt = -g_L(V - E_L) + g_L Δ_T exp((V - V_T)/Δ_T) + I_syn - w

    Adaptation equation:
      τ_w dw/dt = a(V - E_L) - w

    After spike (V ≥ V_cutoff):
      V ← V_reset,  w ← w + b

    Neuron types from SAME equations (different params):
      Regular Spiking  (RS):  a=4, b=80.5, τ_w=144  (default)
      Fast Spiking     (FS):  a=0, b=0,    τ_w=144  (no adaptation)
      Intrinsic Burst  (IB):  a=2, b=60,   τ_w=20   (fast adapt → burst)
      Late Spiking     (LS):  a=-2, b=0               (delayed firing)

    Biological reference values:
      E_L = v_rest = -70 mV  (cortical pyramidal resting potential)
      V_T = v_thresh = -55 mV  (Na⁺ spike threshold, McCormick et al. 1985)
      V_reset = -75 mV   (post-spike AHP)
      g_L = 30 nS    (Destexhe & Paré 1999)
      C_m = 281 pF   (Brette & Gerstner 2005)
      refrac = 2 ms  (absolute refractory, Bean 2007)
    """
    # ── LIF-compatible fields (E_L = v_rest, V_T = v_thresh) ─────────
    v_rest: float = -70.0
    v_thresh: float = -55.0
    v_reset: float = -75.0
    tau_m: float = 20.0
    refrac_period: int = 2  # steps (= ms at dt=1)

    # ── AdEx-specific fields (Brette & Gerstner 2005) ────────────────
    delta_t: float = 2.0           # Spike initiation sharpness (mV)
    v_spike_cutoff: float = -30.0  # Spike detection threshold (mV), above V_T
    tau_w: float = 144.0           # Adaptation current time constant (ms)
    a: float = 4.0                 # Subthreshold adaptation conductance (nS)
    b: float = 80.5                # Spike-triggered adaptation increment (pA)
    g_L: float = 30.0              # Leak conductance (nS, Destexhe & Paré 1999)
    C_m: float = 281.0             # Membrane capacitance (pF)

    # Synaptic scaling (Turrigiano 2008)
    scaling_interval: int = 1000  # steps between homeostatic scaling events

    # Neuromodulatory compression factors
    ne_trace_compression: float = 3.0   # NE → up to (1 + factor)× trace compression
    ach_membrane_compression: float = 1.0  # ACh → up to (1 + factor)× τ_m compression

    # ── Derived (computed in __post_init__) ───────────────────────────
    # Membrane decay factor per timestep (LIF-compat, used by subclasses)
    mem_decay: float = field(init=False, default=0.0)
    # Membrane gain complement (1 - decay)
    mem_gain: float = field(init=False, default=0.0)
    # Threshold gap: v_thresh - v_rest (mV) — used for weight scaling
    gap: float = field(init=False, default=0.0)
    # Minimum current to reach threshold from rest in one tau_m
    i_thresh: float = field(init=False, default=0.0)
    # Adaptation decay: exp(-dt / tau_w)
    w_decay: float = field(init=False, default=0.0)
    # Adaptation gain: 1 - exp(-dt / tau_w)
    w_gain: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        assert self.tau_m > 0, f"tau_m must be positive, got {self.tau_m}"
        assert self.v_reset < self.v_thresh, f"v_reset ({self.v_reset}) must be < v_thresh ({self.v_thresh})"
        assert self.v_rest < self.v_thresh, f"v_rest ({self.v_rest}) must be < v_thresh ({self.v_thresh})"
        assert self.refrac_period >= 0, f"refrac_period must be >= 0, got {self.refrac_period}"
        assert self.delta_t > 0, f"delta_t must be positive, got {self.delta_t}"
        assert self.v_spike_cutoff > self.v_thresh, (
            f"v_spike_cutoff ({self.v_spike_cutoff}) must be > v_thresh ({self.v_thresh})"
        )
        assert self.tau_w > 0, f"tau_w must be positive, got {self.tau_w}"
        assert self.g_L > 0, f"g_L must be positive, got {self.g_L}"
        assert self.C_m > 0, f"C_m must be positive, got {self.C_m}"
        object.__setattr__(self, 'mem_decay', self.ctx.decay(self.tau_m))
        object.__setattr__(self, 'mem_gain', self.ctx.complement(self.tau_m))
        gap = abs(self.v_thresh - self.v_rest)
        object.__setattr__(self, 'gap', gap)
        # I_thresh: current that brings v from rest to thresh in one tau_m
        # v_thresh = v_rest + I × (1 - decay) => I = gap / (1 - decay)
        object.__setattr__(self, 'i_thresh', gap * (1.0 - self.mem_decay))
        # AdEx adaptation decay factors
        object.__setattr__(self, 'w_decay', self.ctx.decay(self.tau_w))
        object.__setattr__(self, 'w_gain', self.ctx.complement(self.tau_w))


@dataclass(frozen=True, kw_only=True)
class STDPConfig(BaseConfig):
    """Spike-Timing-Dependent Plasticity kernel parameters.

    Reference: Bi & Poo (2001) "Synaptic modification by correlated
    activity: Hebb's postulate revisited"

    Asymmetric window:
      Δw = +A_plus  × exp(-|Δt| / τ_plus)   if pre before post (LTP)
      Δw = -A_minus × exp(-|Δt| / τ_minus)   if post before pre (LTD)

    Bi & Poo values: A_minus/A_plus ≈ 1.05, τ_plus ≈ 17ms, τ_minus ≈ 34ms
    This slight LTD bias ensures long-term weight stability.
    """
    tau_plus: float = 17.0    # LTP trace time constant (ms)
    tau_minus: float = 34.0   # LTD trace time constant (ms)
    a_plus: float = 0.01      # LTP amplitude
    a_minus: float = 0.0105   # LTD amplitude (1.05× A_plus, Bi & Poo)
    tau_eligibility: float = 20.0  # Third-factor eligibility trace (ms)

    # ── Derived ───────────────────────────────────────────────────────
    pre_decay: float = field(init=False, default=0.0)
    post_decay: float = field(init=False, default=0.0)
    elig_decay: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        assert self.tau_plus > 0, f"tau_plus must be positive, got {self.tau_plus}"
        assert self.tau_minus > 0, f"tau_minus must be positive, got {self.tau_minus}"
        assert self.a_plus > 0, f"a_plus must be positive, got {self.a_plus}"
        assert self.tau_eligibility > 0, f"tau_eligibility must be positive, got {self.tau_eligibility}"
        object.__setattr__(self, 'pre_decay', self.ctx.decay(self.tau_plus))
        object.__setattr__(self, 'post_decay', self.ctx.decay(self.tau_minus))
        object.__setattr__(self, 'elig_decay', self.ctx.decay(self.tau_eligibility))


@dataclass(frozen=True, kw_only=True)
class HomeostaticConfig(BaseConfig):
    """BCM-derived homeostatic threshold adaptation.

    Reference: Bienenstock, Cooper & Munro (1982), Turrigiano (2008)

    The adaptation learning rate is derived from BCM theory:
      lr = 1 / (homeostatic_tau × target_rate)

    This ensures the threshold correction magnitude is independent of
    the absolute target rate — a neuron targeting 1% or 10% converges
    at the same relative speed.
    """
    target_rate: float = 0.05        # Spikes per timestep (5%)
    homeostatic_tau: float = 1000.0  # Slow averaging window (ms)
    thresh_min: float = -68.0        # Physiological threshold floor (mV)
    thresh_max: float = -45.0        # Physiological threshold ceiling (mV)

    # Dark Matter Neurons (reserve capacity for continual learning)
    dark_matter_ratio: float = 0.0
    dark_matter_thresh_offset: float = 20.0  # Extra mV for dark neurons
    ne_thresh_drop: float = 15.0             # Max NE-driven threshold reduction

    # ── Derived ───────────────────────────────────────────────────────
    homeo_decay: float = field(init=False, default=0.0)
    thresh_adapt_lr: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        assert 0 < self.target_rate < 1, f"target_rate must be in (0, 1), got {self.target_rate}"
        assert self.homeostatic_tau > 0, f"homeostatic_tau must be positive, got {self.homeostatic_tau}"
        assert self.thresh_min < self.thresh_max, f"thresh_min ({self.thresh_min}) must be < thresh_max ({self.thresh_max})"
        assert 0 <= self.dark_matter_ratio <= 1, f"dark_matter_ratio must be in [0, 1], got {self.dark_matter_ratio}"
        object.__setattr__(self, 'homeo_decay', self.ctx.decay(self.homeostatic_tau))
        # BCM-derived: lr = 1 / (tau × target_rate)
        lr = 1.0 / (self.homeostatic_tau * max(self.target_rate, 1e-6))
        object.__setattr__(self, 'thresh_adapt_lr', lr)


@dataclass(frozen=True, kw_only=True)
class SynapseConfig(BaseConfig):
    """Conductance-based synapse parameters.

    Reference: Jahr & Stevens (1990), Destexhe et al. (1998)

    Four channel types with biophysical kinetics:
      AMPA:   τ_rise ≈ 0.2ms, τ_decay ≈ 2ms   (fast excitatory)
      NMDA:   τ_rise ≈ 2ms,   τ_decay ≈ 100ms  (slow, voltage-gated)
      GABA-A: τ_rise ≈ 0.5ms, τ_decay ≈ 5ms    (fast inhibitory)
      GABA-B: τ_rise ≈ 30ms,  τ_decay ≈ 100ms   (slow inhibitory)

    NMDA voltage-dependent Mg²⁺ block (Jahr & Stevens 1990):
      B(V) = 1 / (1 + [Mg²⁺]/3.57 × exp(-0.062 × V))
      At V_rest=-70mV: B ≈ 0.02 (blocked)
      At V_thresh=-55mV: B ≈ 0.12 (partially open)
      At 0mV: B ≈ 1.0 (fully open)
    """
    # AMPA
    tau_ampa: float = 2.0
    # NMDA
    tau_nmda: float = 100.0
    mg_concentration: float = 1.0  # mM (extracellular [Mg²⁺])
    # GABA-A
    tau_gaba_a: float = 5.0
    # GABA-B
    tau_gaba_b: float = 100.0
    # Reversal potentials
    e_exc: float = 0.0       # Excitatory reversal (mV)
    e_inh: float = -75.0     # Inhibitory reversal (mV)
    # AMPA/NMDA ratio (Myme et al. 2003: ~2:1 to 4:1 in cortex)
    ampa_nmda_ratio: float = 3.0

    # ── Derived ───────────────────────────────────────────────────────
    ampa_decay: float = field(init=False, default=0.0)
    nmda_decay: float = field(init=False, default=0.0)
    gaba_a_decay: float = field(init=False, default=0.0)
    gaba_b_decay: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        assert self.tau_ampa > 0, f"tau_ampa must be positive, got {self.tau_ampa}"
        assert self.tau_nmda > 0, f"tau_nmda must be positive, got {self.tau_nmda}"
        assert self.tau_ampa < self.tau_nmda, f"tau_ampa ({self.tau_ampa}) must be < tau_nmda ({self.tau_nmda})"
        assert self.mg_concentration > 0, f"mg_concentration must be positive, got {self.mg_concentration}"
        object.__setattr__(self, 'ampa_decay', self.ctx.decay(self.tau_ampa))
        object.__setattr__(self, 'nmda_decay', self.ctx.decay(self.tau_nmda))
        object.__setattr__(self, 'gaba_a_decay', self.ctx.decay(self.tau_gaba_a))
        object.__setattr__(self, 'gaba_b_decay', self.ctx.decay(self.tau_gaba_b))

    @staticmethod
    def nmda_mg_block(v: float | np.ndarray) -> float | np.ndarray:
        """NMDA voltage-dependent Mg²⁺ block factor B(V).

        B(V) = 1 / (1 + [Mg²⁺]/3.57 × exp(-0.062 × V))

        Reference: Jahr & Stevens (1990)
        """
        return 1.0 / (1.0 + (1.0 / 3.57) * np.exp(-0.062 * v))


@dataclass(frozen=True, kw_only=True)
class CompetitiveConfig(BaseConfig):
    """k-WTA parameters derived from target sparsity and population size.

    Key derivation:
      k_winners = ceil(target_sparsity × num_neurons)
      i_inh = (v_thresh - v_rest) × num_neurons / k_winners × inhibition_strength

    The inhibition strength ensures losers are pushed below rest.
    """
    target_sparsity: float = 0.15  # Fraction of neurons active per window
    inhibition_strength: float = 1.5  # Multiplier on i_inh (1.0 = just-sufficient)
    window_ms: float = 100.0  # k-WTA evaluation window (ms)

    def __post_init__(self) -> None:
        assert 0 < self.target_sparsity < 1, f"target_sparsity must be in (0, 1), got {self.target_sparsity}"
        assert self.inhibition_strength > 0, f"inhibition_strength must be positive, got {self.inhibition_strength}"
        assert self.window_ms > 0, f"window_ms must be positive, got {self.window_ms}"

    @staticmethod
    def derive_k(target_sparsity: float, num_neurons: int) -> int:
        """Compute k_winners from target sparsity and population size."""
        return max(1, math.ceil(target_sparsity * num_neurons))

    @staticmethod
    def derive_i_inh(
        gap: float,
        num_neurons: int,
        k_winners: int,
        strength: float = 1.5,
    ) -> float:
        """Derive inhibition magnitude from biophysics.

        i_inh must push losers below V_rest with margin. Scaled by
        the ratio N/k to ensure k-WTA dynamics are maintained regardless
        of population size.
        """
        return gap * (num_neurons / max(k_winners, 1)) * strength


# =====================================================================
# Predictive Coding & Pyramidal
# =====================================================================

@dataclass(frozen=True, kw_only=True)
class PredictiveCodingConfig(BaseConfig):
    """Predictive Coding layer parameters.

    Reference: Friston (2010), Rao & Ballard (1999)

    Single-step dynamics matching the spiking paradigm:
      v += ACh × error_gradient + (1-ACh) × top_down
    No inner relaxation loops.
    """
    feedback_strength: float = 0.5
    feedback_learning_rate: float = 0.005
    feedback_norm: bool = True

    def __post_init__(self) -> None:
        assert self.feedback_learning_rate > 0, f"feedback_learning_rate must be positive, got {self.feedback_learning_rate}"


@dataclass(frozen=True, kw_only=True)
class PyramidalConfig(BaseConfig):
    """Multi-compartment pyramidal neuron parameters.

    Reference: Larkum, Zhu & Bhatt (1999), Payeur et al. (2021)

    Apical trunk: passive cable with Ca²⁺ spike mechanism.
      tau_apical = 100-200ms (electrotonically remote from soma)
      Ca²⁺ activation: m_Ca = sigmoid((V_apical - V_half) / k_ca)
      BAC firing: soma spike + apical Ca²⁺ spike within temporal window → burst

    Burst-dependent plasticity (Payeur et al. 2021):
      burst → 3-5× STDP eligibility boost
    """
    tau_apical: float = 150.0       # Apical membrane τ (ms) — was 50, should be 100-200
    apical_threshold: float = 0.3   # Normalized Ca²⁺ spike threshold
    apical_boost: float = 10.0      # Somatic threshold reduction (mV) during plateau
    burst_stdp_factor: float = 3.0  # Payeur et al.: 3-5× boost
    apical_lr: float = 0.005        # Apical weight Hebbian learning rate
    plateau_duration_ms: int = 50   # Ca²⁺ plateau duration (ms)
    background_noise_std: float = 2.0  # Membrane noise σ (mV)
    # Ca²⁺ spike voltage-dependent activation (Larkum 2013)
    ca_v_half: float = 0.4  # Half-activation of apical Ca²⁺ channel
    ca_k: float = 0.1       # Activation steepness
    # Top-down prediction scaling for generate_prediction()
    feedback_strength: float = 0.5

    # ── Derived ───────────────────────────────────────────────────────
    apical_decay: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        assert self.tau_apical > 0, f"tau_apical must be positive, got {self.tau_apical}"
        assert self.burst_stdp_factor > 0, f"burst_stdp_factor must be positive, got {self.burst_stdp_factor}"
        object.__setattr__(self, 'apical_decay', self.ctx.decay(self.tau_apical))


@dataclass(frozen=True, kw_only=True)
class ErrorNeuronConfig(BaseConfig):
    """Error/State neuron layer for continuous predictive coding.

    Reference: Bogacz (2017), Bastos et al. (2012)

    State neurons (pyramidal L2/3): slow τ ~20ms, maintain belief μ
    Error neurons (stellate L4): fast τ ~4ms, compute ε = input - g(μ)
    """
    n_state: int = 64
    n_error: int = 30
    tau_state: float = 20.0
    tau_error: float = 4.0
    w_bu_lr: float = 0.005   # Error→State (Hebbian)
    w_td_lr: float = 0.005   # State→Error (Anti-Hebbian, Rao & Ballard)
    refrac_period: int = 2
    # ACh gain range from Hill equation dose-response
    ach_gain_min: float = 0.7
    ach_gain_max: float = 1.5
    ach_ec50: float = 0.4     # Half-max ACh concentration
    ach_hill_n: float = 1.5   # Hill coefficient

    # ── Derived ───────────────────────────────────────────────────────
    state_decay: float = field(init=False, default=0.0)
    error_decay: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        assert self.n_state > 0, f"n_state must be positive, got {self.n_state}"
        assert self.n_error > 0, f"n_error must be positive, got {self.n_error}"
        assert self.tau_state > 0, f"tau_state must be positive, got {self.tau_state}"
        assert self.tau_error > 0, f"tau_error must be positive, got {self.tau_error}"
        object.__setattr__(self, 'state_decay', self.ctx.decay(self.tau_state))
        object.__setattr__(self, 'error_decay', self.ctx.decay(self.tau_error))


# =====================================================================
# Inhibitory Pool (Interneurons)
# =====================================================================

@dataclass(frozen=True, kw_only=True)
class InhibitoryPoolConfig(BaseConfig):
    """GABAergic inhibitory pool parameters.

    Reference: Brunel & Wang (2003), Markram et al. (2004)

    PV+ basket cells: τ_m ~8-10ms (fast-spiking), lower spike threshold.
    Target sparsity controls initial E/I weight calibration.
    Dual GABA channels: GABA-A (fast, ~70-80%) + GABA-B (slow, ~20-30%).
    """
    n_interneurons: int = 16
    tau_m_inh: float = 8.0
    v_rest: float = -70.0
    v_thresh: float = -58.0  # Lower than excitatory (fast-spiking PV+)
    v_reset: float = -75.0
    w_ei_mean: float = 0.8
    w_ie_mean: float = 0.6
    gaba_b_ratio: float = 0.25  # Isaacson & Scanziani (2011): ~20-30%
    target_sparsity: float = 0.15
    # Inhibitory STDP (Woodin et al. 2003)
    inh_stdp_lr: float = 0.001
    # E/I balance homeostatic rate
    ei_balance_lr: float = 0.0005

    # ── Derived ───────────────────────────────────────────────────────
    inh_decay: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        assert self.n_interneurons > 0, f"n_interneurons must be positive, got {self.n_interneurons}"
        assert self.tau_m_inh > 0, f"tau_m_inh must be positive, got {self.tau_m_inh}"
        assert 0 < self.target_sparsity < 1, f"target_sparsity must be in (0, 1), got {self.target_sparsity}"
        object.__setattr__(self, 'inh_decay', self.ctx.decay(self.tau_m_inh))


# =====================================================================
# Neuromodulation & Receptor
# =====================================================================

@dataclass(frozen=True, kw_only=True)
class NeuromodulatorConfig(BaseConfig):
    """Four-channel neuromodulatory system.

    Decay time constants derived from reuptake/degradation kinetics:
      DA:  τ ≈ 200ms  — DAT reuptake in striatum (Cragg & Rice 2004)
      ACh: τ ≈ 25ms   — AChE hydrolysis (Sarter et al. 2009)
      NE:  τ ≈ 75ms   — NET reuptake (Morilak et al. 2005)
      5-HT: τ ≈ 150ms — SERT reuptake (volume transmission slower)

    Tonic DA: continuous leaky integrator τ ≈ 60s (Grace 1991).
    """
    # Phasic time constants (ms)
    tau_da: float = 200.0
    tau_ach: float = 25.0
    tau_ne: float = 75.0
    tau_sero: float = 150.0

    # Tonic DA: continuous leaky integrator (Grace 1991)
    tau_tonic_da: float = 60_000.0  # 60 seconds — minute-scale VTA background

    # Baselines
    baseline_da: float = 0.5
    baseline_ach: float = 0.5
    baseline_ne: float = 0.3
    baseline_sero: float = 0.6
    baseline_tonic_da: float = 0.0

    # Dynamic timescale modulation
    tau_ne_compression: float = 4.0
    tau_ach_compression: float = 2.0

    # Serotonin input weights (dorsal raphe anatomy)
    # DRN receives ~70% cortical (sensory/world model), ~30% amygdala/VTA
    sero_world_weight: float = 0.7
    sero_behavioral_weight: float = 0.3

    # DA RMS adaptation time constant (VTA gain adaptation)
    # Biological: minutes scale; sim: τ ≈ 10s = 10000 steps
    da_rms_decay: float = 0.9999

    # ── Derived ───────────────────────────────────────────────────────
    da_decay: float = field(init=False, default=0.0)
    ach_decay: float = field(init=False, default=0.0)
    ne_decay: float = field(init=False, default=0.0)
    sero_decay: float = field(init=False, default=0.0)
    tonic_da_decay: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        assert self.tau_da > 0, f"tau_da must be positive, got {self.tau_da}"
        assert self.tau_ach > 0, f"tau_ach must be positive, got {self.tau_ach}"
        assert self.tau_ne > 0, f"tau_ne must be positive, got {self.tau_ne}"
        assert self.tau_sero > 0, f"tau_sero must be positive, got {self.tau_sero}"
        assert self.tau_tonic_da > 0, f"tau_tonic_da must be positive, got {self.tau_tonic_da}"
        object.__setattr__(self, 'da_decay', self.ctx.decay(self.tau_da))
        object.__setattr__(self, 'ach_decay', self.ctx.decay(self.tau_ach))
        object.__setattr__(self, 'ne_decay', self.ctx.decay(self.tau_ne))
        object.__setattr__(self, 'sero_decay', self.ctx.decay(self.tau_sero))
        object.__setattr__(self, 'tonic_da_decay', self.ctx.decay(self.tau_tonic_da))


@dataclass(frozen=True, kw_only=True)
class ReceptorProfile(BaseConfig):
    """Per-layer receptor expression profile.

    Each layer declares which receptor subtypes it expresses and at what
    density (0-1). The NeuromodulatorSystem distributes global transmitter
    levels; each layer computes local effects via:

      effect = transmitter_level × density × dose_response(transmitter_level)

    Reference: Doya (2002), Seamans & Yang (2004)
    """
    # Dopamine receptors
    d1_density: float = 0.0
    d2_density: float = 0.0
    # Acetylcholine receptors
    m1_density: float = 0.0
    m4_density: float = 0.0
    nachr_density: float = 0.0
    # Noradrenaline receptors
    alpha1_density: float = 0.0
    alpha2_density: float = 0.0
    beta_density: float = 0.0
    # Serotonin receptors
    ht1a_density: float = 0.0
    ht2a_density: float = 0.0

    def to_density_dict(self) -> dict["ReceptorType", float]:
        """Convert to {ReceptorType: density} mapping for receptor.py."""
        return {
            ReceptorType.D1: self.d1_density,
            ReceptorType.D2: self.d2_density,
            ReceptorType.M1: self.m1_density,
            ReceptorType.M4: self.m4_density,
            ReceptorType.NACHR: self.nachr_density,
            ReceptorType.ALPHA1: self.alpha1_density,
            ReceptorType.ALPHA2: self.alpha2_density,
            ReceptorType.BETA: self.beta_density,
            ReceptorType.HT1A: self.ht1a_density,
            ReceptorType.HT2A: self.ht2a_density,
        }


# Predefined receptor profiles for different brain regions
CORTICAL_L4_RECEPTORS = ReceptorProfile(
    nachr_density=0.8, m1_density=0.3, alpha1_density=0.5, beta_density=0.3,
)
CORTICAL_L5_RECEPTORS = ReceptorProfile(
    d1_density=0.6, m1_density=0.4, alpha1_density=0.4, ht2a_density=0.3,
)
PFC_RECEPTORS = ReceptorProfile(
    d1_density=0.7, m1_density=0.5, alpha1_density=0.3, ht2a_density=0.2,
)
STRIATUM_D1_RECEPTORS = ReceptorProfile(
    d1_density=0.9, m4_density=0.3, alpha1_density=0.2,
)
STRIATUM_D2_RECEPTORS = ReceptorProfile(
    d2_density=0.9, m4_density=0.3, alpha1_density=0.2,
)


# =====================================================================
# Oscillator
# =====================================================================

@dataclass(frozen=True, kw_only=True)
class OscillatorConfig(BaseConfig):
    """Theta-gamma nested oscillation parameters.

    Reference: Lisman & Jensen (2013) "The theta-gamma neural code"

    Theta (4-8 Hz): Drives episodic encoding, gates memory storage/retrieval.
      Period ≈ 125-250ms.
    Gamma (30-100 Hz): Local binding within theta phase, paces k-WTA.
      Period ≈ 10-33ms.

    Phase-amplitude coupling (PAC):
      gamma_amplitude = base + modulation_depth × cos(theta_phase)
    """
    # Theta rhythm
    theta_freq_hz: float = 6.0   # Center frequency (Hz)
    theta_min_hz: float = 4.0
    theta_max_hz: float = 8.0
    # Gamma rhythm
    gamma_freq_hz: float = 40.0  # Center frequency (Hz)
    gamma_min_hz: float = 30.0
    gamma_max_hz: float = 100.0
    # Phase-amplitude coupling depth (0 = no coupling, 1 = full modulation)
    pac_depth: float = 0.6
    # NE/5-HT modulation of theta frequency
    ne_theta_shift: float = 2.0   # Hz added at NE=1
    sero_theta_shift: float = -1.0  # Hz subtracted at 5-HT=1 (longer cycles)

    def __post_init__(self) -> None:
        assert 0 < self.theta_min_hz <= self.theta_freq_hz <= self.theta_max_hz, "theta freq range invalid"
        assert 0 < self.gamma_min_hz <= self.gamma_freq_hz <= self.gamma_max_hz, "gamma freq range invalid"
        assert 0 <= self.pac_depth <= 1, f"pac_depth must be in [0, 1], got {self.pac_depth}"


# =====================================================================
# Astrocyte & Glial
# =====================================================================

@dataclass(frozen=True, kw_only=True)
class AstrocyteConfig(BaseConfig):
    """Astrocyte field for local precision estimation via Ca²⁺ dynamics.

    Reference: De Pittà, Volman, Berry & Ben-Jacob (2011)

    tau_ca = 5000ms (5 seconds) — biological astrocyte Ca²⁺ τ = 2-10s.
    D-Serine release: sigmoid, not step function.
    Gap junction diffusion between zones.

    ATP Energy Budget (Krok 1.3):
      Continuous, not binary. ATP modulates spike threshold (V_T)
      and leak conductance (g_L) — network silences smoothly under
      energy depletion. Na⁺/K⁺-ATPase slowdown ≈ rising V_T + rising g_L.
    """
    n_zones: int = 16
    tau_ca: float = 5000.0      # Ca²⁺ time constant (ms) — was 500, now biological
    ca_accumulation: float = 0.1
    ca_threshold: float = 0.5   # Only used as sigmoid midpoint now
    ca_release_k: float = 0.15  # Sigmoid steepness for D-Serine release
    d_serine_max: float = 0.3   # Max release per step
    gain_baseline: float = 1.0
    gain_max: float = 2.0
    metabolic_scale: float = 0.5
    # Gap junction diffusion coefficient (De Pittà et al. 2011)
    gap_junction_D: float = 0.01  # Diffusion coefficient between zones

    # ── ATP Energy Budget ─────────────────────────────────────────────
    atp_max: float = 1.0            # Normalised ceiling
    atp_regen_rate: float = 0.001   # Recovery per ms (~1 s to full)
    atp_spike_cost: float = 0.02    # Cost per spike per zone per step
    atp_threshold_shift: float = 10.0  # Max V_T shift (mV) at zero ATP
    atp_leak_gain: float = 0.5      # Max g_L multiplier increase at zero ATP

    # ── Derived ───────────────────────────────────────────────────────
    ca_decay: float = field(init=False, default=0.0)
    d_serine_decay: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        assert self.n_zones > 0, f"n_zones must be positive, got {self.n_zones}"
        assert self.tau_ca > 0, f"tau_ca must be positive, got {self.tau_ca}"
        assert self.gain_max >= self.gain_baseline, f"gain_max ({self.gain_max}) must be >= gain_baseline ({self.gain_baseline})"
        object.__setattr__(self, 'ca_decay', self.ctx.decay(self.tau_ca))
        # D-Serine decays with τ ≈ 200ms (gliotransmitter clearance)
        object.__setattr__(self, 'd_serine_decay', self.ctx.decay(200.0))


# =====================================================================
# Attention
# =====================================================================

@dataclass(frozen=True, kw_only=True)
class AttentionConfig(BaseConfig):
    """Spatial attention system with bottom-up saliency and IOR.

    Reference: Reynolds & Heeger (2009), Posner & Cohen (1984)

    Temperature modulated by NE (inverse-U, Usher & Damasio):
      T = T_base × (1 + |NE - NE_optimal|²)
    """
    gain_strength: float = 2.0
    base_temperature: float = 1.0
    ne_optimal: float = 0.5      # Optimal NE for focused attention
    learning_rate: float = 0.005
    decay: float = 0.9           # Temporal smoothing
    # Bottom-up saliency weight (vs top-down)
    bottom_up_weight: float = 0.4  # α for saliency mix
    # Inhibition of Return (Posner & Cohen 1984)
    ior_tau: float = 400.0         # IOR decay τ (ms)
    ior_strength: float = 0.3      # IOR inhibition magnitude

    # ── Derived ───────────────────────────────────────────────────────
    ior_decay: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        assert self.base_temperature > 0, f"base_temperature must be positive, got {self.base_temperature}"
        assert self.ior_tau > 0, f"ior_tau must be positive, got {self.ior_tau}"
        object.__setattr__(self, 'ior_decay', self.ctx.decay(self.ior_tau))

    def ne_modulated_temperature(self, ne: float) -> float:
        """Inverse-U relationship: NE at optimal → low T (focused)."""
        return self.base_temperature * (1.0 + (ne - self.ne_optimal) ** 2)


# =====================================================================
# Memory Systems
# =====================================================================

@dataclass(frozen=True, kw_only=True)
class WorkingMemoryConfig(BaseConfig):
    """Working Memory prefrontal attractor dynamics.

    Reference: Goldman-Rakic (1995), O'Reilly & Frank (2006)

    tau_m = 300ms for slow sustained dynamics.
    Dual gating: ACh (sensory) AND DA (update signal).
    """
    tau_m: float = 300.0
    v_rest: float = -70.0
    v_thresh: float = -55.0
    v_reset: float = -75.0
    refrac_period: int = 2
    # Gating thresholds (O'Reilly & Frank 2006: conjunction gate)
    ach_gate_threshold: float = 0.5
    da_gate_threshold: float = 0.4
    # Lateral attractor
    lateral_strength: float = 0.5
    lateral_lr: float = 0.01
    learning_rate: float = 0.01
    # STDP traces
    tau_e: float = 20.0
    tau_pre: float = 20.0
    tau_post: float = 20.0

    # ── Derived ───────────────────────────────────────────────────────
    mem_decay: float = field(init=False, default=0.0)
    content_decay: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        assert self.tau_m > 0, f"tau_m must be positive, got {self.tau_m}"
        assert self.v_reset < self.v_thresh, f"v_reset ({self.v_reset}) must be < v_thresh ({self.v_thresh})"
        object.__setattr__(self, 'mem_decay', self.ctx.decay(self.tau_m))
        object.__setattr__(self, 'content_decay', self.ctx.decay(self.tau_m))


@dataclass(frozen=True, kw_only=True)
class EpisodicMemoryConfig(BaseConfig):
    """Hippocampal one-shot episodic memory.

    Reference: Rolls (2013), O'Neill et al. (2010)

    Interference-based forgetting: new memories overwrite most similar
    (not oldest). Consolidated memories resist overwrite.
    """
    ne_threshold: float = 0.3
    similarity_thresh: float = 0.85
    capacity: int = 500
    # Pattern separation (DG-like sparse coding)
    dg_sparsity: float = 0.05       # Target sparsity for dentate gyrus encoding
    dg_expansion_factor: int = 5    # DG has ~5× more neurons than CA3
    # Consolidation resistance
    consolidation_threshold: int = 3  # Replay count to become resistant

    def __post_init__(self) -> None:
        assert self.capacity > 0, f"capacity must be positive, got {self.capacity}"
        assert 0 < self.similarity_thresh <= 1, f"similarity_thresh must be in (0, 1], got {self.similarity_thresh}"


@dataclass(frozen=True, kw_only=True)
class SequenceMemoryConfig(BaseConfig):
    """Temporal sequence learning parameters."""
    learning_rate: float = 0.01
    decay: float = 0.999
    max_weight: float = 1.0
    # DG-like pattern separation (D5)
    expansion_factor: int = 4     # DG expansion ratio (Rolls 2013)
    sparsity_k: float = 0.1       # Fraction of active neurons after k-WTA

    def __post_init__(self) -> None:
        assert self.learning_rate > 0, f"learning_rate must be positive, got {self.learning_rate}"
        assert 0 < self.decay <= 1, f"decay must be in (0, 1], got {self.decay}"
        assert self.max_weight > 0, f"max_weight must be positive, got {self.max_weight}"


@dataclass(frozen=True, kw_only=True)
class ReplayBufferConfig(BaseConfig):
    """Sleep consolidation parameters.

    Reference: Walker & Stickgold (2006), Diekelmann & Born (2010)

    Two-phase sleep:
      SWS: Reverse replay (sharp-wave ripples), cortical consolidation
      REM: Forward replay (theta sequences), world model refinement
    """
    capacity: int = 1000
    sws_replay_fraction: float = 0.7  # Fraction of sleep budget for SWS
    rem_replay_fraction: float = 0.3  # Fraction for REM
    gamma: float = 0.99

    def __post_init__(self) -> None:
        assert self.capacity > 0, f"capacity must be positive, got {self.capacity}"
        assert 0 < self.sws_replay_fraction + self.rem_replay_fraction <= 1, "replay fractions must sum to <= 1"
        assert 0 < self.gamma <= 1, f"gamma must be in (0, 1], got {self.gamma}"


# =====================================================================
# World Model
# =====================================================================

@dataclass(frozen=True, kw_only=True)
class WorldModelConfig(BaseConfig):
    """SNN World Model (ErrorNeuronLayer encoder + Hebbian decoder).

    Reference: Friston et al. (2015)
    """
    hidden_size: int = 64
    decode_lr: float = 0.02
    encoder_lr: float = 0.005
    n_neurons_per_dim: int = 15   # Population coding density (Pouget et al. 2000)
    # Multi-step rehearsal (Friston et al. 2015)
    max_rehearsal_depth: int = 3  # Planning depth (modulated by 5-HT)

    def __post_init__(self) -> None:
        assert self.hidden_size > 0, f"hidden_size must be positive, got {self.hidden_size}"
        assert self.decode_lr > 0, f"decode_lr must be positive, got {self.decode_lr}"
        assert self.n_neurons_per_dim > 0, f"n_neurons_per_dim must be positive, got {self.n_neurons_per_dim}"


@dataclass(frozen=True, kw_only=True)
class ActiveInferenceConfig(BaseConfig):
    """Active Inference action selection parameters.

    Reference: Friston (2010), Bromberg-Martin et al. (2010)
    """
    epistemic_weight: float = 0.5
    ne_epistemic_boost: float = 0.5
    pragmatic_temperature: float = 1.0
    uncertainty_method: str = "novelty"

    def __post_init__(self) -> None:
        assert self.pragmatic_temperature > 0, f"pragmatic_temperature must be positive, got {self.pragmatic_temperature}"
        assert self.uncertainty_method in ("novelty", "entropy", "variance"), (
            f"uncertainty_method must be one of novelty/entropy/variance, got {self.uncertainty_method}"
        )


# =====================================================================
# Basal Ganglia
# =====================================================================

@dataclass(frozen=True, kw_only=True)
class AgentConfig(BaseConfig):
    """Agent-level parameters extracted from snn_agent.py magic numbers.

    These control reward shaping, plasticity gating, exploration dynamics,
    and sleep scheduling — previously hardcoded in SNNAgent.observe().
    """
    intrinsic_reward_weight: float = 0.1       # curiosity weight in effective_reward
    da_offset: float = 0.0                     # shift for DA → learning_rate_modulation
    td_clip: float = 50.0                      # gradient clipping on TD error
    consolidation_midpoint: float = 0.7        # sigmoid inflection for consolidation gate
    consolidation_steepness: float = 8.0       # sigmoid steepness
    consolidation_floor: float = 1.0           # minimum plasticity scale (1.0 = gate disabled)
    noise_smoothing: float = 0.8               # EMA coefficient for exploration noise
    min_exploration: float = 0.15              # exploration noise floor
    sleep_gain_scale: float = 0.5              # quality → sleep_gain multiplier
    sleep_gain_max: float = 2.5                # sleep_gain ceiling

    def __post_init__(self) -> None:
        assert self.td_clip > 0, f"td_clip must be positive, got {self.td_clip}"
        assert 0 < self.consolidation_floor <= 1, f"consolidation_floor must be in (0, 1], got {self.consolidation_floor}"
        assert 0 < self.min_exploration < 1, f"min_exploration must be in (0, 1), got {self.min_exploration}"
        assert self.sleep_gain_max > 0, f"sleep_gain_max must be positive, got {self.sleep_gain_max}"


@dataclass(frozen=True, kw_only=True)
class BasalGangliaConfig(BaseConfig):
    """Integrated BG system with D1/D2 pathway separation.

    Reference: Frank (2005), Gurney et al. (2001), Wilson & Kawaguchi (1996)

    MSN dynamics:
      Down state: τ_m ≈ 80ms, high threshold (quiescent)
      Up state: τ_m ≈ 25ms, low threshold (ready to fire)
    """
    gamma: float = 0.95
    critic_lr: float = 1e-3
    actor_lr: float = 1e-2
    # Eligibility trace time constants
    tau_e_actor: float = 20.0
    tau_e_critic: float = 50.0
    # NE compression (Aston-Jones & Cohen 2005)
    tau_ne_compression: float = 4.0
    # LIF parameters for BG populations
    # MSN Up-state parameters (Wilson & Kawaguchi 1996)
    tau_m_msn_up: float = 25.0     # Up-state τ (fast-spiking, ready)
    tau_m_msn_down: float = 80.0   # Down-state τ (quiescent)
    tau_m_critic: float = 15.0     # Ventral striatal neuron τ
    v_rest: float = -70.0
    v_thresh: float = -55.0
    v_reset: float = -75.0
    refrac_period: int = 2
    membrane_noise_std: float = 1.0  # mV background cortical noise (Destexhe et al. 2003: 1-1.5 mV in vivo)
    hidden_size: int = 128
    w_clip: float = 1.0
    w_clip_critic: float = 5.0
    # D1/D2 pathway balance (Frank 2005)
    d1_bias: float = 0.6    # D1 pathway relative strength at DA=0.5
    d2_bias: float = 0.4    # D2 pathway relative strength at DA=0.5
    # Population coding: neurons per action channel (Georgopoulos 1986;
    # Humphries, Stewart & Gurney 2006).  Each motor action is represented
    # by a population, not a single neuron.  Population sum gives robust
    # rate estimates for spike-based action selection.
    neurons_per_action: int = 32
    # Exploration
    exploration_noise: float = 0.3
    # Homeostatic synaptic scaling (Turrigiano 2004, 2008)
    # Slow multiplicative weight adjustment targeting stable per-neuron
    # firing rate.  Prevents weight erosion and runaway excitation
    # without task-specific clip values.
    homeo_target_rate: float = 0.01    # Target firing rate per neuron (Planert et al. 2010: MSN 1-10 Hz → 0.001-0.01 at dt=1ms)
    homeo_tau: float = 5000.0         # Slow rate-averaging τ (ms)
    homeo_interval: int = 200         # Forward steps between scaling events
    homeo_max_change: float = 0.02    # Max fractional change per event
    # Bidirectional DA modulation (Shen et al. 2008)
    ltd_ratio: float = 0.7            # LTD/LTP magnitude ratio (Shen et al. 2008)
    d2_ltd_protection: float = 0.5    # D2 LTD ×0.5 under positive TD (Shen et al. 2008)
    # Synaptic degradation — protein turnover (Bhatt et al. 2009)
    readout_decay: float = 1e-5       # Per-step decay on readout weights

    def __post_init__(self) -> None:
        assert 0 < self.gamma <= 1, f"gamma must be in (0, 1], got {self.gamma}"
        assert self.critic_lr > 0, f"critic_lr must be positive, got {self.critic_lr}"
        assert self.actor_lr > 0, f"actor_lr must be positive, got {self.actor_lr}"
        assert self.tau_m_msn_up > 0, f"tau_m_msn_up must be positive, got {self.tau_m_msn_up}"
        assert self.tau_m_critic > 0, f"tau_m_critic must be positive, got {self.tau_m_critic}"


# =====================================================================
# Weight Initialization Utilities
# =====================================================================

def compute_weight_std(
    fan_in: int,
    fan_out: int,
    psp_target: float = 2.0,
    target_rate: float = 0.05,
) -> float:
    """Derive weight initialization std from desired PSP amplitude.

    σ² = PSP_target² / (fan_in × p_fire)

    Ensures that under expected input (fan_in × target_rate active inputs),
    the total PSP approximates psp_target mV.

    Args:
        fan_in:      Number of input connections.
        fan_out:     Number of output neurons (unused, kept for API).
        psp_target:  Desired total PSP amplitude (mV).
        target_rate: Expected fraction of active inputs.

    Returns:
        Standard deviation for weight initialization.
    """
    expected_active = max(1.0, fan_in * target_rate)
    return psp_target / np.sqrt(expected_active)


def init_weights(
    fan_in: int,
    fan_out: int,
    psp_target: float = 2.0,
    target_rate: float = 0.05,
    excitatory: bool = True,
) -> np.ndarray:
    """Initialize weights with principled scaling.

    Feedforward: w ~ |N(0, σ²)| if excitatory (Dale's law)
    Inhibitory:  w ~ -|N(0, σ²)| (Dale's law: w ≤ 0)

    Args:
        fan_in:      Input dimension.
        fan_out:     Output dimension.
        psp_target:  Desired PSP (mV).
        target_rate: Expected input activity.
        excitatory:  If True, weights ≥ 0 (Dale's law).

    Returns:
        Weight matrix (fan_in, fan_out), dtype float32.
    """
    std = compute_weight_std(fan_in, fan_out, psp_target, target_rate)
    w = np.random.normal(0.0, std, (fan_in, fan_out)).astype(np.float32)
    if excitatory:
        w = np.abs(w)
    else:
        w = -np.abs(w)
    return w
