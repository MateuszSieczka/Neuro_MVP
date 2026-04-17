"""
Frozen state containers — the pytree leaves carried through ``lax.scan``.

Every module in the JAX port separates *parameters* (constants for a run)
from *state* (dynamic, carried step-to-step).  Both are ``eqx.Module``
subclasses so they automatically participate in JAX's pytree machinery:

* ``NeuronParams``  — AdEx parameters + precomputed decay factors.
* ``NeuronState``   — per-neuron dynamic variables (V, w, traces, …).
* ``SynapticParams`` / ``SynapticState`` — dual-exponential AMPA / NMDA /
  GABA-A / GABA-B channels.
* ``OscillatorState`` — theta + gamma phase accumulators + PAC envelope.
* ``HomeostaticState`` — adaptive threshold and mean firing rate.

The Params objects are *constants* relative to the inner scan: JAX will
cache-hash them and avoid re-tracing.  State objects are the scan carry.

Factory helpers at the bottom (``init_neuron_state`` etc.) build
zero-initialised states from shapes — they are not traced (called once at
graph construction time).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array


# ======================================================================
# AdEx neuron
# ======================================================================


class NeuronParams(eqx.Module):
    """AdEx kernel parameters.

    Membrane:  C_m dV/dt = -g_L(V-E_L) + g_L·Δ_T·exp((V-V_T)/Δ_T) + I - w
    Adapt:     τ_w dw/dt = a(V-E_L) - w
    Reset:     V≥V_cutoff ⇒ V←V_reset,  w←w+b

    Setting ``a=0``, ``b=0``, ``delta_t→0`` degenerates the kernel to
    pure LIF — used when exporting to Akida/TrueNorth-class hardware
    that lacks adaptation currents.  No code branch required.

    All fields are ``float32`` scalars (traced leaves), not ``static`` —
    this lets a single jit-compiled step function dispatch over
    different neuron types (RS/FS/IB/LS) by swapping params.
    """

    # Resting / threshold / reset potentials (mV)
    v_rest: Array
    v_thresh: Array
    v_reset: Array
    v_spike_cutoff: Array
    delta_t: Array

    # Passive membrane
    C_m: Array
    g_L: Array

    # Adaptation
    tau_w: Array
    a: Array
    b: Array

    # Refractory (steps)
    refrac_period: Array  # int32 scalar

    # Precomputed, derived from ctx.dt + tau_w
    w_decay: Array
    w_gain: Array


class NeuronState(eqx.Module):
    """Per-neuron dynamic state for one AdEx population.

    Shape convention: every field is ``(n,)`` where ``n`` is the
    population size, except the eligibility trace which is ``(n_pre,
    n)`` for STDP weight updates.
    """

    v: Array              # (n,)   membrane potential (mV)
    w_adapt: Array        # (n,)   adaptation current (pA)
    refrac: Array         # (n,)   refractory counter (int32, steps remaining)
    x_pre: Array          # (n,)   pre-synaptic STDP trace (from inputs)
    x_post: Array         # (n,)   post-synaptic STDP trace (own spikes)
    spikes: Array         # (n,)   last spike output (float32 0/1)


def init_neuron_state(
    n: int,
    v_rest: float = -70.0,
    *,
    dtype=DTYPE,
) -> NeuronState:
    """Construct a fresh zero-initialised ``NeuronState``.

    ``v`` is seeded at ``v_rest``; everything else is zero.  Intended
    for use outside jit (graph construction / reset).
    """
    zeros = jnp.zeros(n, dtype=dtype)
    return NeuronState(
        v=jnp.full(n, v_rest, dtype=dtype),
        w_adapt=zeros,
        refrac=jnp.zeros(n, dtype=jnp.int32),
        x_pre=zeros,
        x_post=zeros,
        spikes=zeros,
    )


# ======================================================================
# Conductance-based synaptic channels (dual-exponential kinetics)
# ======================================================================


class SynapticParams(eqx.Module):
    """Precomputed dual-exponential channel parameters.

    Dual-exponential kinetics (Destexhe et al. 1998):
        g(t) = N · (exp(-t/τ_d) - exp(-t/τ_r))
    with ``N`` chosen so ``max g(t) = 1``.  We cache the per-channel
    decay multipliers (``exp(-dt/τ)``) and the peak-normalisation
    constant ``N`` so the hot path is pure arithmetic — no transcendentals.

    Four channels: AMPA, NMDA, GABA-A, GABA-B.  Reversals are scalars
    shared across the postsynaptic layer.  ``mg_concentration`` feeds
    the Jahr–Stevens Mg²⁺ block for NMDA.
    """

    # Decay / rise multipliers (per-step factors)
    ampa_decay: Array
    ampa_rise_decay: Array
    nmda_decay: Array
    nmda_rise_decay: Array
    gaba_a_decay: Array
    gaba_a_rise_decay: Array
    gaba_b_decay: Array
    gaba_b_rise_decay: Array

    # Peak-normalisation factors so max(g) = 1 per unit input
    ampa_norm: Array
    nmda_norm: Array
    gaba_a_norm: Array
    gaba_b_norm: Array

    # Reversal potentials (mV)
    e_exc: Array     # AMPA + NMDA share E_exc ≈ 0 mV
    e_inh: Array     # GABA-A + GABA-B share E_inh ≈ -75 mV

    # NMDA extracellular [Mg²⁺] (mM)
    mg_concentration: Array

    # AMPA / NMDA drive split (Myme et al. 2003: ~3:1 in cortex)
    ampa_nmda_ratio: Array


class SynapticState(eqx.Module):
    """Rise + decay conductance state for the four channels.

    Each channel maintains ``g_*`` (decay-side) and ``g_*_rise``
    (rise-side); the effective conductance is their difference
    renormalised to peak at 1 at the kinetic peak time (computed by
    the consumer).
    """

    g_ampa: Array
    g_ampa_rise: Array
    g_nmda: Array
    g_nmda_rise: Array
    g_gaba_a: Array
    g_gaba_a_rise: Array
    g_gaba_b: Array
    g_gaba_b_rise: Array


def init_synaptic_state(n: int, *, dtype=DTYPE) -> SynapticState:
    z = jnp.zeros(n, dtype=dtype)
    return SynapticState(
        g_ampa=z, g_ampa_rise=z,
        g_nmda=z, g_nmda_rise=z,
        g_gaba_a=z, g_gaba_a_rise=z,
        g_gaba_b=z, g_gaba_b_rise=z,
    )


# ======================================================================
# Oscillator (theta + gamma with PAC)
# ======================================================================


class OscillatorState(eqx.Module):
    """Theta + gamma phase + current PAC envelope amplitude.

    Phase fields wrap in ``[0, 2π)``.  ``gamma_amplitude`` is the
    theta-modulated gamma envelope (Lisman & Jensen 2013).
    """

    theta_phase: Array       # scalar, radians
    gamma_phase: Array       # scalar, radians
    gamma_amplitude: Array   # scalar, 0..1


def init_oscillator_state() -> OscillatorState:
    zero = jnp.zeros((), dtype=DTYPE)
    one = jnp.ones((), dtype=DTYPE)
    return OscillatorState(theta_phase=zero, gamma_phase=zero, gamma_amplitude=one)


# ======================================================================
# Homeostatic plasticity (shared by CorticalArea layers)
# ======================================================================


class HomeostaticState(eqx.Module):
    """Turrigiano (2008) synaptic scaling — mean rate + threshold shift.

    Shared across any layer subtype that maintains an adaptive
    threshold (formerly duplicated in CompetitiveLIFLayer /
    PyramidalLayer / ErrorNeuronLayer).
    """

    avg_rate: Array            # (n,)   EMA of spike rate
    v_thresh_adaptive: Array   # (n,)   per-neuron threshold shift (mV)


def init_homeostatic_state(n: int, v_thresh: float, *, dtype=DTYPE) -> HomeostaticState:
    return HomeostaticState(
        avg_rate=jnp.zeros(n, dtype=dtype),
        v_thresh_adaptive=jnp.full(n, v_thresh, dtype=dtype),
    )


# ======================================================================
# Eligibility traces for three-factor STDP
# ======================================================================


class EligibilityState(eqx.Module):
    """Pairwise eligibility trace ``e[i, j]`` between pre-i and post-j.

    Kept as a separate container because it's the only
    ``(n_pre, n_post)``-shaped state — everything else is ``(n,)``.
    """

    e: Array  # (n_pre, n_post)


def init_eligibility_state(n_pre: int, n_post: int, *, dtype=DTYPE) -> EligibilityState:
    return EligibilityState(e=jnp.zeros((n_pre, n_post), dtype=dtype))


# ======================================================================
# Astrocyte field (Ca²⁺, D-Serine, ATP energy budget)
# ======================================================================


class AstrocyteState(eqx.Module):
    """Per-zone astrocyte state (De Pittà et al. 2011, Krok 1.3 ATP).

    Ca²⁺ integrates local error energy; D-Serine is sigmoid readout;
    ATP is a normalised energy pool (1.0 = full, 0.0 = depleted).
    All leaves are shape `(n_zones,)`.
    """

    calcium: Array
    d_serine: Array
    atp: Array


def init_astrocyte_state(n_zones: int, atp_max: float = 1.0, *, dtype=DTYPE) -> AstrocyteState:
    z = jnp.zeros(n_zones, dtype=dtype)
    return AstrocyteState(
        calcium=z,
        d_serine=z,
        atp=jnp.full(n_zones, atp_max, dtype=dtype),
    )

