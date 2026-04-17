"""
Synapse — pure JAX functional dual-exponential conductance channels.

The legacy ``SynapticChannels`` (NumPy, mutable ``np.ndarray``) is
replaced by free functions that consume and return ``SynapticState``
pytrees.  No class state, no aliasing — every step is explicit.

Channels (Destexhe et al. 1998, Jahr & Stevens 1990):
    AMPA  — fast excitatory,   τ_r≈0.4 ms,  τ_d≈2 ms
    NMDA  — slow excitatory,   τ_r≈10 ms,   τ_d≈100 ms,  Mg²⁺-gated
    GABA-A — fast inhibitory,  τ_r≈0.25 ms, τ_d≈5 ms
    GABA-B — slow inhibitory,  τ_r≈30 ms,   τ_d≈100 ms

Dual-exponential kinetics (Roth & van Rossum 2009, eq 7):
    g(t) = N · (exp(-t/τ_d) - exp(-t/τ_r))
    t_peak = τ_r τ_d / (τ_d - τ_r) · ln(τ_d / τ_r)
    N = 1 / (exp(-t_peak/τ_d) - exp(-t_peak/τ_r))

Implementation: maintain a decay-side trace ``g_x`` and a rise-side
trace ``g_x_rise``; a spike increments both by ``N · w``; decay both
exponentially; the effective conductance at step ``t`` is
``max(g_x - g_x_rise, 0)``.

NMDA voltage-dependent Mg²⁺ block (Jahr & Stevens 1990):
    B(V) = 1 / (1 + [Mg²⁺]/3.57 · exp(-0.062 V))

All functions are jit-safe and side-effect-free.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array, BackendContext
from .state import SynapticParams, SynapticState


# ======================================================================
# Parameter factory — computes derived constants once, outside hot path
# ======================================================================


def init_synaptic_params(
    ctx: BackendContext,
    *,
    tau_ampa_rise: float = 0.4,
    tau_ampa_decay: float = 2.0,
    tau_nmda_rise: float = 10.0,
    tau_nmda_decay: float = 100.0,
    tau_gaba_a_rise: float = 0.25,
    tau_gaba_a_decay: float = 5.0,
    tau_gaba_b_rise: float = 30.0,
    tau_gaba_b_decay: float = 100.0,
    e_exc: float = 0.0,
    e_inh: float = -75.0,
    mg_concentration: float = 1.0,
    ampa_nmda_ratio: float = 3.0,
) -> SynapticParams:
    """Precompute per-step multipliers and peak-normalisation factors."""

    def _norm(tau_d: float, tau_r: float) -> float:
        # Peak-amplitude normalisation for the difference of exponentials.
        # ``tau_d`` must strictly exceed ``tau_r`` (decay slower than rise).
        assert tau_d > tau_r, (
            f"Decay τ={tau_d} must exceed rise τ={tau_r} for dual-exp."
        )
        t_peak = (tau_r * tau_d / (tau_d - tau_r)) * float(jnp.log(tau_d / tau_r))
        raw = float(jnp.exp(-t_peak / tau_d) - jnp.exp(-t_peak / tau_r))
        return 1.0 / raw

    return SynapticParams(
        ampa_decay=ctx.decay(tau_ampa_decay),
        ampa_rise_decay=ctx.decay(tau_ampa_rise),
        nmda_decay=ctx.decay(tau_nmda_decay),
        nmda_rise_decay=ctx.decay(tau_nmda_rise),
        gaba_a_decay=ctx.decay(tau_gaba_a_decay),
        gaba_a_rise_decay=ctx.decay(tau_gaba_a_rise),
        gaba_b_decay=ctx.decay(tau_gaba_b_decay),
        gaba_b_rise_decay=ctx.decay(tau_gaba_b_rise),
        ampa_norm=jnp.asarray(_norm(tau_ampa_decay, tau_ampa_rise), DTYPE),
        nmda_norm=jnp.asarray(_norm(tau_nmda_decay, tau_nmda_rise), DTYPE),
        gaba_a_norm=jnp.asarray(_norm(tau_gaba_a_decay, tau_gaba_a_rise), DTYPE),
        gaba_b_norm=jnp.asarray(_norm(tau_gaba_b_decay, tau_gaba_b_rise), DTYPE),
        e_exc=jnp.asarray(e_exc, DTYPE),
        e_inh=jnp.asarray(e_inh, DTYPE),
        mg_concentration=jnp.asarray(mg_concentration, DTYPE),
        ampa_nmda_ratio=jnp.asarray(ampa_nmda_ratio, DTYPE),
    )


# ======================================================================
# Step primitives
# ======================================================================


def nmda_mg_block(v: Array, mg_concentration: Array) -> Array:
    """Voltage-dependent Mg²⁺ block factor (Jahr & Stevens 1990).

    B(V) = 1 / (1 + [Mg]/3.57 · exp(-0.062·V))
    """
    return 1.0 / (1.0 + (mg_concentration / 3.57) * jnp.exp(-0.062 * v))


def decay_channels(state: SynapticState, params: SynapticParams) -> SynapticState:
    """Multiply every conductance trace by its per-step decay factor.

    Applied once per step *before* spike-driven increments so that the
    instantaneous rise is visible on the same step.
    """
    return SynapticState(
        g_ampa=state.g_ampa * params.ampa_decay,
        g_ampa_rise=state.g_ampa_rise * params.ampa_rise_decay,
        g_nmda=state.g_nmda * params.nmda_decay,
        g_nmda_rise=state.g_nmda_rise * params.nmda_rise_decay,
        g_gaba_a=state.g_gaba_a * params.gaba_a_decay,
        g_gaba_a_rise=state.g_gaba_a_rise * params.gaba_a_rise_decay,
        g_gaba_b=state.g_gaba_b * params.gaba_b_decay,
        g_gaba_b_rise=state.g_gaba_b_rise * params.gaba_b_rise_decay,
    )


def receive_excitatory(
    state: SynapticState,
    params: SynapticParams,
    drive: Array,
) -> SynapticState:
    """Add excitatory drive (AMPA + NMDA).

    ``drive`` is the per-postsynaptic summed weighted spike contribution
    (shape ``(n_post,)``).  Split between AMPA and NMDA by
    ``ampa_nmda_ratio``; both decay and rise traces receive the same
    norm-scaled increment so their difference peaks at ``drive``.
    """
    ratio = params.ampa_nmda_ratio
    total = ratio + 1.0
    ampa_inc = drive * (ratio / total) * params.ampa_norm
    nmda_inc = drive * (1.0 / total) * params.nmda_norm
    return eqx.tree_at(
        lambda s: (s.g_ampa, s.g_ampa_rise, s.g_nmda, s.g_nmda_rise),
        state,
        (
            state.g_ampa + ampa_inc,
            state.g_ampa_rise + ampa_inc,
            state.g_nmda + nmda_inc,
            state.g_nmda_rise + nmda_inc,
        ),
    )


def receive_inhibitory(
    state: SynapticState,
    params: SynapticParams,
    drive: Array,
    *,
    gaba_b_ratio: float = 0.25,
) -> SynapticState:
    """Add inhibitory drive (GABA-A + GABA-B, Isaacson 2011 ratio)."""
    ga = drive * (1.0 - gaba_b_ratio) * params.gaba_a_norm
    gb = drive * gaba_b_ratio * params.gaba_b_norm
    return eqx.tree_at(
        lambda s: (s.g_gaba_a, s.g_gaba_a_rise, s.g_gaba_b, s.g_gaba_b_rise),
        state,
        (
            state.g_gaba_a + ga,
            state.g_gaba_a_rise + ga,
            state.g_gaba_b + gb,
            state.g_gaba_b_rise + gb,
        ),
    )


def effective_conductances(state: SynapticState) -> tuple[Array, Array, Array, Array]:
    """Return ``(g_ampa, g_nmda, g_gaba_a, g_gaba_b)`` effective values.

    Each is ``max(decay_trace - rise_trace, 0)`` — the characteristic
    rise-then-decay waveform normalised so peak is driven by spike
    weight.
    """
    g_ampa = jnp.maximum(state.g_ampa - state.g_ampa_rise, 0.0)
    g_nmda = jnp.maximum(state.g_nmda - state.g_nmda_rise, 0.0)
    g_gaba_a = jnp.maximum(state.g_gaba_a - state.g_gaba_a_rise, 0.0)
    g_gaba_b = jnp.maximum(state.g_gaba_b - state.g_gaba_b_rise, 0.0)
    return g_ampa, g_nmda, g_gaba_a, g_gaba_b


def compute_current(
    state: SynapticState,
    params: SynapticParams,
    v_post: Array,
) -> tuple[Array, Array]:
    """Total synaptic current for the AdEx Euler step.

    Returns ``(I_syn, g_total)`` — the total Ohm's-law current and its
    total conductance (used by the exponential-Euler Jacobian
    correction: dI/dV = -g_total).

    Conductance-based current (Jahr & Stevens 1990, Destexhe 1998):
        I_exc = (g_ampa + g_nmda · B(V)) · (E_exc − V)
        I_inh = (g_gaba_a + g_gaba_b)  · (E_inh − V)
    """
    g_ampa, g_nmda, g_gaba_a, g_gaba_b = effective_conductances(state)
    mg = nmda_mg_block(v_post, params.mg_concentration)

    g_exc = g_ampa + g_nmda * mg
    g_inh = g_gaba_a + g_gaba_b

    i_exc = g_exc * (params.e_exc - v_post)
    i_inh = g_inh * (params.e_inh - v_post)

    return (i_exc + i_inh).astype(DTYPE), (g_exc + g_inh).astype(DTYPE)


def synapse_step(
    state: SynapticState,
    params: SynapticParams,
    exc_drive: Array,
    inh_drive: Array,
    v_post: Array,
    *,
    gaba_b_ratio: float = 0.25,
) -> tuple[SynapticState, Array, Array]:
    """Full synapse step: decay → receive → integrate current.

    ``exc_drive``/``inh_drive`` are pre-summed weighted spike drives
    per postsynaptic neuron.  Returns the new state plus the current
    and total conductance for the AdEx consumer.
    """
    decayed = decay_channels(state, params)
    with_exc = receive_excitatory(decayed, params, exc_drive)
    with_both = receive_inhibitory(with_exc, params, inh_drive, gaba_b_ratio=gaba_b_ratio)
    i_syn, g_total = compute_current(with_both, params, v_post)
    return with_both, i_syn, g_total
