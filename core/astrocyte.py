"""
Astrocyte field — pure functional Ca²⁺ / D-Serine / ATP dynamics.

Reference:
  De Pittà, Volman, Berry & Ben-Jacob (2011) — Ca²⁺ → D-Serine sigmoid.
  Araque et al. (2014) — tripartite synapse.
  Attwell & Laughlin (2001) — Na⁺/K⁺-ATPase ≈ 10⁹ ATP/spike.
  Aubert & Costalat (2005) — fast astrocytic glycolysis (τ ≈ 2 s).
  Rangaraju et al. (2014) — presynaptic ATP transients, seconds.

State (per zone, shape ``(n_zones,)``):
    calcium   — second-messenger integrator, τ_ca ≈ 5 s
    d_serine  — gliotransmitter readout, sigmoid of Ca²⁺, τ ≈ 200 ms
    atp       — normalised **local** glial/synaptic energy pool,
                [0, 1], τ ≈ 2 s (Aubert & Costalat 2005; Rangaraju
                2014). NOTE: this is the *local* pool consumed by
                active synapses, NOT the whole-brain ATP reservoir
                (τ ≈ 200 s, Attwell 2001). At sustained 50 Hz firing
                it reaches a dynamic equilibrium of ≈ 0.1·atp_max,
                approached with the τ above.

Outputs consumed by the neuron layer (all shape ``(n_zones,)``, then
indexed by ``zone_idx`` when feeding ``AstroMod``):
    precision        = 1 / (1 + Ca)
    synaptic_gain    = baseline + (max-baseline) · d_serine
    threshold_shift  = atp_threshold_shift · (1 − atp)
    leak_gain        = 1 + atp_leak_gain · (1 − atp)
    metabolic_lr     = 1 + metabolic_scale · Ca

All operations are jit/vmap-safe; gap-junction diffusion uses an
explicit Laplacian with Neumann BCs.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array, BackendContext
from .state import AstrocyteState


# ======================================================================
# Parameters
# ======================================================================


class AstrocyteParams(eqx.Module):
    """Astrocyte constants — precomputed decay factors for the hot path."""

    ca_decay: Array
    ca_accumulation: Array
    ca_threshold: Array
    ca_release_k: Array
    d_serine_max: Array
    d_serine_decay: Array

    gap_junction_D: Array

    atp_max: Array
    atp_regen_rate: Array       # per ms
    atp_spike_cost: Array       # per spike per ms
    atp_threshold_shift: Array  # max mV shift at zero ATP
    atp_leak_gain: Array        # max g_L multiplier increase at zero ATP

    gain_baseline: Array
    gain_max: Array
    metabolic_scale: Array


def init_astrocyte_params(
    ctx: BackendContext,
    *,
    tau_ca: float = 5000.0,
    ca_accumulation: float = 0.1,
    ca_threshold: float = 0.5,
    ca_release_k: float = 0.15,
    d_serine_max: float = 0.3,
    tau_d_serine: float = 200.0,
    gap_junction_D: float = 0.01,
    atp_max: float = 1.0,
    # Local glial/synaptic ATP pool, NOT whole-brain reservoir.
    # τ_recover = 1 / atp_regen_rate ≈ 2 s  (Aubert & Costalat 2005;
    #   Rangaraju et al. 2014 show presynaptic ATP transients on a
    #   seconds timescale at 10–50 Hz firing).
    # Equilibrium ATP at firing rate r (spikes / ms) satisfies
    #   ATP_eq = 1 − r · atp_spike_cost / atp_regen_rate.
    # Defaults give ATP_eq ≈ 0.1 at r = 0.05 (50 Hz), reproducing the
    # 50 Hz / 5 s depletion target in the P0.11 plan without the
    # whole-brain timescale mismatch.
    atp_regen_rate: float = 5e-4,
    atp_spike_cost: float = 9e-3,
    # ATP-depletion modulation of AdEx threshold and leak.
    # Grounding (Attwell & Laughlin 2001; Cisternas et al. 2014):
    #   Na\u207a/K\u207a-ATPase consumes ~70% of neuronal ATP; complete
    #   depletion raises [Na\u207a]\u1d62 by ~20 mM, which shifts V_rest by
    #   ~+10 mV via the Nernst equation.  Because V_thresh is set in
    #   the AdEx params, shifting V_rest upward is mathematically
    #   equivalent to shifting V_thresh by the same amount, hence
    #   ``atp_threshold_shift = 10 mV`` at zero ATP.
    #   Leak conductance: Cisternas 2014 reports ~40\u201360% g_L
    #   increase at 80\u2013100% ATP depletion (inward Na leak no longer
    #   counteracted by the pump), hence ``atp_leak_gain = 0.5``.
    atp_threshold_shift: float = 10.0,
    atp_leak_gain: float = 0.5,
    gain_baseline: float = 1.0,
    gain_max: float = 2.0,
    metabolic_scale: float = 0.5,
) -> AstrocyteParams:
    f = lambda x: jnp.asarray(x, DTYPE)
    return AstrocyteParams(
        ca_decay=ctx.decay(tau_ca),
        ca_accumulation=f(ca_accumulation),
        ca_threshold=f(ca_threshold),
        ca_release_k=f(max(ca_release_k, 1e-6)),
        d_serine_max=f(d_serine_max),
        d_serine_decay=ctx.decay(tau_d_serine),
        gap_junction_D=f(gap_junction_D),
        atp_max=f(atp_max),
        atp_regen_rate=f(atp_regen_rate),
        atp_spike_cost=f(atp_spike_cost),
        atp_threshold_shift=f(atp_threshold_shift),
        atp_leak_gain=f(atp_leak_gain),
        gain_baseline=f(gain_baseline),
        gain_max=f(gain_max),
        metabolic_scale=f(metabolic_scale),
    )


# ======================================================================
# Zone aggregation (neuron → zone mapping)
# ======================================================================


def aggregate_to_zones(
    values: Array, zone_idx: Array, n_zones: int,
) -> Array:
    """Mean-reduce ``values`` by zone assignment.

    ``zone_idx`` is an int32 array of shape matching ``values`` giving
    the zone index of each element.  Returns a ``(n_zones,)`` array.

    Implemented as two segment_sums (sum + count) for jit safety.
    """
    import jax
    sums = jax.ops.segment_sum(values.astype(DTYPE), zone_idx, num_segments=n_zones)
    counts = jax.ops.segment_sum(
        jnp.ones_like(values, dtype=DTYPE), zone_idx, num_segments=n_zones,
    )
    return sums / jnp.maximum(counts, 1.0)


# ======================================================================
# Dynamics
# ======================================================================


def _gap_junction_laplacian(ca: Array) -> Array:
    """1D Laplacian with Neumann (zero-flux) BCs via ghost points.

    Interior: Ca[i-1] + Ca[i+1] − 2·Ca[i]
    Boundary: ghost-point reflection — 2·(neighbour − edge).
    For ``n_zones ≤ 2`` returns zeros (degenerate chain).
    """
    n = ca.shape[0]
    if n <= 2:
        return jnp.zeros_like(ca)
    interior = jnp.zeros_like(ca).at[1:-1].set(ca[:-2] + ca[2:] - 2.0 * ca[1:-1])
    interior = interior.at[0].set(2.0 * (ca[1] - ca[0]))
    interior = interior.at[-1].set(2.0 * (ca[-2] - ca[-1]))
    return interior


def astrocyte_step(
    state: AstrocyteState,
    params: AstrocyteParams,
    ctx: BackendContext,
    zone_rates: Array,
) -> AstrocyteState:
    """One astrocyte update step.

    Args:
        state: current ``AstrocyteState``.
        params: precomputed ``AstrocyteParams``.
        ctx: backend context (``ctx.dt`` in ms).
        zone_rates: ``(n_zones,)`` aggregated spike rate per zone
            (produced by ``aggregate_to_zones`` from per-neuron spike
            output).  Values ≥ 0.

    Returns: new ``AstrocyteState``.
    """
    rates = jnp.abs(zone_rates).astype(DTYPE)
    rates_sq = rates * rates
    dt = ctx.dt

    # ── Ca²⁺ accumulation (De Pittà 2011): weighted EMA of rate² ──────
    one_minus_decay = 1.0 - params.ca_decay
    ca = (
        state.calcium * params.ca_decay
        + params.ca_accumulation * rates_sq * one_minus_decay
    )

    # ── Gap-junction diffusion (1D Laplacian, Neumann BCs) ─────────────
    ca = ca + params.gap_junction_D * _gap_junction_laplacian(ca)
    ca = jnp.maximum(ca, 0.0)

    # ── D-Serine release: sigmoid of Ca²⁺ (De Pittà 2011) ─────────────
    sigmoid_arg = (ca - params.ca_threshold) / params.ca_release_k
    release = params.d_serine_max / (1.0 + jnp.exp(-sigmoid_arg))
    d_ser = state.d_serine * params.d_serine_decay + release
    d_ser = jnp.clip(d_ser, 0.0, 1.0)

    # ── ATP dynamics: regen − spike cost ──────────────────────────────
    atp = state.atp + params.atp_regen_rate * (params.atp_max - state.atp) * dt
    atp = atp - params.atp_spike_cost * rates * dt
    atp = jnp.clip(atp, 0.0, params.atp_max)

    return AstrocyteState(
        calcium=ca.astype(DTYPE),
        d_serine=d_ser.astype(DTYPE),
        atp=atp.astype(DTYPE),
    )


# ======================================================================
# Readouts (consumed by neuron_step via AstroMod)
# ======================================================================


def precision(state: AstrocyteState) -> Array:
    """Per-zone precision = 1/(1+Ca).  Low Ca ⇒ high precision."""
    return (1.0 / (1.0 + state.calcium)).astype(DTYPE)


def threshold_shift(state: AstrocyteState, params: AstrocyteParams) -> Array:
    """Per-zone V_T shift (mV) from ATP depletion."""
    return (params.atp_threshold_shift * (1.0 - state.atp)).astype(DTYPE)


def leak_gain(state: AstrocyteState, params: AstrocyteParams) -> Array:
    """Per-zone g_L multiplier from ATP depletion."""
    return (1.0 + params.atp_leak_gain * (1.0 - state.atp)).astype(DTYPE)



