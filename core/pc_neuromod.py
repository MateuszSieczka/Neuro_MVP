"""Neuromodulators as precision controllers (Faza U, §4 / plan §U.5).

The four ascending neuromodulators are not separate mechanisms in this
substrate — they are **controllers of precision** (Yu & Dayan 2005;
FitzGerald, Dolan & Friston 2015; Parr & Friston 2017).  This module
turns the graph's own signals (node error ε, global free energy,
value-node belief μ) into the gains the existing substrate hooks already
consume; it adds no new dynamics.

    channel  driver (graph-native)            hook it drives
    ───────  ──────────────────────────────  ─────────────────────────────
    ACh      sensory novelty  (mean|ε_sens|)  sensory-node Π  (attention)
    NE       surprise volatility (Δ free E)   EFE β  (exploration weight)
    DA       value-node TD error  (ε_value)   value-node Π   (reward gain)
    5-HT     world stability (1 − mean|ε|)    planning horizon

* ACh and DA emerge as a ``precision_gains`` dict fed straight to
  :func:`core.pc_brain.pc_brain_cognitive_step` (→ ``scale_node_precision``).
* NE sets the epistemic weight ``β`` of :func:`core.pc_active.efe_select`
  (high NE / volatility ⇒ more exploration; Aston-Jones & Cohen 2005).
* 5-HT sets the planning horizon (Doya 2002): a stable world earns
  patience / deeper rollout.

Curiosity (the epistemic term of expected free energy) is the
**learning-progress** signal salvaged from the legacy world model
(Oudeyer 2007 IAC; Schmidhuber 1991): two EMAs of the ``world_model``
node's error at different timescales, ``LP = pe_long − pe_short``,
positive while the model is still mastering its current regime.  Feed it
as the epistemic argument of :func:`core.pc_active.epistemic_value` /
``efe_select`` — it replaces the noise-blind inverse-precision proxy
(the "noisy-TV" failure mode of pure-PE curiosity).

All timescales are in cognitive-step units (no wall clock), the same
convention as the rest of the substrate.  Discarded with the spiking
world: external scalar reward and the receptor layer.  The DA channel's
TD error is *internal* — the value node's own prediction error against
its §6 temporal edge — not an externally supplied signed TD signal.

References
----------
  Yu & Dayan (2005)            — ACh/NE expected vs unexpected uncertainty.
  Aston-Jones & Cohen (2005)   — LC-NE adaptive gain / exploration.
  FitzGerald, Dolan, Friston (2015) — dopaminergic precision.
  Parr & Friston (2017)        — neuromodulation as precision in active inference.
  Doya (2002)                  — 5-HT and temporal discounting / horizon.
  Oudeyer & Kaplan (2007)      — intrinsic motivation / learning progress.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array
from .pc_graph import (
    PCGraphParams, PCGraphState,
    pc_graph_errors, graph_free_energy, REGION_INDEX,
)
from .pc_precision import step_alpha


# =====================================================================
# Params / state
# =====================================================================


class NeuromodParams(eqx.Module):
    """Step-unit EMA rates + baselines + gain strengths + node indices."""

    # EMA rates (per cognitive step).
    ach_alpha: Array
    ne_alpha: Array
    da_alpha: Array
    sero_alpha: Array
    fe_alpha: Array            # free-energy reference (NE volatility baseline)
    stab_alpha: Array          # global error EMA (5-HT stability)
    wm_short_alpha: Array      # curiosity short-τ |ε_wm|
    wm_long_alpha: Array       # curiosity long-τ  |ε_wm|

    # Channel baselines (tonic levels).
    baseline_ach: Array
    baseline_ne: Array
    baseline_da: Array
    baseline_sero: Array

    # Precision-gain mapping (level → multiplicative Π gain).
    ach_gain_strength: Array
    da_gain_strength: Array
    gain_floor: Array          # gains never fall below this (stability)

    # EFE β mapping (NE → epistemic weight).
    beta_base: Array
    beta_ne_strength: Array
    beta_floor: Array

    # Planning-horizon mapping (5-HT → [horizon_min, horizon_max]).
    horizon_min: Array
    horizon_max: Array

    sensory_idx: int = eqx.field(static=True)
    value_idx: int = eqx.field(static=True)
    wm_idx: int = eqx.field(static=True)


def init_neuromod_params(
    *,
    sensory_idx: int | None = None,
    value_idx: int | None = None,
    wm_idx: int | None = None,
    tau_ach: float = 10.0,
    tau_ne: float = 20.0,
    tau_da: float = 20.0,
    tau_sero: float = 50.0,
    tau_fe: float = 20.0,
    tau_stability: float = 50.0,
    tau_wm_short: float = 5.0,
    tau_wm_long: float = 50.0,
    baseline_ach: float = 0.5,
    baseline_ne: float = 0.3,
    baseline_da: float = 0.5,
    baseline_sero: float = 0.6,
    ach_gain_strength: float = 1.0,
    da_gain_strength: float = 1.0,
    gain_floor: float = 0.1,
    beta_base: float = 1.0,
    beta_ne_strength: float = 1.0,
    beta_floor: float = 0.0,
    horizon_min: float = 1.0,
    horizon_max: float = 4.0,
    dtype=DTYPE,
) -> NeuromodParams:
    """Neuromodulatory controller params.

    ``tau_*`` are EMA timescales in cognitive steps (short for phasic
    novelty, long for tonic stability).  ``*_gain_strength`` scale how
    far a channel pushes its node's precision from 1; ``gain_floor``
    keeps relaxation stable when a channel is fully suppressed.  Node
    indices default to the canonical region graph.
    """
    if horizon_min <= 0.0 or horizon_max < horizon_min:
        raise ValueError("require 0 < horizon_min ≤ horizon_max")
    a = step_alpha
    f = lambda x: jnp.asarray(x, dtype)
    return NeuromodParams(
        ach_alpha=a(tau_ach).astype(dtype), ne_alpha=a(tau_ne).astype(dtype),
        da_alpha=a(tau_da).astype(dtype), sero_alpha=a(tau_sero).astype(dtype),
        fe_alpha=a(tau_fe).astype(dtype), stab_alpha=a(tau_stability).astype(dtype),
        wm_short_alpha=a(tau_wm_short).astype(dtype),
        wm_long_alpha=a(tau_wm_long).astype(dtype),
        baseline_ach=f(baseline_ach), baseline_ne=f(baseline_ne),
        baseline_da=f(baseline_da), baseline_sero=f(baseline_sero),
        ach_gain_strength=f(ach_gain_strength),
        da_gain_strength=f(da_gain_strength), gain_floor=f(gain_floor),
        beta_base=f(beta_base), beta_ne_strength=f(beta_ne_strength),
        beta_floor=f(beta_floor),
        horizon_min=f(horizon_min), horizon_max=f(horizon_max),
        sensory_idx=REGION_INDEX["sensory"] if sensory_idx is None else int(sensory_idx),
        value_idx=REGION_INDEX["value"] if value_idx is None else int(value_idx),
        wm_idx=REGION_INDEX["world_model"] if wm_idx is None else int(wm_idx),
    )


class NeuromodState(eqx.Module):
    """The four channel levels + the EMAs / memory that drive them."""

    ach: Array                 # scalar — acetylcholine level
    ne: Array                  # scalar — noradrenaline level
    da: Array                  # scalar — dopamine level
    sero: Array                # scalar — serotonin level
    value_prev: Array          # scalar — last cycle's value belief (diagnostic)
    fe_ema: Array              # scalar — running free-energy reference
    stab_ema: Array            # scalar — running global mean|ε|
    wm_pe_short: Array         # scalar — short-τ EMA of mean|ε_wm|
    wm_pe_long: Array          # scalar — long-τ  EMA of mean|ε_wm|


def init_neuromod_state(params: NeuromodParams, *, dtype=DTYPE) -> NeuromodState:
    """Start at tonic baselines with zeroed histories (warm, neutral)."""
    z = jnp.asarray(0.0, dtype)
    return NeuromodState(
        ach=params.baseline_ach.astype(dtype),
        ne=params.baseline_ne.astype(dtype),
        da=params.baseline_da.astype(dtype),
        sero=params.baseline_sero.astype(dtype),
        value_prev=z, fe_ema=z, stab_ema=z,
        wm_pe_short=z, wm_pe_long=z,
    )


# =====================================================================
# Step — graph signals → channel levels
# =====================================================================


def _ema(prev: Array, x: Array, alpha: Array) -> Array:
    return (1.0 - alpha) * prev + alpha * x


def neuromod_step(
    state: NeuromodState, params: NeuromodParams,
    graph: PCGraphState, gparams: PCGraphParams,
) -> NeuromodState:
    """Advance the four channels one step from the relaxed graph's signals.

    Drivers are all the substrate's own quantities: sensory novelty
    (``mean|ε_sensory|``) raises ACh; free-energy volatility (deviation of
    this cycle's free energy from its running reference) raises NE; the
    value node's temporal-difference error (its own prediction error
    ``ε_value``, the §6 temporal edge's residual) raises DA; and a stable
    world (low global ``mean|ε|``) raises 5-HT.  Call on the *relaxed*
    graph (after ``pc_graph_relax``), before or after learning — it only
    reads.
    """
    eps = pc_graph_errors(graph, gparams)
    novelty = jnp.mean(jnp.abs(eps[params.sensory_idx]))
    global_err = jnp.mean(
        jnp.stack([jnp.mean(jnp.abs(e)) for e in eps])
    )
    fe = graph_free_energy(graph, gparams)
    value = jnp.mean(graph.mu[params.value_idx])
    wm_pe = jnp.mean(jnp.abs(eps[params.wm_idx]))

    # ACh — phasic novelty on top of tonic baseline.
    ach = _ema(state.ach, params.baseline_ach + novelty, params.ach_alpha)

    # NE — unexpected uncertainty: free energy departing from its own
    # running reference (volatility), not its absolute level.
    fe_ema = _ema(state.fe_ema, fe, params.fe_alpha)
    volatility = jnp.abs(fe - state.fe_ema)
    ne = _ema(state.ne, params.baseline_ne + volatility, params.ne_alpha)

    # DA — temporal-difference error = the value node's own prediction
    # error ε_value.  With the §6 value(t−1)→value(t) temporal edge present
    # this ε *is* the TD error r + γV(s′) − V(s) — the bootstrap γV(s′) is
    # the temporal edge's prediction, so no separate critic/TD update is
    # needed (closes the §4 DA-proxy deferral).  Without a temporal value
    # edge it degrades gracefully to the value node's static surprise.
    # Only the positive part drives phasic DA (Schultz 1997).
    rpe = jnp.maximum(jnp.mean(eps[params.value_idx]), 0.0)
    da = _ema(state.da, params.baseline_da + rpe, params.da_alpha)

    # 5-HT — world stability (low global error ⇒ patience).
    stab_ema = _ema(state.stab_ema, global_err, params.stab_alpha)
    stability = jnp.clip(1.0 - stab_ema, 0.0, 1.0)
    sero = _ema(state.sero, stability, params.sero_alpha)

    # Curiosity learning-progress EMAs (Oudeyer 2007).
    wm_pe_short = _ema(state.wm_pe_short, wm_pe, params.wm_short_alpha)
    wm_pe_long = _ema(state.wm_pe_long, wm_pe, params.wm_long_alpha)

    return NeuromodState(
        ach=ach.astype(DTYPE), ne=ne.astype(DTYPE),
        da=da.astype(DTYPE), sero=sero.astype(DTYPE),
        value_prev=value.astype(DTYPE), fe_ema=fe_ema.astype(DTYPE),
        stab_ema=stab_ema.astype(DTYPE),
        wm_pe_short=wm_pe_short.astype(DTYPE), wm_pe_long=wm_pe_long.astype(DTYPE),
    )


# =====================================================================
# Read-outs — into the substrate's existing hooks
# =====================================================================


def neuromod_precision_gains(
    state: NeuromodState, params: NeuromodParams,
) -> dict[int, Array]:
    """``{sensory_idx: ACh gain, value_idx: DA gain}`` for ``precision_gains``.

    A gain is ``1 + strength·(level − baseline)``, floored at
    ``gain_floor`` so a suppressed channel never zeroes a node's
    precision.  Pass straight to
    :func:`core.pc_brain.pc_brain_cognitive_step` (``precision_gains=``):
    ACh sharpens the sensory node (perceptual attention), DA the value
    node (reward salience) — Parr & Friston 2017.
    """
    ach_gain = jnp.maximum(
        params.gain_floor,
        1.0 + params.ach_gain_strength * (state.ach - params.baseline_ach),
    )
    da_gain = jnp.maximum(
        params.gain_floor,
        1.0 + params.da_gain_strength * (state.da - params.baseline_da),
    )
    return {params.sensory_idx: ach_gain, params.value_idx: da_gain}


def neuromod_beta(state: NeuromodState, params: NeuromodParams) -> Array:
    """NE → EFE epistemic weight ``β`` (exploration gain).

    ``β = β_base + strength·(NE − baseline_ne)``, floored at
    ``beta_floor``.  High NE (volatile, surprising world) widens
    exploration in :func:`core.pc_active.efe_select`; a calm world
    collapses toward greedy-pragmatic action.
    """
    return jnp.maximum(
        params.beta_floor,
        params.beta_base + params.beta_ne_strength * (state.ne - params.baseline_ne),
    )


def neuromod_curiosity(state: NeuromodState) -> Array:
    """Signed learning progress ``pe_long − pe_short`` (Oudeyer 2007).

    Positive while the world model's error is *falling* (a regime still
    being mastered — worth revisiting); ≈ 0 with high error means
    irreducible noise (not worth chasing).  The epistemic value to feed
    :func:`core.pc_active.epistemic_value` / ``efe_select``.
    """
    return state.wm_pe_long - state.wm_pe_short


def neuromod_horizon(state: NeuromodState, params: NeuromodParams) -> Array:
    """5-HT → planning horizon in ``[horizon_min, horizon_max]`` (Doya 2002).

    A stable, well-modelled world (high 5-HT) earns a longer horizon —
    map to rollout depth or relaxation steps at the call site.
    """
    return params.horizon_min + state.sero * (params.horizon_max - params.horizon_min)


def neuromod_levels(state: NeuromodState) -> Array:
    """Pack ``(ACh, NE, DA, 5-HT)`` into a ``(4,)`` vector for diagnostics."""
    return jnp.stack([state.ach, state.ne, state.da, state.sero])
