"""Brain graph — inter-region wiring in pure JAX.

References
----------
  Swanson (2012)               — Brain architecture
  Muller et al. (2018)         — Cortical travelling waves
  Felleman & Van Essen (1991)  — Distributed hierarchical processing
  Buzsaki & Draguhn (2004)     — Neuronal oscillations in cortical networks
  Markov et al. (2014)         — Cortical interareal connectivity
  Swanson & Lichtman (2016)    — Cajal's structural plan of the nervous system

Role (critical analysis)
------------------------
Individual modules (cortex, thalamus, cerebellum, basal_ganglia, …)
expose heterogeneous APIs. A *universal* container that stores them in
a Python ``dict`` cannot be fully ``jax.jit``-compiled. ``brain_graph``
instead provides:

  1. **Wiring primitives** — ``DelayBuffer`` (O(1) conduction delay
     per edge, JIT-safe), ``edge_apply`` (projection + delay), phase
     offsets for traveling-wave oscillator bus.
  2. **A concrete reference build** — ``MinimalBrain`` = one thalamus +
     one cortical area + one cerebellum, sharing a global oscillator
     and neuromodulator bus. This IS the Phase 2 end-to-end agent;
     Phase 3 will extend it (BG, VTA, episodic, multi-area) by cloning
     the same pattern.

The brain graph owns the edges, not the regions. Each edge = weight
matrix + delay ring buffer. Regions remain body-agnostic; embodiment
enters only at the ``sensory`` input (what afferent arrays carry) and
at the optional ``climbing_error`` for cerebellum (actual−predicted
sensory outcome, supplied by the body interface).

Traveling-wave oscillator bus
-----------------------------
A single theta/gamma oscillator is advanced each step; each region is
assigned a **phase offset** (radians) representing its position in the
cortical hierarchy (Muller 2018). Downstream gating can read
``theta_phase + region_offset`` to implement travelling-wave
coordination. Phase 2 advances the oscillator and exposes phases; Phase
3 routes them into per-region gain/attention gating.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, BackendContext, split_key
from .state import OscillatorState, init_oscillator_state

from .oscillator import (
    OscillatorParams, init_oscillator_params, oscillator_step,
)
from .neuromodulator import (
    NeuromodulatorParams, NeuromodulatorState,
    init_neuromodulator_params, init_neuromodulator_state,
    neuromodulator_step,
)
from .thalamus import (
    RelayParams, RelayState, TRNParams, TRNState,
    init_relay_params, init_relay_state,
    init_trn_params, init_trn_state,
    thalamic_step,
)
from .cortex import (
    CorticalAreaParams, CorticalAreaState, CorticalInputs,
    init_cortical_area_params, init_cortical_area_state,
    cortical_area_step,
)
from .cerebellum import (
    CerebellumParams, CerebellumState,
    init_cerebellum_params, init_cerebellum_state,
    cerebellum_step, cerebellum_update,
)


# =====================================================================
# Primitive: fixed-delay ring buffer
# =====================================================================


class DelayBuffer(eqx.Module):
    """O(1) conduction-delay ring buffer for a ``(n,)`` signal.

    Writes the newest value at ``head``; reads the oldest (which is the
    value written ``delay_steps`` steps ago after the first wrap).

    JIT-safe: ``delay_steps`` and ``n`` are static fields; the buffer
    array is a traced pytree leaf of shape ``(delay_steps, n)``.

    The first ``delay_steps`` pops return the initial zero fill —
    equivalent to "silence before the conduction path has propagated".
    """

    buf: Array                                # (delay_steps, n)
    head: Array                               # int32 scalar
    delay_steps: int = eqx.field(static=True)
    n: int = eqx.field(static=True)


def init_delay_buffer(n: int, delay_steps: int, *, dtype=DTYPE) -> DelayBuffer:
    """Zero-initialised delay buffer. ``delay_steps`` must be \u2265 1."""
    d = max(1, int(delay_steps))
    return DelayBuffer(
        buf=jnp.zeros((d, n), dtype=dtype),
        head=jnp.asarray(0, jnp.int32),
        delay_steps=d, n=n,
    )


def delay_push_pop(db: DelayBuffer, x: Array) -> tuple[DelayBuffer, Array]:
    """Read oldest, write newest, advance head. Returns (new_db, delayed)."""
    delayed = db.buf[db.head]
    new_buf = db.buf.at[db.head].set(x.astype(db.buf.dtype))
    new_head = ((db.head + 1) % db.delay_steps).astype(jnp.int32)
    return DelayBuffer(
        buf=new_buf, head=new_head,
        delay_steps=db.delay_steps, n=db.n,
    ), delayed


# =====================================================================
# MinimalBrain \u2014 reference Phase-2 agent
# =====================================================================


class MinimalBrainParams(eqx.Module):
    """Params for a reference single-area brain.

    Region params are owned; edge matrices are plain arrays with static
    delay step counts. Scalar phase offsets (radians) implement the
    traveling-wave bus (Muller 2018).
    """

    thalamus_relay: RelayParams
    thalamus_trn: TRNParams
    cortex: CorticalAreaParams
    cerebellum: CerebellumParams
    oscillator: OscillatorParams
    neuromodulator: NeuromodulatorParams

    # Inter-region projections
    w_l5_ct: Array                # (n_l5, n_ct) cortex L5 \u2192 thalamus CT
    w_l5_mossy: Array             # (n_l5, mossy_size) cortex L5 \u2192 cerebellum mossy

    # Traveling-wave phase offsets (radians, in [0, 2\u03c0))
    phase_offset_thalamus: Array  # scalar
    phase_offset_cortex: Array    # scalar
    phase_offset_cerebellum: Array

    # Static delay step counts
    delay_ct_steps: int = eqx.field(static=True)
    delay_mossy_steps: int = eqx.field(static=True)


class MinimalBrainState(eqx.Module):
    thalamus_relay: RelayState
    thalamus_trn: TRNState
    cortex: CorticalAreaState
    cerebellum: CerebellumState
    oscillator: OscillatorState
    neuromodulator: NeuromodulatorState

    # Delay lines
    delay_ct: DelayBuffer          # carries L5 spikes \u2192 thalamus CT drive
    delay_mossy: DelayBuffer       # carries L5 rate \u2192 cerebellum mossy


class MinimalBrainOutput(NamedTuple):
    state: MinimalBrainState
    # External consumers
    cortex_belief: Array           # (n_l23_state,) L2/3 rate \u2014 main readout
    cortex_l5_rate: Array          # (n_l5,) motor-command precursor
    cerebellum_nuclei: Array       # (n_dn,) motor correction (available)
    # Diagnostics
    relay_spikes: Array
    trn_spikes: Array
    theta_phase: Array             # scalar, radians
    gamma_amp: Array               # scalar, 0..1
    neuromod: NeuromodulatorState  # same as state.neuromodulator, convenience


def init_minimal_brain_params(
    ctx: BackendContext,
    *,
    sensory_size: int = 16,
    n_tc: int = 64,
    n_ct: int = 32,
    n_trn: int = 32,
    cortex_n_l4: int = 64,
    cortex_n_l23_state: int = 64,
    cortex_n_l23_error: int = 64,
    cortex_n_l5: int = 32,
    mossy_size: int = 32,
    cerebellum_n_purkinje: int = 32,
    delay_ct_ms: float = 2.0,
    delay_mossy_ms: float = 10.0,
    phase_offsets_rad: tuple[float, float, float] = (0.0, 0.15, 0.3),
    w_l5_ct_mean: float = 0.5,
    w_l5_mossy_mean: float = 0.5,
    seed: int = 0,
) -> MinimalBrainParams:
    """Build params for the reference MinimalBrain build.

    Phase offsets (thalamus, cortex, cerebellum) default to
    ``(0, 0.15, 0.3)`` radians \u2014 a small forward-propagating wave along
    the canonical hierarchy (thalamus leads cortex by ~25 ms at 6 Hz).
    """
    tr_p = init_relay_params(
        ctx, n_afferent=sensory_size, n_tc=n_tc, n_ct=n_ct,
    )
    trn_p = init_trn_params(ctx, n_tc_total=n_tc, n_ct=n_ct, n_trn=n_trn)
    cx_p = init_cortical_area_params(
        ctx, input_size=n_tc,
        n_l4=cortex_n_l4, n_l23_state=cortex_n_l23_state,
        n_l23_error=cortex_n_l23_error, n_l5=cortex_n_l5,
    )
    cb_p = init_cerebellum_params(
        ctx, mossy_size=mossy_size, n_purkinje=cerebellum_n_purkinje,
    )
    osc_p = init_oscillator_params()
    nm_p = init_neuromodulator_params(ctx)

    # Inter-region projection weights (half-normal, small)
    master = jax.random.PRNGKey(seed)
    k1, k2 = split_key(master, 2)
    w_l5_ct = jnp.abs(
        jax.random.normal(k1, (cortex_n_l5, n_ct), dtype=DTYPE)
    ) * jnp.asarray(w_l5_ct_mean, DTYPE)
    w_l5_mossy = jnp.abs(
        jax.random.normal(k2, (cortex_n_l5, mossy_size), dtype=DTYPE)
    ) * jnp.asarray(w_l5_mossy_mean, DTYPE)

    # Delays in steps
    d_ct = max(1, int(round(delay_ct_ms / ctx.dt)))
    d_mossy = max(1, int(round(delay_mossy_ms / ctx.dt)))

    f = lambda x: jnp.asarray(x, DTYPE)
    return MinimalBrainParams(
        thalamus_relay=tr_p, thalamus_trn=trn_p,
        cortex=cx_p, cerebellum=cb_p,
        oscillator=osc_p, neuromodulator=nm_p,
        w_l5_ct=w_l5_ct, w_l5_mossy=w_l5_mossy,
        phase_offset_thalamus=f(phase_offsets_rad[0]),
        phase_offset_cortex=f(phase_offsets_rad[1]),
        phase_offset_cerebellum=f(phase_offsets_rad[2]),
        delay_ct_steps=d_ct, delay_mossy_steps=d_mossy,
    )


def init_minimal_brain_state(
    key: PRNGKey, params: MinimalBrainParams, *, dtype=DTYPE,
) -> MinimalBrainState:
    """Zero-initialised region states + delay buffers."""
    k_tr, k_trn, k_cx, k_cb = split_key(key, 4)
    tr_s = init_relay_state(k_tr, params.thalamus_relay)
    trn_s = init_trn_state(k_trn, params.thalamus_trn)
    cx_s = init_cortical_area_state(k_cx, params.cortex)
    cb_s = init_cerebellum_state(k_cb, params.cerebellum)
    osc_s = init_oscillator_state()
    nm_s = init_neuromodulator_state(params.neuromodulator)

    n_ct = params.thalamus_relay.n_ct
    mossy_size = params.cerebellum.mossy_size
    return MinimalBrainState(
        thalamus_relay=tr_s, thalamus_trn=trn_s,
        cortex=cx_s, cerebellum=cb_s,
        oscillator=osc_s, neuromodulator=nm_s,
        delay_ct=init_delay_buffer(n_ct, params.delay_ct_steps, dtype=dtype),
        delay_mossy=init_delay_buffer(
            mossy_size, params.delay_mossy_steps, dtype=dtype,
        ),
    )


# =====================================================================
# Step \u2014 topologically ordered single-dt advance
# =====================================================================


def minimal_brain_step(
    state: MinimalBrainState,
    params: MinimalBrainParams,
    ctx: BackendContext,
    sensory: Array,
    *,
    climbing_error: Array | None = None,
    reward: float | Array = 0.0,
    novelty: float | Array | None = None,
    apply_cortex_stdp: bool = True,
    apply_cerebellum_update: bool = True,
) -> MinimalBrainOutput:
    """One ``dt`` of the MinimalBrain wiring.

    Topological order:
      1. Oscillator + neuromodulator advance (global bus).
      2. Pop delayed signals (CT drive, mossy drive).
      3. Thalamus(sensory, ct_delayed) \u2192 relay spikes.
      4. Cortex(ff=relay, ach, da) \u2192 L5 spikes + L5 rate + L2/3 error.
      5. Cerebellum(mossy_delayed, climbing_error).
      6. Push new delays from cortex L5 outputs.
      7. Update neuromodulator from cortex L2/3 PE + reward.

    All branches use JAX primitives; the whole function is JIT-safe.
    """
    # ---- 1. Oscillator advances (depends on PREVIOUS step's NE / 5-HT) ----
    new_osc, osc_out = oscillator_step(
        state.oscillator, params.oscillator, ctx,
        ne_level=state.neuromodulator.noradrenaline,
        sero_level=state.neuromodulator.serotonin,
        sws_mode=False,
    )

    # ---- 2. Pop delayed signals ----
    delay_ct_new, ct_delayed = delay_push_pop(
        state.delay_ct,
        jnp.zeros((params.thalamus_relay.n_ct,), DTYPE),  # placeholder, rewritten below
    )
    delay_mossy_new, mossy_delayed = delay_push_pop(
        state.delay_mossy,
        jnp.zeros((params.cerebellum.mossy_size,), DTYPE),  # placeholder
    )
    # NOTE: the push values above are placeholders; we'll overwrite after
    # computing this step's cortex outputs. For correctness we MUST read
    # the old buffer first (done) and then push the new value at the same
    # head slot. We'll handle this with a second push below.

    # ---- 3. Thalamus ----
    thal = thalamic_step(
        state.thalamus_relay, params.thalamus_relay,
        state.thalamus_trn, params.thalamus_trn,
        ctx, sensory, ct_delayed,
        ach=state.neuromodulator.acetylcholine,
        ne=state.neuromodulator.noradrenaline,
    )

    # ---- 4. Cortex ----
    cx_inputs = CorticalInputs(
        ff_input=thal.relay_spikes,
        td_prediction=None,
        ach=state.neuromodulator.acetylcholine,
        da=state.neuromodulator.dopamine,
        ne=state.neuromodulator.noradrenaline,
        receptor_gain=jnp.asarray(1.0, DTYPE),
    )
    cx_out = cortical_area_step(
        state.cortex, params.cortex, ctx, cx_inputs,
        apply_stdp=apply_cortex_stdp,
    )

    # ---- 5. Cerebellum ----
    cb_out = cerebellum_step(
        state.cerebellum, params.cerebellum, ctx, mossy_delayed,
    )
    cb_state_after_step = cb_out.state
    if apply_cerebellum_update and climbing_error is not None:
        cb_state_final = cerebellum_update(
            cb_state_after_step, params.cerebellum,
            climbing_error.astype(DTYPE),
            modulator=1.0,
        )
    else:
        cb_state_final = cb_state_after_step

    # ---- 6. Push new values into delay buffers (at the SAME head slot
    # we popped from). We re-push by overwriting the slot that ``head``
    # pointed to before advancing.
    # Strategy: rewrite the buf at pre-advance head of the NEW buffer
    # (which is the slot we just read and already overwrote with zeros).
    # Cleaner: skip the placeholder push above and inline it here.
    l5_new_ct_drive = cx_out.l5_spikes @ params.w_l5_ct
    l5_new_mossy_drive = cx_out.l5_rate @ params.w_l5_mossy

    pre_head_ct = state.delay_ct.head
    pre_head_mossy = state.delay_mossy.head
    delay_ct_final = eqx.tree_at(
        lambda db: db.buf,
        delay_ct_new,
        delay_ct_new.buf.at[pre_head_ct].set(l5_new_ct_drive.astype(DTYPE)),
    )
    delay_mossy_final = eqx.tree_at(
        lambda db: db.buf,
        delay_mossy_new,
        delay_mossy_new.buf.at[pre_head_mossy].set(l5_new_mossy_drive.astype(DTYPE)),
    )

    # ---- 7. Neuromodulator ----
    # PE magnitude = cortex L2/3 error rate mean; td_error not used here
    # (BG+VTA in Phase 3 will provide it).
    pe_signal = cx_out.ff_out                      # (n_l23_error,)
    nm_new = neuromodulator_step(
        state.neuromodulator, params.neuromodulator,
        prediction_error=pe_signal,
        td_error=0.0,
        reward=reward,
        novelty=novelty,
    )

    new_state = MinimalBrainState(
        thalamus_relay=thal.relay, thalamus_trn=thal.trn,
        cortex=cx_out.state,
        cerebellum=cb_state_final,
        oscillator=new_osc,
        neuromodulator=nm_new,
        delay_ct=delay_ct_final,
        delay_mossy=delay_mossy_final,
    )

    return MinimalBrainOutput(
        state=new_state,
        cortex_belief=cx_out.belief,
        cortex_l5_rate=cx_out.l5_rate,
        cerebellum_nuclei=cb_out.nuclei,
        relay_spikes=thal.relay_spikes,
        trn_spikes=thal.trn_spikes,
        theta_phase=new_osc.theta_phase,
        gamma_amp=new_osc.gamma_amplitude,
        neuromod=nm_new,
    )


# =====================================================================
# Traveling-wave helper
# =====================================================================


def region_phase(osc_state: OscillatorState, phase_offset: Array) -> Array:
    """Theta phase at a named region: ``(theta + offset) mod 2\u03c0``.

    Used by downstream modules to schedule encoding/retrieval windows
    coherently across the cortical hierarchy (Muller 2018 waves).
    """
    return jnp.mod(osc_state.theta_phase + phase_offset, 2.0 * jnp.pi)
