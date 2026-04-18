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
    td_error: float | Array = 0.0,
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
        td_error=td_error,
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


# =====================================================================
# ActionBrain — Phase 3: closes the sense–act–learn–reward loop
# =====================================================================
#
# Extends MinimalBrain with
#   * basal_ganglia: D1/D2 actor + ventral-striatum critic
#   * VTA: TD-style RPE broadcast to the DA bus
#   * world_model: dense-coded forward model (inferior olive proxy)
#   * fixed random projection sensory PE → Purkinje cell climbing fibres
#
# Timing discipline
# -----------------
# 1 "decision cycle" = ``substeps`` brain ``dt``s (default 20 ms = 20 dt).
# Inside the cycle the cortex, thalamus and cerebellum iterate; BG
# MSN spike counts accumulate (Gold & Shadlen 2007 evidence integration);
# the actor produces an action at the end of the cycle.
#
# Transition learning uses the TD(0) sliding-window pattern:
#   * V(s_t) was stored in VTA at the end of the PREVIOUS perceive
#     (``vta_store_prediction``).
#   * At the START of the current perceive we have r_{t}, s_{t+1} (the
#     body just reported them). The critic has been iterated on s_{t+1}
#     so ``critic.activation`` now approximates V(s_{t+1}).
#   * rpe = r_t + γ·V(s_{t+1}) · (1−done_t) − V(s_t)  (``vta_compute_rpe``)
#   * Weights update (critic + actor + VTA value readout).
#   * World model update with (s_t, a_t, s_{t+1}) → PE → climbing signal.
#   * Then perceive s_{t+1} for ``substeps`` dt; store V(s_{t+1}); pick a_{t+1}.
#
# Intrinsic drives
# ----------------
# The total reward broadcast to VTA / neuromodulator:
#   r_total = r_extrinsic + β_c · curiosity − β_h · (1 − atp_mean)
# with β_c = 0.1 (Pathak 2017 ICM curiosity weight lower bound) and
# β_h = 0.05 (homeostatic drives are a small modulation; Keramati &
# Gutkin 2014). These are reference-paper defaults; no task tuning.
#
# References
# ----------
# Frank (2005); Collins & Frank (2014); Schultz (1997); Tobler et al.
# (2005); Wolpert et al. (1998); Sutton & Barto (2018) TD(0);
# Gold & Shadlen (2007); Pathak et al. (2017) ICM; Keramati & Gutkin
# (2014) homeostatic RL; Schweighofer et al. (2008) 5-HT·γ.


from .basal_ganglia import (
    CriticParams, CriticState,
    init_critic_params, init_critic_state,
    critic_step, critic_update, critic_reset_transient,
    ActorParams, ActorState, ActorInputs,
    init_actor_params, init_actor_state,
    actor_step, actor_select_action, actor_update,
    actor_reset_evidence, actor_reset_transient,
)
from .vta import (
    VTAParams, VTAState,
    init_vta_params, init_vta_state,
    vta_store_prediction, vta_compute_rpe, vta_update, vta_reset_transient,
)
from .world_model import (
    WorldModelParams, WorldModelState,
    init_world_model_params, init_world_model_state,
    wm_predict, wm_update, wm_curiosity_signal, wm_reset_transient,
)


# ---------------------------------------------------------------------
# Parallel cortico-BG-thalamo-cortical loops
# ---------------------------------------------------------------------
#
# Alexander, DeLong & Strick (1986) demonstrated that the basal ganglia
# implement at least five anatomically segregated parallel loops —
# motor, oculomotor, dorsolateral prefrontal, lateral orbitofrontal,
# anterior cingulate — each selecting actions in its own domain and
# sharing only the dopaminergic broadcast from SNc/VTA and the cortical
# glutamatergic drive. Credit assignment across domains is delegated to
# the world model (shared sensory outcome → shared DA signal).
#
# In Phase 4 we instantiate **two** such loops:
#
#   * ``actor_body``    — skeletomotor loop (putamen analogue), selects
#     one of ``n_body_actions`` discrete motor commands.
#   * ``actor_saccade`` — oculomotor loop (caudate / FEF-SC analogue),
#     selects one of ``n_saccade_actions`` discrete fovea shifts.
#
# Both populations receive the SAME striatal drive (cortex L2/3 belief
# ⊕ raw sensory) and the SAME phasic DA from VTA, but they maintain
# independent MSN voltages, independent lateral inhibition and
# independent evidence accumulators. This is the only architecture
# that scales linearly with the number of action domains — a
# prerequisite for Phase 5 (babbling / speech motor) and beyond.
#
# References
# ----------
# Alexander, DeLong & Strick (1986) — parallel segregated circuits.
# Hikosaka et al. (2000)           — saccade selection via BG.
# Middleton & Strick (2000)        — distinct cortical territories.
# Tan (1993); Claus & Boutilier (1998) — independent learners with a
#                                         shared global reward.


#: Discrete saccade action space emitted by the oculomotor loop.
#: Index 0 = recentre fovea, 1..8 = eight cardinal / diagonal shifts.
#: The brain emits only the index; the body interprets it.
SACCADE_ACTION_DIM: int = 9


# ---------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------


class ActionBrainParams(eqx.Module):
    """All MinimalBrain params + two parallel BG loops + VTA + WM."""

    # Region modules (body-agnostic)
    thalamus_relay: RelayParams
    thalamus_trn: TRNParams
    cortex: CorticalAreaParams
    cerebellum: CerebellumParams
    oscillator: OscillatorParams
    neuromodulator: NeuromodulatorParams
    critic: CriticParams
    actor_body: ActorParams
    actor_saccade: ActorParams
    vta: VTAParams
    world_model: WorldModelParams

    # Inter-region projections (L5 → CT drive, L5 → mossy fibre)
    w_l5_ct: Array              # (n_l5, n_ct)
    w_l5_mossy: Array           # (n_l5, mossy_size)

    # Inferior-olive proxy: sensory PE → climbing fibres (Purkinje)
    w_io_pc: Array              # (sensory_size, n_purkinje)

    # Traveling-wave phase offsets (radians)
    phase_offset_thalamus: Array
    phase_offset_cortex: Array
    phase_offset_cerebellum: Array

    # Intrinsic-drive coefficients (paper-documented, not task-tuned)
    beta_curiosity: Array       # Pathak 2017 ICM lower bound
    beta_homeostasis: Array     # Keramati & Gutkin 2014

    # Static
    n_body_actions: int = eqx.field(static=True)
    n_saccade_actions: int = eqx.field(static=True)
    sensory_size: int = eqx.field(static=True)
    substeps: int = eqx.field(static=True)
    delay_ct_steps: int = eqx.field(static=True)
    delay_mossy_steps: int = eqx.field(static=True)


class ActionBrainState(eqx.Module):
    """All MinimalBrain state + dual-loop BG + VTA/WM + decision bookkeeping."""

    thalamus_relay: RelayState
    thalamus_trn: TRNState
    cortex: CorticalAreaState
    cerebellum: CerebellumState
    oscillator: OscillatorState
    neuromodulator: NeuromodulatorState
    critic: CriticState
    actor_body: ActorState
    actor_saccade: ActorState
    vta: VTAState
    world_model: WorldModelState

    delay_ct: DelayBuffer
    delay_mossy: DelayBuffer

    # Last committed motor commands from each parallel loop. The world
    # model consumes the concatenation as its action space so it can
    # model sensory outcomes that depend on both where the body moved
    # and where the eye looked.
    last_body_action: Array          # (n_body_actions,) one-hot
    last_body_action_id: Array       # scalar int32
    last_saccade_action: Array       # (n_saccade_actions,) one-hot
    last_saccade_action_id: Array    # scalar int32

    last_sensory: Array              # (sensory_size,)
    last_rpe: Array                  # scalar — diagnostic only
    last_total_reward: Array         # scalar — diagnostic only


class ActionBrainOutput(NamedTuple):
    state: ActionBrainState
    body_action: Array           # scalar int32 — skeletomotor command
    saccade_action: Array        # scalar int32 — oculomotor command
    rpe: Array                   # scalar — RPE for the previous transition
    total_reward: Array          # scalar — extrinsic + intrinsic
    curiosity: Array             # scalar — world-model curiosity ∈ [0, 1]
    cortex_belief: Array         # (n_l23_state,)
    cortex_l5_rate: Array        # (n_l5,)
    cerebellum_nuclei: Array     # (n_dn,)
    relay_spikes: Array
    theta_phase: Array
    neuromod: NeuromodulatorState


def init_action_brain_params(
    ctx: BackendContext,
    *,
    sensory_size: int,
    n_body_actions: int,
    n_saccade_actions: int = SACCADE_ACTION_DIM,
    # thalamus
    n_tc: int = 64,
    n_ct: int = 32,
    n_trn: int = 32,
    # cortex
    cortex_n_l4: int = 64,
    cortex_n_l23_state: int = 64,
    cortex_n_l23_error: int = 64,
    cortex_n_l5: int = 32,
    # cerebellum
    mossy_size: int = 32,
    cerebellum_n_purkinje: int = 32,
    # BG
    critic_hidden: int = 64,
    actor_n_per_action: int = 4,
    # world model
    wm_hidden: int = 64,
    wm_n_error: int = 64,
    wm_n_neurons_per_dim: int = 8,
    # delays
    delay_ct_ms: float = 2.0,
    delay_mossy_ms: float = 10.0,
    # waves
    phase_offsets_rad: tuple[float, float, float] = (0.0, 0.15, 0.3),
    # inter-region weights
    w_l5_ct_mean: float = 0.5,
    w_l5_mossy_mean: float = 0.5,
    w_io_pc_sigma: float = 0.3,
    # decision cycle
    substeps: int = 20,
    # intrinsic drives (paper-grounded, not task-tuned)
    beta_curiosity: float = 0.1,
    beta_homeostasis: float = 0.05,
    seed: int = 0,
) -> ActionBrainParams:
    """Build an ActionBrain params pytree.

    Two BG actors are instantiated over the same striatal drive (cortex
    L2/3 belief ⊕ raw sensory): a body-action actor of ``motor_dim =
    n_body_actions`` and a saccade actor of ``motor_dim =
    n_saccade_actions``. The world model's action space is the
    concatenation ``n_body_actions + n_saccade_actions`` so it can
    account for sensory outcomes that depend on both.

    20 substeps ≈ 20 ms is the canonical cortical decision interval
    (Schall 2002; Gold & Shadlen 2007); both parallel loops share this
    window (Rayner 1998 reports 200-300 ms inter-saccade intervals,
    i.e. ≥10 decision cycles per saccade — evidence accumulation is
    naturally slower for low-salience inputs).
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

    state_size = cortex_n_l23_state + sensory_size
    critic_p = init_critic_params(
        ctx, state_size=state_size, hidden_size=critic_hidden,
    )
    actor_body_p = init_actor_params(
        ctx, state_size=state_size,
        motor_dim=n_body_actions,
        n_per_action=actor_n_per_action,
    )
    actor_saccade_p = init_actor_params(
        ctx, state_size=state_size,
        motor_dim=n_saccade_actions,
        n_per_action=actor_n_per_action,
    )
    vta_p = init_vta_params(ctx, hidden_size=critic_hidden)
    wm_p = init_world_model_params(
        ctx, state_size=sensory_size,
        action_size=n_body_actions + n_saccade_actions,
        hidden_size=wm_hidden, n_error=wm_n_error,
        n_neurons_per_dim=wm_n_neurons_per_dim,
    )

    master = jax.random.PRNGKey(seed)
    k1, k2, k3 = split_key(master, 3)
    w_l5_ct = jnp.abs(
        jax.random.normal(k1, (cortex_n_l5, n_ct), dtype=DTYPE)
    ) * jnp.asarray(w_l5_ct_mean, DTYPE)
    w_l5_mossy = jnp.abs(
        jax.random.normal(k2, (cortex_n_l5, mossy_size), dtype=DTYPE)
    ) * jnp.asarray(w_l5_mossy_mean, DTYPE)
    w_io_pc = jax.random.normal(
        k3, (sensory_size, cerebellum_n_purkinje), dtype=DTYPE,
    ) * jnp.asarray(w_io_pc_sigma, DTYPE)

    d_ct = max(1, int(round(delay_ct_ms / ctx.dt)))
    d_mossy = max(1, int(round(delay_mossy_ms / ctx.dt)))

    f = lambda x: jnp.asarray(x, DTYPE)
    return ActionBrainParams(
        thalamus_relay=tr_p, thalamus_trn=trn_p,
        cortex=cx_p, cerebellum=cb_p,
        oscillator=osc_p, neuromodulator=nm_p,
        critic=critic_p,
        actor_body=actor_body_p,
        actor_saccade=actor_saccade_p,
        vta=vta_p, world_model=wm_p,
        w_l5_ct=w_l5_ct, w_l5_mossy=w_l5_mossy, w_io_pc=w_io_pc,
        phase_offset_thalamus=f(phase_offsets_rad[0]),
        phase_offset_cortex=f(phase_offsets_rad[1]),
        phase_offset_cerebellum=f(phase_offsets_rad[2]),
        beta_curiosity=f(beta_curiosity),
        beta_homeostasis=f(beta_homeostasis),
        n_body_actions=int(n_body_actions),
        n_saccade_actions=int(n_saccade_actions),
        sensory_size=int(sensory_size),
        substeps=int(substeps),
        delay_ct_steps=d_ct, delay_mossy_steps=d_mossy,
    )


def init_action_brain_state(
    key: PRNGKey, params: ActionBrainParams, *, dtype=DTYPE,
) -> ActionBrainState:
    keys = split_key(key, 9)
    tr_s = init_relay_state(keys[0], params.thalamus_relay)
    trn_s = init_trn_state(keys[1], params.thalamus_trn)
    cx_s = init_cortical_area_state(keys[2], params.cortex)
    cb_s = init_cerebellum_state(keys[3], params.cerebellum)
    osc_s = init_oscillator_state()
    nm_s = init_neuromodulator_state(params.neuromodulator)
    critic_s = init_critic_state(keys[4], params.critic)
    actor_body_s = init_actor_state(keys[5], params.actor_body)
    actor_saccade_s = init_actor_state(keys[6], params.actor_saccade)
    vta_s = init_vta_state(keys[7], params.vta)
    wm_s = init_world_model_state(keys[8], params.world_model)

    n_ct = params.thalamus_relay.n_ct
    mossy_size = params.cerebellum.mossy_size
    return ActionBrainState(
        thalamus_relay=tr_s, thalamus_trn=trn_s,
        cortex=cx_s, cerebellum=cb_s,
        oscillator=osc_s, neuromodulator=nm_s,
        critic=critic_s,
        actor_body=actor_body_s,
        actor_saccade=actor_saccade_s,
        vta=vta_s, world_model=wm_s,
        delay_ct=init_delay_buffer(n_ct, params.delay_ct_steps, dtype=dtype),
        delay_mossy=init_delay_buffer(
            mossy_size, params.delay_mossy_steps, dtype=dtype,
        ),
        last_body_action=jnp.zeros(params.n_body_actions, dtype),
        last_body_action_id=jnp.asarray(0, jnp.int32),
        last_saccade_action=jnp.zeros(params.n_saccade_actions, dtype),
        last_saccade_action_id=jnp.asarray(0, jnp.int32),
        last_sensory=jnp.zeros(params.sensory_size, dtype),
        last_rpe=jnp.asarray(0.0, dtype),
        last_total_reward=jnp.asarray(0.0, dtype),
    )


# ---------------------------------------------------------------------
# Internal: single ``dt`` of the perception cortex loop with BG/WM/critic
# ---------------------------------------------------------------------


def _perceive_substep(
    state: ActionBrainState,
    params: ActionBrainParams,
    ctx: BackendContext,
    sensory: Array,
    td_error: Array,
    novelty: Array,
) -> tuple[ActionBrainState, tuple[Array, Array, Array]]:
    """One ``dt`` of perception: thalamus, cortex, cerebellum, critic,
    actor, world-model prediction, neuromodulator.

    Returns the advanced state plus the current-step cortex/thalamus
    readouts ``(cortex_belief, cortex_l5_rate, relay_spikes)`` so the
    scanning caller can expose them without peeking at internal region
    fields.
    """
    # Rebuild the MinimalBrain-shaped subtree and step it. This reuses
    # the exact wiring (delay buffers, neuromod, oscillator) already
    # validated in Phase 2.
    mb_state = MinimalBrainState(
        thalamus_relay=state.thalamus_relay,
        thalamus_trn=state.thalamus_trn,
        cortex=state.cortex, cerebellum=state.cerebellum,
        oscillator=state.oscillator, neuromodulator=state.neuromodulator,
        delay_ct=state.delay_ct, delay_mossy=state.delay_mossy,
    )
    mb_params = MinimalBrainParams(
        thalamus_relay=params.thalamus_relay,
        thalamus_trn=params.thalamus_trn,
        cortex=params.cortex, cerebellum=params.cerebellum,
        oscillator=params.oscillator, neuromodulator=params.neuromodulator,
        w_l5_ct=params.w_l5_ct, w_l5_mossy=params.w_l5_mossy,
        phase_offset_thalamus=params.phase_offset_thalamus,
        phase_offset_cortex=params.phase_offset_cortex,
        phase_offset_cerebellum=params.phase_offset_cerebellum,
        delay_ct_steps=params.delay_ct_steps,
        delay_mossy_steps=params.delay_mossy_steps,
    )
    mb_out = minimal_brain_step(
        mb_state, mb_params, ctx, sensory,
        climbing_error=None,
        reward=jnp.asarray(0.0, DTYPE),      # reward only at learn phase
        td_error=td_error,                    # phasic DA follows VTA rpe
        novelty=novelty,                      # ACh tracks curiosity
        apply_cortex_stdp=True,
        apply_cerebellum_update=False,        # no IO signal during perceive
    )

    # BG critic/actor read cortico-thalamic convergent drive: cortex
    # L2/3 belief (continuous rate) concatenated with the raw sensory
    # afferent vector. This models the parallel thalamostriatal
    # pathway (Smith 2004; McFarland & Haber 2002) that bypasses cortex
    # and lets striatum learn directly from subcortical sensory input.
    striatal_drive = jnp.concatenate([mb_out.cortex_belief, sensory], axis=0)
    critic_out = critic_step(
        state.critic, params.critic, ctx, striatal_drive,
    )

    # Parallel cortico-BG-thalamo-cortical loops (Alexander 1986):
    # both actors receive the same striatal drive and the same phasic
    # DA broadcast, but maintain independent MSN populations, lateral
    # inhibition and evidence accumulators.
    actor_inputs = ActorInputs(
        da=mb_out.neuromod.dopamine,
        tonic_da=mb_out.neuromod.tonic_da,
        epistemic_drive=novelty,
    )
    body_out = actor_step(
        state.actor_body, params.actor_body, ctx, striatal_drive,
        inputs=actor_inputs,
    )
    saccade_out = actor_step(
        state.actor_saccade, params.actor_saccade, ctx, striatal_drive,
        inputs=actor_inputs,
    )

    # World-model forward prediction consumes the JOINT last motor
    # command — body and saccade both shape what the brain will see
    # next (the body by moving the agent, the saccade by moving the
    # fovea).
    joint_last_action = jnp.concatenate(
        [state.last_body_action, state.last_saccade_action], axis=0,
    )
    wm_out = wm_predict(
        state.world_model, params.world_model, ctx,
        sensory, joint_last_action,
        ach=mb_out.neuromod.acetylcholine,
    )

    return ActionBrainState(
        thalamus_relay=mb_out.state.thalamus_relay,
        thalamus_trn=mb_out.state.thalamus_trn,
        cortex=mb_out.state.cortex,
        cerebellum=mb_out.state.cerebellum,
        oscillator=mb_out.state.oscillator,
        neuromodulator=mb_out.state.neuromodulator,
        critic=critic_out.state,
        actor_body=body_out.state,
        actor_saccade=saccade_out.state,
        vta=state.vta,
        world_model=wm_out.state,
        delay_ct=mb_out.state.delay_ct,
        delay_mossy=mb_out.state.delay_mossy,
        last_body_action=state.last_body_action,
        last_body_action_id=state.last_body_action_id,
        last_saccade_action=state.last_saccade_action,
        last_saccade_action_id=state.last_saccade_action_id,
        last_sensory=state.last_sensory,
        last_rpe=state.last_rpe,
        last_total_reward=state.last_total_reward,
    ), (mb_out.cortex_belief, mb_out.cortex_l5_rate, mb_out.relay_spikes)


# ---------------------------------------------------------------------
# Top-level: perceive→learn→perceive cycle (one body decision)
# ---------------------------------------------------------------------


def action_brain_step(
    state: ActionBrainState,
    params: ActionBrainParams,
    ctx: BackendContext,
    sensory: Array,
    prev_reward: float | Array,
    prev_done: float | Array,
    key: PRNGKey,
) -> ActionBrainOutput:
    """Run one full decision cycle on sensory ``s_{t+1}``.

    Inputs:
      * ``sensory``      : ``s_{t+1}`` just received from
        ``body.act(body_action_t, saccade_action_t)``.
      * ``prev_reward``  : ``r_t`` emitted during that transition.
      * ``prev_done``    : terminal flag for the transition.
      * ``key``          : PRNG for action sampling (independently
        split between the body-actor and saccade-actor tie-breaks).

    Output:
      * ``body_action``    : the newly committed skeletomotor command.
      * ``saccade_action`` : the newly committed oculomotor command.
      * Diagnostics including the RPE that corrected the previous
        transition, the aggregated intrinsic+extrinsic total reward,
        curiosity, and selected cortical readouts.

    Side effect: the world model is *updated* with the realised
    transition ``(s_t, (a_body_t ⊕ a_saccade_t), s_{t+1})``, and the
    cerebellum receives the resulting sensory-prediction error as its
    climbing signal. Both actor populations receive the same scalar
    RPE and update independently against their own eligibility traces.

    All operations are pure; JIT via ``jax.lax.scan`` for the substep
    loop inside perception.
    """
    r_ext = jnp.asarray(prev_reward, DTYPE)
    done = jnp.asarray(prev_done, DTYPE)

    # --- 1. Perceive s_{t+1} first so critic.activation = V(s_{t+1}).
    # Clear BG evidence on BOTH parallel loops at the start of a new
    # decision window so spike counts accumulate cleanly over this
    # window only.
    state = eqx.tree_at(
        lambda s: (s.actor_body, s.actor_saccade),
        state,
        (
            actor_reset_evidence(state.actor_body),
            actor_reset_evidence(state.actor_saccade),
        ),
    )

    # During perception the phasic-DA / ACh drives carry the *previous*
    # cycle's RPE and curiosity (neuromod bus has finite lag; new
    # estimates are available only after we have finished perceiving
    # and computed the transition error).
    prev_rpe = state.last_rpe
    prev_curiosity = wm_curiosity_signal(state.world_model, params.world_model)

    def scan_body(st: ActionBrainState, _):
        new_st, readouts = _perceive_substep(
            st, params, ctx, sensory,
            td_error=prev_rpe, novelty=prev_curiosity,
        )
        return new_st, readouts

    state, readouts_hist = jax.lax.scan(
        scan_body, state, None, length=params.substeps,
    )
    belief_hist, l5_rate_hist, relay_hist = readouts_hist
    cortex_belief = belief_hist[-1]
    cortex_l5_rate = l5_rate_hist[-1]
    relay_spikes = relay_hist[-1]

    # --- 2. Close the loop on the PREVIOUS transition --------------
    # 2a. World-model learning on (s_t, a_t, s_{t+1}) — sensory PE
    #     serves as the cerebellar climbing signal. Action is the
    #     joint body+saccade one-hot committed on the previous cycle.
    joint_last_action = jnp.concatenate(
        [state.last_body_action, state.last_saccade_action], axis=0,
    )
    wm_out = wm_update(
        state.world_model, params.world_model, ctx,
        state.last_sensory, joint_last_action, sensory,
        m_t=1.0,
        ach=state.neuromodulator.acetylcholine,
    )
    sensory_pe = wm_out.prediction_error                 # (sensory_size,)
    climbing = sensory_pe @ params.w_io_pc               # (n_purkinje,)
    cb_state = cerebellum_update(
        state.cerebellum, params.cerebellum, climbing, modulator=1.0,
    )

    # 2b. Intrinsic drives feed r_total.
    curiosity = wm_curiosity_signal(wm_out.state, params.world_model)
    atp_mean = jnp.mean(wm_out.state.astro.atp)
    atp_deficit = jnp.clip(1.0 - atp_mean, 0.0, 1.0)
    r_total = (
        r_ext
        + params.beta_curiosity * curiosity
        - params.beta_homeostasis * atp_deficit
    )

    # 2c. TD(0) RPE: V(s_t)=state.vta.stored_v, V(s_{t+1})=critic.activation
    #     (just computed by the perceive scan above).
    vta_out = vta_compute_rpe(
        state.vta, params.vta,
        critic_activation=state.critic.activation,
        reward=r_total,
        is_terminal=done,
        serotonin=state.neuromodulator.serotonin,
        n_substeps=params.substeps,
    )
    rpe = vta_out.rpe

    # 2d. Weight updates driven by this RPE. The same scalar RPE drives
    #     both parallel loops; credit is assigned per-domain through
    #     the INDEPENDENT eligibility traces each actor maintains.
    vta_state = vta_update(vta_out.state, params.vta, rpe)
    critic_state = critic_update(state.critic, params.critic, rpe)
    actor_body_state = actor_update(
        state.actor_body, params.actor_body, rpe,
        state.last_body_action_id,
    )
    actor_saccade_state = actor_update(
        state.actor_saccade, params.actor_saccade, rpe,
        state.last_saccade_action_id,
    )

    # --- 3. Store V(s_{t+1}) in VTA for next cycle's TD target. -----
    vta_state = vta_store_prediction(vta_state, critic_state.activation)

    # --- 4. Select a_{t+1} from each loop's evidence accumulated in
    #     the scan. Independent PRNG keys so tie-breaking in one loop
    #     does not correlate with the other.
    k_body, k_saccade = split_key(key, 2)
    body_action_id = actor_select_action(
        actor_body_state, params.actor_body, k_body,
    )
    saccade_action_id = actor_select_action(
        actor_saccade_state, params.actor_saccade, k_saccade,
    )
    body_action_oh = (
        jnp.arange(params.n_body_actions) == body_action_id
    ).astype(DTYPE)
    saccade_action_oh = (
        jnp.arange(params.n_saccade_actions) == saccade_action_id
    ).astype(DTYPE)

    cerebellum_nuclei = cb_state.dn_rate

    new_state = ActionBrainState(
        thalamus_relay=state.thalamus_relay,
        thalamus_trn=state.thalamus_trn,
        cortex=state.cortex,
        cerebellum=cb_state,
        oscillator=state.oscillator,
        neuromodulator=state.neuromodulator,
        critic=critic_state,
        actor_body=actor_body_state,
        actor_saccade=actor_saccade_state,
        vta=vta_state,
        world_model=wm_out.state,
        delay_ct=state.delay_ct, delay_mossy=state.delay_mossy,
        last_body_action=body_action_oh,
        last_body_action_id=body_action_id,
        last_saccade_action=saccade_action_oh,
        last_saccade_action_id=saccade_action_id,
        last_sensory=sensory.astype(DTYPE),
        last_rpe=rpe,
        last_total_reward=r_total,
    )

    return ActionBrainOutput(
        state=new_state,
        body_action=body_action_id,
        saccade_action=saccade_action_id,
        rpe=rpe,
        total_reward=r_total,
        curiosity=curiosity,
        cortex_belief=cortex_belief,
        cortex_l5_rate=cortex_l5_rate,
        cerebellum_nuclei=cerebellum_nuclei,
        relay_spikes=relay_spikes,
        theta_phase=state.oscillator.theta_phase,
        neuromod=state.neuromodulator,
    )
