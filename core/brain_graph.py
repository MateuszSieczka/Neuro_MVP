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
  2. **ActionBrain** — single thalamus + cortical area + cerebellum +
     dual BG loops + VTA + world model, sharing a global oscillator
     and neuromodulator bus. Closes the sense–act–learn–reward loop.

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

from typing import NamedTuple, TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from sensory.sensory_stack import SensoryStackParams, SensoryStackState
    from sensory.proprioception import ProprioceptionParams

from .backend import DTYPE, Array, PRNGKey, BackendContext, split_key
from .state import OscillatorState, init_oscillator_state

from .oscillator import (
    OscillatorParams, init_oscillator_params, oscillator_step,
)
from .neuromodulator import (
    NeuromodulatorParams, NeuromodulatorState,
    init_neuromodulator_params, init_neuromodulator_state,
    neuromodulator_step, transmitter_vector,
    adenosine_update,
)
from .receptor import (
    ReceptorParams, init_receptor_params, compute_layer_modulation,
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
    cortical_area_step, cortical_area_update,
)
from .cerebellum import (
    CerebellumParams, CerebellumState,
    init_cerebellum_params, init_cerebellum_state,
    cerebellum_step, cerebellum_update,
)
from .attention import (
    AttentionParams, AttentionState,
    init_attention_params, init_attention_state,
    attention_step, attention_learn,
)
from .basal_ganglia import (
    CriticParams, CriticState,
    init_critic_params, init_critic_state,
    critic_step, critic_update, critic_commit_eligibility,
    critic_reset_transient,
    ActorParams, ActorState, ActorInputs,
    init_actor_params, init_actor_state,
    actor_step, actor_select_action, actor_update,
    actor_commit_eligibility,
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
from .precision_bus import (
    PrecisionChannel, init_precision_channel,
    precision_update, precision_standardize,
)
from .pfc import (
    PFCParams, PFCState,
    init_pfc_params, init_pfc_state,
    pfc_step, pfc_select_reset,
)
from .replay_buffer import (
    ReplayParams, ReplayState, Experience,
    init_replay_params, init_replay_state, replay_store,
)
from .sleep import (
    SleepPhase, SleepParams, SleepState,
    init_sleep_params, init_sleep_state, sleep_step,
    is_sws, is_wake, is_rem,
)
from .sleep_replay import sws_replay_step, rem_rollout_step
from .ec import (
    EntorhinalParams, EntorhinalState,
    init_ec_params, init_ec_state, ec_step,
)
from .hippocampus import (
    HippocampusParams, HippocampusState, HippocampusOutput,
    init_hippocampus_params, init_hippocampus_state, hippocampus_step,
)
from .learning_pipeline import (
    critic_learn_step, actors_learn_step,
    cortex_learn_step, attention_learn_step,
)
from .m1 import (
    M1Params, M1State,
    init_m1_params, init_m1_state,
    m1_step, m1_learn_readout,
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
# Receptor gain helper
# =====================================================================


# Default receptor densities per region (fractional, [0, 1]) along
# :data:`core.receptor.RECEPTOR_ORDER`:
# ``(D1, D2, M1, M4, NACHR, ALPHA1, ALPHA2, BETA, HT1A, HT2A)``.
#
# These are literature-guided ballpark values for the dominant
# excitatory pyramidal neuron of each area (not exact quantifications
# \u2014 cortical receptor counts vary 5-10\u00d7 across studies). They are
# deliberately conservative; phase-specific regions (PFC, V1, ATL,
# striatum) override individual subtype densities in their init_*.
#
#   Cortex (generic pyramidal):  high M1/HT2A/ALPHA1, moderate D1/NACHR,
#                                low D2/M4.                              (Seamans & Yang 2004)
#   PFC override (Phase 0.5):     bump D1 \u2192 0.7, M1 \u2192 0.6.               (Goldman-Rakic 1995)
#   Thalamic relay (not used \u2014 McCormick bias covers ACh/NE gating).
#   Striatum (BG already uses receptor.hill_response directly \u2014 keep).
CORTEX_DEFAULT_DENSITY: tuple[float, ...] = (
    0.4,  # D1
    0.2,  # D2
    0.5,  # M1
    0.1,  # M4
    0.3,  # NACHR
    0.4,  # ALPHA1
    0.3,  # ALPHA2
    0.4,  # BETA
    0.3,  # HT1A
    0.5,  # HT2A
)


def _region_receptor_gain(
    receptor_params: ReceptorParams,
    neuromod: NeuromodulatorState,
    nm_params: NeuromodulatorParams,
    density: Array,
) -> Array:
    """Multiplicative excitability gain from current neuromod levels.

    Normalised so that at the neuromodulator **baseline** set-points
    (``baseline_da / _ach / _ne / _sero`` in ``NeuromodulatorParams``)
    the gain equals exactly 1.0. Phasic deviations above / below the
    tonic set-point therefore produce proportional multiplicative
    modulation, which is what the receptor\u2013G-protein cascade does at
    steady state (Aston-Jones & Cohen 2005; Schultz 1998).

    This is not a "hack" \u2014 the baseline is an intrinsic property of
    the neuromodulator pytree (not a tuned free parameter); calibrating
    gain to 1.0 at that baseline preserves the existing weight scales
    (rheobase targets in ``cortex.init_cortical_area_state``) while
    enabling D1/D2/M1/\u03b1/\u03b2/5-HT effects to enter the loop.
    """
    tx = transmitter_vector(neuromod)
    baseline_tx = jnp.stack([
        nm_params.baseline_da, nm_params.baseline_ach,
        nm_params.baseline_ne, nm_params.baseline_sero,
    ])
    curr = compute_layer_modulation(receptor_params, tx, density)
    base = compute_layer_modulation(receptor_params, baseline_tx, density)
    return curr.gain_mod / jnp.maximum(base.gain_mod, jnp.asarray(0.1, DTYPE))




# =====================================================================
# ActionBrain — closes the sense–act–learn–reward loop
# =====================================================================
# ActionBrain — Phase 3: closes the sense–act–learn–reward loop
# =====================================================================
#
# Extends wiring primitives with
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
# The critic TD target is pure extrinsic:
#   r_total = r_extrinsic
# so V(s) converges to the expected extrinsic return without novelty
# contamination (Pathak 2017 note 4 on curiosity-as-reward artefacts).
# Epistemic drive enters at the ACTOR level (post-critic), segregated
# by modality, following the Dayan & Yu 2003 dual-process view:
#   (i)   curiosity -> ACh/NE (perception gain; Yu & Dayan 2005).
#   (ii)  body actor RPE += curiosity (transition-surprise exploration;
#         Friston 2017 EFE epistemic component).
#   (iii) saccade actor RPE += info_gain (Itti & Baldi 2009).
#   (iv)  curiosity -> replay-buffer salience (off-policy WM refinement).
# ATP homeostasis acts locally on AdEx dynamics (V_T shift, g_L gain;
# Morris 2003) and does NOT enter r_total.
#
# References
# ----------
# Frank (2005); Collins & Frank (2014); Schultz (1997); Tobler et al.
# (2005); Wolpert et al. (1998); Sutton & Barto (2018) TD(0);
# Gold & Shadlen (2007); Yu & Dayan (2005) uncertainty; Itti & Baldi
# (2009) surprise; Schweighofer et al. (2008) 5-HT.gamma.



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
    """Core region params + two parallel BG loops + VTA + WM.

    :attr:`sensory_stack` wires a foveated retina → LGN → V1 chain
    inside the brain. The eye is CNS (Rodieck 1998):
    :func:`action_brain_step` expects ``image`` + ``fixation_xy`` and
    computes V1 L2/3 belief as the ``sensory`` afferent, plus a
    Bayesian-surprise saccade reward (Itti & Baldi 2009).
    """

    # Sensory front-end: foveated retina → LGN → V1.
    sensory_stack: SensoryStackParams

    # Region modules (body-agnostic)
    thalamus_relay: RelayParams
    thalamus_trn: TRNParams
    cortex: CorticalAreaParams
    cerebellum: CerebellumParams
    oscillator: OscillatorParams
    neuromodulator: NeuromodulatorParams
    receptor: ReceptorParams
    attention: AttentionParams
    critic: CriticParams
    actor_body: ActorParams
    actor_saccade: ActorParams
    vta: VTAParams
    world_model: WorldModelParams
    pfc: PFCParams

    # Medial-temporal memory system (Phase 5B): EC gathers
    # neocortical/PFC/motor streams into the perforant path, HC
    # performs DG-sparse encoding + CA3 pattern completion + CA1
    # mismatch comparison.  HC output routes back into the wake cycle
    # as a novelty/ACh boost and serves as the source of salience for
    # replay sampling during SWS.
    ec: EntorhinalParams
    hippocampus: HippocampusParams

    # Inter-region projections (L5 → CT drive, L5 → mossy fibre)
    w_l5_ct: Array              # (n_l5, n_ct)
    w_l5_mossy: Array           # (n_l5, mossy_size)

    # Wolpert 1998 efference copy: motor command (body \u2295 saccade one-hot)
    # routed directly into cerebellum mossy fibres for forward-model
    # learning.  Shape ``(n_body + n_saccade, mossy_size)``.
    w_efference_mossy: Array

    # Inferior-olive proxy: sensory PE → climbing fibres (Purkinje)
    w_io_pc: Array              # (sensory_size, n_purkinje)

    # Receptor densities along RECEPTOR_ORDER
    cortex_receptor_density: Array

    # Traveling-wave phase offsets (radians)
    phase_offset_thalamus: Array
    phase_offset_cortex: Array
    phase_offset_cerebellum: Array

    # Precision-tracking channel templates (Friston 2010 precision
    # weighting, applied to reward-like signals before actor RPE
    # composition). Each template carries an alpha = ctx.complement(τ)
    # derived from a biophysically-motivated timescale; actor bonuses
    # are then added to rpe after z-scoring (see
    # ``precision_standardize`` in :mod:`core.precision_bus`). The
    # templates sit in params because their ``alpha`` is a function
    # of ``ctx.dt`` (static across the sim); the running mean/var live
    # in the corresponding state fields.
    precision_r_ext_init: PrecisionChannel
    precision_curiosity_init: PrecisionChannel
    precision_info_gain_init: PrecisionChannel

    # Experience-replay buffer (Phase 5A): written each cognitive
    # cycle with the (s_t, a_t, r_t, s_{t+1}) transition just closed,
    # prioritised by the world-model's smoothed |PE| surprise signal
    # (wm_curiosity_signal — Schaul et al. 2016 prioritised experience
    # replay uses |TD error|; pe_short_abs is its EMA-smoothed analogue
    # and is bootstrap-safe unlike signed learning-progress).  Read by
    # Phase 5B's SWS reverse replay + REM rollout.
    replay: ReplayParams

    # Sleep-phase state machine (Phase 5A).  In Phase 5A the agent is
    # always awake in practice (sleep_step is invoked but no phase-
    # specific learning fires yet); the pytree fields exist so that
    # Phase 5B can drop in the SWS/REM dispatch without widening
    # ActionBrainState's signature again.
    sleep: SleepParams

    # Phase 6A: continuous motor substrate.
    # M1 is a learned linear readout on cortex L5 producing a bounded
    # joint command (Lemon 2008; Doya 2000).  Proprioception encoder
    # population-codes synthetic joint angles + velocities (Georgopoulos
    # 1986; Pouget & Sejnowski 1997).  Cerebellar motor-PE is projected
    # onto Purkinje cells via ``w_motor_pc`` (Wolpert 1998 internal
    # forward model) and the deep-nuclei output is projected back onto
    # the joint-command via ``w_dn_motor`` for additive M1 correction.
    # When ``bypass_m1`` is True (default, Phase 6A regression-safe),
    # the entire continuous path is skipped and the body receives the
    # BG discrete action unchanged — pre-6A bit-identical behaviour.
    m1: "M1Params"
    proprio: "ProprioceptionParams"
    w_motor_pc: Array              # (n_proprio_enc, n_purkinje)
    w_dn_motor: Array              # (n_dn, motor_dim)

    # Static
    n_body_actions: int = eqx.field(static=True)
    n_saccade_actions: int = eqx.field(static=True)
    sensory_size: int = eqx.field(static=True)
    substeps: int = eqx.field(static=True)
    delay_ct_steps: int = eqx.field(static=True)
    delay_mossy_steps: int = eqx.field(static=True)
    bypass_m1: bool = eqx.field(static=True, default=True)
    # Phase 6B: when True, the motor-PE block treats
    # ``state.last_joint_angles`` / ``last_joint_velocities`` as the
    # real (driver-supplied) proprioceptive state rather than applying
    # a synthetic ±0.1 rad delta derived from the discrete action id
    # (Phase 6A placeholder).  The MJX driver is expected to overwrite
    # those two fields with normalised real ``qpos`` / ``qvel`` after
    # each ``body.act_continuous``; for discrete-action envs the flag
    # stays False and behaviour is bit-identical to Phase 6A.
    use_real_proprio: bool = eqx.field(static=True, default=False)


class ActionBrainState(eqx.Module):
    """Core region state + dual-loop BG + VTA/WM + decision bookkeeping.

    :attr:`prev_pe_rate` is the V1 L2/3 prediction-error rate from the
    previous decision cycle, used to compute the saccade info-gain
    ``relu(prev_pe_rate - curr_pe_rate)`` (Itti & Baldi 2009).
    """

    # Sensory front-end state (foveated retina → LGN → V1).
    sensory_stack: SensoryStackState
    prev_pe_rate: Array              # scalar — last V1 L2/3 mean error rate

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
    pfc: PFCState
    attention: AttentionState

    # Medial-temporal memory system (Phase 5B).
    ec: EntorhinalState
    hippocampus: HippocampusState

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
    last_info_gain: Array            # scalar — saccade info-gain, diagnostic

    # Online precision channels for reward-composition scale
    # invariance (Friston 2010 precision; Oudeyer 2007 IAC).
    # Updated each decision cycle with the observed scalar; consumed
    # by z-scoring the epistemic bonuses added to actor RPE.
    precision_r_ext: PrecisionChannel
    precision_curiosity: PrecisionChannel
    precision_info_gain: PrecisionChannel

    # Experience-replay ring buffer (Phase 5A).  Stored each cognitive
    # cycle with the transition that was just closed.
    replay: ReplayState

    # Sleep-phase state machine (Phase 5A).  Updated each cognitive
    # cycle from the cortical ATP mean (sleep pressure).
    sleep: SleepState

    # Phase 6A: continuous motor substrate state.
    m1: M1State
    last_joint_angles: Array        # (n_joints,) synthetic proprio angles (t)
    last_joint_velocities: Array    # (n_joints,) synthetic proprio velocities (t)
    last_predicted_joint_angles: Array  # (n_joints,) M1 predicted angles for PE


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


def init_default_sensory_stack_params(ctx: BackendContext, sensory_size: int) -> SensoryStackParams:
    from sensory.retina import RetinaConfig
    from sensory.sensory_stack import init_sensory_stack_params
    return init_sensory_stack_params(
        ctx,
        retina_cfg=RetinaConfig(
            fovea_size=4, n_pyramid=1, periphery_tile=2,
        ),
        n_l4=max(sensory_size, 4),
        n_l23_state=max(sensory_size // 2, 4),
        n_l23_error=max(sensory_size // 4, 4),
        n_l5=max(sensory_size // 4, 4),
        )
        

def init_action_brain_params(
    ctx: BackendContext,
    *,
    sensory_size: int,
    n_body_actions: int,
    n_saccade_actions: int = SACCADE_ACTION_DIM,
    sensory_stack_params: SensoryStackParams | None = None,
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
    seed: int = 0,
    # replay buffer (Phase 5A)
    replay_capacity: int = 10_000,
    # sleep (Phase 5A) — biophysically calibrated defaults live in
    # ``init_sleep_params``; expose a single override knob here for
    # tests that need to force a transition without long simulations.
    sleep_params: SleepParams | None = None,
    # Phase 6A continuous motor substrate
    n_joints: int = 2,
    n_cells_per_joint: int = 16,
    m1_readout_lr: float = 1e-3,
    m1_cb_alpha: float = 0.2,
    bypass_m1: bool = True,
    # Phase 6B: real-proprioception wiring (see ``use_real_proprio``
    # docstring on ``ActionBrainParams``).  Default False keeps every
    # discrete-action env / existing test bit-identical.
    use_real_proprio: bool = False,
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

    The effective sensory afferent for thalamus / striatum / world-model
    is the V1 L4 rate population (Hubel & Wiesel 1962 simple-cell layer;
    corticostriatal projection — Smith 2004; Reiner 2010).
    ``sensory_size`` is overridden by ``sensory_stack_params.v1.n_l4``.
    """
    if sensory_stack_params is None:
        sensory_stack_params = init_default_sensory_stack_params(ctx, sensory_size)
    ss_sensory_size = int(sensory_stack_params.v1.n_l4)
    if sensory_size != ss_sensory_size:
        import warnings
        warnings.warn(
            f"sensory_size={sensory_size} overridden to "
            f"{ss_sensory_size} by sensory_stack_params.v1.n_l4",
            stacklevel=2,
        )
    sensory_size = ss_sensory_size
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
    rcp = init_receptor_params()
    attn_p = init_attention_params(ctx)

    # PFC: persistent content attractor biasing BG (Frank & Badre 2012).
    # Reads cortex L2/3 belief; output rate projected to striatum.
    pfc_p = init_pfc_params(ctx, input_size=cortex_n_l23_state)

    state_size = cortex_n_l23_state + sensory_size + pfc_p.n_content
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
    k1, k2, k3, k4 = split_key(master, 4)
    w_l5_ct = jnp.abs(
        jax.random.normal(k1, (cortex_n_l5, n_ct), dtype=DTYPE)
    ) * jnp.asarray(w_l5_ct_mean, DTYPE)
    w_l5_mossy = jnp.abs(
        jax.random.normal(k2, (cortex_n_l5, mossy_size), dtype=DTYPE)
    ) * jnp.asarray(w_l5_mossy_mean, DTYPE)
    w_io_pc = jax.random.normal(
        k3, (sensory_size, cerebellum_n_purkinje), dtype=DTYPE,
    ) * jnp.asarray(w_io_pc_sigma, DTYPE)
    # Wolpert 1998: efference copy projection onto mossy fibres.
    # Sparse-positive random (same scale as L5\u2192mossy) so that one-hot
    # motor commands inject a bounded afferent pulse without dominating
    # the cortical drive.  Shape: (body + saccade, mossy_size).
    n_efference = int(n_body_actions + n_saccade_actions)
    w_efference_mossy = jnp.abs(
        jax.random.normal(k4, (n_efference, mossy_size), dtype=DTYPE)
    ) * jnp.asarray(w_l5_mossy_mean, DTYPE)

    d_ct = max(1, int(round(delay_ct_ms / ctx.dt)))
    d_mossy = max(1, int(round(delay_mossy_ms / ctx.dt)))

    f = lambda x: jnp.asarray(x, DTYPE)
    cortex_density = jnp.asarray(CORTEX_DEFAULT_DENSITY, DTYPE)
    # Precision-bus channel templates. τ = 10 s (10 000 dt at dt = 1 ms)
    # matches D2 autoreceptor desensitisation (Benoit-Marand 2011) —
    # the slowest homeostatic timescale in the DA loop, so the
    # precision estimate smooths single-trial noise while still
    # tracking task-level drift.
    pc_r_ext_init = init_precision_channel(ctx, tau_ms=10_000.0)
    pc_curiosity_init = init_precision_channel(ctx, tau_ms=10_000.0)
    pc_info_gain_init = init_precision_channel(ctx, tau_ms=10_000.0)

    # Replay buffer template (Phase 5A).  State size = flattened
    # sensory afferent (V1 L4 rate) — the same vector fed to thalamus
    # / striatum / world_model — so replay_store can persist the
    # exact input the policy conditioned on.  Capacity 10 000 ≈ 10 s
    # of wake at dt=1 ms × 1 decision/cycle, or ~3 min at 20 ms/cycle;
    # either way sufficient for one HC-consolidation session
    # (Stickgold 2013).
    replay_p = init_replay_params(
        capacity=int(replay_capacity),
        state_size=int(sensory_size),
    )
    sleep_p = sleep_params if sleep_params is not None else init_sleep_params()

    # --- Medial-temporal memory system (Phase 5B) ------------------
    # EC afferents = [cortex_l23_belief, pfc_content, body⊕saccade].
    # Size the single EC cortical microcircuit so its output width
    # (``n_l23_state``) matches the HC input vocabulary exactly.
    n_motor = int(n_body_actions + n_saccade_actions)
    ec_p = init_ec_params(
        ctx,
        n_sensory=int(cortex_n_l23_state),
        n_pfc=int(pfc_p.n_content),
        n_motor=n_motor,
    )
    hc_p = init_hippocampus_params(input_dim=int(ec_p.output_dim))

    # Phase 6A: continuous motor substrate (M1 + proprio + cerebellar
    # motor-PE projections).  M1 readout head drives motor_dim =
    # n_joints channels into [-1, 1]; proprioception encoder Gaussian-
    # codes angles+velocities; fixed random projections w_motor_pc /
    # w_dn_motor close the efference-copy loop through cerebellum.
    from sensory.proprioception import (
        init_proprioception_params, proprio_output_dim,
    )
    motor_dim = int(n_joints)
    m1_p = init_m1_params(
        n_l5=int(cortex_n_l5), motor_dim=motor_dim,
        readout_lr=float(m1_readout_lr),
        cb_alpha=float(m1_cb_alpha),
    )
    proprio_p = init_proprioception_params(
        n_joints=int(n_joints),
        n_cells_per_joint=int(n_cells_per_joint),
    )
    n_proprio_enc = proprio_output_dim(proprio_p)
    k_motor_pc, k_dn_motor = split_key(jax.random.PRNGKey(seed + 1), 2)
    # Proprioception PE → Purkinje climbing (inferior-olive motor
    # channel; Wolpert 1998).  Same scale as ``w_io_pc`` for matched
    # climbing-fibre magnitudes.
    w_motor_pc = jax.random.normal(
        k_motor_pc, (n_proprio_enc, cerebellum_n_purkinje), dtype=DTYPE,
    ) * jnp.asarray(w_io_pc_sigma, DTYPE)
    # Deep nuclei → motor correction vector (half-normal positive; DN
    # output is disinhibition-driven excitatory on motor structures).
    w_dn_motor = jnp.abs(
        jax.random.normal(
            k_dn_motor, (cb_p.n_dn, motor_dim), dtype=DTYPE,
        )
    ) / jnp.sqrt(jnp.asarray(cb_p.n_dn, DTYPE))
    return ActionBrainParams(
        sensory_stack=sensory_stack_params,
        thalamus_relay=tr_p, thalamus_trn=trn_p,
        cortex=cx_p, cerebellum=cb_p,
        oscillator=osc_p, neuromodulator=nm_p, receptor=rcp,
        attention=attn_p,
        critic=critic_p,
        actor_body=actor_body_p,
        actor_saccade=actor_saccade_p,
        vta=vta_p, world_model=wm_p,
        pfc=pfc_p,
        ec=ec_p, hippocampus=hc_p,
        w_l5_ct=w_l5_ct, w_l5_mossy=w_l5_mossy, w_io_pc=w_io_pc,
        w_efference_mossy=w_efference_mossy,
        cortex_receptor_density=cortex_density,
        phase_offset_thalamus=f(phase_offsets_rad[0]),
        phase_offset_cortex=f(phase_offsets_rad[1]),
        phase_offset_cerebellum=f(phase_offsets_rad[2]),
        precision_r_ext_init=pc_r_ext_init,
        precision_curiosity_init=pc_curiosity_init,
        precision_info_gain_init=pc_info_gain_init,
        replay=replay_p,
        sleep=sleep_p,
        m1=m1_p,
        proprio=proprio_p,
        w_motor_pc=w_motor_pc,
        w_dn_motor=w_dn_motor,
        n_body_actions=int(n_body_actions),
        n_saccade_actions=int(n_saccade_actions),
        sensory_size=int(sensory_size),
        substeps=int(substeps),
        delay_ct_steps=d_ct, delay_mossy_steps=d_mossy,
        bypass_m1=bool(bypass_m1),
        use_real_proprio=bool(use_real_proprio),
    )


def init_action_brain_state(
    key: PRNGKey, params: ActionBrainParams, *, dtype=DTYPE,
) -> ActionBrainState:
    keys = split_key(key, 16)
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
    pfc_s = init_pfc_state(keys[9], params.pfc)
    attn_s = init_attention_state(
        keys[10],
        n_assoc=params.cortex.n_l5,
        n_columns=params.thalamus_relay.n_tc,
    )

    from sensory.sensory_stack import init_sensory_stack_state
    ss_s = init_sensory_stack_state(keys[11], params.sensory_stack)
    replay_s = init_replay_state(params.replay, dtype=dtype)
    sleep_s = init_sleep_state(keys[12], initial_phase=SleepPhase.WAKE)
    ec_s = init_ec_state(keys[13], params.ec, dtype=dtype)
    hc_s = init_hippocampus_state(keys[14], params.hippocampus, dtype=dtype)
    m1_s = init_m1_state(keys[15], params.m1)

    n_ct = params.thalamus_relay.n_ct
    mossy_size = params.cerebellum.mossy_size
    n_joints = int(params.proprio.n_joints)
    zero_j = jnp.zeros(n_joints, dtype)
    return ActionBrainState(
        sensory_stack=ss_s,
        prev_pe_rate=jnp.asarray(0.0, dtype),
        thalamus_relay=tr_s, thalamus_trn=trn_s,
        cortex=cx_s, cerebellum=cb_s,
        oscillator=osc_s, neuromodulator=nm_s,
        critic=critic_s,
        actor_body=actor_body_s,
        actor_saccade=actor_saccade_s,
        vta=vta_s, world_model=wm_s,
        pfc=pfc_s,
        attention=attn_s,
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
        last_info_gain=jnp.asarray(0.0, dtype),
        precision_r_ext=params.precision_r_ext_init,
        precision_curiosity=params.precision_curiosity_init,
        precision_info_gain=params.precision_info_gain_init,
        replay=replay_s,
        sleep=sleep_s,
        ec=ec_s,
        hippocampus=hc_s,
        m1=m1_s,
        last_joint_angles=zero_j,
        last_joint_velocities=zero_j,
        last_predicted_joint_angles=zero_j,
    )


# ---------------------------------------------------------------------
# Internal: single ``dt`` of the perception cortex loop with BG/WM/critic
# ---------------------------------------------------------------------


@eqx.filter_jit
def _perceive_substep(
    state: ActionBrainState,
    params: ActionBrainParams,
    ctx: BackendContext,
    sensory: Array,
    td_error: Array,
    novelty: Array,
    key: PRNGKey,
    sws_mode: Array,
) -> tuple[ActionBrainState, tuple[Array, Array, Array, Array]]:
    """One ``dt`` of perception: oscillator, thalamus, cortex, cerebellum,
    critic, actor, world-model prediction, neuromodulator.

    ``sws_mode`` is a bool scalar routed from the sleep-phase state
    machine (``sleep.phase == SWS``). During SWS the oscillator
    switches to the ~1 Hz slow-wave clamp and suppresses gamma
    (Steriade 1993); downstream regions see this through the
    oscillator output.  During wake or REM it is ``False`` and the
    normal theta/gamma bus applies.

    Returns the advanced state plus per-substep readouts
    ``(cortex_belief, cortex_l5_rate, relay_spikes, attn_gains)``
    so the scanning caller can expose them without peeking at internal
    region fields.
    """
    # ---- 1. Oscillator (depends on PREVIOUS step's NE / 5-HT) ----
    new_osc, _osc_out = oscillator_step(
        state.oscillator, params.oscillator, ctx,
        ne_level=state.neuromodulator.noradrenaline,
        sero_level=state.neuromodulator.serotonin,
        sws_mode=sws_mode,
    )

    # ---- 2. Pop delayed signals ----
    delay_ct_new, ct_delayed = delay_push_pop(
        state.delay_ct,
        jnp.zeros((params.thalamus_relay.n_ct,), DTYPE),
    )
    delay_mossy_new, mossy_delayed = delay_push_pop(
        state.delay_mossy,
        jnp.zeros((params.cerebellum.mossy_size,), DTYPE),
    )

    # ---- 3. Thalamus ----
    # Top-down attention gate (Saalmann 2012; Reynolds & Heeger 2009).
    # Bottom-up saliency = L2/3 prediction error rate (Itti & Koch 2001;
    # Feldman & Friston 2010).
    attn_out = attention_step(
        state.attention, params.attention,
        assoc_activity=state.cortex.rate_l5,
        bottom_up_errors=state.cortex.l23.error_rate,
        global_ach=state.neuromodulator.acetylcholine,
        ne_level=state.neuromodulator.noradrenaline,
    )

    thal = thalamic_step(
        state.thalamus_relay, params.thalamus_relay,
        state.thalamus_trn, params.thalamus_trn,
        ctx, sensory, ct_delayed,
        ach=state.neuromodulator.acetylcholine,
        ne=state.neuromodulator.noradrenaline,
        afferent_gain=attn_out.gains,
    )

    # ---- 4. Cortex ----
    cortex_gain = _region_receptor_gain(
        params.receptor, state.neuromodulator, params.neuromodulator,
        params.cortex_receptor_density,
    )
    theta_exc = jnp.asarray(1.0, DTYPE) + jnp.asarray(0.1, DTYPE) * jnp.cos(
        new_osc.theta_phase + params.phase_offset_cortex
    )
    cx_inputs = CorticalInputs(
        ff_input=thal.relay_spikes,
        td_prediction=None,
        ach=state.neuromodulator.acetylcholine,
        da=state.neuromodulator.dopamine,
        ne=state.neuromodulator.noradrenaline,
        receptor_gain=cortex_gain,
        excitability_mod=theta_exc,
    )
    cx_out = cortical_area_step(
        state.cortex, params.cortex, ctx, cx_inputs,
        apply_ipool_stdp=True,
    )

    # ---- 5. Cerebellum (no climbing signal during perceive) ----
    cb_out = cerebellum_step(
        state.cerebellum, params.cerebellum, ctx, mossy_delayed,
    )

    # ---- 6. Delay buffer push (overwrite the slot just popped) ----
    l5_new_ct_drive = cx_out.l5_spikes @ params.w_l5_ct
    l5_new_mossy_drive = cx_out.l5_rate @ params.w_l5_mossy
    efference_copy = jnp.concatenate(
        [state.last_body_action, state.last_saccade_action], axis=0,
    )
    l5_new_mossy_drive = l5_new_mossy_drive + (
        efference_copy.astype(DTYPE) @ params.w_efference_mossy.astype(DTYPE)
    )
    pre_head_ct = state.delay_ct.head
    pre_head_mossy = state.delay_mossy.head
    delay_ct_final = eqx.tree_at(
        lambda db: db.buf, delay_ct_new,
        delay_ct_new.buf.at[pre_head_ct].set(l5_new_ct_drive.astype(DTYPE)),
    )
    delay_mossy_final = eqx.tree_at(
        lambda db: db.buf, delay_mossy_new,
        delay_mossy_new.buf.at[pre_head_mossy].set(l5_new_mossy_drive.astype(DTYPE)),
    )

    # ---- 7. Neuromodulator ----
    pe_signal = cx_out.ff_out
    nm_new = neuromodulator_step(
        state.neuromodulator, params.neuromodulator,
        prediction_error=pe_signal,
        td_error=td_error,
        reward=jnp.asarray(0.0, DTYPE),
        novelty=novelty,
    )

    # ---- 8. PFC (Frank & Badre 2012; Hasselmo 2005) ----
    k_pfc, _ = split_key(key, 2)
    pfc_out = pfc_step(
        state.pfc, params.pfc, ctx,
        cortex_belief=cx_out.belief,
        ach=nm_new.acetylcholine,
        da=nm_new.dopamine,
        key=k_pfc,
        theta_phase=new_osc.theta_phase,
        phase_offset=params.phase_offset_cortex,
    )

    # ---- 9. BG: critic + parallel actor loops ----
    striatal_drive = jnp.concatenate(
        [cx_out.belief, sensory, pfc_out.content_rate], axis=0,
    )
    critic_out = critic_step(
        state.critic, params.critic, ctx, striatal_drive,
    )
    actor_inputs = ActorInputs(
        da=nm_new.dopamine,
        tonic_da=nm_new.tonic_da,
        epistemic_drive=novelty,
        gamma_amp=new_osc.gamma_amplitude,
    )
    body_out = actor_step(
        state.actor_body, params.actor_body, ctx, striatal_drive,
        inputs=actor_inputs,
    )
    saccade_out = actor_step(
        state.actor_saccade, params.actor_saccade, ctx, striatal_drive,
        inputs=actor_inputs,
    )

    # ---- 10. World-model forward prediction ----
    joint_last_action = jnp.concatenate(
        [state.last_body_action, state.last_saccade_action], axis=0,
    )
    wm_out = wm_predict(
        state.world_model, params.world_model, ctx,
        sensory, joint_last_action,
        ach=nm_new.acetylcholine,
    )

    return ActionBrainState(
        sensory_stack=state.sensory_stack,
        prev_pe_rate=state.prev_pe_rate,
        thalamus_relay=thal.relay,
        thalamus_trn=thal.trn,
        cortex=cx_out.state,
        cerebellum=cb_out.state,
        oscillator=new_osc,
        neuromodulator=nm_new,
        critic=critic_out.state,
        actor_body=body_out.state,
        actor_saccade=saccade_out.state,
        vta=state.vta,
        world_model=wm_out.state,
        pfc=pfc_out.state,
        attention=attn_out.state,
        delay_ct=delay_ct_final,
        delay_mossy=delay_mossy_final,
        last_body_action=state.last_body_action,
        last_body_action_id=state.last_body_action_id,
        last_saccade_action=state.last_saccade_action,
        last_saccade_action_id=state.last_saccade_action_id,
        last_sensory=state.last_sensory,
        last_rpe=state.last_rpe,
        last_total_reward=state.last_total_reward,
        last_info_gain=state.last_info_gain,
        precision_r_ext=state.precision_r_ext,
        precision_curiosity=state.precision_curiosity,
        precision_info_gain=state.precision_info_gain,
        replay=state.replay,
        sleep=state.sleep,
        ec=state.ec,
        hippocampus=state.hippocampus,
        m1=state.m1,
        last_joint_angles=state.last_joint_angles,
        last_joint_velocities=state.last_joint_velocities,
        last_predicted_joint_angles=state.last_predicted_joint_angles,
    ), (cx_out.belief, cx_out.l5_rate, thal.relay_spikes, attn_out.gains)


@eqx.filter_jit
def _perceive_scan_block(
    state: ActionBrainState,
    params: ActionBrainParams,
    ctx: BackendContext,
    sensory: Array,
    prev_rpe: Array,
    prev_curiosity: Array,
    sws_flag: Array,
    substep_keys: Array,
) -> tuple[ActionBrainState, tuple[Array, Array, Array, Array]]:
    """Run ``params.substeps`` perception substeps via ``jax.lax.scan``.

    Factored out of ``action_brain_cognitive_step`` so that the scan
    body is compiled ONCE (cached by ``@eqx.filter_jit``) and reused
    across subsequent calls — critical when callers dispatch the
    cognitive step from a Python loop (e.g. bandit rollouts,
    gridworld episodes).  Without this helper, every call to the
    surrounding ``action_brain_cognitive_step`` (which is itself NOT
    jitted) would re-trace and re-compile the whole scan program,
    producing 10-to-100× compile overhead.
    """
    def _scan_body(carry_state, k_sub):
        new_state, readouts = _perceive_substep(
            carry_state, params, ctx, sensory,
            td_error=prev_rpe, novelty=prev_curiosity,
            key=k_sub, sws_mode=sws_flag,
        )
        return new_state, readouts

    state, readouts_stacked = jax.lax.scan(
        _scan_body, state, substep_keys,
    )
    last_readouts = jax.tree_util.tree_map(
        lambda x: x[-1], readouts_stacked,
    )
    return state, last_readouts


@eqx.filter_jit
def _sensory_scan_block(
    ss_state,
    ss_params,
    ctx: BackendContext,
    image: Array,
    fixation_xy: Array,
    ach: Array,
    da: Array,
    ne: Array,
    length: int,
) -> tuple[Any, tuple[Array, Array]]:
    """Run ``length`` sensory-stack substeps via ``jax.lax.scan``.

    Same rationale as ``_perceive_scan_block``: cache the compiled
    scan across Python-loop callers.
    """
    from sensory.sensory_stack import sensory_stack_step

    def _scan_body(carry_ss, _):
        o = sensory_stack_step(
            carry_ss, ss_params, ctx,
            image, fixation_xy,
            ach=ach, da=da, ne=ne, apply_ipool_stdp=True,
        )
        return o.state, (o.l4_rate, o.pe_rate)

    return jax.lax.scan(_scan_body, ss_state, None, length=length)


# ---------------------------------------------------------------------
# Top-level: perceive→learn→perceive cycle (one body decision)
# ---------------------------------------------------------------------


def action_brain_cognitive_step(
    state: ActionBrainState,
    params: ActionBrainParams,
    ctx: BackendContext,
    sensory: Array,
    prev_reward: float | Array = 0.0,
    prev_done: float | Array = 0.0,
    key: PRNGKey | None = None,
    *,
    info_gain: float | Array | None = None,
) -> ActionBrainOutput:
    """One decision cycle using a pre-computed sensory representation.

    This is the cognitive core of the brain: perceive → learn → act.
    Use this when the sensory input is already available as a flat
    array (e.g. from a non-visual body or for targeted subsystem
    tests).  The full visual pipeline version is
    :func:`action_brain_step`.

    Each ``@eqx.filter_jit``-decorated leaf function compiles once and
    is cached; the Python loop dispatches individual calls, keeping
    each XLA compilation bounded by one subsystem's graph size.
    """
    if key is None:
        raise ValueError(
            "action_brain_cognitive_step requires an explicit PRNG key"
        )

    if info_gain is None:
        info_gain = jnp.asarray(0.0, DTYPE)

    r_ext = jnp.asarray(prev_reward, DTYPE)
    done = jnp.asarray(prev_done, DTYPE)

    # --- 0a. Sleep-phase state machine (Phase 5B).
    #     Sleep pressure is driven by an adenosine Process-S integrator
    #     (Porkka-Heiskanen 1997; Achermann & Borbély 1992): extracellular
    #     adenosine accumulates monotonically during sustained wake and
    #     clears exponentially during NREM.  We advance the integrator
    #     here once per cognitive cycle and feed ``energy = 1 - adenosine``
    #     to ``sleep_step`` in place of the earlier raw-ATP proxy.  The
    #     raw ATP mean from cortical astrocytes is a fast metabolic
    #     signal (t ~ 200 s) that reaches a wake equilibrium well below
    #     the sws threshold and would therefore force immediate sleep;
    #     adenosine is the slow, biologically-correct VLPO drive
    #     (Saper 2010).  After the update ``sws_flag`` is the live
    #     ``is_sws`` flag — the oscillator's SWS clamp now engages
    #     during NREM (Steriade 1993).
    awake_now = is_wake(state.sleep)
    nm_with_aden = adenosine_update(
        state.neuromodulator, params.neuromodulator, is_awake=awake_now,
    )
    state = eqx.tree_at(lambda s: s.neuromodulator, state, nm_with_aden)
    energy = jnp.asarray(1.0, DTYPE) - nm_with_aden.adenosine
    new_sleep = sleep_step(state.sleep, params.sleep, ctx, energy)
    sws_flag = is_sws(new_sleep)

    # --- 0b. Conditional PFC reset on episode boundary.  Biologically:
    # when a task episode ends, PFC abandons the current goal/context
    # attractor and returns to baseline (Fuster 2001).  JAX-safe via
    # elementwise ``where`` — no Python branch.
    state = eqx.tree_at(
        lambda s: s.pfc, state,
        pfc_select_reset(state.pfc, params.pfc, done),
    )

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

    # Split keys: one per perception substep (PFC gate needs membrane
    # noise each dt), plus two for action selection at the end and
    # one for M1 exploration noise (Phase 6B node-perturbation
    # REINFORCE; Williams 1992 / Tumer & Brainard 2007).
    k_scan, k_body, k_saccade, k_m1 = split_key(key, 4)
    substep_keys = jax.random.split(k_scan, params.substeps)

    # ``jax.lax.scan`` over substeps: the body function is traced ONCE
    # and iterated ``substeps`` times inside XLA.  A prior Python
    # ``for`` loop unrolled the entire perceive graph (cortex 6L +
    # thalamus + BG×2 + PFC + WM + neuromod + ACh + attention) by
    # ``substeps`` copies, which blew up XLA compile time on GPU.
    def _scan_perceive(carry_state, k_sub):
        new_state, readouts = _perceive_substep(
            carry_state, params, ctx, sensory,
            td_error=prev_rpe, novelty=prev_curiosity,
            key=k_sub,
            sws_mode=sws_flag,
        )
        return new_state, readouts

    state, readouts_stacked = jax.lax.scan(
        _scan_perceive, state, substep_keys,
    )
    last_readouts = jax.tree_util.tree_map(
        lambda x: x[-1], readouts_stacked,
    )

    cortex_belief, cortex_l5_rate, relay_spikes, last_attn_gains = last_readouts

    # --- 1b. Medial-temporal memory system (Phase 5B).
    #     EC gathers neocortex belief + PFC goal + last motor command
    #     (Witter 2007 perforant-path source) and projects the
    #     concatenated afferent through its canonical microcircuit; HC
    #     (DG → CA3 → CA1) then performs one-shot episodic encoding
    #     and pattern-completed recall under the current theta phase.
    #     The CA1 mismatch scalar drives a basal-forebrain cholinergic
    #     release (McGaughy 2008) that boosts the global ACh level and
    #     therefore the next cycle's cortical plasticity gain (Hasselmo
    #     2006).  HC runs ONCE per cognitive cycle, not per substep —
    #     information-theoretically the perforant-path carries one
    #     theta-indexed snapshot per decision window (Hasselmo 2005),
    #     which is exactly the resolution we sample at.
    joint_last_action_hc = jnp.concatenate(
        [state.last_body_action, state.last_saccade_action], axis=0,
    )
    ec_state, ec_belief = ec_step(
        state.ec, params.ec, ctx,
        sensory_belief=cortex_belief,
        pfc_content=state.pfc.output_rate,
        last_motor=joint_last_action_hc,
        ach=state.neuromodulator.acetylcholine,
        da=state.neuromodulator.dopamine,
    )
    hc_out: HippocampusOutput = hippocampus_step(
        state.hippocampus, params.hippocampus,
        ec_in=ec_belief,
        theta_phase=state.oscillator.theta_phase,
        ne_level=state.neuromodulator.noradrenaline,
        reward=r_ext,
        action=state.last_body_action_id,
    )
    hc_state = hc_out.state
    # CA1 mismatch → basal-forebrain ACh release (McGaughy 2008).  The
    # ACh channel stays bounded to [0, 1] by the neuromodulator
    # homeostat on subsequent cycles; we simply add the mismatch and
    # clip.
    nm_boosted = eqx.tree_at(
        lambda s: s.acetylcholine, state.neuromodulator,
        jnp.clip(
            state.neuromodulator.acetylcholine + hc_out.mismatch,
            jnp.asarray(0.0, DTYPE), jnp.asarray(1.0, DTYPE),
        ),
    )
    state = eqx.tree_at(
        lambda s: (s.ec, s.hippocampus, s.neuromodulator),
        state, (ec_state, hc_state, nm_boosted),
    )

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
        n_substeps=params.substeps,
    )
    sensory_pe = wm_out.prediction_error                 # (sensory_size,)
    climbing_sensory = sensory_pe @ params.w_io_pc       # (n_purkinje,)

    # --- 2a.2 Phase 6A: synthetic proprioception + motor PE --------
    # Derive a pseudo-joint kinematics signal from the previously
    # committed body action (one-hot → signed direction per joint) so
    # the downstream M1 / cerebellum forward-model loop has an
    # exercised data path even while the current bodies remain
    # discrete (plan 6A.4).  Phase 6B (``use_real_proprio=True``)
    # swaps this for the driver-supplied real MJX joint sensors
    # already stored in ``state.last_joint_angles`` /
    # ``state.last_joint_velocities`` without changing consumer
    # wiring.
    prev_body_id = state.last_body_action_id
    proprio_p = params.proprio
    n_joints_int = int(proprio_p.n_joints)
    if params.use_real_proprio:
        # Phase 6B: the MJX driver wrote the normalised real qpos /
        # qvel from the previous cycle into these two fields right
        # after ``body.act_continuous``.  Use them directly as the
        # ``actual'' of motor PE (Wolpert 1998 forward-model error =
        # real − predicted).  No synthetic delta is added.
        new_angles = state.last_joint_angles
        new_velocities = state.last_joint_velocities
    else:
        # Map action id 0..2*n_joints into per-joint signed direction
        # (sign-split, the inverse of ``discretise_joint_command``).
        joint_idx = jnp.mod(prev_body_id, jnp.asarray(n_joints_int, jnp.int32))
        sign_bit = jnp.where(
            prev_body_id >= jnp.asarray(n_joints_int, jnp.int32),
            jnp.asarray(-1.0, DTYPE), jnp.asarray(1.0, DTYPE),
        )
        one_hot_joint = (
            jnp.arange(n_joints_int) == joint_idx
        ).astype(DTYPE)
        joint_direction = one_hot_joint * sign_bit           # (n_joints,)
        # Simple first-order kinematics: angles integrate direction,
        # velocities = direction scaled by dt-proxy.
        delta_angle = jnp.asarray(0.1, DTYPE) * joint_direction
        new_angles = jnp.clip(
            state.last_joint_angles + delta_angle,
            jnp.asarray(-1.0, DTYPE), jnp.asarray(1.0, DTYPE),
        )
        new_velocities = delta_angle
    from sensory.proprioception import proprio_encode as _proprio_encode
    proprio_actual_enc = _proprio_encode(
        proprio_p, new_angles, new_velocities,
    )
    # Predicted proprio comes from the cerebellar/M1 forward estimate
    # stored at the previous cycle (``last_predicted_joint_angles``).
    proprio_pred_enc = _proprio_encode(
        proprio_p, state.last_predicted_joint_angles, new_velocities,
    )
    motor_pe_vec = proprio_actual_enc - proprio_pred_enc
    climbing_motor = motor_pe_vec @ params.w_motor_pc    # (n_purkinje,)
    climbing = climbing_sensory + climbing_motor
    cb_state = cerebellum_update(
        state.cerebellum, params.cerebellum, climbing, modulator=1.0,
    )

    # 2b. Active-inference reward composition (Friston 2017; Dayan &
    #     Yu 2003; Schultz 1998).  In the dual-process decomposition,
    #     the VTA/critic TD loop approximates expected EXTRINSIC
    #     return V(s) = E[\u03a3 \u03b3\u1d57 r_ext], so ``r_total`` must be
    #     r_ext alone -- adding curiosity here would corrupt V(s) with
    #     expected-novelty, which converges to 0 after learning and
    #     leaves V(s) biased toward unlearned states (the classic
    #     curiosity-as-reward artefact; Pathak 2017 note 4).
    #
    #     Epistemic drive enters via FOUR independent channels, each
    #     anatomically segregated from the VTA/critic:
    #       (i)   curiosity \u2192 ACh/NE (novelty input to
    #             ``neuromodulator_step`` above): drives perception
    #             gain and arousal (Yu & Dayan 2005 uncertainty \u2192 ACh).
    #       (ii)  info_gain \u2192 saccade RPE only (below, step 2d):
    #             oculomotor-specific epistemic control per
    #             Itti & Baldi 2009 surprise \u2192 SC bias.
    #       (iii) curiosity \u2192 body-actor RPE only (below, step 2d):
    #             transition-surprise drives body exploration without
    #             corrupting V(s) (Friston 2017 EFE decomposition --
    #             epistemic value modulates policy selection, not
    #             state value).
    #       (iv)  curiosity \u2192 mental-rehearsal priority (replay
    #             buffer salience): off-policy WM refinement.
    #
    #     ATP homeostasis does NOT enter r_total -- it acts locally on
    #     neuron energetics through V_T shift and g_L gain (Morris
    #     2003; astrocyte.py).  Adding it as negative reward would
    #     double-count the metabolic constraint already enforced by
    #     the AdEx dynamics.
    curiosity = wm_curiosity_signal(wm_out.state, params.world_model)
    r_total = r_ext

    # 2b-bis. Precision-weighted additive composition of multi-source
    #   drives. Actors combine an extrinsic-reward RPE with intrinsic
    #   epistemic bonuses that live in DIFFERENT natural units
    #   (hedonic reward vs information gain vs prediction surprise).
    #   The VTA's auto_rms normalises the critic RPE path only; naive
    #   addition `rpe + curiosity` therefore lets whichever signal
    #   happens to have larger raw scale swamp the others, which is
    #   precisely the scale-mismatch bug the plan identifies.
    #
    #   Fix: standardise each bonus by its own running variance
    #   (precision_bus.PrecisionChannel, Welford EMA over τ = 10 s)
    #   BEFORE adding to RPE. This is the Friston-Feldman (2010)
    #   precision-weighted prediction-error rule applied to the
    #   reward composition itself:
    #       r_eff = rpe_norm  +  Σ (xᵢ − μᵢ) / √(σᵢ² + ε)
    #   Each bonus then contributes a unit-variance additive signal,
    #   regardless of its raw scale. IVW averaging is NOT used here
    #   because the channels carry different physical quantities
    #   (avg-ing a reward with an info-gain is meaningless); IVW is
    #   reserved for same-quantity multi-sensor fusion
    #   (precision_bus.precision_compose).
    ig = jnp.asarray(info_gain, DTYPE)
    pc_r_ext_new = precision_update(state.precision_r_ext, r_ext)
    pc_curiosity_new = precision_update(state.precision_curiosity, curiosity)
    pc_info_gain_new = precision_update(state.precision_info_gain, ig)
    curiosity_z = precision_standardize(pc_curiosity_new, curiosity)
    info_gain_z = precision_standardize(pc_info_gain_new, ig)

    # 2c. TD(0) RPE: V(s_t)=state.vta.stored_v, V(s_{t+1})=critic.activation
    #     (just computed by the perceive loop above).
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
    critic_state = critic_learn_step(state.critic, params.critic, rpe)
    # Actor RPE receives a modality-specific epistemic bonus (the
    # critic RPE does NOT -- V(s) approximates pure expected
    # extrinsic return; Dayan & Yu 2003 dual-process decomposition):
    #   body actor    : + curiosity (transition surprise over the
    #                   world-model posterior, Friston 2017 EFE
    #                   epistemic component; z-scored so it is
    #                   unit-variance independent of raw scale).
    #   saccade actor : + info_gain (attention-specific surprise,
    #                   Itti & Baldi 2009).
    actor_body_state, actor_saccade_state = actors_learn_step(
        state.actor_body, params.actor_body,
        state.actor_saccade, params.actor_saccade,
        rpe=rpe,
        body_bonus=curiosity_z,
        saccade_bonus=info_gain_z,
    )

    # 2e. Cortical three-factor STDP: RPE modulates eligibility traces
    #     accumulated during the perception loop (Step 1 above).
    cortex_state = cortex_learn_step(
        state.cortex, params.cortex, modulator=rpe,
    )

    # 2f. Attention Hebbian learning: top-down weights adapt so that
    #     attended columns with high cortical activity are reinforced
    #     (Reynolds & Heeger 2009; Feldman & Friston 2010).
    attn_state = attention_learn_step(
        state.attention, params.attention,
        assoc_activity=cortex_l5_rate,
        column_mean_rates=cortex_belief,
        gains=last_attn_gains,
    )

    # 2g. Persist the (s_t, a_t, r_t, s_{t+1}) transition in the
    #     replay buffer.  Salience = wm_curiosity_signal
    #     (= pe_short_abs, EMA-smoothed |PE|).  This is the
    #     Schaul et al. (2016) |TD error|-prioritised replay rule
    #     applied at the surprise-EMA timescale — bootstrap-safe
    #     (unlike signed learning-progress which sign-flips when the
    #     long EMA has not caught up to the short one).  Exposes the
    #     transition to Phase 5B's SWS reverse-replay + REM rollout.
    exp = Experience(
        state=state.last_sensory,
        action=state.last_body_action_id,
        reward=r_ext,
        next_state=sensory.astype(DTYPE),
        prediction_error=jnp.abs(rpe),
        done=done,
        salience=curiosity,
        recorded_da=state.neuromodulator.dopamine,
    )
    new_replay = replay_store(state.replay, params.replay, exp)

    # --- 3. Store V(s_{t+1}) in VTA for next cycle's TD target. -----
    vta_state = vta_store_prediction(vta_state, critic_state.activation)

    # --- 4. Select a_{t+1} from each loop's evidence accumulated in
    #     the loop. Independent PRNG keys so tie-breaking in one loop
    #     does not correlate with the other.
    body_action_id = actor_select_action(
        actor_body_state, params.actor_body, k_body,
    )
    saccade_action_id = actor_select_action(
        actor_saccade_state, params.actor_saccade, k_saccade,
    )
    # Phase 6B credit-assignment fix: keep the actor's OWN selection
    # for eligibility commit even when M1 overrides the executed
    # action below.  Pre-fix bug: the body actor's eligibility was
    # committed for M1's discretised choice, so the BG actor was
    # being credited for actions it did not select → noise gradient.
    actor_body_action_id = body_action_id
    body_action_oh = (
        jnp.arange(params.n_body_actions) == body_action_id
    ).astype(DTYPE)
    saccade_action_oh = (
        jnp.arange(params.n_saccade_actions) == saccade_action_id
    ).astype(DTYPE)

    # --- 4a. Phase 6A M1 continuous head --------------------------
    # Bypass path (default, regression-safe): skip M1 entirely and
    # keep the discrete BG body action unchanged.  Active path: run
    # M1 on cortex L5 rate with cerebellar motor correction from the
    # deep nuclei (Wolpert 1998), learn the readout via
    # node-perturbation REINFORCE gated by VTA RPE (Williams 1992;
    # Fiete & Seung 2006), and override the EXECUTED body_action_id
    # with the sign-split argmax of the joint command (the BG actor
    # still trains on its own selection — see actor_body_action_id).
    if params.bypass_m1:
        m1_new_state = state.m1
        jc_out = state.m1.last_joint_command
        predicted_angles_next = state.last_predicted_joint_angles
    else:
        cb_correction = cb_state.dn_rate @ params.w_dn_motor    # (motor_dim,)
        m1_out = m1_step(
            state.m1, params.m1, cortex_l5_rate,
            key=k_m1,
            ne_level=state.neuromodulator.noradrenaline,
            cb_motor_correction=cb_correction,
        )
        jc_out = m1_out.joint_command
        # Override the BG-selected body action with the discretised
        # M1 command so the env still receives an int.  This is the
        # EXECUTED action; the BG actor's eligibility credit stays on
        # its own selection (actor_body_action_id, captured above).
        from embodiment.body_interface import (
            discretise_joint_command as _disc,
        )
        body_action_id = _disc(jc_out, params.n_body_actions)
        body_action_oh = (
            jnp.arange(params.n_body_actions) == body_action_id
        ).astype(DTYPE)
        # Node-perturbation REINFORCE update on motor_readout: dw
        # correlates the injected exploration noise ξ with the RPE
        # produced by the action that committed to that ξ.  RPE here
        # is the CURRENT cycle's RPE (the action being trained was
        # committed by state.m1 last cycle; the noise / l5 cached on
        # state.m1 carry that information forward through the
        # one-cycle delay).
        m1_new_state = m1_learn_readout(
            state.m1, params.m1,
            rpe=rpe,
            l5_rate_normalised=None,        # use state.m1.last_l5_rate
            exploration_noise=None,         # use state.m1.last_exploration_noise
            cb_motor_err=None,
        )
        # Stash THIS cycle's exploration noise + L5 rate onto the
        # learned-state object so next cycle's update can reach them.
        m1_new_state = eqx.tree_at(
            lambda s: (
                s.last_joint_command,
                s.last_exploration_noise,
                s.last_l5_rate,
            ),
            m1_new_state,
            (jc_out, m1_out.exploration_noise, m1_out.l5_rate_normalised),
        )
        # Forward-model prediction of joint angles for next-cycle
        # motor PE.  Phase 6B architectural fix (post-D8 audit):
        #
        # The body (``MjxArmBody.act_continuous``) treats ``jc`` as
        # a *normalised qpos setpoint* — ``ctrl = jc * joint_range``
        # is fed to a MuJoCo position actuator with PD servo.  Under
        # perfect tracking the next normalised qpos equals ``jc``;
        # the steady-state contract of the body interface is
        # ``q_norm → jc``.  The previous formulation
        # ``q + 0.1·jc`` treated ``jc`` as a *velocity* command
        # (matching the Phase-6A synthetic-proprio fallback at line
        # ~1417) — a structural type error against the real body.
        #
        # Empirical evidence (D8 in colab/phase6b_diag.ipynb): the
        # measured slope of real Δq on jc is 0.019 / 0.031 (R²≈0.04),
        # i.e. the velocity-proxy gain is wrong by ~5×; cerebellum
        # cannot learn its way around a constant ~80 % motor-PE bias
        # so dn_rate saturates and dominates ``jc_out`` via
        # ``α·tanh(cb)``.  By predicting ``jc_out`` directly we adopt
        # the steady-state body contract as the architectural prior;
        # the cerebellum forward model then learns the *deviation*
        # from perfect tracking — i.e. the servo-lag transient —
        # which is the canonical Wolpert (1998) forward-model error
        # signal carried by climbing fibres in the inferior olive.
        # No magic constants remain; the prediction has the same
        # units as the proprio encoder input (normalised qpos).
        predicted_angles_next = jnp.clip(
            jc_out,
            jnp.asarray(-1.0, DTYPE), jnp.asarray(1.0, DTYPE),
        )

    # --- 4b. Freeze eligibility traces + action one-hot at cycle end.
    # The live traces at this point represent the correlation
    # pre(s_t) × post(s_t) built during the perceive loop; combined
    # with the chosen action id they will be used by the next cycle's
    # actor_update / critic_update to apply the RPE for the
    # (s_t → s_{t+1}) transition to the weights that produced the
    # value/policy evaluation of s_t (Sutton & Barto 2018 §6.1, §12).
    # Use the BG actor's OWN selected action id (not M1's override)
    # so the eligibility × RPE update credits the action the actor
    # actually chose — Phase 6B credit-assignment fix.
    actor_body_state = actor_commit_eligibility(
        actor_body_state, params.actor_body, actor_body_action_id,
    )
    actor_saccade_state = actor_commit_eligibility(
        actor_saccade_state, params.actor_saccade, saccade_action_id,
    )
    critic_state = critic_commit_eligibility(critic_state)

    cerebellum_nuclei = cb_state.dn_rate

    new_state = ActionBrainState(
        sensory_stack=state.sensory_stack,
        prev_pe_rate=state.prev_pe_rate,
        thalamus_relay=state.thalamus_relay,
        thalamus_trn=state.thalamus_trn,
        cortex=cortex_state,
        cerebellum=cb_state,
        oscillator=state.oscillator,
        neuromodulator=state.neuromodulator,
        critic=critic_state,
        actor_body=actor_body_state,
        actor_saccade=actor_saccade_state,
        vta=vta_state,
        world_model=wm_out.state,
        pfc=state.pfc,
        attention=attn_state,
        delay_ct=state.delay_ct, delay_mossy=state.delay_mossy,
        last_body_action=body_action_oh,
        last_body_action_id=body_action_id,
        last_saccade_action=saccade_action_oh,
        last_saccade_action_id=saccade_action_id,
        last_sensory=sensory.astype(DTYPE),
        last_rpe=rpe,
        last_total_reward=r_total,
        last_info_gain=ig,
        precision_r_ext=pc_r_ext_new,
        precision_curiosity=pc_curiosity_new,
        precision_info_gain=pc_info_gain_new,
        replay=new_replay,
        sleep=new_sleep,
        ec=state.ec,
        hippocampus=state.hippocampus,
        m1=m1_new_state,
        last_joint_angles=new_angles,
        last_joint_velocities=new_velocities,
        last_predicted_joint_angles=predicted_angles_next,
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


def action_brain_step(
    state: ActionBrainState,
    params: ActionBrainParams,
    ctx: BackendContext,
    image: Array,
    fixation_xy: Array,
    prev_reward: float | Array = 0.0,
    prev_done: float | Array = 0.0,
    key: PRNGKey | None = None,
    *,
    info_gain: float | Array | None = None,
) -> ActionBrainOutput:
    """Run one full decision cycle on foveated input.

    The brain runs the retina -> LGN -> V1 chain itself (eye is CNS;
    Rodieck 1998), uses the V1 L4 rate as the ``sensory`` afferent for
    downstream regions, and computes ``info_gain = relu(prev_pe_rate -
    curr_pe_rate)`` (Itti & Baldi 2009 Bayesian surprise reduction).

    Delegates cognitive processing to :func:`action_brain_cognitive_step`
    after computing the sensory representation.

    Inputs:
      * ``image``        : ``(H, W)`` float32 in [0, 1].
      * ``fixation_xy``  : ``(2,)`` in [0, 1]^2.
      * ``prev_reward``  : ``r_t`` emitted during the prior transition.
      * ``prev_done``    : terminal flag for that transition.
      * ``key``          : PRNG for action sampling.
      * ``info_gain``    : explicit override; when ``None`` the value is
        computed from V1 PE delta.
    """
    if key is None:
        raise ValueError("action_brain_step requires an explicit PRNG key")

    # --- Sensory front-end: retina -> LGN -> V1 ---------------------------
    from sensory.sensory_stack import sensory_stack_step
    ach = state.neuromodulator.acetylcholine
    da = state.neuromodulator.dopamine
    ne = state.neuromodulator.noradrenaline

    ss_state = state.sensory_stack

    # ``jax.lax.scan`` over the sensory-stack substeps.  The body is
    # traced once; outputs ``l4_rate`` and ``pe_rate`` are stacked
    # along axis 0 by ``scan`` with shape ``(substeps, …)``.
    def _scan_sensory(carry_ss, _):
        o = sensory_stack_step(
            carry_ss, params.sensory_stack, ctx,
            image, fixation_xy,
            ach=ach, da=da, ne=ne, apply_ipool_stdp=True,
        )
        return o.state, (o.l4_rate, o.pe_rate)

    ss_state, (l4_stack, pe_stack) = jax.lax.scan(
        _scan_sensory, ss_state, None, length=params.substeps,
    )

    l4_window = jnp.mean(l4_stack, axis=0)
    l4_peak = jnp.max(l4_window)
    sensory = jnp.where(
        l4_peak > 1e-3,
        l4_window / (l4_peak + jnp.asarray(1e-6, DTYPE)),
        l4_window,
    )
    new_pe_rate = jnp.mean(pe_stack)
    computed_ig = jax.nn.relu(state.prev_pe_rate - new_pe_rate)
    if info_gain is None:
        info_gain = computed_ig

    # Update sensory_stack and prev_pe_rate into state before cognitive step
    state = eqx.tree_at(
        lambda s: (s.sensory_stack, s.prev_pe_rate),
        state,
        (ss_state, new_pe_rate),
    )

    return action_brain_cognitive_step(
        state, params, ctx, sensory,
        prev_reward=prev_reward,
        prev_done=prev_done,
        key=key,
        info_gain=info_gain,
    )


# =====================================================================
# Phase-5B sleep-aware top-level dispatcher
# =====================================================================


def _advance_sleep_bookkeeping(
    state: ActionBrainState,
    params: ActionBrainParams,
    ctx: BackendContext,
    *,
    awake: bool,
    sws: bool,
) -> ActionBrainState:
    """Off-wake adenosine/oscillator/sleep bookkeeping for SWS & REM.

    During wake, :func:`action_brain_cognitive_step` advances all of
    these internally on every cognitive cycle.  During sleep we step
    them ONCE per brain cycle (the offline loops are already iterating
    a mini-batch of transitions inside the ``scan``).
    """
    nm_with_aden = adenosine_update(
        state.neuromodulator, params.neuromodulator, is_awake=awake,
    )
    energy = jnp.asarray(1.0, DTYPE) - nm_with_aden.adenosine
    new_sleep = sleep_step(state.sleep, params.sleep, ctx, energy)
    new_osc, _osc_out = oscillator_step(
        state.oscillator, params.oscillator, ctx,
        ne_level=nm_with_aden.noradrenaline,
        sero_level=nm_with_aden.serotonin,
        sws_mode=sws,
    )
    return eqx.tree_at(
        lambda s: (s.neuromodulator, s.sleep, s.oscillator),
        state,
        (nm_with_aden, new_sleep, new_osc),
    )


def _sleep_phase_output(
    state: ActionBrainState, *, cortex_belief_size: int, l5_size: int,
    dn_size: int, relay_size: int,
) -> ActionBrainOutput:
    """Zero-valued :class:`ActionBrainOutput` used during offline sleep.

    Motor, RPE and reward channels are set to zero — the body does
    not actuate during sleep — but the pytree shape matches the wake
    output so downstream code can be written once.
    """
    zero = jnp.asarray(0.0, DTYPE)
    zero_int = jnp.asarray(0, jnp.int32)
    return ActionBrainOutput(
        state=state,
        body_action=zero_int,
        saccade_action=zero_int,
        rpe=zero,
        total_reward=zero,
        curiosity=zero,
        cortex_belief=jnp.zeros(cortex_belief_size, DTYPE),
        cortex_l5_rate=jnp.zeros(l5_size, DTYPE),
        cerebellum_nuclei=jnp.zeros(dn_size, DTYPE),
        relay_spikes=jnp.zeros(relay_size, DTYPE),
        theta_phase=state.oscillator.theta_phase,
        neuromod=state.neuromodulator,
    )


def brain_cycle(
    state: ActionBrainState,
    params: ActionBrainParams,
    ctx: BackendContext,
    image: Array,
    fixation_xy: Array,
    prev_reward: float | Array = 0.0,
    prev_done: float | Array = 0.0,
    key: PRNGKey | None = None,
    *,
    info_gain: float | Array | None = None,
    n_sws_replay: int = 32,
    n_rem_rollout: int = 10,
) -> ActionBrainOutput:
    """Phase-aware top-level brain dispatcher.

    Python-level ``if`` on the CURRENT sleep phase — we do not use
    ``jax.lax.switch`` because the three branches have fundamentally
    different signatures (WAKE needs sensory, sleep branches do not)
    and fundamentally different computational costs (offline replay
    is a short ``scan`` over replay indices whereas the wake branch
    runs the entire neural front-end).  Phase transitions themselves
    are JIT-safe inside :func:`sleep_step`; the dispatch is a
    host-side control-flow decision, consistent with the
    neuromodulatory-switching literature (Saper 2010 flip-flop
    VLPO).
    """
    if key is None:
        raise ValueError("brain_cycle requires an explicit PRNG key")

    phase = int(state.sleep.phase)  # concrete Python int

    if phase == int(SleepPhase.WAKE):
        return action_brain_step(
            state, params, ctx, image, fixation_xy,
            prev_reward=prev_reward, prev_done=prev_done,
            key=key, info_gain=info_gain,
        )

    # Offline sleep: SWS or REM.
    if phase == int(SleepPhase.SWS):
        k_rep, k_next = jax.random.split(key)
        wm_new, replay_new, hc_new = sws_replay_step(
            state.world_model, params.world_model, ctx,
            state.replay, params.replay,
            state.hippocampus, params.hippocampus,
            k_rep,
            n_replay=n_sws_replay,
            n_body_actions=int(params.n_body_actions),
            n_saccade_actions=int(params.n_saccade_actions),
            ach=state.neuromodulator.acetylcholine,
        )
        state = eqx.tree_at(
            lambda s: (s.world_model, s.replay, s.hippocampus),
            state, (wm_new, replay_new, hc_new),
        )
        state = _advance_sleep_bookkeeping(
            state, params, ctx, awake=False, sws=True,
        )
    else:  # REM
        k_rem, k_next = jax.random.split(key)
        wm_new, hc_new = rem_rollout_step(
            state.world_model, params.world_model, ctx,
            state.replay, params.replay,
            state.hippocampus, params.hippocampus,
            k_rem,
            k_steps=n_rem_rollout,
            n_body_actions=int(params.n_body_actions),
            n_saccade_actions=int(params.n_saccade_actions),
            ach=state.neuromodulator.acetylcholine,
        )
        state = eqx.tree_at(
            lambda s: (s.world_model, s.hippocampus),
            state, (wm_new, hc_new),
        )
        state = _advance_sleep_bookkeeping(
            state, params, ctx, awake=False, sws=False,
        )

    return _sleep_phase_output(
        state,
        cortex_belief_size=int(params.cortex.n_l23_state),
        l5_size=int(params.cortex.n_l5),
        dn_size=int(params.cerebellum.n_dn),
        relay_size=int(params.thalamus_relay.n_tc),
    )
