"""Cerebellum — generic forward-model (Marr-Albus-Ito) in pure JAX.

References:
  Marr (1969)       — A theory of cerebellar cortex
  Albus (1971)      — A theory of cerebellar function
  Ito (2001)        — Cerebellar long-term depression
  Wolpert (1998)    — Internal models in the cerebellum
  Raymond & Lisberger (1998) — Neural learning rules for the VOR
  Schmahmann (2019) — The cerebellum and cognition
  Strick et al. (2009) — Cerebellum and the nonmotor brain

Circuit
-------
    mossy  ─[fixed sparse random]─▶ granule (kWTA, ~5% active)
    mossy  ─[fixed diffuse]─────────▶ deep nuclei (excit collaterals)
    granule─[LEARNED dense]─────────▶ purkinje   (rate-based readout)
    purkinje─[fixed 1-to-1]────────▶ deep nuclei (inhibitory)
    climbing_fiber ─────────────────▶ purkinje   (teaching signal, LTD)

Function
--------
Generic forward model: given a *context* (mossy), predict a *correction*
(deep nuclei output). The same circuit learns motor trajectories, VOR
gains, eye-blink timing, or cognitive prediction (Schmahmann 2019) —
what differs is only the external wiring supplied by ``brain_graph``.

Body-dependence
---------------
The circuit is body-AGNOSTIC. Embodiment enters only via what mossy
fibers carry (typically ``concat([proprioception, efference_copy])``)
and what climbing fibers carry (``actual - predicted`` sensory error
from the body/sim interface). This module takes both as plain arrays.

Output
------
``nuclei`` has the sign convention: positive = excitatory correction,
negative = suppression. It is NOT a motor command — it is a contextual
signal injected into M1 / thalamus by ``brain_graph``. Motor commands
come from cortex; the motor loop is cortex → body_interface, with
cerebellum merely refining the cortical command.

Rate-based (not spiking)
------------------------
Purkinje cells fire tonically at 40–80 Hz; their average rate *is* the
prediction. Spiking would cost ~50× more FLOPs for zero functional gain
at the system level. The granule layer retains biological kWTA sparsity
through a top-k threshold (no spikes either — just a sparse code).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, BackendContext, split_key


# =====================================================================
# Params / State / IO
# =====================================================================


class CerebellumParams(eqx.Module):
    """Static params — mossy→granule + mossy→DN fabrics are fixed."""

    w_mg: Array              # (mossy_size, n_granule) sparse random, FIXED
    w_mdn: Array             # (mossy_size, n_dn) diffuse collaterals, FIXED
    pc_dn_gain: Array        # scalar — Purkinje inhibition strength onto DN

    granule_trace_decay: Array   # eligibility decay (~100 ms for 20 ms CF window)
    pc_rate_decay: Array         # low-pass on Purkinje output
    dn_rate_decay: Array         # low-pass on nuclei output

    ltd_rate: Array              # Marr-Albus LTD (dominant)
    ltp_rate: Array              # tonic LTP floor (stabiliser)
    w_clip: Array                # abs-value cap on w_gp

    mossy_size: int = eqx.field(static=True)
    n_granule: int = eqx.field(static=True)
    n_purkinje: int = eqx.field(static=True)
    n_dn: int = eqx.field(static=True)
    granule_k: int = eqx.field(static=True)        # kWTA winners
    pc_fanin: int = eqx.field(static=True)         # for scaling readout


class CerebellumState(eqx.Module):
    w_gp: Array             # (n_granule, n_purkinje) LEARNED
    granule_trace: Array    # (n_granule,) eligibility (pre-syn)
    pc_rate: Array          # (n_purkinje,) low-passed readout
    dn_rate: Array          # (n_dn,) low-passed nuclei output


class CerebellumOutput(NamedTuple):
    state: CerebellumState
    nuclei: Array           # (n_dn,) — external correction signal
    granule: Array          # (n_granule,) — sparse expanded code
    purkinje: Array         # (n_purkinje,) — raw Purkinje readout


# =====================================================================
# Init
# =====================================================================


def init_cerebellum_params(
    ctx: BackendContext,
    mossy_size: int,
    *,
    granule_expansion: int = 50,
    n_purkinje: int = 32,
    n_dn: int | None = None,
    granule_sparsity: float = 0.05,
    mossy_per_granule: int = 4,
    pc_dn_gain: float = 1.0,
    tau_granule_trace: float = 100.0,
    tau_pc_rate: float = 20.0,
    tau_dn_rate: float = 20.0,
    ltd_rate: float = 5e-3,
    ltp_rate: float = 5e-5,
    w_clip: float = 2.0,
    seed: int = 0,
) -> CerebellumParams:
    """Build cerebellum params.

    ``granule_expansion`` follows Marr: ``N_gr = expansion · mossy_size``
    (biologically ~500; default 50 is a tractable approximation).
    ``mossy_per_granule`` is the typical small fan-in (~4 biologically,
    Palkovits 1971) implemented as a sparse random mask on ``w_mg``.
    ``n_dn`` defaults to ``n_purkinje`` for a one-to-one PC→DN mapping.
    """
    n_granule = max(1, granule_expansion * mossy_size)
    n_dn_val = int(n_purkinje if n_dn is None else n_dn)
    k = max(1, int(round(n_granule * granule_sparsity)))

    f = lambda x: jnp.asarray(x, DTYPE)
    master = jax.random.PRNGKey(seed)
    k_mg_val, k_mg_mask, k_mdn = split_key(master, 3)

    # mossy → granule: each granule picks ``mossy_per_granule`` mossy inputs
    fan = min(mossy_per_granule, mossy_size)
    # Build a (mossy_size, n_granule) mask with exactly ``fan`` ones per column
    # by sampling uniform noise and thresholding the top-fan per column.
    noise = jax.random.uniform(
        k_mg_mask, (mossy_size, n_granule), dtype=DTYPE)
    top_thresh = jnp.sort(noise, axis=0)[mossy_size - fan]
    mask = (noise >= top_thresh[None, :]).astype(DTYPE)
    # Signed weights so granule activity reflects feature combinations
    raw = jax.random.normal(k_mg_val, (mossy_size, n_granule), dtype=DTYPE)
    # Scale so expected pre-activation has unit std regardless of fan
    w_mg = mask * raw / jnp.sqrt(jnp.asarray(fan, DTYPE))

    # mossy → deep nuclei: dense small random excitatory collaterals
    w_mdn = jnp.abs(
        jax.random.normal(k_mdn, (mossy_size, n_dn_val), dtype=DTYPE)
    ) / jnp.sqrt(jnp.asarray(mossy_size, DTYPE))

    return CerebellumParams(
        w_mg=w_mg, w_mdn=w_mdn, pc_dn_gain=f(pc_dn_gain),
        granule_trace_decay=f(ctx.decay(tau_granule_trace)),
        pc_rate_decay=f(ctx.decay(tau_pc_rate)),
        dn_rate_decay=f(ctx.decay(tau_dn_rate)),
        ltd_rate=f(ltd_rate), ltp_rate=f(ltp_rate), w_clip=f(w_clip),
        mossy_size=mossy_size, n_granule=n_granule,
        n_purkinje=n_purkinje, n_dn=n_dn_val,
        granule_k=k, pc_fanin=n_granule,
    )


def init_cerebellum_state(
    key: PRNGKey, params: CerebellumParams, *, dtype=DTYPE,
) -> CerebellumState:
    """Initialise learned weights near zero (blank-slate forward model)."""
    w_gp = jax.random.normal(
        key, (params.n_granule, params.n_purkinje), dtype=dtype,
    ) * (1.0 / jnp.sqrt(jnp.asarray(params.pc_fanin, dtype)))
    z = lambda shape: jnp.zeros(shape, dtype)
    return CerebellumState(
        w_gp=w_gp,
        granule_trace=z(params.n_granule),
        pc_rate=z(params.n_purkinje),
        dn_rate=z(params.n_dn),
    )


# =====================================================================
# Step
# =====================================================================


def _kwta(pre_act: Array, k: int) -> Array:
    """k-winners-take-all (positive-only, binary mask × magnitude).

    Returns a (n,) array where only the top-k entries of ``pre_act``
    survive, ReLU'd. Differentiable except at the threshold.
    """
    # Threshold = k-th largest value
    # jnp.sort ascending; index -k yields the k-th largest
    sorted_vals = jnp.sort(pre_act)
    thresh = sorted_vals[-k]
    active = pre_act >= thresh
    return jnp.where(active, jnp.maximum(pre_act, 0.0), 0.0)


def cerebellum_step(
    state: CerebellumState,
    params: CerebellumParams,
    ctx: BackendContext,
    mossy_input: Array,
) -> CerebellumOutput:
    """One dt of forward-model inference (NO learning — call ``update``).

    Pipeline:
      1. Granule = kWTA(mossy @ W_mg)                 — sparse expansion
      2. Purkinje = tanh(granule @ w_gp)              — bounded readout
      3. DN = mossy @ W_mdn  − pc_dn_gain · purkinje  — net correction
      4. Eligibility traces updated for later LTD.
    """
    mossy = mossy_input.astype(DTYPE)

    # 1. Granule layer — sparse code
    pre = mossy @ params.w_mg
    granule = _kwta(pre, params.granule_k)

    # 2. Purkinje readout (rate-based, bounded)
    pc_raw = granule @ state.w_gp
    purkinje = jnp.tanh(pc_raw)

    # 3. Deep nuclei — excitatory mossy − inhibitory Purkinje (PC→DN 1-to-1
    #    if n_dn == n_purkinje, else broadcast/truncate via repeat)
    mossy_drive = mossy @ params.w_mdn
    if params.n_dn == params.n_purkinje:
        pc_onto_dn = purkinje
    else:
        rep = int(params.n_dn // params.n_purkinje) + 1
        pc_onto_dn = jnp.tile(purkinje, rep)[: params.n_dn]
    nuclei_raw = mossy_drive - params.pc_dn_gain * pc_onto_dn

    # 4. Low-pass rates + eligibility trace on granule activity
    gtd = params.granule_trace_decay
    granule_trace = state.granule_trace * gtd + (1.0 - gtd) * granule
    prd = params.pc_rate_decay
    pc_rate = state.pc_rate * prd + (1.0 - prd) * purkinje
    drd = params.dn_rate_decay
    dn_rate = state.dn_rate * drd + (1.0 - drd) * nuclei_raw

    new_state = CerebellumState(
        w_gp=state.w_gp,
        granule_trace=granule_trace,
        pc_rate=pc_rate,
        dn_rate=dn_rate,
    )
    return CerebellumOutput(
        state=new_state,
        nuclei=dn_rate,
        granule=granule,
        purkinje=purkinje,
    )


# =====================================================================
# Learning — Marr-Albus-Ito LTD via climbing fibers
# =====================================================================


def cerebellum_update(
    state: CerebellumState,
    params: CerebellumParams,
    climbing_error: Array,
    *,
    modulator: float | Array = 1.0,
) -> CerebellumState:
    """Marr-Albus-Ito supervised-descent rule.

    ``climbing_error`` is ``(n_purkinje,)`` — typically the signed
    mismatch between actual and predicted sensorimotor outcome as
    computed upstream (inferior olive in biology, ``brain_graph`` here).

    Rule:
      dw_gp = −lr_LTD · granule_trace ⊗ climbing_error   (Ito 2001)
              + lr_LTP · granule_trace ⊗ (1 − |error|)    (tonic LTP)
      w_gp ← clip(w_gp + dw_gp, −w_clip, +w_clip)

    The LTP floor keeps synapses from drifting to zero when CF activity
    is absent (a well-known problem of pure-LTD models; Coesmans 2004).
    ``modulator`` (0..1) is a global learning-rate gate (e.g. attention
    or arousal from brain_graph).
    """
    err = climbing_error.astype(DTYPE)
    mod = jnp.asarray(modulator, DTYPE)
    gt = state.granule_trace[:, None]         # (n_gr, 1)
    e = err[None, :]                          # (1, n_pc)

    ltd = -params.ltd_rate * mod * gt * e
    ltp = params.ltp_rate * mod * gt * (1.0 - jnp.abs(e))
    w_new = jnp.clip(state.w_gp + ltd + ltp, -params.w_clip, params.w_clip)
    return eqx.tree_at(lambda s: s.w_gp, state, w_new)


# =====================================================================
# Reset
# =====================================================================


def cerebellum_reset_transient(
    state: CerebellumState, params: CerebellumParams,
) -> CerebellumState:
    """Clear traces and low-pass filters; preserve learned ``w_gp``."""
    z = lambda shape: jnp.zeros(shape, DTYPE)
    return CerebellumState(
        w_gp=state.w_gp,
        granule_trace=z(params.n_granule),
        pc_rate=z(params.n_purkinje),
        dn_rate=z(params.n_dn),
    )
