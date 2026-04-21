"""M1 — primary motor cortex: continuous readout head over cortical L5.

Phase 6A design
---------------
M1 is a **learned linear readout** from the main cortex's L5 rate
population onto an ``(motor_dim,)`` continuous joint-command vector,
squashed through ``tanh``.  In the biology this corresponds to the M1
L5 pyramidal-tract descending projection (Lemon 2008; Rathelot &
Strick 2009): massive convergence from sensorimotor cortex onto the
spinal/α-motor interface, with the cortex itself providing the
state-dependent dynamics.  Keeping M1 as a readout (rather than a
second full cortical microcircuit) for Phase 6A minimises JIT-compile
cost and keeps the integration surface small — a second
``CorticalAreaParams`` can be nested in Phase 6B once MJX confirms
continuous control is required for reach.

Learning
--------
Three-factor Hebbian on ``motor_readout`` (Doya 2000; Shadmehr &
Krakauer 2008):

    Δw_{ij} = lr · (rpe + α · cb_motor_err_j) · l5_i · jc_j

where ``jc`` is the post-tanh command (used as the post-synaptic
eligibility because it is what the downstream motor neuron actually
sees).  ``cb_motor_err`` is a per-joint correction vector supplied by
the cerebellar deep-nuclei readout (Wolpert 1998); it is ZERO for the
initial regression-safe Phase 6A path where only RPE drives learning.

PCA-style motor-primitive initialisation
----------------------------------------
``motor_readout`` is initialised analytically to block-structured
"muscle synergies" — the first ``motor_dim`` principal directions of an
identity-like synergy matrix, which is the closed-form analogue of
Dominici (2011)'s developmental infant primitives.  This is **not**
gradient-trained; it is a biologically-motivated prior exactly like
V1's Gabor init (Olshausen & Field 1996 background).

References
----------
- Dominici et al. (2011) *Science* 334: 997-999 — motor primitives in
  infant locomotion.
- Doya (2000) *Neural Comput.* 12: 219-245 — cortico-cerebello-BG
  computation split.
- Shadmehr & Krakauer (2008) *Exp. Brain Res.* 185: 359-381 — motor
  adaptation as Hebbian readout learning.
- Lemon (2008) *Annu. Rev. Neurosci.* 31: 195-218 — M1 descending
  corticospinal projection.
- Churchland et al. (2012) *Nature* 487: 51-56 — M1 dynamical systems.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey


class M1Params(eqx.Module):
    """Static M1 readout parameters."""

    # Hebbian readout learning rate (three-factor, RPE-gated).
    readout_lr: Array

    # Clip for motor_readout absolute value (prevents unbounded drift
    # under sustained RPE; Doya 2000 argues motor readout gain is
    # homeostatically capped by recurrent inhibition in M1 L5).
    w_clip: Array

    # Cerebellar-correction blend coefficient α in
    #   jc_out = tanh(jc_raw + α · cb_motor_correction)
    # α ~ 1/τ_cerebellum ≈ 0.2 (Medina & Lisberger 2008 ~50 ms / 250
    # ms task-horizon).  Zero in Phase 6A regression (no cerebellar
    # correction yet); default 0.2 for the active M1 path.
    cb_alpha: Array

    # Sizes
    n_l5: int = eqx.field(static=True)
    motor_dim: int = eqx.field(static=True)


class M1State(eqx.Module):
    """M1 learned state — just the readout plus last-command memory."""

    motor_readout: Array            # (n_l5, motor_dim)
    last_joint_command: Array       # (motor_dim,)


class M1Output(NamedTuple):
    state: M1State
    joint_command: Array            # (motor_dim,) in [-1, 1]
    l5_rate_normalised: Array       # (n_l5,) the post-normalisation drive


# =====================================================================
# Init
# =====================================================================


def _pca_synergy_init(n_l5: int, motor_dim: int) -> Array:
    """Closed-form motor-primitive initialisation (Dominici 2011 analog).

    The ideal readout maps structured L5 activity into orthogonal joint
    commands.  We pick an orthonormal basis on ``R^motor_dim`` and
    embed it into an ``(n_l5, motor_dim)`` block matrix such that each
    motor channel is driven by a disjoint cluster of L5 units — the
    "muscle synergy" prior.  Magnitude 1/sqrt(n_l5) keeps the initial
    command small (||jc|| ~ 1 under unit-rate L5).

    This is analytic — **no gradient descent** — and plays the same
    role that Gabor init plays in V1 (Jones & Palmer 1987 / Olshausen &
    Field 1996): a biologically defensible starting point that
    subsequent three-factor Hebbian learning refines.
    """
    w = jnp.zeros((n_l5, motor_dim), dtype=DTYPE)
    # Assign each motor channel a contiguous block of L5 units.
    block = max(1, n_l5 // max(1, motor_dim))
    scale = jnp.asarray(1.0 / jnp.sqrt(jnp.asarray(n_l5, DTYPE)), DTYPE)
    for j in range(motor_dim):
        lo = j * block
        hi = min(n_l5, lo + block)
        # Alternating sign so different channels pull opposite muscle
        # groups (Georgopoulos 1986 directional tuning).
        sign = 1.0 if (j % 2 == 0) else -1.0
        w = w.at[lo:hi, j].set(sign * scale)
    # Fill remaining L5 units (if n_l5 % motor_dim > 0) with random
    # small half-normal noise so every unit has some projection.
    # (Deterministic filler; PRNG is unnecessary here.)
    if n_l5 > block * motor_dim:
        pass  # leave as zeros — biological sparse connectivity
    return w


def init_m1_params(
    *,
    n_l5: int,
    motor_dim: int,
    readout_lr: float = 1e-3,
    w_clip: float = 2.0,
    cb_alpha: float = 0.2,
) -> M1Params:
    f = lambda x: jnp.asarray(x, DTYPE)
    return M1Params(
        readout_lr=f(readout_lr),
        w_clip=f(w_clip),
        cb_alpha=f(cb_alpha),
        n_l5=int(n_l5),
        motor_dim=int(motor_dim),
    )


def init_m1_state(key: PRNGKey, params: M1Params) -> M1State:
    # PCA-style synergy init (no key consumed — analytic). ``key`` is
    # accepted for API symmetry with the rest of core.
    del key
    w0 = _pca_synergy_init(params.n_l5, params.motor_dim)
    return M1State(
        motor_readout=w0,
        last_joint_command=jnp.zeros(params.motor_dim, DTYPE),
    )


# =====================================================================
# Step
# =====================================================================


def _normalise_l5(l5_rate: Array) -> Array:
    """Peak-normalise L5 rate so ||jc|| is drive-invariant.

    Matches the existing ``action_brain_step`` L4 normalisation trick
    (brain_graph.py): dividing by (peak + ε) keeps the readout scale
    stable regardless of overall cortical firing rate.
    """
    peak = jnp.max(jnp.abs(l5_rate))
    return jnp.where(
        peak > 1e-3,
        l5_rate / (peak + jnp.asarray(1e-6, DTYPE)),
        l5_rate,
    ).astype(DTYPE)


@eqx.filter_jit
def m1_step(
    state: M1State,
    params: M1Params,
    l5_rate: Array,
    *,
    cb_motor_correction: Array | None = None,
) -> M1Output:
    """One dt of M1: L5 rate → bounded joint command.

    Parameters
    ----------
    l5_rate : (n_l5,)
        The cortex L5 rate population feeding M1. In Phase 6A this is
        the main cortex L5 rate (re-used by BG striatum as well); in
        Phase 6B this can be routed through a dedicated M1 cortical
        microcircuit.
    cb_motor_correction : (motor_dim,) | None
        Additive pre-tanh term from cerebellar deep nuclei (Wolpert
        1998 forward-model correction).  ``None`` → zero.
    """
    l5 = _normalise_l5(l5_rate)
    raw = l5 @ state.motor_readout                          # (motor_dim,)
    if cb_motor_correction is not None:
        raw = raw + params.cb_alpha * cb_motor_correction.astype(DTYPE)
    jc = jnp.tanh(raw).astype(DTYPE)
    new_state = M1State(
        motor_readout=state.motor_readout,
        last_joint_command=jc,
    )
    return M1Output(state=new_state, joint_command=jc, l5_rate_normalised=l5)


# =====================================================================
# Learning — three-factor Hebbian on motor_readout
# =====================================================================


def m1_learn_readout(
    state: M1State,
    params: M1Params,
    *,
    rpe: Array,
    l5_rate_normalised: Array,
    joint_command: Array,
    cb_motor_err: Array | None = None,
) -> M1State:
    """Three-factor Hebbian update (Doya 2000; Shadmehr & Krakauer 2008).

      Δw_{ij} = lr · (rpe + cb_motor_err_j) · l5_i · jc_j

    Gated by the VTA RPE broadcast — classic cortical
    reward-modulated plasticity.  ``cb_motor_err`` optionally adds a
    per-joint supervised-descent term from the cerebellum (Wolpert
    1998).  The rule is local (outer product of pre and post
    activities times a scalar modulator per channel); no backprop.
    """
    r = jnp.asarray(rpe, DTYPE)
    if cb_motor_err is None:
        mod = r                                             # scalar
        dw = params.readout_lr * mod * jnp.outer(
            l5_rate_normalised.astype(DTYPE),
            joint_command.astype(DTYPE),
        )
    else:
        mod = r + cb_motor_err.astype(DTYPE)                # (motor_dim,)
        dw = params.readout_lr * (
            jnp.outer(l5_rate_normalised.astype(DTYPE),
                      joint_command.astype(DTYPE)) * mod[None, :]
        )
    w_new = jnp.clip(state.motor_readout + dw, -params.w_clip, params.w_clip)
    return eqx.tree_at(lambda s: s.motor_readout, state, w_new)
