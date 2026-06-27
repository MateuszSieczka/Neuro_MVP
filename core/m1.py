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

Learning (Phase 6B fix)
-----------------------
Node-perturbation REINFORCE (Williams 1992; Fiete & Seung 2006;
biological analogue: motor-cortex variability as policy-gradient
exploration, Tumer & Brainard 2007; Dhawale et al. 2017):

    jc = tanh(raw + ξ)       with  ξ ~ 𝒩(0, σ²(NE)),  raw = wᵀ·l5
    Δw_{ij} = lr · (rpe − b) · l5_i · ξ_j / σ²

The injected noise ξ is the *exploration direction* the readout
committed to this cycle; ``l5·ξ/σ²`` is the score ∂lnπ/∂w of the
Gaussian policy (Williams 1992 eq. for 𝒩(μ, σ²)), so correlating it
with the baseline-subtracted RPE is an unbiased policy-gradient
estimator (Williams 1992 §6; Fiete & Seung 2006).  The ``1/σ²`` factor
makes the estimate scale-correct under NE-driven changes in the
exploration amplitude.

σ is gated by noradrenaline (Aston-Jones & Cohen 2005 LC arousal
→ motor variability): σ = σ_base · (σ_floor + ne_gain · NE).  Default
σ_base = 0.15 matches the 5–20 % motor-dynamic-range variability
reported in songbird HVC→RA (Tumer & Brainard 2007) and rodent M1
(Dhawale et al. 2017).

M1 produces the command only.  The cerebellum is a forward model
(Wolpert 1998) wired in ``brain_graph`` to predict the proprioceptive
consequence of the command, not to correct it — so no cerebellar term
enters ``m1_step`` (see its docstring).

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

    # Node-perturbation REINFORCE learning rate (Williams 1992).
    readout_lr: Array

    # Clip for motor_readout absolute value (prevents unbounded drift
    # under sustained RPE; Doya 2000 argues motor readout gain is
    # homeostatically capped by recurrent inhibition in M1 L5).
    w_clip: Array

    # Exploration-noise std at NE = 0 baseline (fraction of tanh
    # dynamic range).  Tumer & Brainard 2007 / Dhawale et al. 2017:
    # 5–20 % motor variability in M1.
    sigma_base: Array
    # NE-coupled gain on σ: σ = σ_base · (σ_floor + ne_gain · NE).
    # Aston-Jones & Cohen 2005: tonic LC discharge linearly scales
    # behavioural variability.
    sigma_floor: Array
    sigma_ne_gain: Array

    # Sizes
    n_l5: int = eqx.field(static=True)
    motor_dim: int = eqx.field(static=True)


class M1State(eqx.Module):
    """M1 learned state."""

    motor_readout: Array            # (n_l5, motor_dim)
    last_joint_command: Array       # (motor_dim,)
    # Most-recent injected exploration noise ξ -- preserved across
    # ``_sync_brain_to_body`` so the next cycle's REINFORCE update can
    # correlate the RPE arriving NOW with the noise that produced the
    # action that EARNED that RPE.
    last_exploration_noise: Array   # (motor_dim,)
    # Most-recent normalised L5 rate that drove the readout (the
    # "pre" side of the eligibility outer product).
    last_l5_rate: Array             # (n_l5,)
    # Exploration-noise std σ used to sample ``last_exploration_noise``.
    # The Gaussian-policy score is ∂lnπ/∂μ = ξ/σ² (Williams 1992 eq.
    # for a 𝒩(μ, σ²) policy); the REINFORCE update must therefore weight
    # the eligibility by 1/σ² so that the per-sample gradient estimate is
    # unbiased regardless of the NE-driven exploration amplitude.  Cached
    # here so the next cycle's update can reach the σ that produced the
    # action which earned the arriving RPE.
    last_sigma: Array               # () scalar
    # EMA of recent RPE -- the policy-gradient learner's own value
    # baseline (Sutton & Barto 1998 §13.4 REINFORCE-with-baseline).
    # VTA already subtracts a slow critic V(s); however with V(s)
    # equilibrating slowly during early learning the residual mean of
    # ``rpe`` consumed by M1 is positive-biased (verified empirically:
    # mean rpe ≈ +0.03 across 200-cycle babble even though the policy
    # is untrained), causing systematic W drift independent of policy
    # quality.  A local M1-side EMA absorbs this residual.
    rpe_baseline: Array             # () scalar


class M1Output(NamedTuple):
    state: M1State
    joint_command: Array            # (motor_dim,) in [-1, 1]
    l5_rate_normalised: Array       # (n_l5,) the post-normalisation drive
    exploration_noise: Array        # (motor_dim,) ξ sampled this step


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
    # Scale = 1/block (NOT 1/sqrt(n_l5)).  Justification: peak-normalised
    # L5 (||l5||_∞ = 1 by construction) co-firing the entire ``block``
    # of synergy-aligned units gives raw[j] = ±block · scale = ±1, which
    # is the edge of tanh's linear regime.  The previous 1/sqrt(n_l5)
    # scale (=0.177 for n_l5=32) produced raw[j] ≈ ±2.83 at coactivation
    # → hard tanh saturation at init (verified empirically: pre_tanh
    # mean = 6.6 even with W frozen at PCA init).  This is the
    # synergy-block analogue of fan-in normalisation (Glorot & Bengio
    # 2010); biologically: averaging convergence onto each M1 L5 cell
    # (Lemon 2008) keeps drive O(1) regardless of cluster size.
    scale = jnp.asarray(1.0 / float(block), DTYPE)
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
    w_clip: float | None = None,
    sigma_base: float = 0.15,
    sigma_floor: float = 0.5,
    sigma_ne_gain: float = 1.0,
) -> M1Params:
    f = lambda x: jnp.asarray(x, DTYPE)
    # Default clip = 4 × scale_init = 4/block.  Init scale is 1/block
    # (see _pca_synergy_init); allowing W to grow up to 4× its init
    # magnitude per cell keeps the synergy-block sum bounded by 4 (so
    # raw ∈ [-4, 4] worst case, on the saturating shoulder of tanh but
    # not deep in the flat region).  This is the empirical "weights
    # stay within O(few) × init scale" rule (Krogh & Hertz 1992); the
    # previous static value w_clip=2.0 allowed ~32× growth and was
    # never reached in practice but was structurally unprincipled.
    if w_clip is None:
        block = max(1, n_l5 // max(1, motor_dim))
        w_clip = 4.0 / float(block)
    return M1Params(
        readout_lr=f(readout_lr),
        w_clip=f(w_clip),
        sigma_base=f(sigma_base),
        sigma_floor=f(sigma_floor),
        sigma_ne_gain=f(sigma_ne_gain),
        n_l5=int(n_l5),
        motor_dim=int(motor_dim),
    )


def init_m1_state(key: PRNGKey, params: M1Params) -> M1State:
    # PCA-style synergy init (no key consumed — analytic). ``key`` is
    # accepted for API symmetry with the rest of core.
    del key
    w0 = _pca_synergy_init(params.n_l5, params.motor_dim)
    # Baseline σ at NE = 0 (Aston-Jones & Cohen 2005 tonic LC floor);
    # never zero because ``sigma_floor`` > 0, so 1/σ² in the learning
    # rule is always finite.
    sigma0 = params.sigma_base * params.sigma_floor
    return M1State(
        motor_readout=w0,
        last_joint_command=jnp.zeros(params.motor_dim, DTYPE),
        last_exploration_noise=jnp.zeros(params.motor_dim, DTYPE),
        last_l5_rate=jnp.zeros(params.n_l5, DTYPE),
        last_sigma=sigma0.astype(DTYPE),
        rpe_baseline=jnp.asarray(0.0, DTYPE),
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
    key: PRNGKey,
    ne_level: Array,
) -> M1Output:
    """One dt of M1: L5 rate → bounded, *noisy* joint command.

    Sample ξ ~ 𝒩(0, σ(NE)²) and emit ``jc = tanh(l5·w + ξ)``.  The noise
    is the exploration channel for node-perturbation REINFORCE
    (Williams 1992; biology: motor-cortex variability, Tumer & Brainard
    2007).  Without this noise the readout is deterministic given L5 and
    there is no policy-gradient signal → the readout could not learn.

    M1 emits the *command*; it receives no cerebellar term.  The
    cerebellum is a **forward model** (Wolpert 1998): it predicts the
    proprioceptive consequence of the command (used by ``brain_graph``
    to close the motor-prediction loop), not an inverse controller.
    Folding the forward-model output back into the command would have
    the wrong sign (a predicted undershoot would *reduce* the command)
    and would conflate the forward and inverse models; the inverse map
    (state → command) is learned by this readout's REINFORCE rule.

    Parameters
    ----------
    l5_rate : (n_l5,)
        Cortex L5 rate population feeding M1.
    key : PRNGKey
        Per-step key for noise sampling.
    ne_level : ()
        Current noradrenaline level ∈ [0, 1] (Aston-Jones & Cohen
        2005 LC tonic discharge); scales σ multiplicatively.
    """
    l5 = _normalise_l5(l5_rate)
    raw = l5 @ state.motor_readout                          # (motor_dim,)
    sigma = params.sigma_base * (
        params.sigma_floor + params.sigma_ne_gain * jnp.asarray(ne_level, DTYPE)
    )
    xi = sigma * jax.random.normal(key, raw.shape, DTYPE)
    jc = jnp.tanh(raw + xi).astype(DTYPE)
    new_state = M1State(
        motor_readout=state.motor_readout,
        last_joint_command=jc,
        last_exploration_noise=xi.astype(DTYPE),
        last_l5_rate=l5.astype(DTYPE),
        last_sigma=sigma.astype(DTYPE),
        rpe_baseline=state.rpe_baseline,
    )
    return M1Output(
        state=new_state,
        joint_command=jc,
        l5_rate_normalised=l5,
        exploration_noise=xi.astype(DTYPE),
    )


# =====================================================================
# Learning — three-factor Hebbian on motor_readout
# =====================================================================


def m1_learn_readout(
    state: M1State,
    params: M1Params,
    *,
    rpe: Array,
    l5_rate_normalised: Array | None = None,
    exploration_noise: Array | None = None,
) -> M1State:
    """Node-perturbation REINFORCE update (Williams 1992; Fiete & Seung 2006).

      Δw_{ij} = lr · (rpe − b) · l5_i · ξ_j / σ²

    For a Gaussian exploration policy ``a = μ + ξ`` with
    ``ξ ~ 𝒩(0, σ²)`` and ``μ = wᵀ·l5``, the score function is
    ``∂lnπ/∂w_{ij} = l5_i · ξ_j / σ²`` (Williams 1992 eq. for the
    normal distribution).  Correlating that score with the
    baseline-subtracted return is the unbiased policy-gradient estimator
    ``∇_w E[R | π_w]`` (Fiete & Seung 2006 node perturbation; biological
    analogue: motor-cortex variability as exploration, Tumer & Brainard
    2007).

    The ``1/σ²`` factor is essential: σ is gated by noradrenaline
    (``m1_step``), so without it the per-sample gradient is mis-scaled
    whenever arousal changes the exploration amplitude, inflating
    variance and biasing learning toward high-NE episodes.  We carry the
    reference variance ``σ_base²`` into the learning rate (the lr is
    defined at the NE = 0 baseline σ_base·σ_floor… i.e. ``σ_base`` sets
    the unit), so the applied scale is ``(σ_base / σ)²`` — exactly the
    Williams score with the reference variance folded into ``lr``.  This
    keeps the update magnitude stable at the reference arousal while
    correctly re-weighting samples taken under different σ.

    ``l5_rate_normalised`` and ``exploration_noise`` default to the
    values cached on ``state`` (``last_l5_rate`` / ``last_exploration_noise``)
    so the caller does not have to thread them through across the
    one-cycle delay between action and reward.

    The rule is local: outer product of presynaptic L5 rate and
    postsynaptic noise, gated by the global RPE scalar.  No backprop, no
    surrogate gradient.
    """
    r = jnp.asarray(rpe, DTYPE)
    # REINFORCE-with-baseline (Sutton & Barto 1998 §13.4).  The EMA
    # rate α_b = readout_lr couples the baseline timescale to the
    # policy-update timescale: τ_b = 1/lr.  This is the principled
    # choice — baseline tracks the policy's own update horizon, fast
    # enough to follow non-stationary returns, slow enough that the
    # baseline estimate has lower variance than the per-step rpe it
    # subtracts (Greensmith et al. 2004 variance bounds).  VTA's V(s)
    # critic provides one baseline already; this M1-local one absorbs
    # the *residual* positive bias that survives because V(s) trains
    # slower than M1 (verified: D3.2 of saturation-diag notebook —
    # raw rpe mean +0.026, baseline-subtracted mean +0.003).
    alpha_b = params.readout_lr
    new_baseline = (
        (jnp.asarray(1.0, DTYPE) - alpha_b) * state.rpe_baseline
        + alpha_b * r
    )
    r_eff = r - new_baseline
    pre = (
        state.last_l5_rate if l5_rate_normalised is None
        else l5_rate_normalised
    ).astype(DTYPE)
    xi = (
        state.last_exploration_noise if exploration_noise is None
        else exploration_noise
    ).astype(DTYPE)
    # Gaussian-policy score with the reference variance folded into lr:
    # (σ_base / σ)²  ·  l5 ⊗ ξ   (Williams 1992).  σ has a positive floor
    # (sigma_floor > 0) so the ratio is always finite.
    score_scale = (params.sigma_base / state.last_sigma) ** 2
    elig = jnp.outer(pre, xi)                               # (n_l5, motor_dim)
    dw = params.readout_lr * r_eff * score_scale * elig
    w_new = jnp.clip(state.motor_readout + dw, -params.w_clip, params.w_clip)
    state = eqx.tree_at(lambda s: s.motor_readout, state, w_new)
    return eqx.tree_at(lambda s: s.rpe_baseline, state, new_baseline)
