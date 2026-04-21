"""Hippocampus wrapper — DG + CA3 + CA1 composite.

Thin composite module tying together three existing primitives
into a single hippocampal pytree with theta-gated dynamics:

  * **DG** = :mod:`core.episodic_memory` — sparse pattern-separated
    codes of the entorhinal input, NE-gated one-shot storage
    (Sara 2009 noradrenergic gating of novel-memory encoding).
  * **CA3** = :mod:`core.sequence_memory` — auto-associative
    transition memory operating on the DG-sparse code; recurrent
    weights learn :math:`P(s_{t+1} | s_t)` so incomplete cues can
    be completed from their temporal context (Treves & Rolls 1994;
    Hasselmo & Eichenbaum 2005).
  * **CA1** = novelty comparator — compares the CA3 recall against
    the live EC input and emits a mismatch scalar that drives ACh
    release in basal-forebrain cholinergic terminals
    (McGaughy 2008; Lisman & Grace 2005 HC–VTA loop).

Theta gating (Hasselmo 2002 "dynamical model of theta-paced
encoding vs retrieval"):

    encoding_gate = 0.5 · (1 + cos(theta − π/2))    # peaks on ascending
    recall_gate   = 1 − encoding_gate               # peaks on descending

The encoding gate modulates the DG write probability; the recall
gate modulates how strongly CA3 pattern completion feeds CA1.

The module is JIT-safe: all branches are mask-composed with
``jnp.where``; :func:`hippocampus_step` does not consume a Python
conditional on any JAX scalar.

References
----------
  Rolls (2013)                 — Pattern separation/completion in HC.
  Hasselmo (2002)              — Theta phase encoding/retrieval.
  Treves & Rolls (1994)        — CA3 auto-associator.
  McGaughy (2008)              — CA1 mismatch → BF cholinergic release.
  Sara (2009)                  — NE gating of novel-memory storage.
  Lisman & Grace (2005)        — HC–VTA loop and novelty DA.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, BackendContext, split_key
from .episodic_memory import (
    EpisodicParams, EpisodicState,
    init_episodic_params, init_episodic_state,
    try_store, recall, dg_encode,
)
from .sequence_memory import (
    SeqMemParams, SeqMemState,
    init_seqmem_params, init_seqmem_state,
    seqmem_step, seqmem_novelty,
)


# =====================================================================
# Params / state
# =====================================================================


class HippocampusParams(eqx.Module):
    """DG + CA3 + CA1 scalar mismatch weight.

    ``input_dim`` is the width of the EC → HC projection (the perforant
    path).  DG and CA3 both see vectors of this width; CA1 mismatch is
    computed in the same space for direct comparison.
    """

    dg: EpisodicParams
    ca3: SeqMemParams
    ca1_mismatch_weight: Array           # scalar — scales ACh boost
    input_dim: int = eqx.field(static=True)


class HippocampusState(eqx.Module):
    dg: EpisodicState
    ca3: SeqMemState
    ca1_prev_recall: Array               # (input_dim,) last cycle's CA3 recall
    last_novelty: Array                  # scalar ∈ [0, 1] — DG-based
    last_mismatch: Array                 # scalar ≥ 0 — CA1 comparator
    last_recall: Array                   # (input_dim,) diagnostic readout


def init_hippocampus_params(
    input_dim: int,
    *,
    dg_capacity: int = 500,
    dg_expansion_factor: int = 5,
    dg_sparsity: float = 0.05,
    ne_threshold: float = 0.3,
    similarity_thresh: float = 0.85,
    ca3_expansion_factor: int = 4,
    ca3_sparsity_k: float = 0.1,
    ca1_mismatch_weight: float = 0.5,
) -> HippocampusParams:
    dg_p = init_episodic_params(
        state_dim=int(input_dim),
        capacity=int(dg_capacity),
        ne_threshold=float(ne_threshold),
        similarity_thresh=float(similarity_thresh),
        dg_sparsity=float(dg_sparsity),
        dg_expansion_factor=int(dg_expansion_factor),
    )
    ca3_p = init_seqmem_params(
        n_in=int(input_dim),
        expansion_factor=int(ca3_expansion_factor),
        sparsity_k=float(ca3_sparsity_k),
    )
    return HippocampusParams(
        dg=dg_p, ca3=ca3_p,
        ca1_mismatch_weight=jnp.asarray(ca1_mismatch_weight, DTYPE),
        input_dim=int(input_dim),
    )


def init_hippocampus_state(
    key: PRNGKey, params: HippocampusParams, *, dtype=DTYPE,
) -> HippocampusState:
    k_dg, k_ca3 = split_key(key, 2)
    z = jnp.asarray(0.0, dtype)
    return HippocampusState(
        dg=init_episodic_state(k_dg, params.dg, dtype=dtype),
        ca3=init_seqmem_state(k_ca3, params.ca3, dtype=dtype),
        ca1_prev_recall=jnp.zeros(params.input_dim, dtype),
        last_novelty=z,
        last_mismatch=z,
        last_recall=jnp.zeros(params.input_dim, dtype),
    )


# =====================================================================
# Step
# =====================================================================


class HippocampusOutput(NamedTuple):
    state: HippocampusState
    ca1_recall: Array        # (input_dim,) CA3 pattern-completion → CA1
    novelty: Array           # scalar ∈ [0, 1] — CA3 temporal-error derived
    mismatch: Array          # scalar ≥ 0 — ||ca3_recall_prev − ec_in||


def _theta_gates(theta_phase: Array) -> tuple[Array, Array]:
    """Hasselmo (2002) encoding/retrieval modulation.

    Ascending theta (θ ≈ 0..π) peaks the encoding gate; descending
    theta (θ ≈ π..2π) peaks the retrieval gate.  Both lie in [0, 1]
    and sum to 1 identically so that the same input budget is shared.
    """
    theta = jnp.asarray(theta_phase, DTYPE)
    enc = 0.5 * (1.0 + jnp.cos(theta - jnp.asarray(math.pi / 2.0, DTYPE)))
    rec = 1.0 - enc
    return enc, rec


@eqx.filter_jit
def hippocampus_step(
    state: HippocampusState,
    params: HippocampusParams,
    ec_in: Array,
    *,
    theta_phase: Array | float,
    ne_level: Array | float,
    reward: Array | float = 0.0,
    action: Array | int = 0,
) -> HippocampusOutput:
    """Run DG pattern-separation → CA3 completion → CA1 comparator.

    Parameters
    ----------
    ec_in:
        ``(input_dim,)`` perforant-path vector from EC L2/3.  This is
        both the write cue for DG and the read cue for CA3, and the
        reference against which the previous cycle's CA3 recall is
        compared inside CA1.
    theta_phase:
        Scalar theta phase in radians (from the global oscillator).
        Gates the balance between encoding and retrieval per
        Hasselmo (2002).
    ne_level:
        Current NE level; only when it exceeds ``dg.ne_threshold`` does
        DG commit a new episode (Sara 2009 novelty-gated encoding).
    reward, action:
        Side-channel fields stored alongside the DG key so that the
        episode is a full ``(s, a, r)`` tuple for Phase 5B replay.

    Returns
    -------
    HippocampusOutput
        * ``ca1_recall``: pattern-completed vector from CA3 (used by
          the brain graph as a memory-augmented afferent).
        * ``novelty``: |CA3 temporal error|-derived scalar driving
          replay salience and neuromodulatory ACh.
        * ``mismatch``: |CA3 recall_{t-1} − ec_in|-derived scalar
          driving CA1's novelty signal to the basal-forebrain.
    """
    ec_f = ec_in.astype(DTYPE)
    enc_gate, rec_gate = _theta_gates(theta_phase)

    # --- CA1 mismatch comparator (against PREVIOUS recall — because it
    #     was the prediction the HC made on the last cycle).
    mismatch = jnp.mean(jnp.abs(state.ca1_prev_recall - ec_f))
    mismatch = jnp.clip(
        mismatch * params.ca1_mismatch_weight, 0.0, 1.0,
    )

    # --- DG write: NE-gated, novelty-gated, theta-encoding-gated.
    # The encoding gate is applied as a multiplicative scalar on the
    # effective NE level so that on descending theta the storage
    # probability collapses smoothly to zero (Hasselmo 2002).
    ne_eff = jnp.asarray(ne_level, DTYPE) * enc_gate
    store_out = try_store(
        state.dg, params.dg,
        s=ec_f, a=action, r=reward, s_next=ec_f,
        ne_level=ne_eff,
    )
    dg_new = store_out.state

    # --- CA3: seqmem learns the transition from previous EC input to
    # current EC input; the prediction it emits is our CA1 recall
    # content.  Scaled by retrieval gate on output so recall is
    # theta-rhythmic (Hasselmo 2002 again).
    ca3_out = seqmem_step(state.ca3, params.ca3, ec_f)
    ca1_recall = (ca3_out.state.predicted_next * rec_gate).astype(DTYPE)
    novelty = seqmem_novelty(ca3_out.state)

    new_state = HippocampusState(
        dg=dg_new,
        ca3=ca3_out.state,
        ca1_prev_recall=ca1_recall,
        last_novelty=novelty,
        last_mismatch=mismatch,
        last_recall=ca1_recall,
    )
    return HippocampusOutput(
        state=new_state,
        ca1_recall=ca1_recall,
        novelty=novelty,
        mismatch=mismatch,
    )
