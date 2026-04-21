"""Entorhinal cortex (EC) — pure JAX.

The EC is the convergence hub between neocortex and hippocampus
(Witter 2007 — medial/lateral EC integrate sensory, prefrontal and
motor streams and project the compound signal to DG/CA3 via the
perforant path).  For the MVP we abstract its six-layer
cytoarchitecture to a single canonical cortical microcircuit
(:class:`core.cortex.CorticalAreaParams`) whose feedforward input is
the concatenation

    afferent = concat[
        cortex_l23_belief,   # associational sensory ( ~ neocortex )
        pfc_content,         # goal / working-memory bias
        last_motor_joint,    # efference copy (body ⊕ saccade one-hot)
    ]

matching the three dominant EC input streams documented in Witter
(2007) Figure 2.  The output consumed by the hippocampus is the
L2/3 *state* population rate (``belief``), which is the information
the perforant path actually delivers to DG (Hasselmo 2005).

Place / grid cells and the 6-layer EC (superficial vs deep laminar
segregation) are deferred to Phase 6 when continuous body topology
comes online (Hafting 2005 grid RFs require MJX kinematics).

References
----------
  Witter (2007)              — Intrinsic and extrinsic circuits of EC.
  Hasselmo (2005)            — Theta-rhythmic phase coding in EC.
  van Strien & Cappaert (2009) — Anatomy of the hippocampal formation.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, BackendContext
from .cortex import (
    CorticalAreaParams, CorticalAreaState, CorticalInputs, CorticalOutput,
    init_cortical_area_params, init_cortical_area_state, cortical_area_step,
)


# =====================================================================
# Params / state
# =====================================================================


class EntorhinalParams(eqx.Module):
    """Wrap one :class:`CorticalAreaParams` + the three afferent sizes.

    ``output_dim`` = ``cortex.n_l23_state`` is the width of the
    perforant-path projection to DG.  Static integer sizes live here
    so ``ec_step`` can build the concatenated afferent without peeking
    at the afferent tensor shapes.
    """

    cortex: CorticalAreaParams
    n_sensory: int = eqx.field(static=True)
    n_pfc: int = eqx.field(static=True)
    n_motor: int = eqx.field(static=True)

    @property
    def output_dim(self) -> int:
        return int(self.cortex.n_l23_state)

    @property
    def input_size(self) -> int:
        return int(self.n_sensory + self.n_pfc + self.n_motor)


class EntorhinalState(eqx.Module):
    """One :class:`CorticalAreaState` (no extra state is needed)."""

    cortex: CorticalAreaState


def init_ec_params(
    ctx: BackendContext,
    *,
    n_sensory: int,
    n_pfc: int,
    n_motor: int,
    n_l4: int = 128,
    n_l23_state: int = 128,
    n_l23_error: int = 128,
    n_l5: int = 32,
) -> EntorhinalParams:
    """Build an EC as a single 128-wide canonical cortical microcircuit."""
    cx = init_cortical_area_params(
        ctx, input_size=int(n_sensory + n_pfc + n_motor),
        n_l4=n_l4, n_l23_state=n_l23_state,
        n_l23_error=n_l23_error, n_l5=n_l5,
    )
    return EntorhinalParams(
        cortex=cx,
        n_sensory=int(n_sensory),
        n_pfc=int(n_pfc),
        n_motor=int(n_motor),
    )


def init_ec_state(
    key: PRNGKey, params: EntorhinalParams, *, dtype=DTYPE,
) -> EntorhinalState:
    return EntorhinalState(
        cortex=init_cortical_area_state(key, params.cortex, dtype=dtype),
    )


# =====================================================================
# Step
# =====================================================================


@eqx.filter_jit
def ec_step(
    state: EntorhinalState,
    params: EntorhinalParams,
    ctx: BackendContext,
    sensory_belief: Array,
    pfc_content: Array,
    last_motor: Array,
    *,
    ach: Array | float = 0.5,
    da: Array | float = 0.5,
    receptor_gain: Array | float = 1.0,
    excitability_mod: Array | float = 1.0,
) -> tuple[EntorhinalState, Array]:
    """Advance EC by one ``dt`` and return ``(new_state, belief)``.

    The returned ``belief`` is the L2/3 state rate — the perforant-path
    output delivered to DG / CA3 / CA1 (Witter 2007 Fig 2).  Callers
    that want the full :class:`CorticalOutput` should use
    :func:`core.cortex.cortical_area_step` directly.
    """
    afferent = jnp.concatenate(
        [
            sensory_belief.astype(DTYPE),
            pfc_content.astype(DTYPE),
            last_motor.astype(DTYPE),
        ],
        axis=0,
    )
    cx_in = CorticalInputs(
        ff_input=afferent,
        td_prediction=None,
        ach=ach,
        da=da,
        receptor_gain=receptor_gain,
        excitability_mod=excitability_mod,
    )
    cx_out: CorticalOutput = cortical_area_step(
        state.cortex, params.cortex, ctx, cx_in,
        apply_ipool_stdp=True,
    )
    new_state = EntorhinalState(cortex=cx_out.state)
    return new_state, cx_out.belief
