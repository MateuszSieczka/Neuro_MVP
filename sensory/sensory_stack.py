"""Sensory stack — foveated retina → LGN → V1 chain (Phase 0.7 MVP).

Composes the already-validated Phase-4 sensory primitives into a single
pure-functional module. Returns both a **compact V1 belief vector**
suitable as the thalamic afferent and the **V1 prediction-error rate**
used by the saccade-actor info-gain reward (Itti & Baldi 2009).

Design notes
------------
- **Pipeline**: image + fixation → retina_step → lgn_normalize →
  cortical_area_step (= V1). V2/V4 are intentionally deferred (Phase 6
  concept emergence).
- **Output**: V1 L2/3 state-rate (`belief`) as afferent + V1 L2/3
  error-rate sum (scalar) as saliency / info-gain signal.
- **State**: retina_state (tiny) + V1 cortical state. V1 uses the same
  ``CorticalAreaParams`` infrastructure as the rest of the brain, so
  STDP / astrocyte modulation / receptor pharmacology all compose
  automatically.
- **Saccade info-gain**: r_info = max(prev_v1_pe - current_v1_pe, 0).
  That drop models Bayesian surprise reduction after an informative
  saccade.
- **JIT-safe**: all dynamic arrays; `RetinaConfig` is static.

References
----------
- Tatler (2011) — foveated active sampling.
- Itti & Baldi (2009) — Bayesian surprise as saccade reward.
- Saalmann (2012) — pulvinar attention modulation.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from core.backend import DTYPE, Array, PRNGKey, BackendContext, split_key
from core.cortex import (
    CorticalAreaParams, CorticalAreaState, CorticalInputs,
    cortical_area_step,
)
from .retina import RetinaConfig, RetinaState, retina_step, init_retina_state
from .lgn import lgn_normalize
from .v1 import init_v1_params, init_v1_state


# =====================================================================
# Params / state
# =====================================================================


class SensoryStackParams(eqx.Module):
    """Static sensory-stack hyperparameters."""

    v1: CorticalAreaParams
    retina_cfg: RetinaConfig = eqx.field(static=True)
    # LGN normalisation constants (Kaplan & Shapley 1984; Heeger 1992)
    lgn_target_mean: Array
    lgn_baseline: Array
    lgn_semi_saturation: Array


class SensoryStackState(eqx.Module):
    retina: RetinaState
    v1: CorticalAreaState


class SensoryStackOutput(NamedTuple):
    state: SensoryStackState
    belief: Array      # (n_l23_state,) V1 L2/3 state rate — predictive-coding readout
    l4_rate: Array     # (n_l4,)        V1 L4 rate EMA — corticostriatal afferent
    pe_rate: Array     # scalar — mean V1 L2/3 error rate, for info-gain


# =====================================================================
# Builders
# =====================================================================


def init_sensory_stack_params(
    ctx: BackendContext,
    *,
    retina_cfg: RetinaConfig | None = None,
    n_l4: int = 256,
    n_l23_state: int = 128,
    n_l23_error: int = 64,
    n_l5: int = 64,
    lgn_target_mean: float = 0.25,
    lgn_baseline: float = 0.15,
    lgn_semi_saturation: float = 0.05,
) -> SensoryStackParams:
    cfg = retina_cfg or RetinaConfig()
    v1_p = init_v1_params(
        ctx, cfg,
        n_l4=n_l4, n_l23_state=n_l23_state,
        n_l23_error=n_l23_error, n_l5=n_l5,
        l4_expected_input_rate=lgn_target_mean,
    )
    f = lambda x: jnp.asarray(x, DTYPE)
    return SensoryStackParams(
        v1=v1_p, retina_cfg=cfg,
        lgn_target_mean=f(lgn_target_mean),
        lgn_baseline=f(lgn_baseline),
        lgn_semi_saturation=f(lgn_semi_saturation),
    )


def init_sensory_stack_state(
    key: PRNGKey, params: SensoryStackParams, *, gabor_init: bool = True,
) -> SensoryStackState:
    (k_v1,) = split_key(key, 1)
    return SensoryStackState(
        retina=init_retina_state(params.retina_cfg),
        v1=init_v1_state(
            k_v1, params.v1, params.retina_cfg, gabor_init=gabor_init,
        ),
    )


# =====================================================================
# Step
# =====================================================================


@eqx.filter_jit
def sensory_stack_step(
    state: SensoryStackState,
    params: SensoryStackParams,
    ctx: BackendContext,
    image: Array,
    fixation_xy: Array,
    *,
    ach: Array | float = 0.5,
    da: Array | float = 0.5,
    ne: Array | float = 0.5,
    receptor_gain: Array | float = 1.0,
    excitability_mod: Array | float = 1.0,
    apply_ipool_stdp: bool = True,
) -> SensoryStackOutput:
    """Run one dt of the retina → LGN → V1 chain.

    ``image`` is ``(H, W)`` float32 in [0, 1]; ``fixation_xy`` is
    ``(2,)`` float32 normalised to [0, 1]^2.
    """
    # Retina
    new_retina, sample = retina_step(
        state.retina, params.retina_cfg, image, fixation_xy,
    )
    afferent = sample.as_afferent()

    # LGN normalisation
    lgn_af = lgn_normalize(
        afferent,
        target_mean=params.lgn_target_mean,
        baseline=params.lgn_baseline,
        semi_saturation=params.lgn_semi_saturation,
    )

    # V1 cortex step — L2/3 state rate is the belief, error rate is PE.
    v1_inputs = CorticalInputs(
        ff_input=lgn_af,
        td_prediction=None,
        ach=ach, da=da, ne=ne,
        receptor_gain=receptor_gain,
        excitability_mod=excitability_mod,
    )
    v1_out = cortical_area_step(
        state.v1, params.v1, ctx, v1_inputs, apply_ipool_stdp=apply_ipool_stdp,
    )

    new_state = SensoryStackState(retina=new_retina, v1=v1_out.state)

    # PE rate: mean across L2/3 error population, scalar.
    pe_rate = jnp.mean(v1_out.ff_out)

    return SensoryStackOutput(
        state=new_state, belief=v1_out.belief,
        l4_rate=v1_out.state.rate_l4, pe_rate=pe_rate,
    )
