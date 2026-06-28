"""V2 and V4/IT — higher-order visual cortices (skeleton).

Design
------
Both V2 and V4/IT are standard cortical areas that receive their
feed-forward drive from the preceding area's L2/3 error channel
(``CorticalOutput.ff_out``). They share cortex.py's STDP and inhibitory
plasticity; no hand-designed features here.

Sizes follow the Phase 4 plan: V2 around 500 neurons, V4/IT around 300
— intentionally small to keep the JIT trace cheap during scaffolding.
The plan explicitly defers pretrained features to optional external
adapters; these builders contain no domain knowledge beyond the
biological defaults of ``init_cortical_area_params``.

References
----------
- Felleman & Van Essen (1991) *Cereb. Cortex* 1: 1-47 — ventral stream
  hierarchy.
- Gattass et al. (2005) — V2/V4 properties.
- DiCarlo, Zoccolan & Rust (2012) *Neuron* 73: 415-434 — IT as object
  representation.
"""

from __future__ import annotations

from core.backend import BackendContext, PRNGKey
from core.cortex import (
    CorticalAreaParams,
    CorticalAreaState,
    init_cortical_area_params,
    init_cortical_area_state,
)


def init_v2_params(
    ctx: BackendContext,
    input_size: int,
    *,
    n_l4: int = 128,
    n_l23_state: int = 128,
    n_l23_error: int = 64,
    n_l5: int = 64,
) -> CorticalAreaParams:
    """V2 params sized for V1 ``ff_out`` as input (default 64 channels)."""
    return init_cortical_area_params(
        ctx,
        input_size=input_size,
        n_l4=n_l4,
        n_l23_state=n_l23_state,
        n_l23_error=n_l23_error,
        n_l5=n_l5,
    )


def init_v2_state(
    key: PRNGKey, params: CorticalAreaParams,
) -> CorticalAreaState:
    return init_cortical_area_state(key, params)


def init_v4it_params(
    ctx: BackendContext,
    input_size: int,
    *,
    n_l4: int = 96,
    n_l23_state: int = 96,
    n_l23_error: int = 48,
    n_l5: int = 48,
) -> CorticalAreaParams:
    """V4/IT params sized for V2 ``ff_out`` as input."""
    return init_cortical_area_params(
        ctx,
        input_size=input_size,
        n_l4=n_l4,
        n_l23_state=n_l23_state,
        n_l23_error=n_l23_error,
        n_l5=n_l5,
    )


def init_v4it_state(
    key: PRNGKey, params: CorticalAreaParams,
) -> CorticalAreaState:
    return init_cortical_area_state(key, params)


__all__ = [
    "init_v2_params",
    "init_v2_state",
    "init_v4it_params",
    "init_v4it_state",
]
