"""Receptor pharmacology — pure JAX (Doya 2002; Seamans & Yang 2004;
Surmeier et al. 2007).

The legacy dict-of-enum representation is replaced by a single fixed-order
vector-of-parameters pytree so that the receptor layer vectorises cleanly
over all subtypes in one call. The canonical order is :data:`RECEPTOR_ORDER`.

Effects compose multiplicatively (Silver 2010): ``gain = ∏ (1 + eff_i)``
which mirrors the independent G-protein cascades (Gs, Gi, Gq).
"""

from __future__ import annotations

from enum import Enum, auto
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from .backend import DTYPE, Array


class ReceptorType(Enum):
    """Neurotransmitter receptor subtypes."""
    # Dopamine
    D1 = auto()   # Excitatory, cAMP↑, Go pathway (Surmeier et al. 2007)
    D2 = auto()   # Inhibitory, cAMP↓, NoGo pathway
    # Acetylcholine
    M1 = auto()   # Cortical excitatory (muscarinic)
    M4 = auto()   # Striatal inhibitory (muscarinic)
    NACHR = auto()  # Fast nicotinic (thalamic input gating)
    # Noradrenaline
    ALPHA1 = auto()
    ALPHA2 = auto()
    BETA = auto()
    # Serotonin
    HT1A = auto()
    HT2A = auto()


RECEPTOR_ORDER: tuple[ReceptorType, ...] = (
    ReceptorType.D1,
    ReceptorType.D2,
    ReceptorType.M1,
    ReceptorType.M4,
    ReceptorType.NACHR,
    ReceptorType.ALPHA1,
    ReceptorType.ALPHA2,
    ReceptorType.BETA,
    ReceptorType.HT1A,
    ReceptorType.HT2A,
)

_RECEPTOR_TABLE: dict[ReceptorType, tuple[float, float, float, float]] = {
    ReceptorType.D1:     (0.4, 1.5, 1.0, +1.0),
    ReceptorType.D2:     (0.3, 1.2, 1.0, -1.0),
    ReceptorType.M1:     (0.4, 1.0, 1.0, +1.0),
    ReceptorType.M4:     (0.5, 1.0, 0.8, -1.0),
    ReceptorType.NACHR:  (0.3, 2.0, 1.0, +1.0),
    ReceptorType.ALPHA1: (0.5, 1.0, 1.0, +1.0),
    ReceptorType.ALPHA2: (0.2, 1.5, 0.6, -1.0),
    ReceptorType.BETA:   (0.6, 1.0, 0.8, +1.0),
    ReceptorType.HT1A:   (0.4, 1.0, 1.0, -1.0),
    ReceptorType.HT2A:   (0.5, 1.5, 1.0, +1.0),
}

TRANSMITTER_INDICES: dict[str, tuple[int, ...]] = {
    "da":   (0, 1),
    "ach":  (2, 3, 4),
    "ne":   (5, 6, 7),
    "sero": (8, 9),
}

PLASTICITY_MASK: tuple[bool, ...] = tuple(
    rt in {ReceptorType.D1, ReceptorType.BETA, ReceptorType.M1, ReceptorType.HT2A}
    for rt in RECEPTOR_ORDER
)


class ReceptorParams(eqx.Module):
    """Vector-of-parameters pytree across all subtypes in :data:`RECEPTOR_ORDER`."""

    ec50: Array
    hill_n: Array
    r_max: Array
    sign: Array
    transmitter_idx: Array
    plasticity_mask: Array


def init_receptor_params(*, dtype=DTYPE) -> ReceptorParams:
    """Build the canonical receptor pytree from the literature table."""
    ec50 = jnp.asarray([_RECEPTOR_TABLE[rt][0] for rt in RECEPTOR_ORDER], dtype=dtype)
    hill = jnp.asarray([_RECEPTOR_TABLE[rt][1] for rt in RECEPTOR_ORDER], dtype=dtype)
    rmax = jnp.asarray([_RECEPTOR_TABLE[rt][2] for rt in RECEPTOR_ORDER], dtype=dtype)
    sign = jnp.asarray([_RECEPTOR_TABLE[rt][3] for rt in RECEPTOR_ORDER], dtype=dtype)

    tx_names = ("da", "ach", "ne", "sero")
    tx_idx = [0] * len(RECEPTOR_ORDER)
    for tx_i, tx_name in enumerate(tx_names):
        for sub_i in TRANSMITTER_INDICES[tx_name]:
            tx_idx[sub_i] = tx_i
    return ReceptorParams(
        ec50=ec50,
        hill_n=hill,
        r_max=rmax,
        sign=sign,
        transmitter_idx=jnp.asarray(tx_idx, dtype=jnp.int32),
        plasticity_mask=jnp.asarray(PLASTICITY_MASK, dtype=jnp.bool_),
    )


def hill_response(
    concentration: Array, ec50: Array, hill_n: Array, r_max: Array
) -> Array:
    """Vectorised Hill equation ``r_max · c^n / (c^n + ec50^n)``."""
    c = jnp.clip(concentration, 0.0, 1.0).astype(DTYPE)
    c_n = c ** hill_n
    ec_n = ec50 ** hill_n
    return r_max * c_n / (c_n + ec_n + jnp.asarray(1e-10, DTYPE))


def receptor_effects(
    params: ReceptorParams,
    transmitter_levels: Array,
    densities: Array,
) -> Array:
    """Signed per-subtype effect vector (shape ``(n_subtypes,)``).

    ``transmitter_levels`` is ``(4,)`` in order ``(da, ach, ne, sero)``;
    ``densities`` is ``(n_subtypes,)`` with values in ``[0, 1]``.
    """
    conc = transmitter_levels[params.transmitter_idx]
    response = hill_response(conc, params.ec50, params.hill_n, params.r_max)
    return params.sign * densities * response


class LayerModulation(NamedTuple):
    """Multiplicative gains emitted by :func:`aggregate_effects`.

    ``gain_mod`` scales membrane excitability / input current.
    ``plasticity_mod`` scales STDP learning rate. Both clipped at ``0.1``.
    """

    gain_mod: Array
    plasticity_mod: Array


def aggregate_effects(
    params: ReceptorParams, effects: Array
) -> LayerModulation:
    """Compose per-subtype effects into ``(gain_mod, plasticity_mod)``."""
    gain = jnp.prod(1.0 + effects)
    gain = jnp.maximum(gain, jnp.asarray(0.1, DTYPE))

    plast_contrib = jnp.where(params.plasticity_mask, 1.0 + effects, 1.0)
    plast = jnp.prod(plast_contrib)
    plast = jnp.maximum(plast, jnp.asarray(0.1, DTYPE))

    return LayerModulation(gain_mod=gain, plasticity_mod=plast)


def compute_layer_modulation(
    params: ReceptorParams,
    transmitter_levels: Array,
    densities: Array,
) -> LayerModulation:
    """Convenience wrapper: transmitter + density → ``LayerModulation``."""
    eff = receptor_effects(params, transmitter_levels, densities)
    return aggregate_effects(params, eff)
