"""
Receptor — dose-response curves for neurotransmitter receptor subtypes.

Reference: Doya (2002), Seamans & Yang (2004), Surmeier et al. (2007)

Each receptor subtype has a Hill equation dose-response curve:
  response = R_max × [L]^n / ([L]^n + EC50^n)

where:
  [L]   = ligand (transmitter) concentration [0, 1]
  EC50  = half-maximal effective concentration
  n     = Hill coefficient (cooperativity)
  R_max = maximal response

Receptor effects are either excitatory (+1) or inhibitory (-1),
applied multiplicatively to the target variable (threshold, gain,
learning rate, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from .config import ReceptorType


@dataclass(frozen=True, kw_only=True)
class ReceptorParams:
    """Pharmacological parameters for a single receptor subtype.

    All values from published dose-response literature.
    """
    receptor_type: ReceptorType
    ec50: float         # Half-maximal concentration [0, 1]
    hill_n: float       # Hill coefficient (cooperativity)
    r_max: float        # Maximal response magnitude
    sign: float         # +1.0 (excitatory) or -1.0 (inhibitory)
    description: str = ""


# ── Receptor pharmacology database ───────────────────────────────────
# Values derived from published dose-response curves and receptor binding studies.

RECEPTOR_PARAMS: dict[ReceptorType, ReceptorParams] = {
    # Dopamine D1 (Surmeier et al. 2007): excitatory, enhances NMDA, Go pathway
    ReceptorType.D1: ReceptorParams(
        receptor_type=ReceptorType.D1,
        ec50=0.4, hill_n=1.5, r_max=1.0, sign=1.0,
        description="D1: cAMP↑, PKA activation, NMDA potentiation",
    ),
    # Dopamine D2 (Surmeier et al. 2007): inhibitory, reduces excitability, NoGo
    ReceptorType.D2: ReceptorParams(
        receptor_type=ReceptorType.D2,
        ec50=0.3, hill_n=1.2, r_max=1.0, sign=-1.0,
        description="D2: cAMP↓, reduced excitability, NoGo pathway",
    ),
    # Muscarinic M1 (Hasselmo 2006): cortical excitatory
    ReceptorType.M1: ReceptorParams(
        receptor_type=ReceptorType.M1,
        ec50=0.4, hill_n=1.0, r_max=1.0, sign=1.0,
        description="M1: cortical excitability increase, WM maintenance",
    ),
    # Muscarinic M4 (Shen et al. 2015): striatal inhibitory
    ReceptorType.M4: ReceptorParams(
        receptor_type=ReceptorType.M4,
        ec50=0.5, hill_n=1.0, r_max=0.8, sign=-1.0,
        description="M4: striatal MSN inhibition, pause in BG output",
    ),
    # Nicotinic (Dani & Bertrand 2007): fast excitatory, thalamic gating
    ReceptorType.NACHR: ReceptorParams(
        receptor_type=ReceptorType.NACHR,
        ec50=0.3, hill_n=2.0, r_max=1.0, sign=1.0,
        description="nAChR: fast excitation, thalamic input gating",
    ),
    # Alpha-1 NE (Berridge & Waterhouse 2003): excitatory arousal
    ReceptorType.ALPHA1: ReceptorParams(
        receptor_type=ReceptorType.ALPHA1,
        ec50=0.5, hill_n=1.0, r_max=1.0, sign=1.0,
        description="α1: cortical arousal, enhanced signal processing",
    ),
    # Alpha-2 NE (Arnsten 2000): presynaptic inhibition (autoreceptor)
    ReceptorType.ALPHA2: ReceptorParams(
        receptor_type=ReceptorType.ALPHA2,
        ec50=0.2, hill_n=1.5, r_max=0.6, sign=-1.0,
        description="α2: presynaptic autoreceptor, reduces NE release",
    ),
    # Beta NE (Sara 2009): slow modulatory
    ReceptorType.BETA: ReceptorParams(
        receptor_type=ReceptorType.BETA,
        ec50=0.6, hill_n=1.0, r_max=0.8, sign=1.0,
        description="β: slow modulatory, enhances plasticity",
    ),
    # 5-HT1A (Doya 2002): inhibitory, patience/temporal discounting
    ReceptorType.HT1A: ReceptorParams(
        receptor_type=ReceptorType.HT1A,
        ec50=0.4, hill_n=1.0, r_max=1.0, sign=-1.0,
        description="5-HT1A: inhibitory, increases temporal patience",
    ),
    # 5-HT2A (Doya 2002): excitatory, modulates perception
    ReceptorType.HT2A: ReceptorParams(
        receptor_type=ReceptorType.HT2A,
        ec50=0.5, hill_n=1.5, r_max=1.0, sign=1.0,
        description="5-HT2A: excitatory, enhances cortical responsiveness",
    ),
}


def hill_response(
    concentration: float,
    ec50: float,
    hill_n: float,
    r_max: float = 1.0,
) -> float:
    """Hill equation dose-response curve.

    response = R_max × [L]^n / ([L]^n + EC50^n)

    Args:
        concentration: Ligand concentration [0, 1].
        ec50:          Half-maximal effective concentration.
        hill_n:        Hill coefficient.
        r_max:         Maximum response.

    Returns:
        Response magnitude in [0, R_max].
    """
    c = float(np.clip(concentration, 0.0, 1.0))
    if c < 1e-10:
        return 0.0
    c_n = c ** hill_n
    return r_max * c_n / (c_n + ec50 ** hill_n)


def receptor_effect(
    receptor_type: ReceptorType,
    transmitter_level: float,
    density: float = 1.0,
) -> float:
    """Compute signed receptor effect from transmitter level and density.

    effect = sign × density × hill_response(transmitter, ec50, n, R_max)

    Args:
        receptor_type:    Which receptor subtype.
        transmitter_level: Global transmitter concentration [0, 1].
        density:          Local receptor density [0, 1].

    Returns:
        Signed effect: positive = excitatory, negative = inhibitory.
        Magnitude in [-R_max × density, +R_max × density].
    """
    if receptor_type not in RECEPTOR_PARAMS:
        return 0.0
    params = RECEPTOR_PARAMS[receptor_type]
    response = hill_response(
        transmitter_level, params.ec50, params.hill_n, params.r_max,
    )
    return params.sign * density * response


def compute_layer_modulation(
    transmitter_levels: dict[str, float],
    receptor_densities: dict[ReceptorType, float],
) -> dict[ReceptorType, float]:
    """Compute all receptor effects for a layer given global transmitter levels.

    Args:
        transmitter_levels: {"da": 0.5, "ach": 0.3, "ne": 0.6, "sero": 0.4}
        receptor_densities: {ReceptorType.D1: 0.8, ReceptorType.M1: 0.3, ...}

    Returns:
        Dict mapping each expressed receptor → signed effect magnitude.
    """
    # Map transmitter names to receptor types
    _transmitter_receptors: dict[str, list[ReceptorType]] = {
        "da": [ReceptorType.D1, ReceptorType.D2],
        "ach": [ReceptorType.M1, ReceptorType.M4, ReceptorType.NACHR],
        "ne": [ReceptorType.ALPHA1, ReceptorType.ALPHA2, ReceptorType.BETA],
        "sero": [ReceptorType.HT1A, ReceptorType.HT2A],
    }

    effects: dict[ReceptorType, float] = {}
    for transmitter_name, level in transmitter_levels.items():
        receptors = _transmitter_receptors.get(transmitter_name, [])
        for rt in receptors:
            density = receptor_densities.get(rt, 0.0)
            if density > 0.0:
                effects[rt] = receptor_effect(rt, level, density)

    return effects


def aggregate_receptor_effects(
    effects: dict[ReceptorType, float],
) -> tuple[float, float]:
    """Aggregate individual receptor effects into gain and plasticity modulation.

    Returns:
        (gain_mod, plasticity_mod) where 1.0 = no modulation.
        gain_mod scales membrane excitability / input current.
        plasticity_mod scales STDP learning rate.
    """
    # Gain weights: how strongly each receptor type affects excitability.
    # Signs already encoded in the receptor effect values.
    _GAIN_WEIGHTS: dict[ReceptorType, float] = {
        ReceptorType.D1: 1.0,
        ReceptorType.D2: 1.0,
        ReceptorType.M1: 0.5,
        ReceptorType.M4: 0.5,
        ReceptorType.NACHR: 0.7,
        ReceptorType.ALPHA1: 0.6,
        ReceptorType.ALPHA2: 0.4,
        ReceptorType.BETA: 0.3,
        ReceptorType.HT1A: 0.4,
        ReceptorType.HT2A: 0.5,
    }
    # Plasticity weights: D1+NMDA, Beta-NE, M1 enhance plasticity.
    _PLASTICITY_WEIGHTS: dict[ReceptorType, float] = {
        ReceptorType.D1: 0.8,
        ReceptorType.BETA: 0.5,
        ReceptorType.M1: 0.3,
        ReceptorType.HT2A: 0.2,
    }

    gain_delta = sum(
        _GAIN_WEIGHTS.get(rt, 0.0) * eff for rt, eff in effects.items()
    )
    gain_mod = max(1.0 + gain_delta * 0.3, 0.1)

    plast_delta = sum(
        _PLASTICITY_WEIGHTS.get(rt, 0.0) * eff for rt, eff in effects.items()
    )
    plasticity_mod = max(1.0 + plast_delta * 0.5, 0.1)

    return gain_mod, plasticity_mod
