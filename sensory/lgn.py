"""LGN adapter — contrast gain control + tonic floor on the retinal afferent.

A normalisation *stage* of the vision adapter (LEGACY_INTEGRATION.md §0.5):
it keeps the rate vector that clamps the sensory node in a sane, roughly
contrast-invariant range, so the same brain copes with dim and bright
scenes.  Pure function of the retinal afferent; no substrate dependency.

The lateral geniculate nucleus does two things that matter here:

1. **Tonic spontaneous firing** — LGN relay cells fire ~5-15 Hz even
   without a stimulus (Kaplan & Shapley 1984; Sherman 2001).  Rectified
   DoG afferents are ~90% zeros; a tonic baseline keeps the sensory clamp
   off the floor on dim/uniform scenes so downstream nodes have something
   to explain.
2. **Contrast gain control** — the mean drive above baseline is normalised
   so the operating point is roughly contrast-independent (Shapley &
   Enroth-Cugell 1984; Mante, Bonin & Carandini 2005), with a soft
   Naka-Rushton saturation of the gain (Heeger 1992) instead of a hard
   clip — smooth derivatives everywhere.

References
----------
- Kaplan & Shapley (1984) *Exp. Brain Res.* — S-potential / tonic firing.
- Shapley & Enroth-Cugell (1984) *Prog. Retinal Res.* — retinal gain.
- Sherman (2001) *Trends Neurosci.* 24: 122-126 — tonic vs burst relay.
- Heeger (1992) *Vis. Neurosci.* 9: 181-197 — divisive normalisation.
- Mante, Bonin & Carandini (2005) *Nat. Neurosci.* 8: 1690-1697.
"""

from __future__ import annotations

import jax.numpy as jnp

from core.backend import DTYPE, Array

#: Post-normalisation operating-point mean of the afferent.  A dense rate
#: target the cortical hierarchy relaxes comfortably around (the same
#: ~0.25 the proprioception/place-code afferents sit at).
LGN_TARGET_MEAN: float = 0.25
#: Tonic spontaneous component added to every channel, ~10-15 Hz as a
#: fraction of the ~60 Hz peak (Kaplan & Shapley 1984; Sherman 2001).
LGN_BASELINE: float = 0.15
#: Additive semi-saturation in the gain denominator (Heeger 1992) — caps
#: amplification on very dim scenes so noise is not boosted to signal.
LGN_SEMI_SATURATION: float = 0.05
#: Asymptotic maximum contrast gain (~4× around the adaptation mean;
#: Mante, Bonin & Carandini 2005 Fig. 3).
LGN_MAX_GAIN: float = 4.0


def lgn_normalize(
    afferent: Array,
    *,
    target_mean: float = LGN_TARGET_MEAN,
    baseline: float = LGN_BASELINE,
    semi_saturation: float = LGN_SEMI_SATURATION,
    max_gain: float = LGN_MAX_GAIN,
) -> Array:
    """Contrast-normalised afferent with a tonic baseline, in ``[0, 1]``.

    ``normalised = clip(baseline + gain · afferent, 0, 1)`` where the raw
    gain ``(target_mean − baseline) / (mean(afferent) + semi_saturation)``
    is soft-saturated by ``max_gain · tanh(g / max_gain)`` (Naka-Rushton /
    Weber-Fechner).  On typical sparse retinal input the output mean sits
    near ``target_mean``.

    Must be applied **once**, on raw retinal afferents — applying it twice
    stacks gains and saturates everything.
    """
    a = jnp.asarray(afferent, DTYPE)
    mu = a.mean()
    target = jnp.asarray(target_mean, DTYPE)
    base = jnp.asarray(baseline, DTYPE)
    gain_raw = (target - base) / (mu + jnp.asarray(semi_saturation, DTYPE))
    g_max = jnp.asarray(max_gain, DTYPE)
    gain = g_max * jnp.tanh(gain_raw / g_max)
    out = base + gain * a
    return jnp.clip(out, 0.0, 1.0).astype(DTYPE)
