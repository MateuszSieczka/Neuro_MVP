"""LGN adapter — contrast gain control between retina and thalamus.

Biologically the lateral geniculate nucleus performs three things that
matter for our scale-invariance story:

1. **Tonic spontaneous firing**. LGN relay cells in the awake tonic
   mode fire around 5-15 Hz even without a stimulus (Kaplan & Shapley
   1984, Sherman 2001 "Tonic and burst firing: Dual modes of
   thalamocortical relay"). This tonic drive keeps the thalamo-cortical
   loop warm; without it, cortex is silent during dim/uniform scenes.
2. **Contrast gain control**. The mean *additional* firing above the
   tonic baseline is normalised so that the overall operating point is
   roughly independent of scene contrast (Shapley & Enroth-Cugell 1984,
   Mante, Bonin & Carandini 2005 *Nat Neurosci*).
3. **TRN gating** (Phase 4.2+). The thalamic reticular nucleus will
   modulate LGN transmission under PFC control. Phase 4.0 leaves this
   at unit gain.

Why we need it
--------------
The retina emits rectified DoG responses — most pixels sit at exactly
zero, and only contrast edges fire. Without any tonic component the
afferent vector has ~90% zeros and a tiny mean (~0.03). Cortical
rheobase is calibrated assuming dense afferents of mean ~0.25 (see
``embodiment.body_interface.uniform_dense`` and the Gaussian place
codes). Without an LGN baseline the chain sits below threshold, the
critic eligibility collapses, and learning cannot start. Adding the
tonic baseline is the anatomically correct fix.

Design
------
Stateless pure function of the retinal afferent vector::

    normalised = baseline + gain * afferent
    gain       = (target_mean - baseline) / (mean(afferent) + eps)

with a clip to ``[0, 1]``. ``baseline`` is the tonic component and
``target_mean`` is the operating-point mean of the normalised vector.
A small ``semi_saturation`` constant in the denominator prevents the
gain from blowing up on very dim scenes (Heeger 1992).

References
----------
- Kaplan & Shapley (1984) "The origin of the S (slow) potential in
  the mammalian lateral geniculate nucleus." *Exp. Brain Res.*
- Shapley & Enroth-Cugell (1984) "Visual adaptation and retinal gain
  controls." *Progress in Retinal Research*.
- Sherman (2001) *Trends Neurosci.* 24: 122-126 — tonic vs burst
  relay modes.
- Heeger (1992) "Normalization of cell responses in cat striate
  cortex." *Vis. Neurosci.* 9: 181-197.
- Mante, Bonin & Carandini (2005) *Nat. Neurosci.* 8: 1690-1697.
- Sherman & Guillery (2006) *Exploring the Thalamus and Its Role in
  Cortical Function*.
"""

from __future__ import annotations

import jax.numpy as jnp

from core.backend import DTYPE, Array


def lgn_normalize(
    afferent: Array,
    *,
    target_mean: float = 0.25,
    baseline: float = 0.15,
    semi_saturation: float = 0.05,
) -> Array:
    """Contrast-normalised afferent with tonic baseline, in ``[0, 1]``.

    Parameters
    ----------
    afferent:
        ``(N,)`` float32 retinal rates (e.g. ``RetinalSample.as_afferent()``).
        Expected to be sparse (rectified DoG).
    target_mean:
        Desired post-normalisation mean. ``0.25`` matches the Poisson
        rates the cortex / BG are calibrated for.
    baseline:
        Tonic spontaneous rate added to every afferent. ``0.15``
        corresponds to ~10-15 Hz spontaneous firing as a fraction of
        the ~60 Hz peak rate (Kaplan & Shapley 1984, Sherman 2001). It
        is what keeps the thalamo-cortical loop above rheobase on
        uniform scenes.
    semi_saturation:
        Additive constant in the gain denominator (Heeger 1992). Caps
        the amplification on very dim scenes so that noise is not
        boosted to signal level.

    Returns
    -------
    ``(N,)`` float32 in ``[0, 1]`` with mean ≈ ``target_mean`` on
    typical retinal inputs.

    Notes
    -----
    Must be called on raw retinal afferents, not already-normalised
    ones. Applying twice stacks gains and saturates everything.
    """
    a = jnp.asarray(afferent, DTYPE)
    mu = a.mean()
    # Gain amplifies the *signal* component so that after adding the
    # tonic baseline the overall mean sits at target_mean.
    target = jnp.asarray(target_mean, DTYPE)
    base = jnp.asarray(baseline, DTYPE)
    gain = (target - base) / (mu + jnp.asarray(semi_saturation, DTYPE))
    # Clamp gain to a physiologically plausible range: below 1 would
    # correspond to attenuating already-bright afferents, above ~20 to
    # cranking dark noise to signal levels. Mante 2005 Fig. 3 shows
    # contrast gain varying ~4x around baseline.
    gain = jnp.clip(gain, 0.5, 10.0)
    out = base + gain * a
    return jnp.clip(out, 0.0, 1.0).astype(DTYPE)
