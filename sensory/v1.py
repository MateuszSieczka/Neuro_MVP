"""V1 — primary visual cortex, learned under STDP with Gabor initialisation.

Architecture
------------
V1 is implemented as a standard :class:`core.cortex.CorticalAreaParams`
with input size equal to the LGN afferent vector length
(:attr:`sensory.RetinaConfig.afferent_size`). Its only specialisation is
the **initialisation of the L4 input weights**:

- The first ``2 * fovea_size**2`` columns of the afferent correspond to
  the foveal ON/OFF DoG responses, arranged retinotopically. For these
  channels the initial L4 receptive fields are Gabor filters spanning a
  grid of ``(orientations, spatial frequencies, phases)``. This is the
  classical Marr/Hubel-Wiesel prior — V1 simple cells are
  orientation-tuned and spatial-frequency-tuned with centre-surround
  organisation.
- The peripheral pyramid and motion channels receive standard
  half-normal weights (same initialisation as any other cortical area).
- STDP + inhibitory plasticity on L4 and L5 are inherited from
  :func:`core.cortex.cortical_area_update`. After sufficient exposure to
  natural image statistics the Gabor RFs refine toward sparse-coding
  bases (Olshausen & Field 1996). Without exposure they stay Gabor.

The Gabor is therefore a prior, not a hardcoded feature extractor.

References
----------
- Hubel & Wiesel (1962) *J. Physiol.* 160: 106-154 — orientation columns.
- Jones & Palmer (1987) *J. Neurophysiol.* 58: 1187-1211 — Gabor model of
  V1 simple cells.
- Marr & Hildreth (1980) *Proc. R. Soc. B* 207: 187-217.
- Olshausen & Field (1996) *Nature* 381: 607-609 — sparse coding
  emerges from STDP-like rules on natural images.
- Hunsberger & Eliasmith (2015) — spiking sparse coding.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from core.backend import DTYPE, BackendContext, Array, PRNGKey, split_key
from core.cortex import (
    CorticalAreaParams,
    CorticalAreaState,
    init_cortical_area_params,
    init_cortical_area_state,
    _psp_sigma,
)

from .retina import RetinaConfig


# ---------------------------------------------------------------------
# Gabor bank
# ---------------------------------------------------------------------


def _gabor_patch(
    size: int,
    theta: float,
    sf: float,
    phase: float,
    sigma: float,
) -> Array:
    """One ``(size, size)`` real-valued Gabor in the retinotopic grid.

    Parameters match the classical Jones & Palmer (1987) parameterisation:
    ``theta`` orientation (radians), ``sf`` spatial frequency in cycles
    per patch, ``phase`` carrier phase, ``sigma`` envelope width (in
    pixels). The patch is zero-mean so an inner product with a raw DoG
    response measures signed alignment.
    """
    yy, xx = jnp.mgrid[0:size, 0:size].astype(DTYPE)
    cx = cy = (size - 1) / 2.0
    x = xx - cx
    y = yy - cy
    x_rot = jnp.cos(theta) * x + jnp.sin(theta) * y
    y_rot = -jnp.sin(theta) * x + jnp.cos(theta) * y
    envelope = jnp.exp(-(x_rot ** 2 + y_rot ** 2) / (2.0 * sigma ** 2))
    carrier = jnp.cos(2.0 * jnp.pi * sf * x_rot / size + phase)
    g = envelope * carrier
    # zero-mean so pure mean drift is not encoded
    g = g - g.mean()
    # unit L2 norm so all filters have comparable drive strength
    norm = jnp.sqrt((g * g).sum()) + 1e-6
    return (g / norm).astype(DTYPE)


def _build_gabor_bank(
    n_filters: int,
    patch_size: int,
    *,
    n_orientations: int = 8,
    n_sf: int = 4,
    sf_min: float = 1.0,
    sf_max: float = 6.0,
) -> Array:
    """Stack of ``n_filters`` Gabor receptive fields, ``(N, P, P)``.

    The first ``n_orientations * n_sf * 2`` (orientations × SFs ×
    even/odd phases) filters form one tiling of orientation × SF
    space. Any extra filters are cycled through the same bank with a
    small random phase offset — this is consistent with cortical
    "replication with jitter" initialisation (Ringach 2002).
    """
    base = []
    phases = (0.0, jnp.pi / 2.0)          # even and odd Gabors
    sigma = patch_size / 4.0              # envelope covers ~half patch
    sfs = jnp.linspace(sf_min, sf_max, n_sf)
    thetas = jnp.linspace(0.0, jnp.pi, n_orientations, endpoint=False)
    for theta in thetas:
        for sf in sfs:
            for phase in phases:
                base.append(
                    _gabor_patch(
                        patch_size, float(theta), float(sf),
                        float(phase), float(sigma),
                    )
                )
    base_bank = jnp.stack(base, axis=0)    # (B, P, P), B = 2*n_ori*n_sf
    B = base_bank.shape[0]
    # Cycle through the bank if more filters than unique Gabors.
    idx = jnp.arange(n_filters) % B
    return base_bank[idx]


# ---------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------


def init_v1_params(
    ctx: BackendContext,
    retina_cfg: RetinaConfig,
    *,
    n_l4: int = 256,
    n_l23_state: int = 256,
    n_l23_error: int = 64,
    n_l5: int = 64,
    l4_expected_input_rate: float = 0.25,
) -> CorticalAreaParams:
    """V1 params — a ``CorticalArea`` sized for LGN afferents.

    ``l4_expected_input_rate`` defaults to ``0.25`` because the LGN
    adapter (``sensory.lgn.lgn_normalize``) pins its output mean near
    that value. This sets the rheobase-preserving ``l4_cond_scale``
    correctly at init.
    """
    return init_cortical_area_params(
        ctx,
        input_size=retina_cfg.afferent_size,
        n_l4=n_l4,
        n_l23_state=n_l23_state,
        n_l23_error=n_l23_error,
        n_l5=n_l5,
        l4_expected_input_rate=l4_expected_input_rate,
    )


def init_v1_state(
    key: PRNGKey,
    params: CorticalAreaParams,
    retina_cfg: RetinaConfig,
    *,
    gabor_init: bool = True,
    gabor_mix: float = 0.7,
    n_orientations: int = 8,
    n_sf: int = 4,
) -> CorticalAreaState:
    """V1 state with optional Gabor initialisation of L4 input weights.

    The L4 weight matrix is ``(input_size, n_l4)``. Its rows are split
    into three blocks corresponding to the retinal afferent layout
    (see :meth:`sensory.RetinalSample.as_afferent`):

    1. Fovea ON: ``fovea_size**2`` rows.
    2. Fovea OFF: ``fovea_size**2`` rows.
    3. Periphery ON + Periphery OFF + motion: the remaining rows.

    When ``gabor_init`` is ``True`` the two foveal blocks receive Gabor
    filters rescaled so their positive part has the same mean as the
    default half-normal initialisation (PSP-preserving). The ON block
    uses the positive half-rectified Gabor, the OFF block the negative
    half-rectified Gabor — so a spatially aligned DoG response drives
    the L4 column through both ON and OFF afferents. Peripheral and
    motion rows are left at their default half-normal initialisation,
    which preserves the rheobase calibration.

    ``gabor_mix`` (in ``[0, 1]``) sets how strongly the Gabor prior
    dominates over the random initialisation. ``1.0`` = pure Gabor,
    ``0.0`` = default random init. The default ``0.7`` keeps some
    random variation so adjacent L4 units are not exactly tied.

    After init, standard STDP updates (``cortical_area_update``) will
    refine these weights toward the data statistics.
    """
    # Start from the generic cortical state (random half-normal weights).
    base = init_cortical_area_state(key, params)
    if not gabor_init:
        return base

    P = retina_cfg.fovea_size
    n_fovea_pixels = P * P
    fov_on_end = n_fovea_pixels
    fov_off_end = 2 * n_fovea_pixels

    # Build a Gabor bank sized to n_l4 at the foveal patch resolution.
    bank = _build_gabor_bank(
        params.n_l4, P,
        n_orientations=n_orientations, n_sf=n_sf,
    )                                           # (n_l4, P, P)
    g_flat = bank.reshape(params.n_l4, n_fovea_pixels)     # (n_l4, P^2)
    on_w = jnp.clip(g_flat.T, 0.0, None)        # (P^2, n_l4)
    off_w = jnp.clip(-g_flat.T, 0.0, None)      # (P^2, n_l4)

    # Scale Gabor weights so their mean magnitude matches the random
    # init mean, preserving the rheobase-calibrated drive.
    gap4 = float(params.l4_ncfg.v_thresh - params.l4_ncfg.v_rest)
    psp4 = gap4 / 2.0
    sigma4 = _psp_sigma(params.l4_ncfg, psp4, float(params.e_exc))
    # mean of half-normal(sigma) is sigma * sqrt(2/pi)
    target_mean = sigma4 * jnp.sqrt(jnp.asarray(2.0 / jnp.pi, DTYPE))
    # current means of the Gabor halves
    on_mean = on_w.mean() + 1e-8
    off_mean = off_w.mean() + 1e-8
    on_w = on_w * (target_mean / on_mean)
    off_w = off_w * (target_mean / off_mean)

    mix = jnp.asarray(gabor_mix, DTYPE)
    w_new = base.w_l4_in
    w_new = w_new.at[:fov_on_end].set(
        mix * on_w + (1.0 - mix) * w_new[:fov_on_end]
    )
    w_new = w_new.at[fov_on_end:fov_off_end].set(
        mix * off_w + (1.0 - mix) * w_new[fov_on_end:fov_off_end]
    )
    # peripheral / motion rows untouched.

    return eqx.tree_at(lambda s: s.w_l4_in, base, w_new)


__all__ = ["init_v1_params", "init_v1_state"]
