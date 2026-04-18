"""Scale-invariant retina — Gaussian pyramid + foveal patch + DoG + motion.

This module is the *only* place in the brain that ever sees raw pixels.
Its sole job is to turn an image of arbitrary resolution into a
**fixed-size** vector of Poisson-rate afferents. The shape of the
output depends only on the static :class:`RetinaConfig`, never on the
input image dimensions. This is the mechanism that lets the same
downstream brain drive cameras at 64x64 and 4K without retraining.

Biology and design decisions
----------------------------
- **Foveal patch**: a high-resolution square window around the current
  fixation point. Biologically the fovea covers ~2 deg of visual angle
  but contains ~50% of retinal ganglion cells (Curcio et al. 1990).
  We mimic this by cropping a normalised window ``fovea_extent`` of the
  image around ``fixation_xy`` and resampling it to a fixed
  ``fovea_size`` x ``fovea_size`` grid by bilinear interpolation.
- **Peripheral pyramid**: successively downsampled views of the full
  image. Each level is resampled to the same ``periphery_tile`` size,
  so the peripheral tensor has fixed shape ``(n_pyramid, tile, tile)``
  regardless of input resolution. Level 0 is the coarsest view of the
  whole scene; later levels band-pass the image at progressively
  higher spatial frequencies (after DoG).
- **Difference of Gaussians**: approximates on-/off-centre retinal
  ganglion cells (Rodieck 1965). We compute
  ``dog = G(sigma_c) * I - G(sigma_s) * I`` on the already-resampled
  patches. ON = max(dog, 0), OFF = max(-dog, 0). Because we use
  separable 1D Gaussians the kernel is cheap and JIT-safe.
- **Temporal differencing**: the coarsest peripheral tile is
  subtracted from its value one step earlier. This feeds a
  motion-like channel, analog to transient Y-cells (Enroth-Cugell &
  Robson 1966). Kept as a single ``(tile, tile)`` channel to stay
  minimal; can be extended per-level if needed.
- **Rate encoding, not spike encoding**: the retina emits values in
  ``[0, 1]`` that are interpreted as Poisson firing rates by the
  thalamic relay (``core/thalamus.py``). This is consistent with the
  rest of the brain, where afferents enter as rates and the stochastic
  spike generation happens downstream.

This module is intentionally *stateless* apart from ``RetinaState``
which carries only the previous coarse tile for temporal differencing.
Everything else is a pure function of the current frame and
``fixation_xy``.

References
----------
- Rodieck (1965) *Vision Res.* 5: 583-601 — DoG receptive fields.
- Curcio et al. (1990) *J. Comp. Neurol.* 292: 497-523 — ganglion
  cell density.
- Enroth-Cugell & Robson (1966) *J. Physiol.* 187: 517-552 — X/Y cells.
- Burt & Adelson (1983) *IEEE TComm* 31: 532-540 — Laplacian pyramid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import image as jimage

from core.backend import DTYPE, Array


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RetinaConfig:
    """Static retinal parameters.

    Output shape of a :class:`RetinalSample` is completely determined
    by this config. It does **not** depend on input image size.

    Attributes
    ----------
    fovea_size:
        Side of the square foveal patch after resampling. Defaults to
        16 — fine enough to resolve orientation without blowing up
        the afferent vector.
    fovea_extent:
        Half-side of the foveal window in normalised image coordinates
        ``[0, 1]``. ``0.1`` -> the fovea covers the central 20% of the
        image along each axis, roughly matching the ratio of human
        foveal angular size to a ~10 deg useful field.
    n_pyramid:
        Number of peripheral pyramid levels. Each level halves the
        working resolution before resampling to ``periphery_tile``.
    periphery_tile:
        Side of each peripheral tile after resampling. The peripheral
        tensor shape is ``(n_pyramid, periphery_tile, periphery_tile)``.
    sigma_center, sigma_surround:
        DoG kernel widths, in pixels on the resampled patches. Ratio
        roughly 1:2.5 follows Marr & Hildreth (1980).
    dog_kernel_size:
        Truncated Gaussian kernel length (odd). Must be >= ceil(3 sigma).
    temporal_gain:
        Amplification applied to ``|frame - prev_frame|`` before
        clipping. Mimics Y-cell transient response.

    Notes
    -----
    All fields are floats or ints so the config is a hashable dataclass
    and can be a static argument of ``jax.jit``.
    """

    fovea_size: int = 16
    fovea_extent: float = 0.1
    n_pyramid: int = 3
    periphery_tile: int = 8
    sigma_center: float = 0.6
    sigma_surround: float = 1.5
    dog_kernel_size: int = 5
    temporal_gain: float = 4.0

    @property
    def afferent_size(self) -> int:
        """Length of the flat afferent vector used by the LGN adapter."""
        fovea = 2 * self.fovea_size * self.fovea_size  # ON + OFF
        peri = 2 * self.n_pyramid * self.periphery_tile * self.periphery_tile
        motion = self.periphery_tile * self.periphery_tile
        return fovea + peri + motion


# ---------------------------------------------------------------------
# State & sample
# ---------------------------------------------------------------------


class RetinaState(NamedTuple):
    """Carried across retina_step calls.

    The retina is almost stateless. Only the previous coarse peripheral
    tile is kept, and only for temporal differencing (motion channel).
    """

    prev_coarse: Array  # (periphery_tile, periphery_tile), float32 [0,1]


class RetinalSample(NamedTuple):
    """Fixed-shape retinal output.

    All tensors are in ``[0, 1]`` and interpreted downstream as Poisson
    rates. Shapes depend only on :class:`RetinaConfig`.

    Attributes
    ----------
    fovea_on, fovea_off:
        ``(fovea_size, fovea_size)`` ON- and OFF-centre DoG responses
        on the high-resolution foveal patch.
    periphery_on, periphery_off:
        ``(n_pyramid, periphery_tile, periphery_tile)`` same, on
        each pyramid level.
    motion:
        ``(periphery_tile, periphery_tile)`` rectified temporal
        difference of the coarsest peripheral tile.
    fixation_xy:
        ``(2,)`` normalised ``[0, 1]^2`` fixation point, passed through
        so that downstream modules (V1, saccade critic) know where the
        fovea was centred.
    """

    fovea_on: Array
    fovea_off: Array
    periphery_on: Array
    periphery_off: Array
    motion: Array
    fixation_xy: Array

    def as_afferent(self) -> Array:
        """Flat ``(afferent_size,)`` vector in retinotopic order.

        Layout: [fovea_on, fovea_off, periphery_on (level-major),
        periphery_off, motion]. The order is fixed and stable so the
        LGN adapter can rely on it.
        """
        return jnp.concatenate(
            [
                self.fovea_on.reshape(-1),
                self.fovea_off.reshape(-1),
                self.periphery_on.reshape(-1),
                self.periphery_off.reshape(-1),
                self.motion.reshape(-1),
            ]
        )


# ---------------------------------------------------------------------
# Helpers: Gaussian kernel, separable convolution, bilinear crop
# ---------------------------------------------------------------------


def _gauss_kernel_1d(sigma: float, size: int) -> Array:
    """Unnormalised 1D Gaussian of length ``size`` (odd)."""
    half = (size - 1) / 2.0
    x = jnp.arange(size, dtype=DTYPE) - half
    k = jnp.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(DTYPE)


def _separable_conv2d(img: Array, kern: Array) -> Array:
    """Same-padded separable 2D convolution on a single-channel image.

    ``img`` is ``(H, W)`` and ``kern`` is ``(K,)`` (1D). The kernel is
    applied along both axes. Uses ``REFLECT`` padding to avoid dark
    halos at the border, which would otherwise create spurious DoG
    responses near the image edges.
    """
    k = kern.astype(DTYPE)
    K = k.shape[0]
    pad = K // 2
    # pad reflectively on both axes
    padded = jnp.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    # conv along axis 1 (x)
    row_windows = jax.lax.conv_general_dilated(
        padded[None, None, :, :],
        k.reshape(1, 1, 1, K),
        window_strides=(1, 1),
        padding="VALID",
    )[0, 0]
    # conv along axis 0 (y)
    out = jax.lax.conv_general_dilated(
        row_windows[None, None, :, :],
        k.reshape(1, 1, K, 1),
        window_strides=(1, 1),
        padding="VALID",
    )[0, 0]
    return out.astype(DTYPE)


def _dog(img: Array, cfg: RetinaConfig) -> Array:
    """Difference of Gaussians on a single-channel patch.

    Returns a signed image in roughly ``[-1, 1]``. Downstream callers
    split it into ON/OFF by half-rectification.
    """
    kc = _gauss_kernel_1d(cfg.sigma_center, cfg.dog_kernel_size)
    ks = _gauss_kernel_1d(cfg.sigma_surround, cfg.dog_kernel_size)
    gc = _separable_conv2d(img, kc)
    gs = _separable_conv2d(img, ks)
    return (gc - gs).astype(DTYPE)


def _split_on_off(dog: Array) -> tuple[Array, Array]:
    """Half-rectify a signed DoG map into ON and OFF rate channels."""
    on = jnp.clip(dog, 0.0, 1.0).astype(DTYPE)
    off = jnp.clip(-dog, 0.0, 1.0).astype(DTYPE)
    return on, off


def _resize(img: Array, h: int, w: int) -> Array:
    """Bilinear resample a single-channel image to ``(h, w)``."""
    return jimage.resize(img, (int(h), int(w)), method="linear").astype(DTYPE)


def _foveal_patch(img: Array, fixation_xy: Array, cfg: RetinaConfig) -> Array:
    """Crop a window around ``fixation_xy`` and resample to fovea_size.

    ``fixation_xy`` is in normalised ``[0, 1]^2`` coordinates
    ``(x, y)`` where ``x`` is column, ``y`` is row. We build a regular
    sampling grid around the fixation in normalised coordinates,
    clamp it, and bilinearly sample the image. We do the sampling
    manually (not via ``jimage.resize`` of a slice) so that the
    fixation can be continuous and differentiable through the
    saccade.
    """
    H, W = img.shape
    fx = jnp.clip(fixation_xy[0], 0.0, 1.0)
    fy = jnp.clip(fixation_xy[1], 0.0, 1.0)
    ext = jnp.asarray(cfg.fovea_extent, DTYPE)
    # grid in normalised image coords around the fixation
    lin = jnp.linspace(-1.0, 1.0, cfg.fovea_size, dtype=DTYPE)
    gx = fx + ext * lin                          # (fovea_size,)
    gy = fy + ext * lin                          # (fovea_size,)
    # to pixel coords
    px = jnp.clip(gx * (W - 1), 0.0, W - 1)
    py = jnp.clip(gy * (H - 1), 0.0, H - 1)
    # bilinear sample: map_coordinates expects (coords_y, coords_x)
    yy, xx = jnp.meshgrid(py, px, indexing="ij")
    coords = jnp.stack([yy, xx], axis=0)
    patch = jax.scipy.ndimage.map_coordinates(
        img, coords, order=1, mode="nearest",
    )
    return patch.astype(DTYPE)


def _build_pyramid(img: Array, cfg: RetinaConfig) -> Array:
    """Gaussian pyramid resampled to fixed ``periphery_tile`` tiles.

    Returns ``(n_pyramid, tile, tile)``. Level 0 is the coarsest
    overview (whole scene resampled to ``tile^2``), higher levels
    progressively retain finer information by halving the scale
    less. Concretely level ``ell`` uses a scale ``2^(n_pyramid-1-ell)``
    so that the last level is near native resolution cropped to
    ``tile^2``.

    Biologically this mimics the coarse-to-fine spatial frequency
    channels of magnocellular and parvocellular ganglion cells.
    """
    H, W = img.shape
    # blur once at each reduction to avoid aliasing
    kern_blur = _gauss_kernel_1d(1.0, cfg.dog_kernel_size)
    smoothed = _separable_conv2d(img, kern_blur)

    levels = []
    for ell in range(cfg.n_pyramid):
        # inverse scale: level 0 is most downsampled
        scale = 2 ** (cfg.n_pyramid - 1 - ell)
        h = max(cfg.periphery_tile, H // scale)
        w = max(cfg.periphery_tile, W // scale)
        coarse = _resize(smoothed, h, w)
        tile = _resize(coarse, cfg.periphery_tile, cfg.periphery_tile)
        levels.append(tile)
    return jnp.stack(levels, axis=0).astype(DTYPE)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def init_retina_state(cfg: RetinaConfig) -> RetinaState:
    """Fresh state: zero previous coarse tile (no motion at t=0)."""
    return RetinaState(
        prev_coarse=jnp.zeros(
            (cfg.periphery_tile, cfg.periphery_tile), dtype=DTYPE,
        ),
    )


def retina_step(
    state: RetinaState,
    cfg: RetinaConfig,
    image: Array,
    fixation_xy: Array,
) -> tuple[RetinaState, RetinalSample]:
    """Run the retina once on a greyscale image.

    Parameters
    ----------
    state:
        Carries the previous coarse tile for motion differencing.
    cfg:
        Static retinal config. Must be the same across calls within a
        ``jax.jit`` trace (it is a hashable dataclass — pass it as a
        static argument).
    image:
        ``(H, W)`` float32 in ``[0, 1]``. Any spatial dimensions are
        accepted; output shape is invariant.
    fixation_xy:
        ``(2,)`` float32 in ``[0, 1]^2``. Set by the saccade head of
        the BG actor, or externally held constant for bodies without
        saccades.

    Returns
    -------
    new_state, sample:
        Retina state to carry forward and a fixed-shape
        :class:`RetinalSample`.

    Notes
    -----
    This function is a pure ``eqx``-compatible function of its inputs.
    It is safe to ``jax.jit`` with ``cfg`` marked static.
    """
    img = jnp.asarray(image, DTYPE)
    fix = jnp.asarray(fixation_xy, DTYPE).reshape((2,))

    # --- foveal DoG ---
    fovea_raw = _foveal_patch(img, fix, cfg)
    fovea_dog = _dog(fovea_raw, cfg)
    fovea_on, fovea_off = _split_on_off(fovea_dog)

    # --- peripheral pyramid DoG ---
    pyramid = _build_pyramid(img, cfg)           # (L, tile, tile)

    def _dog_level(t: Array) -> Array:
        return _dog(t, cfg)

    pyramid_dog = jax.vmap(_dog_level)(pyramid)  # (L, tile, tile)
    periphery_on = jnp.clip(pyramid_dog, 0.0, 1.0).astype(DTYPE)
    periphery_off = jnp.clip(-pyramid_dog, 0.0, 1.0).astype(DTYPE)

    # --- temporal differencing on the coarsest tile (level 0) ---
    coarse = pyramid[0]
    motion_raw = coarse - state.prev_coarse
    motion = jnp.clip(
        jnp.asarray(cfg.temporal_gain, DTYPE) * jnp.abs(motion_raw),
        0.0, 1.0,
    ).astype(DTYPE)

    sample = RetinalSample(
        fovea_on=fovea_on,
        fovea_off=fovea_off,
        periphery_on=periphery_on,
        periphery_off=periphery_off,
        motion=motion,
        fixation_xy=fix,
    )
    new_state = RetinaState(prev_coarse=coarse)
    return new_state, sample
