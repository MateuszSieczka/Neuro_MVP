"""Scale-invariant retina — Gaussian pyramid + foveal patch + DoG + motion.

An I/O adapter (LEGACY_INTEGRATION.md §0.5): the *only* place that ever
sees raw pixels, turning an image of arbitrary resolution into a
**fixed-size flat rate vector** — the afferent that clamps the substrate's
sensory node (``core.pc_brain.pc_brain_cognitive_step(sensory=…)``).  The
output shape depends only on the static :class:`RetinaConfig`, never on the
input image dimensions, so the same brain drives a 64×64 and a 4K camera
without retraining: scale-invariance is the retina's job, not the cortex's.

Values are in ``[0, 1]`` and read directly as the clamped sensory rates
(Rao & Ballard 1999 rate-mode predictive coding) — there is no Poisson
spike stage: the substrate is rate-mode, the encoder hands it a rate
vector and stops at the edge.

Biology and design decisions
----------------------------
- **Foveal patch**: a high-resolution square window around the current
  fixation point. Biologically the fovea covers ~2° of visual angle but
  holds ~50% of retinal ganglion cells (Curcio et al. 1990).  We crop a
  normalised window ``fovea_extent`` around ``fixation_xy`` and resample
  it to a fixed ``fovea_size`` × ``fovea_size`` grid by bilinear
  interpolation.  The fixation is continuous so a saccade can move it
  smoothly (the saccade loop lives in :mod:`sensory.vision`).
- **Peripheral pyramid**: successively downsampled views of the full
  image, each resampled to ``periphery_tile`` so the peripheral tensor is
  a fixed ``(n_pyramid, tile, tile)`` regardless of input resolution.
- **Difference of Gaussians**: ON/OFF-centre retinal ganglion cells
  (Rodieck 1965): ``dog = G(σ_c)·I − G(σ_s)·I``; ON = ``max(dog, 0)``,
  OFF = ``max(−dog, 0)``.  Separable 1-D Gaussians keep it cheap/JIT-safe.
- **Temporal differencing**: the coarsest peripheral tile minus its value
  one step earlier — a motion-like transient channel (Y-cells,
  Enroth-Cugell & Robson 1966).

Stateless apart from :class:`RetinaState`, which carries only the previous
coarse tile for the motion channel.  Everything else is a pure function of
the current frame and ``fixation_xy``.

References
----------
- Rodieck (1965) *Vision Res.* 5: 583-601 — DoG receptive fields.
- Curcio et al. (1990) *J. Comp. Neurol.* 292: 497-523 — ganglion density.
- Enroth-Cugell & Robson (1966) *J. Physiol.* 187: 517-552 — X/Y cells.
- Burt & Adelson (1983) *IEEE TComm* 31: 532-540 — Laplacian pyramid.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    The output shape of a :class:`RetinalSample` is completely determined
    by this config; it does **not** depend on input image size.  All
    fields are ints/floats so the config is hashable and can be a static
    argument of ``jax.jit``.

    Attributes
    ----------
    fovea_size:
        Side of the square foveal patch after resampling.
    fovea_extent:
        Half-side of the foveal window in normalised image coordinates
        ``[0, 1]``.  ``0.1`` ⇒ the fovea covers the central 20% per axis.
    n_pyramid:
        Number of peripheral pyramid levels.
    periphery_tile:
        Side of each peripheral tile after resampling; the peripheral
        tensor is ``(n_pyramid, periphery_tile, periphery_tile)``.
    sigma_center, sigma_surround:
        DoG kernel widths in pixels; ratio ~1:2.5 (Marr & Hildreth 1980).
    dog_kernel_size:
        Truncated Gaussian kernel length (odd), ``>= ceil(3·sigma)``.
    temporal_gain:
        Amplification on ``|frame − prev_frame|`` before clipping.
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
    def fovea_block(self) -> int:
        """Length of one half (ON *or* OFF) of the foveal sub-vector.

        The Gabor generative-edge prior
        (:func:`core.pc_graph.init_region_graph`) needs the foveal block
        geometry to place its patches; this exposes it without the core
        module importing :mod:`sensory`.
        """
        return self.fovea_size * self.fovea_size

    @property
    def afferent_size(self) -> int:
        """Length of the flat afferent (the sensory-node dimensionality)."""
        fovea = 2 * self.fovea_block                 # ON + OFF
        peri = 2 * self.n_pyramid * self.periphery_tile * self.periphery_tile
        motion = self.periphery_tile * self.periphery_tile
        return fovea + peri + motion


# ---------------------------------------------------------------------
# State & sample
# ---------------------------------------------------------------------


class RetinaState(NamedTuple):
    """Carried across :func:`retina_step` calls.

    Almost stateless: only the previous coarse peripheral tile is kept,
    and only for temporal differencing (the motion channel).
    """

    prev_coarse: Array  # (periphery_tile, periphery_tile), float32 [0,1]


class RetinalSample(NamedTuple):
    """Fixed-shape retinal output; all tensors in ``[0, 1]``."""

    fovea_on: Array         # (fovea_size, fovea_size)
    fovea_off: Array        # (fovea_size, fovea_size)
    periphery_on: Array     # (n_pyramid, periphery_tile, periphery_tile)
    periphery_off: Array    # (n_pyramid, periphery_tile, periphery_tile)
    motion: Array           # (periphery_tile, periphery_tile)
    fixation_xy: Array      # (2,) normalised [0,1]² fovea centre, passed through

    def as_afferent(self) -> Array:
        """Flat ``(afferent_size,)`` vector in fixed retinotopic order.

        Layout: ``[fovea_on, fovea_off, periphery_on (level-major),
        periphery_off, motion]`` — stable so the Gabor edge-init can rely
        on the foveal ON/OFF blocks sitting first.
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
    """Unnormalised → normalised 1-D Gaussian of length ``size`` (odd)."""
    half = (size - 1) / 2.0
    x = jnp.arange(size, dtype=DTYPE) - half
    k = jnp.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(DTYPE)


def _separable_conv2d(img: Array, kern: Array) -> Array:
    """Same-padded separable 2-D convolution on a single-channel image.

    Reflective padding avoids dark border halos that would otherwise
    create spurious DoG responses near the edges.
    """
    k = kern.astype(DTYPE)
    K = k.shape[0]
    pad = K // 2
    padded = jnp.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    row_windows = jax.lax.conv_general_dilated(
        padded[None, None, :, :],
        k.reshape(1, 1, 1, K),
        window_strides=(1, 1),
        padding="VALID",
    )[0, 0]
    out = jax.lax.conv_general_dilated(
        row_windows[None, None, :, :],
        k.reshape(1, 1, K, 1),
        window_strides=(1, 1),
        padding="VALID",
    )[0, 0]
    return out.astype(DTYPE)


def _dog(img: Array, cfg: RetinaConfig) -> Array:
    """Difference of Gaussians on a single-channel patch (signed ~[-1, 1])."""
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
    """Crop a window around ``fixation_xy`` and resample to ``fovea_size``.

    ``fixation_xy`` is ``(x, y)`` in normalised ``[0, 1]²`` (x = column).
    Sampling is done manually (not by slicing) so the fixation can be
    continuous and differentiable through a saccade.
    """
    H, W = img.shape
    fx = jnp.clip(fixation_xy[0], 0.0, 1.0)
    fy = jnp.clip(fixation_xy[1], 0.0, 1.0)
    ext = jnp.asarray(cfg.fovea_extent, DTYPE)
    lin = jnp.linspace(-1.0, 1.0, cfg.fovea_size, dtype=DTYPE)
    gx = fx + ext * lin
    gy = fy + ext * lin
    px = jnp.clip(gx * (W - 1), 0.0, W - 1)
    py = jnp.clip(gy * (H - 1), 0.0, H - 1)
    yy, xx = jnp.meshgrid(py, px, indexing="ij")
    coords = jnp.stack([yy, xx], axis=0)
    patch = jax.scipy.ndimage.map_coordinates(img, coords, order=1, mode="nearest")
    return patch.astype(DTYPE)


def _build_pyramid(img: Array, cfg: RetinaConfig) -> Array:
    """Gaussian pyramid resampled to fixed ``periphery_tile`` tiles.

    Returns ``(n_pyramid, tile, tile)``.  Level 0 is the coarsest overview
    (whole scene); higher levels retain finer detail (scale
    ``2^(n_pyramid−1−ell)``).  Mimics the coarse-to-fine spatial-frequency
    channels of magno-/parvocellular ganglion cells.
    """
    H, W = img.shape
    kern_blur = _gauss_kernel_1d(1.0, cfg.dog_kernel_size)
    smoothed = _separable_conv2d(img, kern_blur)
    levels = []
    for ell in range(cfg.n_pyramid):
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
        Static retinal config (a hashable dataclass — pass as a ``jax.jit``
        static argument).  Must be constant across calls within a trace.
    image:
        ``(H, W)`` float32 in ``[0, 1]``.  Any spatial size; output shape
        is invariant.
    fixation_xy:
        ``(2,)`` float32 in ``[0, 1]²`` — set by the saccade selector
        (:mod:`sensory.vision`), or held constant for bodies without
        saccades.

    Returns
    -------
    ``(new_state, sample)`` — pure / ``jax.jit``-safe with ``cfg`` static.
    """
    img = jnp.asarray(image, DTYPE)
    fix = jnp.asarray(fixation_xy, DTYPE).reshape((2,))

    # --- foveal DoG ---
    fovea_raw = _foveal_patch(img, fix, cfg)
    fovea_dog = _dog(fovea_raw, cfg)
    fovea_on, fovea_off = _split_on_off(fovea_dog)

    # --- peripheral pyramid DoG ---
    pyramid = _build_pyramid(img, cfg)               # (L, tile, tile)
    pyramid_dog = jax.vmap(lambda t: _dog(t, cfg))(pyramid)
    periphery_on = jnp.clip(pyramid_dog, 0.0, 1.0).astype(DTYPE)
    periphery_off = jnp.clip(-pyramid_dog, 0.0, 1.0).astype(DTYPE)

    # --- temporal differencing on the coarsest tile (level 0) ---
    coarse = pyramid[0]
    motion_raw = coarse - state.prev_coarse
    motion = jnp.clip(
        jnp.asarray(cfg.temporal_gain, DTYPE) * jnp.abs(motion_raw), 0.0, 1.0,
    ).astype(DTYPE)

    sample = RetinalSample(
        fovea_on=fovea_on, fovea_off=fovea_off,
        periphery_on=periphery_on, periphery_off=periphery_off,
        motion=motion, fixation_xy=fix,
    )
    return RetinaState(prev_coarse=coarse), sample


def retina_afferent(
    state: RetinaState, cfg: RetinaConfig, image: Array, fixation_xy: Array,
) -> tuple[RetinaState, Array]:
    """Convenience: :func:`retina_step` → flat afferent in one call."""
    new_state, sample = retina_step(state, cfg, image, fixation_xy)
    return new_state, sample.as_afferent()
