"""Cochlea + MGN + A1 — auditory front-end skeleton.

Phase 4 parity with :mod:`sensory.retina` / :mod:`sensory.lgn` /
:mod:`sensory.v1`. The auditory pathway gets the same scale-invariance
treatment as vision:

1. **Cochlea** is fixed (biologically hair cells have stereotyped
   tuning curves, Pickles 2012). It turns a time window of a
   waveform into a mel-filterbank spectrogram of fixed shape
   ``(n_bands, n_frames)``.
2. **Attention window** picks a sub-band of the spectrogram. This is
   the auditory analog of the foveal patch — it will be steered by a
   BG actor head in Phase 4.1+. Phase 4.0 keeps it centred.
3. **MGN adapter** performs contrast gain control just like
   :mod:`sensory.lgn`, so A1 gets a Poisson-rate vector in the
   operating range the cortex expects.
4. **A1** is an ordinary cortical area (no analog of Gabor init —
   tonotopic initialisation is possible but less universally agreed
   on than V1 Gabor, so we leave A1 with random init in Phase 4).

Babbling / speech motor / ear-to-mouth loops are deliberately NOT in
this phase (plan: Phase 5). This module just establishes the wiring.

References
----------
- Pickles (2012) *An Introduction to the Physiology of Hearing* (4th ed.)
- Stevens & Volkmann (1940) — mel scale.
- Lyon (2017) *Human and Machine Hearing*.
- Kaas & Hackett (2000) *PNAS* 97: 11793-11799 — auditory cortex
  organisation.
- Sherman & Guillery (2006) — MGN as first-order relay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from core.backend import DTYPE, BackendContext, Array, PRNGKey
from core.cortex import (
    CorticalAreaParams,
    CorticalAreaState,
    init_cortical_area_params,
    init_cortical_area_state,
)
from .lgn import lgn_normalize


# ---------------------------------------------------------------------
# Mel filterbank
# ---------------------------------------------------------------------


def _hz_to_mel(f_hz: Array) -> Array:
    return 2595.0 * jnp.log10(1.0 + f_hz / 700.0)


def _mel_to_hz(m: Array) -> Array:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(
    n_bands: int, n_fft: int, sample_rate: float,
    f_min: float, f_max: float,
) -> Array:
    """Triangular mel-filterbank matrix ``(n_bands, n_fft//2 + 1)``.

    Canonical construction (Stevens & Volkmann 1940): place ``n_bands+2``
    mel-uniform points, convert back to Hz, then bin FFT frequencies
    into triangular windows peaked at each interior point.
    """
    n_bins = n_fft // 2 + 1
    fft_freqs = jnp.linspace(0.0, sample_rate / 2.0, n_bins, dtype=DTYPE)
    mel_min = _hz_to_mel(jnp.asarray(f_min, DTYPE))
    mel_max = _hz_to_mel(jnp.asarray(f_max, DTYPE))
    mel_pts = jnp.linspace(mel_min, mel_max, n_bands + 2, dtype=DTYPE)
    hz_pts = _mel_to_hz(mel_pts)
    # triangle per band
    def _tri(i):
        l, c, r = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (fft_freqs - l) / jnp.maximum(c - l, 1e-6)
        down = (r - fft_freqs) / jnp.maximum(r - c, 1e-6)
        tri = jnp.clip(jnp.minimum(up, down), 0.0, 1.0)
        return tri
    fb = jax.vmap(_tri)(jnp.arange(n_bands))
    return fb.astype(DTYPE)


# ---------------------------------------------------------------------
# Configs + samples
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CochleaConfig:
    """Static cochlear parameters.

    Attributes
    ----------
    n_bands:
        Number of mel channels (auditory ganglion cells). 64 matches
        the Phase 4 plan ("cochleogram: mel-filterbank 64 bands").
    n_fft:
        FFT size applied to each analysis frame.
    frame_size, hop:
        Frame and hop in samples. With a 16 kHz sample rate, frame
        400 / hop 160 corresponds to 25 ms / 10 ms windows — standard
        for speech (Davis & Mermelstein 1980).
    n_frames:
        Number of frames emitted per sample. ``10`` frames at hop
        ``160`` covers 100 ms, a typical auditory attention window.
    sample_rate:
        Waveform sampling rate. ``16_000`` Hz matches the Phase 4 plan
        default.
    f_min, f_max:
        Mel range. ``80 - sample_rate/2`` excludes DC and aliasing.
    """

    n_bands: int = 64
    n_fft: int = 512
    frame_size: int = 400
    hop: int = 160
    n_frames: int = 10
    sample_rate: float = 16_000.0
    f_min: float = 80.0
    f_max: float = 8000.0

    @property
    def window_size(self) -> int:
        """Samples needed per ``cochlea_step`` call."""
        return self.frame_size + self.hop * (self.n_frames - 1)

    @property
    def afferent_size(self) -> int:
        """Flat size of the cochleogram seen by MGN."""
        return self.n_bands * self.n_frames


class Cochleogram(NamedTuple):
    """Fixed-shape output of ``cochlea_step``.

    Attributes
    ----------
    bands:
        ``(n_bands, n_frames)`` log-mel energies rectified to ``[0, 1]``.
    attention_xy:
        ``(2,)`` the current ``(centre_band_norm, width_norm)`` of the
        auditory attention window, passed through for downstream use.
    """

    bands: Array
    attention_xy: Array

    def as_afferent(self) -> Array:
        """Flat ``(n_bands * n_frames,)`` vector."""
        return self.bands.reshape(-1)


# ---------------------------------------------------------------------
# Cochlea step
# ---------------------------------------------------------------------


def _frame_signal(wave: Array, cfg: CochleaConfig) -> Array:
    """Split a waveform into ``n_frames`` overlapping windows.

    ``wave`` is ``(window_size,)`` float32 in approximately ``[-1, 1]``.
    Output is ``(n_frames, frame_size)``.
    """
    idx = (
        jnp.arange(cfg.frame_size)[None, :]
        + jnp.arange(cfg.n_frames)[:, None] * cfg.hop
    )
    return wave[idx].astype(DTYPE)


def _hann(n: int) -> Array:
    return (0.5 * (1.0 - jnp.cos(2 * jnp.pi * jnp.arange(n) / (n - 1)))).astype(DTYPE)


def cochlea_step(cfg: CochleaConfig, wave: Array, attention_xy: Array) -> Cochleogram:
    """Greyscale-of-sound: return a log-mel spectrogram of fixed shape.

    Parameters
    ----------
    cfg:
        Static :class:`CochleaConfig`. Pass as a jit static argument.
    wave:
        ``(window_size,)`` waveform in roughly ``[-1, 1]``.
    attention_xy:
        ``(2,)`` ``(band_centre_norm, band_width_norm)`` in ``[0, 1]^2``.
        Phase 4.0 uses ``(0.5, 1.0)`` (no attenuation). Phase 4.1 will
        multiply the spectrogram by a Gaussian band-window centred on
        ``band_centre_norm`` with half-width ``band_width_norm``.

    Returns
    -------
    :class:`Cochleogram` with band energies in ``[0, 1]`` and the
    attention vector passed through.
    """
    w = jnp.asarray(wave, DTYPE).reshape((cfg.window_size,))
    att = jnp.asarray(attention_xy, DTYPE).reshape((2,))

    frames = _frame_signal(w, cfg)                    # (F, frame_size)
    win = _hann(cfg.frame_size)
    frames = frames * win
    # Real FFT along frame axis; pad to n_fft.
    pad = cfg.n_fft - cfg.frame_size
    frames = jnp.pad(frames, ((0, 0), (0, pad)))
    spec = jnp.fft.rfft(frames, axis=-1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(DTYPE)    # (F, n_fft/2+1)

    fb = _mel_filterbank(
        cfg.n_bands, cfg.n_fft, cfg.sample_rate, cfg.f_min, cfg.f_max,
    )                                                  # (n_bands, n_bins)
    mel = power @ fb.T                                 # (F, n_bands)
    mel = jnp.transpose(mel)                           # (n_bands, F)
    # log compression then normalisation to [0, 1]
    log_mel = jnp.log1p(mel)
    peak = log_mel.max() + 1e-6
    log_mel = log_mel / peak

    # Attention window: Gaussian along the band axis. Phase 4.0 uses a
    # very wide window so this is effectively unit gain; Phase 4.1 will
    # narrow it under BG control.
    band_centre = jnp.clip(att[0], 0.0, 1.0) * (cfg.n_bands - 1)
    half_width = jnp.maximum(att[1], 0.05) * cfg.n_bands
    bands = jnp.arange(cfg.n_bands, dtype=DTYPE)
    gauss = jnp.exp(-0.5 * ((bands - band_centre) / half_width) ** 2)
    mel_att = log_mel * gauss[:, None]

    return Cochleogram(bands=mel_att.astype(DTYPE), attention_xy=att)


# ---------------------------------------------------------------------
# MGN adapter (thin wrapper around LGN-style contrast gain control)
# ---------------------------------------------------------------------


def mgn_normalize(
    afferent: Array,
    *,
    target_mean: float = 0.25,
    baseline: float = 0.15,
    semi_saturation: float = 0.05,
) -> Array:
    """MGN contrast gain control + tonic baseline.

    Biologically the medial geniculate nucleus has a matching role to
    LGN in vision: tonic spontaneous activity (McAlonan, Cavanaugh &
    Wurtz 2008) plus gain control (Sherman & Guillery 2006). The
    parameters here match :func:`sensory.lgn.lgn_normalize` so the A1
    operating point is the same as V1's — a cortex reused as either
    visual or auditory area stays in the same dynamic regime.
    """
    return lgn_normalize(
        afferent,
        target_mean=target_mean,
        baseline=baseline,
        semi_saturation=semi_saturation,
    )


# ---------------------------------------------------------------------
# A1 (cortical area)
# ---------------------------------------------------------------------


def init_a1_params(
    ctx: BackendContext,
    cfg: CochleaConfig,
    *,
    n_l4: int = 128,
    n_l23_state: int = 128,
    n_l23_error: int = 64,
    n_l5: int = 64,
    l4_expected_input_rate: float = 0.25,
) -> CorticalAreaParams:
    """A1 params sized for the MGN-normalised cochleogram."""
    return init_cortical_area_params(
        ctx,
        input_size=cfg.afferent_size,
        n_l4=n_l4,
        n_l23_state=n_l23_state,
        n_l23_error=n_l23_error,
        n_l5=n_l5,
        l4_expected_input_rate=l4_expected_input_rate,
    )


def init_a1_state(
    key: PRNGKey, params: CorticalAreaParams,
) -> CorticalAreaState:
    return init_cortical_area_state(key, params)


__all__ = [
    "CochleaConfig",
    "Cochleogram",
    "cochlea_step",
    "mgn_normalize",
    "init_a1_params",
    "init_a1_state",
]
