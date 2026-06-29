"""Vision adapter — retina → LGN → the sensory clamp, plus saccades.

The composition that turns an image into the flat rate vector clamping the
substrate's sensory node, and drives the fovea by active inference.  An I/O
adapter (LEGACY_INTEGRATION.md §0.5): it lives *outside* ``core`` and adds
no machinery — it only drives hooks the substrate already exposes.

Pipeline (per cognitive cycle)::

    image, fixation → retina_step → lgn_normalize → sensory clamp
                                                   (pc_brain_cognitive_step)

What used to be V1/V2/V4 cortical areas (legacy ``sensory_stack``) is now
the brain's own cortical hierarchy ``cortex_l1→l2→l3``: the adapter does
**not** contain a visual cortex, it hands the clamp to the graph and the
deep cortical node carries the cause.  The first cortical edge
``cortex_l1→sensory`` may be seeded with the V1 Gabor prior
(:func:`foveal_gabor_init` → :class:`core.pc_graph.FovealGaborInit`).

Saccades are active inference, not a saliency heuristic: a candidate
fixation is evaluated by the **sensory prediction error** it would leave
(Bayesian surprise; Itti & Baldi 2009) — the substrate's "what is
currently unexplained" signal — and the fovea moves to the fixation of
maximal expected information gain via :func:`core.pc_active.efe_select`
(argmin expected free energy).  The probe runs the cognitive step with
``learn=False`` so evaluating a saccade never mutates the brain.

References
----------
- Itti & Baldi (2009) *Vision Res.* 49: 1295-1306 — Bayesian surprise as
  the saccade-driving signal.
- Tatler et al. (2011) — foveated active sampling.
- Friston et al. (2015) — active inference / epistemic foraging.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from core.backend import DTYPE, Array, PRNGKey
from core.pc_brain import PCBrainParams, PCBrainState, init_pc_brain
from core.pc_graph import FovealGaborInit, pc_graph_clamp, pc_graph_errors
from core.pc_active import efe_select

from .retina import (
    RetinaConfig, RetinaState, retina_step, init_retina_state,
)
from .lgn import (
    lgn_normalize,
    LGN_TARGET_MEAN, LGN_BASELINE, LGN_SEMI_SATURATION, LGN_MAX_GAIN,
)


# =====================================================================
# Adapter config
# =====================================================================


class VisionParams(NamedTuple):
    """Static vision-adapter config: retina geometry + LGN constants."""

    retina: RetinaConfig
    lgn_target_mean: float
    lgn_baseline: float
    lgn_semi_saturation: float
    lgn_max_gain: float

    @property
    def afferent_size(self) -> int:
        """Sensory-node dimensionality this adapter clamps."""
        return self.retina.afferent_size


def init_vision_params(
    retina_cfg: RetinaConfig | None = None,
    *,
    lgn_target_mean: float = LGN_TARGET_MEAN,
    lgn_baseline: float = LGN_BASELINE,
    lgn_semi_saturation: float = LGN_SEMI_SATURATION,
    lgn_max_gain: float = LGN_MAX_GAIN,
) -> VisionParams:
    return VisionParams(
        retina=retina_cfg or RetinaConfig(),
        lgn_target_mean=lgn_target_mean,
        lgn_baseline=lgn_baseline,
        lgn_semi_saturation=lgn_semi_saturation,
        lgn_max_gain=lgn_max_gain,
    )


def init_vision_state(params: VisionParams) -> RetinaState:
    """Fresh retina state (the adapter's only carried state)."""
    return init_retina_state(params.retina)


# =====================================================================
# Encode: image → sensory clamp
# =====================================================================


def vision_encode(
    state: RetinaState, params: VisionParams, image: Array, fixation_xy: Array,
) -> tuple[RetinaState, Array]:
    """``(image, fixation) → (new_retina_state, afferent)``.

    The afferent is the LGN-normalised retinal rate vector — the flat
    ``[0, 1]`` clamp for the sensory node (length ``afferent_size``).
    """
    new_state, sample = retina_step(state, params.retina, image, fixation_xy)
    afferent = lgn_normalize(
        sample.as_afferent(),
        target_mean=params.lgn_target_mean,
        baseline=params.lgn_baseline,
        semi_saturation=params.lgn_semi_saturation,
        max_gain=params.lgn_max_gain,
    )
    return new_state, afferent


# =====================================================================
# V1 Gabor prior for the cortex_l1→sensory edge
# =====================================================================


def foveal_gabor_init(
    cfg: RetinaConfig,
    *,
    n_orientations: int = 8,
    n_sf: int = 4,
    sf_min: float = 1.0,
    sf_max: float = 6.0,
    mix: float = 0.7,
) -> FovealGaborInit:
    """Build the :class:`FovealGaborInit` matching this retina's layout.

    The foveal ON block starts at 0 and the OFF block at ``fovea_block``
    in :meth:`sensory.retina.RetinalSample.as_afferent`.
    """
    return FovealGaborInit(
        patch_size=cfg.fovea_size,
        on_offset=0,
        off_offset=cfg.fovea_block,
        n_orientations=n_orientations,
        n_sf=n_sf,
        sf_min=sf_min,
        sf_max=sf_max,
        mix=mix,
    )


def init_vision_brain(
    key: PRNGKey,
    params: VisionParams | None = None,
    *,
    motor_size: int = 8,
    gabor: bool = True,
    gabor_mix: float = 0.7,
    **brain_kwargs,
) -> tuple[VisionParams, PCBrainParams, PCBrainState]:
    """Build a brain sized to the vision afferent, optional Gabor prior.

    Convenience wiring: the sensory node is sized to ``afferent_size`` and
    the ``cortex_l1→sensory`` edge is seeded with the foveal Gabor prior
    unless ``gabor=False`` (then the default LeCun init stands).
    """
    vp = params or init_vision_params()
    gfi = foveal_gabor_init(vp.retina, mix=gabor_mix) if gabor else None
    bp, bs = init_pc_brain(
        key,
        sensory_size=vp.afferent_size,
        motor_size=motor_size,
        gabor_foveal_init=gfi,
        **brain_kwargs,
    )
    return vp, bp, bs


# =====================================================================
# Saccades — active fixation selection (argmin expected free energy)
# =====================================================================


def saccade_info_gain(
    brain_state: PCBrainState, brain_params: PCBrainParams,
    retina_state: RetinaState, params: VisionParams,
    image: Array, fixation_xy: Array,
) -> Array:
    """Surprise of foveating ``fixation_xy`` — expectation violation (scalar).

    The substrate's saccade-driving signal: how much the candidate
    observation departs from what the model currently *expects* at the
    sensory node (Rao & Ballard 1999 prediction error; Itti & Baldi 2009
    Bayesian surprise as the gaze-driving quantity).  The candidate
    afferent is clamped onto the sensory node while every other belief is
    held at its standing value — crucially **without relaxing**, so the
    deep causes cannot explain the stimulus away — and the mean ``|ε|``
    against the model's standing top-down prediction is the surprise.

    Measuring at the prior (not after relaxation) is what makes this robust:
    residual ``|ε|`` after relaxation reflects only how well the model
    *can* fit a stimulus (a Gabor-primed edge fits with low residual), not
    how far it is from expectation.  Against a model's *flat* initial
    prior this is bottom-up saliency — the fovea is drawn to contrast /
    structure (Itti & Koch 2001); as the generative model learns a scene
    the same signal sharpens into novelty detection — a learned region
    yields low surprise, an unlearned patch high.
    """
    _, afferent = vision_encode(retina_state, params, image, fixation_xy)
    s_idx = brain_params.sensory_idx
    clamped = pc_graph_clamp(brain_state.graph, {s_idx: afferent})
    errors = pc_graph_errors(clamped, brain_params.graph)
    return jnp.mean(jnp.abs(errors[s_idx])).astype(DTYPE)


class SaccadeChoice(NamedTuple):
    fixation: Array      # (2,) chosen next fixation
    index: Array         # argmin-EFE index into the candidates
    info_gain: Array     # (n_candidates,) per-candidate Bayesian surprise
    G: Array             # (n_candidates,) expected free energy


def select_fixation(
    brain_state: PCBrainState, brain_params: PCBrainParams,
    retina_state: RetinaState, params: VisionParams,
    image: Array, candidates: Array,
    *,
    epistemic_weight: float | Array = 1.0,
) -> SaccadeChoice:
    """Pick the fixation of maximal expected information gain.

    ``candidates`` is ``(n_candidates, 2)`` normalised ``[0, 1]²`` points.
    Each is scored by :func:`saccade_info_gain`; with a purely epistemic
    preference (no pragmatic term) :func:`core.pc_active.efe_select` then
    minimises ``G = −β·info_gain`` — i.e. moves the fovea to the most
    surprising location (Friston 2015 epistemic foraging).  ``β``
    (``epistemic_weight``, NE-modulated) sets exploration strength.
    """
    cands = jnp.asarray(candidates, DTYPE)

    def _gain(fix: Array) -> Array:
        return saccade_info_gain(
            brain_state, brain_params, retina_state, params, image, fix,
        )

    info_gain = jax.vmap(_gain)(cands)                  # (n_candidates,)
    choice = efe_select(
        jnp.zeros_like(info_gain), info_gain, epistemic_weight=epistemic_weight,
    )
    return SaccadeChoice(
        fixation=cands[choice.index],
        index=choice.index,
        info_gain=info_gain,
        G=choice.G,
    )
