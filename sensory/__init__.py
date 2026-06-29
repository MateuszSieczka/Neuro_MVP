"""Sensory front-ends — raw input → the flat rate vector that clamps the
predictive-coding sensory node.

Substrate-agnostic encoders living *outside* ``core``: an I/O adapter in the
integration contract (LEGACY_INTEGRATION.md §0.5).

* proprioception — population-coded joint angles + velocities (§1).
* vision — retina → LGN → sensory clamp + active saccades (§2); the V1
  Gabor prior is an opt-in init of the ``cortex_l1→sensory`` generative
  edge in :mod:`core.pc_graph`, the only substrate-touching part.

Audio (cochlea/MGN) is deferred to a later milestone; ``mgn_normalize``
reduces to :func:`lgn_normalize`, so it is mostly free once needed.
"""

from .population_code import gaussian_population_encode
from .proprioception import (
    DEFAULT_VELOCITY_RANGE_FACTOR,
    ProprioceptionParams,
    init_proprioception_params,
    proprio_encode,
    proprio_output_dim,
)
from .retina import (
    RetinaConfig,
    RetinaState,
    RetinalSample,
    init_retina_state,
    retina_step,
    retina_afferent,
)
from .lgn import (
    lgn_normalize,
    LGN_TARGET_MEAN,
    LGN_BASELINE,
    LGN_SEMI_SATURATION,
    LGN_MAX_GAIN,
)
from .vision import (
    VisionParams,
    init_vision_params,
    init_vision_state,
    vision_encode,
    foveal_gabor_init,
    init_vision_brain,
    saccade_info_gain,
    SaccadeChoice,
    select_fixation,
)

__all__ = [
    # proprioception (§1)
    "gaussian_population_encode",
    "DEFAULT_VELOCITY_RANGE_FACTOR",
    "ProprioceptionParams",
    "init_proprioception_params",
    "proprio_encode",
    "proprio_output_dim",
    # vision (§2)
    "RetinaConfig",
    "RetinaState",
    "RetinalSample",
    "init_retina_state",
    "retina_step",
    "retina_afferent",
    "lgn_normalize",
    "LGN_TARGET_MEAN",
    "LGN_BASELINE",
    "LGN_SEMI_SATURATION",
    "LGN_MAX_GAIN",
    "VisionParams",
    "init_vision_params",
    "init_vision_state",
    "vision_encode",
    "foveal_gabor_init",
    "init_vision_brain",
    "saccade_info_gain",
    "SaccadeChoice",
    "select_fixation",
]
