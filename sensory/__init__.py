"""Sensory front-ends — retina, cochlea and learned cortical hierarchies.

Phase 4 design principles
-------------------------
1. **Scale invariance is the retina's job**, not the cortex's. The
   retina produces a fixed-shape sample regardless of the input
   image resolution (64x64, 256x256, 1024x1024 all map to the same
   vector). The brain downstream never sees pixels, only retinal
   spike patterns. Any change in input resolution stops at this
   boundary — the cortex topology is untouched.
2. **Learned V1+ under STDP, not fixed Gabor.** The retina is fixed
   (biologically ganglion cells have stereotyped receptive fields,
   Rodieck 1998). V1 and above are cortical areas that learn their
   receptive fields from the statistics of the input (Olshausen &
   Field 1996). Gabor-like weights may be used for initialisation
   but never as a final representation.
3. **Saccades are mental actions**, not saliency heuristics. The
   fixation point is set by an extra action head on the basal
   ganglia actor, in exactly the same way that body actions are.
   Bottom-up saliency contributes via the reward signal
   (information gain after a saccade), not by bypassing the BG.

References
----------
- Rodieck (1998) *The First Steps in Seeing*.
- Rosenholtz (2016) "Capabilities and limitations of peripheral
  vision."
- Olshausen & Field (1996) *Nature* 381: 607-609.
- Itti & Koch (2001) *Nat. Rev. Neurosci.* 2: 194-203.
- Findlay & Walker (1999) *Behav. Brain Sci.* 22: 661-721.
- Tatler et al. (2011) "Eye guidance in natural vision: Reinterpreting
  salience."
"""

from .retina import (
    RetinaConfig,
    RetinalSample,
    retina_step,
    init_retina_state,
    RetinaState,
)
from .lgn import lgn_normalize
from .v1 import init_v1_params, init_v1_state
from .ventral import (
    init_v2_params, init_v2_state,
    init_v4it_params, init_v4it_state,
)
from .auditory import (
    CochleaConfig, Cochleogram, cochlea_step,
    mgn_normalize, init_a1_params, init_a1_state,
)
from .sensory_stack import (
    SensoryStackParams, SensoryStackState, SensoryStackOutput,
    init_sensory_stack_params, init_sensory_stack_state,
    sensory_stack_step,
)

__all__ = [
    "RetinaConfig", "RetinalSample", "RetinaState",
    "retina_step", "init_retina_state",
    "lgn_normalize",
    "init_v1_params", "init_v1_state",
    "init_v2_params", "init_v2_state",
    "init_v4it_params", "init_v4it_state",
    "CochleaConfig", "Cochleogram", "cochlea_step",
    "mgn_normalize", "init_a1_params", "init_a1_state",
    "SensoryStackParams", "SensoryStackState", "SensoryStackOutput",
    "init_sensory_stack_params", "init_sensory_stack_state",
    "sensory_stack_step",
]
