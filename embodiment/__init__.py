"""Embodiment — body adapters that drive the predictive-coding brain.

The I/O side of the integration contract (LEGACY_INTEGRATION.md §1): a
continuous :class:`BodyInterface` (joint command in, named sensory vector
out) and the babble→reach pipeline that proves active-inference control on
the MuJoCo-MJX arm.  Substrate-agnostic — lives outside ``core``.
"""

from .body_interface import (
    BodyInterface, SensorySample, SensorySegment, SensoryLayout,
    zero_value_code_in,
)
from .mjx_arm_body import (
    ArmConfig, MjxArmBody, default_arm_config,
    SEG_PROPRIOCEPTION, SEG_TARGET_ERROR_X, SEG_TARGET_ERROR_Y,
    TARGET_ERROR_SEGMENTS,
)
from .babbling import BabbleConfig, BabbleResult, ou_babble_step, run_babbling
from .reach import (
    ReachConfig, ReachResult, build_reacher, run_reach, collect_render_frames,
)

__all__ = [
    # interface
    "BodyInterface", "SensorySample", "SensorySegment", "SensoryLayout",
    "zero_value_code_in",
    # MJX arm
    "ArmConfig", "MjxArmBody", "default_arm_config",
    "SEG_PROPRIOCEPTION", "SEG_TARGET_ERROR_X", "SEG_TARGET_ERROR_Y",
    "TARGET_ERROR_SEGMENTS",
    # babble
    "BabbleConfig", "BabbleResult", "ou_babble_step", "run_babbling",
    # reach
    "ReachConfig", "ReachResult", "build_reacher", "run_reach",
    "collect_render_frames",
]
