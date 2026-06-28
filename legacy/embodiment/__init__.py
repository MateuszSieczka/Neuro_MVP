"""Embodiment layer — abstract body interface + reference adapters.

The brain is body-agnostic; the ``BodyInterface`` ABC is the single
boundary between the neural simulation (``core``) and any physical
world (MuJoCo, Unity, ROS2, or trivial gym-style adapters for early
integration testing).
"""

from .body_interface import BodyInterface, SensorySample, discretise_joint_command
from .bandit import GaussianBanditBody
from .gridworld import GridWorldBody
from .visual_grid import VisualGridBody
from .run_loop import run_episode, EpisodeResult

# Phase 6B MJX adapters are **lazily** importable — importing them at
# package level would fail on machines without mujoco-mjx.  Users do
# ``from embodiment.mjx_arm_body import MjxArmBody`` explicitly.

__all__ = [
    "BodyInterface", "SensorySample", "discretise_joint_command",
    "GaussianBanditBody", "GridWorldBody", "VisualGridBody",
    "run_episode", "EpisodeResult",
]
