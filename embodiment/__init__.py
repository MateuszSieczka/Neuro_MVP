"""Embodiment layer — abstract body interface + reference adapters.

The brain is body-agnostic; the ``BodyInterface`` ABC is the single
boundary between the neural simulation (``core``) and any physical
world (MuJoCo, Unity, ROS2, or trivial gym-style adapters for early
integration testing).
"""

from .body_interface import BodyInterface, SensorySample
from .bandit import GaussianBanditBody
from .gridworld import GridWorldBody
from .visual_grid import VisualGridBody
from .run_loop import run_episode, EpisodeResult

__all__ = [
    "BodyInterface", "SensorySample",
    "GaussianBanditBody", "GridWorldBody", "VisualGridBody",
    "run_episode", "EpisodeResult",
]
