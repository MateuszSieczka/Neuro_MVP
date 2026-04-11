"""
arena.gym_env — Adapter wrapping OpenAI Gymnasium environments into arena.Environment.

Handles:
  - State normalization (running mean/std or fixed bounds)
  - gymnasium API differences (reset returns (obs, info), step returns 5-tuple)
  - Truncation vs termination semantics
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

from .core import Environment


class GymEnv(Environment):
    """
    Wraps a gymnasium environment into the arena Environment protocol.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID (e.g., 'CartPole-v1', 'MountainCar-v0').
    normalize : bool
        If True, apply running normalization to observations.
    clip_obs : float
        After normalization, clip observations to [-clip_obs, clip_obs].
    fixed_bounds : tuple[np.ndarray, np.ndarray] | None
        If provided, use fixed (low, high) bounds for normalization
        instead of running statistics.  Useful for environments with
        known observation ranges.
    """

    def __init__(
        self,
        env_id: str,
        normalize: bool = True,
        clip_obs: float = 5.0,
        fixed_bounds: tuple[np.ndarray, np.ndarray] | None = None,
        reward_scale: float = 1.0,
    ) -> None:
        if gym is None:
            raise ImportError("gymnasium is required: pip install gymnasium")

        self._gym_env = gym.make(env_id)
        self._env_id = env_id
        self._normalize = normalize
        self._clip_obs = clip_obs
        self._reward_scale = reward_scale

        obs_space = self._gym_env.observation_space
        assert isinstance(obs_space, gym.spaces.Box), "Only Box observation spaces"
        self._obs_dim = obs_space.shape[0]

        act_space = self._gym_env.action_space
        assert isinstance(act_space, gym.spaces.Discrete), "Only Discrete action spaces"
        self._n_actions = int(act_space.n)

        # Normalization state
        if fixed_bounds is not None:
            low, high = fixed_bounds
            self._obs_mean = (low + high) / 2.0
            self._obs_std = np.maximum((high - low) / 2.0, 1e-6)
            self._use_running = False
        else:
            self._obs_mean = np.zeros(self._obs_dim, dtype=np.float64)
            self._obs_M2 = np.zeros(self._obs_dim, dtype=np.float64)
            self._obs_count = 0
            self._obs_std = np.ones(self._obs_dim, dtype=np.float64)
            self._use_running = True

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self._normalize:
            return obs.astype(np.float32)

        if self._use_running:
            self._update_running_stats(obs)

        normed = (obs - self._obs_mean) / self._obs_std
        return np.clip(normed, -self._clip_obs, self._clip_obs).astype(np.float32)

    def _update_running_stats(self, obs: np.ndarray) -> None:
        """Welford's online algorithm for running mean/variance."""
        self._obs_count += 1
        delta = obs - self._obs_mean
        self._obs_mean += delta / self._obs_count
        delta2 = obs - self._obs_mean
        self._obs_M2 += delta * delta2
        if self._obs_count >= 2:
            self._obs_std = np.sqrt(np.maximum(self._obs_M2 / (self._obs_count - 1), 1e-6))

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        obs, _info = self._gym_env.reset(**kwargs)
        return self._normalize_obs(obs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._gym_env.step(action)
        done = terminated or truncated
        info["terminated"] = terminated
        info["truncated"] = truncated
        return self._normalize_obs(obs), float(reward) * self._reward_scale, done, info

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def state_size(self) -> int:
        return self._obs_dim

    def close(self) -> None:
        self._gym_env.close()
