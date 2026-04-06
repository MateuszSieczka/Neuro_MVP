"""
arena.py — Lightweight reinforcement-learning test-bed for Neuro_MVP.

The arena is **completely decoupled** from the SNN internals.
It exposes a Gym-like ``Environment`` base class and a generic ``Agent``
protocol so the same test harness works with *any* controller, not just
the SNN.

Architecture
============

  Environment   ←→   Agent   ←→   Metrics
  (state, rew)       (action)     (logs, curves)

Key classes:

  Environment (ABC):   step(action) → (next_state, reward, done, info)
                       reset()      → initial_state
  Agent (ABC):         act(state)   → action
                       observe(state, action, reward, next_state, done)
                       reset()
  Trainer:             Runs episodes, collects metrics, supports evaluation.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# =====================================================================
# Environment base class
# =====================================================================

class Environment(abc.ABC):
    """Gym-like environment interface.  State is always a 1-D float32 array."""

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        """Reset and return initial observation."""

    @abc.abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Execute *action*, return (next_state, reward, done, info)."""

    @property
    @abc.abstractmethod
    def n_actions(self) -> int:
        """Number of discrete actions available."""

    @property
    @abc.abstractmethod
    def state_size(self) -> int:
        """Dimensionality of the observation vector."""


# =====================================================================
# Agent base class
# =====================================================================

class Agent(abc.ABC):
    """Controller interface — maps observations to actions and learns."""

    @abc.abstractmethod
    def act(self, state: np.ndarray) -> int:
        """Choose a discrete action given the current state."""

    @abc.abstractmethod
    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Process a transition for learning."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset transient state between episodes (preserve learned weights)."""


class RandomAgent(Agent):
    """Baseline: picks a random action uniformly.  Learns nothing."""

    def __init__(self, n_actions: int) -> None:
        self._n_actions = n_actions

    def act(self, state: np.ndarray) -> int:
        return int(np.random.randint(self._n_actions))

    def observe(self, state, action, reward, next_state, done) -> None:
        pass

    def reset(self) -> None:
        pass


# =====================================================================
# Metric tracking
# =====================================================================

@dataclass
class EpisodeLog:
    """Record of a single episode."""
    total_reward: float = 0.0
    steps: int = 0
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


@dataclass
class TrainResult:
    """Aggregated training statistics."""
    episode_logs: list[EpisodeLog] = field(default_factory=list)

    @property
    def rewards(self) -> list[float]:
        return [ep.total_reward for ep in self.episode_logs]

    def mean_reward(self, last_n: int | None = None) -> float:
        rews = self.rewards
        if last_n is not None:
            rews = rews[-last_n:]
        return float(np.mean(rews)) if rews else 0.0

    def learning_curve(self, window: int = 50) -> list[float]:
        """Moving-average reward curve for plotting/analysis."""
        rews = self.rewards
        if len(rews) < window:
            return [float(np.mean(rews))] if rews else []
        return [
            float(np.mean(rews[max(0, i - window):i]))
            for i in range(window, len(rews) + 1)
        ]

    def is_improving(self, early_n: int = 100, late_n: int = 100) -> bool:
        """True if late performance is significantly better than early."""
        rews = self.rewards
        if len(rews) < early_n + late_n:
            return False
        early = float(np.mean(rews[:early_n]))
        late = float(np.mean(rews[-late_n:]))
        return late > early + 0.05  # must improve by at least 0.05

    def action_distribution(self, last_n: int | None = None) -> dict[int, float]:
        """Fraction of each action across recent episodes."""
        logs = self.episode_logs
        if last_n is not None:
            logs = logs[-last_n:]
        all_actions = []
        for ep in logs:
            all_actions.extend(ep.actions)
        if not all_actions:
            return {}
        counts: dict[int, int] = {}
        for a in all_actions:
            counts[a] = counts.get(a, 0) + 1
        total = len(all_actions)
        return {a: c / total for a, c in sorted(counts.items())}


# =====================================================================
# Trainer
# =====================================================================

class Trainer:
    """
    Runs episodes and collects metrics.

    Usage::

        trainer = Trainer(env, agent)
        result = trainer.train(n_episodes=500, max_steps=50)
        print(result.mean_reward(last_n=50))
    """

    def __init__(self, env: Environment, agent: Agent) -> None:
        self.env = env
        self.agent = agent

    def train(
        self,
        n_episodes: int,
        max_steps: int = 100,
    ) -> TrainResult:
        result = TrainResult()

        for _ in range(n_episodes):
            log = self._run_episode(max_steps, training=True)
            result.episode_logs.append(log)

        return result

    def evaluate(
        self,
        n_episodes: int,
        max_steps: int = 100,
    ) -> TrainResult:
        """Run episodes WITHOUT learning (observe is skipped)."""
        result = TrainResult()

        for _ in range(n_episodes):
            log = self._run_episode(max_steps, training=False)
            result.episode_logs.append(log)

        return result

    def _run_episode(self, max_steps: int, training: bool) -> EpisodeLog:
        state = self.env.reset()
        self.agent.reset()
        log = EpisodeLog()

        for _ in range(max_steps):
            action = self.agent.act(state)
            next_state, reward, done, info = self.env.step(action)

            if training:
                self.agent.observe(state, action, reward, next_state, done)

            log.actions.append(action)
            log.rewards.append(reward)
            log.total_reward += reward
            log.steps += 1

            state = next_state
            if done:
                break

        return log
