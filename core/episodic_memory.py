"""
Episodic Memory — hippocampal one-shot pattern binding.

Biological grounding:
  Hippocampal area CA3 forms auto-associative memories in a single
  exposure when noradrenaline (locus coeruleus) is high.  The dentate
  gyrus provides pattern separation; CA3 recurrent collaterals provide
  pattern completion.

  This module implements the computational essence:
    - Store:  high NE → snapshot (state, action, reward, next_state)
              with one-shot Hebbian binding (no eligibility trace needed).
    - Recall: cosine-similarity pattern completion from a partial cue.
    - Inject: stored episodes are fed into ReplayBuffer during sleep
              consolidation, enabling offline STDP on rare events that
              would otherwise be underrepresented in the replay buffer.

Design:
  Fixed-capacity ring buffer of (key, value) pairs.
    key   = state spike pattern (num_state,)
    value = full Experience-like tuple (state, action, reward, next_state)
  Storage is gated by NE level ≥ ne_threshold AND novelty (cosine
  distance from all existing keys > similarity_thresh).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EpisodicMemoryConfig


@dataclass
class Episode:
    """A single episodic memory trace."""
    state: np.ndarray
    action: int | np.ndarray
    reward: float
    next_state: np.ndarray
    salience: float = 1.0

    def __post_init__(self) -> None:
        self.state = np.asarray(self.state, dtype=np.float32).copy()
        self.next_state = np.asarray(self.next_state, dtype=np.float32).copy()


class EpisodicMemory:
    """
    One-shot episodic memory with NE-gated storage and cosine recall.

    Usage in the agent loop::

        em = EpisodicMemory(state_dim)
        # ... during online step:
        em.try_store(state, action, reward, next_state, ne_level)
        # ... during sleep:
        episodes = em.recall_all()
        for ep in episodes:
            replay_buffer.store(
                state=ep.state, action=ep.action, reward=ep.reward,
                next_state=ep.next_state, ..., salience=ep.salience,
            )
    """

    def __init__(
        self,
        state_dim: int,
        config: EpisodicMemoryConfig | None = None,
    ) -> None:
        self.config = config or EpisodicMemoryConfig()
        self.state_dim = state_dim

        self._keys: list[np.ndarray] = []     # stored state patterns
        self._episodes: list[Episode] = []     # full episode records
        self._write_idx: int = 0               # ring buffer pointer

    # ------------------------------------------------------------------
    # Storage (gated by NE)
    # ------------------------------------------------------------------

    def try_store(
        self,
        state: np.ndarray,
        action: int | np.ndarray,
        reward: float,
        next_state: np.ndarray,
        ne_level: float,
    ) -> bool:
        """
        Attempt to store an episode.  Returns True if stored.

        Storage occurs only when:
          1. ne_level ≥ config.ne_threshold  (arousal gate)
          2. The state is sufficiently novel (cosine distance from all
             existing keys exceeds config.similarity_thresh).
        """
        if ne_level < self.config.ne_threshold:
            return False

        state_f32 = np.asarray(state, dtype=np.float32)
        if not self._is_novel(state_f32):
            return False

        episode = Episode(
            state=state_f32,
            action=action,
            reward=reward,
            next_state=next_state,
            salience=float(np.clip(ne_level, 0.0, 1.0)),
        )

        if len(self._episodes) < self.config.capacity:
            self._keys.append(state_f32.copy())
            self._episodes.append(episode)
        else:
            # Ring buffer overwrite
            self._keys[self._write_idx] = state_f32.copy()
            self._episodes[self._write_idx] = episode
        self._write_idx = (self._write_idx + 1) % self.config.capacity

        return True

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(self, cue: np.ndarray, top_k: int = 1) -> list[Episode]:
        """
        Pattern-completion recall: find the top_k most similar episodes.

        Args:
            cue:   Partial or full state pattern (state_dim,).
            top_k: Number of episodes to return.

        Returns:
            List of Episode objects, sorted by descending similarity.
        """
        if not self._episodes:
            return []

        cue_f32 = np.asarray(cue, dtype=np.float32)
        cue_norm = np.linalg.norm(cue_f32)
        if cue_norm < 1e-8:
            return []

        similarities = []
        for key in self._keys:
            key_norm = np.linalg.norm(key)
            if key_norm < 1e-8:
                similarities.append(-1.0)
            else:
                similarities.append(float(np.dot(cue_f32, key) / (cue_norm * key_norm)))

        sorted_idx = np.argsort(similarities)[::-1]
        top_k = min(top_k, len(self._episodes))
        return [self._episodes[i] for i in sorted_idx[:top_k]]

    def recall_all(self) -> list[Episode]:
        """Return all stored episodes (for sleep-phase injection)."""
        return list(self._episodes)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._episodes)

    def clear(self) -> None:
        self._keys.clear()
        self._episodes.clear()
        self._write_idx = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_novel(self, state: np.ndarray) -> bool:
        """True if state is sufficiently different from all stored keys."""
        if not self._keys:
            return True

        state_norm = np.linalg.norm(state)
        if state_norm < 1e-8:
            return False

        for key in self._keys:
            key_norm = np.linalg.norm(key)
            if key_norm < 1e-8:
                continue
            cos_sim = float(np.dot(state, key) / (state_norm * key_norm))
            if cos_sim >= self.config.similarity_thresh:
                return False
        return True

    def __len__(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        return f"EpisodicMemory(size={self.size}/{self.config.capacity})"
