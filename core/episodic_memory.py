"""
Episodic Memory — hippocampal one-shot binding with interference forgetting.

Reference:
  Rolls (2013)  "Pattern completion and separation in the hippocampus"
  O'Neill et al. (2010)  Hippocampal replay
  McClelland, McNaughton & O'Reilly (1995)  Complementary learning systems

Changes from legacy:
  1. Interference-based forgetting: new memories overwrite most similar
     (not oldest FIFO). Consolidated memories resist overwrite.
  2. DG-like sparse encoding: random projection + threshold for pattern
     separation before storage.
  3. Uses EpisodicMemoryConfig from config.py (dg_sparsity, consolidation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .config import EpisodicMemoryConfig

if TYPE_CHECKING:
    from .basal_ganglia import BGSnapshot


# =====================================================================
# Data container
# =====================================================================

@dataclass
class Episode:
    """A single episodic memory trace."""
    state: NDArray[np.float32]
    action: int | NDArray[np.float32]
    reward: float
    next_state: NDArray[np.float32]
    salience: float = 1.0
    prediction_error: NDArray[np.float32] | None = None
    encoder_e_bu: NDArray[np.float32] | None = None
    encoder_spikes: NDArray[np.float32] | None = None
    bg_snapshot: "BGSnapshot | None" = None
    aug_state: NDArray[np.float32] | None = None
    # Consolidation tracking
    replay_count: int = 0

    def __post_init__(self) -> None:
        self.state = np.asarray(self.state, dtype=np.float32).copy()
        self.next_state = np.asarray(self.next_state, dtype=np.float32).copy()
        if self.aug_state is not None:
            self.aug_state = np.asarray(self.aug_state, dtype=np.float32).copy()
        if self.prediction_error is not None:
            self.prediction_error = np.asarray(self.prediction_error, dtype=np.float32).copy()
        if self.encoder_e_bu is not None:
            self.encoder_e_bu = np.asarray(self.encoder_e_bu, dtype=np.float32).copy()
        if self.encoder_spikes is not None:
            self.encoder_spikes = np.asarray(self.encoder_spikes, dtype=np.float32).copy()


# =====================================================================
# Episodic Memory
# =====================================================================

class EpisodicMemory:
    """One-shot episodic memory with interference forgetting and DG sparse coding.

    Storage is gated by NE level ≥ threshold AND novelty.
    Forgetting: new memory overwrites the most similar existing memory
    (not the oldest). Consolidated memories (replay_count ≥ threshold)
    resist overwrite.
    """

    def __init__(
        self,
        state_dim: int,
        config: EpisodicMemoryConfig | None = None,
    ) -> None:
        self.config = config or EpisodicMemoryConfig()
        self.state_dim = state_dim
        cfg = self.config

        self._keys: list[NDArray[np.float32]] = []
        self._episodes: list[Episode] = []

        # ── DG-like sparse encoding (Rolls 2013) ─────────────────────
        # Random projection: state → expanded sparse code
        dg_dim = state_dim * cfg.dg_expansion_factor
        self._dg_projection: NDArray[np.float32] = np.random.randn(
            state_dim, dg_dim,
        ).astype(np.float32) * (1.0 / np.sqrt(state_dim))
        self._dg_dim = dg_dim
        self._dg_sparsity = cfg.dg_sparsity

    # ------------------------------------------------------------------
    # DG sparse encoding
    # ------------------------------------------------------------------

    def _dg_encode(self, state: NDArray[np.float32]) -> NDArray[np.float32]:
        """Pattern separation: state → sparse binary DG code."""
        projected = state @ self._dg_projection
        # Keep top-k% as active (competitive threshold)
        k = max(1, int(self._dg_sparsity * self._dg_dim))
        threshold = np.partition(projected, -k)[-k] if projected.size > k else 0.0
        sparse = (projected >= threshold).astype(np.float32)
        return sparse

    # ------------------------------------------------------------------
    # Storage (NE-gated, interference forgetting)
    # ------------------------------------------------------------------

    def try_store(
        self,
        state: NDArray[np.float32],
        action: int | NDArray[np.float32],
        reward: float,
        next_state: NDArray[np.float32],
        ne_level: float,
        prediction_error: NDArray[np.float32] | None = None,
        encoder_e_bu: NDArray[np.float32] | None = None,
        encoder_spikes: NDArray[np.float32] | None = None,
        bg_snapshot: "BGSnapshot | None" = None,
        aug_state: NDArray[np.float32] | None = None,
    ) -> bool:
        """Store an episode if NE-gated and novel. Returns True if stored."""
        if ne_level < self.config.ne_threshold:
            return False

        state_f32 = np.asarray(state, dtype=np.float32)
        dg_key = self._dg_encode(state_f32)

        if not self._is_novel(dg_key):
            return False

        episode = Episode(
            state=state_f32,
            action=action,
            reward=reward,
            next_state=next_state,
            salience=float(np.clip(ne_level, 0.0, 1.0)),
            prediction_error=prediction_error,
            encoder_e_bu=encoder_e_bu,
            encoder_spikes=encoder_spikes,
            bg_snapshot=bg_snapshot,
            aug_state=aug_state,
        )

        if len(self._episodes) < self.config.capacity:
            self._keys.append(dg_key.copy())
            self._episodes.append(episode)
        else:
            # Interference-based forgetting: overwrite most similar
            # non-consolidated memory
            idx = self._find_interference_target(dg_key)
            if idx is not None:
                self._keys[idx] = dg_key.copy()
                self._episodes[idx] = episode
            else:
                # All memories consolidated: overwrite least salient
                sals = [ep.salience for ep in self._episodes]
                idx_min = int(np.argmin(sals))
                self._keys[idx_min] = dg_key.copy()
                self._episodes[idx_min] = episode

        return True

    def _find_interference_target(
        self,
        dg_key: NDArray[np.float32],
    ) -> int | None:
        """Find the most similar non-consolidated memory to overwrite."""
        cfg = self.config
        best_sim = -1.0
        best_idx: int | None = None
        key_norm = np.linalg.norm(dg_key)
        if key_norm < 1e-8:
            return None

        for i, stored_key in enumerate(self._keys):
            # Skip consolidated memories
            if self._episodes[i].replay_count >= cfg.consolidation_threshold:
                continue
            stored_norm = np.linalg.norm(stored_key)
            if stored_norm < 1e-8:
                continue
            sim = float(np.dot(dg_key, stored_key) / (key_norm * stored_norm))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        return best_idx

    # ------------------------------------------------------------------
    # Recall (CA3-like pattern completion)
    # ------------------------------------------------------------------

    def recall(
        self,
        cue: NDArray[np.float32],
        top_k: int = 1,
    ) -> list[Episode]:
        """Pattern-completion recall: find top_k most similar episodes."""
        if not self._episodes:
            return []

        cue_dg = self._dg_encode(np.asarray(cue, dtype=np.float32))
        cue_norm = np.linalg.norm(cue_dg)
        if cue_norm < 1e-8:
            return []

        similarities: list[float] = []
        for key in self._keys:
            key_norm = np.linalg.norm(key)
            if key_norm < 1e-8:
                similarities.append(-1.0)
            else:
                similarities.append(float(np.dot(cue_dg, key) / (cue_norm * key_norm)))

        sorted_idx = np.argsort(similarities)[::-1]
        top_k = min(top_k, len(self._episodes))
        return [self._episodes[i] for i in sorted_idx[:top_k]]

    def recall_all(self) -> list[Episode]:
        """Return all stored episodes (for sleep consolidation)."""
        return list(self._episodes)

    def mark_replayed(self, episode: Episode) -> None:
        """Increment replay count for consolidation tracking."""
        episode.replay_count += 1

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._episodes)

    def clear(self) -> None:
        self._keys.clear()
        self._episodes.clear()

    def _is_novel(self, dg_key: NDArray[np.float32]) -> bool:
        """True if DG key is sufficiently different from all stored keys."""
        if not self._keys:
            return True
        key_norm = np.linalg.norm(dg_key)
        if key_norm < 1e-8:
            return False
        for stored in self._keys:
            stored_norm = np.linalg.norm(stored)
            if stored_norm < 1e-8:
                continue
            cos_sim = float(np.dot(dg_key, stored) / (key_norm * stored_norm))
            if cos_sim >= self.config.similarity_thresh:
                return False
        return True

    def __len__(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        return f"EpisodicMemory(size={self.size}/{self.config.capacity})"
