"""
Replay Buffer — hippocampal offline consolidation with SWS/REM phases.

Reference:
  Walker & Stickgold (2006)  Sleep-dependent memory consolidation
  Diekelmann & Born (2010)   Two-phase sleep model
  Buzsáki (2015)             Sharp-wave ripple replay

Changes from legacy:
  1. Two-phase sleep: SWS (reverse replay, consolidation) + REM (forward,
     world model refinement)
  2. Uses ReplayBufferConfig from config.py (sws/rem fractions)
  3. Separate SWS and REM methods instead of single sleep_phase()
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .config import ReplayBufferConfig
from .sequence_memory import SequenceMemory

if TYPE_CHECKING:
    from .world_model import SNNWorldModel
    from .neuromodulator import NeuromodulatorSystem


# =====================================================================
# Experience
# =====================================================================

@dataclass
class Experience:
    """Single (s, a, r, s') transition with biological context.

    Spike-time representation: spike_trains capture temporal spike
    patterns (not raw float state vectors). Raw state/next_state
    retained for world model compatibility.
    """
    state: NDArray[np.float32]
    action: int
    reward: float
    next_state: NDArray[np.float32]
    prediction_error: NDArray[np.float32]
    # Spike-time representation
    spike_trains: list[NDArray[np.float32]]  # Per-layer spike trains
    synaptic_fingerprint: dict[str, NDArray[np.float32]]  # Per-layer eligibility snapshot
    aug_state: NDArray[np.float32] | None = None
    salience: float = 0.0
    recorded_da: float = 0.0
    curiosity: float = 0.0
    done: bool = False

    def __post_init__(self) -> None:
        self.state = self.state.copy()
        self.next_state = self.next_state.copy()
        self.prediction_error = self.prediction_error.copy()
        self.spike_trains = [t.copy() for t in self.spike_trains]
        self.synaptic_fingerprint = {
            k: v.copy() for k, v in self.synaptic_fingerprint.items()
        }
        if self.aug_state is not None:
            self.aug_state = self.aug_state.copy()


# =====================================================================
# Replay Buffer
# =====================================================================

class ReplayBuffer:
    """Fixed-capacity replay buffer with SWS + REM sleep phases.

    SWS (Slow-Wave Sleep):
      Reverse replay (sharp-wave ripples) — consolidate declarative
      memories. Critic/actor weight updates from TD returns.

    REM (Rapid Eye Movement):
      Forward replay (theta sequences) — refine world model predictions.
      Sequence memory learning from temporal transitions.
    """

    def __init__(
        self,
        config: ReplayBufferConfig | None = None,
        capacity: int = 1000,
    ) -> None:
        cfg = config or ReplayBufferConfig()
        self.config = cfg
        self.capacity = cfg.capacity if config is not None else capacity
        self._buffer: deque[Experience] = deque(maxlen=self.capacity)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(self, exp: Experience) -> None:
        self._buffer.append(replace(exp))

    # ------------------------------------------------------------------
    # Unified sleep (dispatches to SWS + REM)
    # ------------------------------------------------------------------

    def sleep_phase(
        self,
        world_model: "SNNWorldModel",
        neuromodulator: "NeuromodulatorSystem",
        bg: object,
        n_experiences: int | None = None,
        sequence_memories: dict[str, SequenceMemory] | None = None,
        gamma: float | None = None,
        sleep_gain: float = 1.0,
        oscillator: object | None = None,
    ) -> list[float]:
        """Two-phase sleep consolidation with biological SWS oscillation.

        Phase 1 (SWS): Reverse replay → critic/actor consolidation.
          Oscillator enters ~1 Hz slow oscillation mode.
          InhibitoryPool gain elevated 2-3× (GABA surge).
          Up phase: noise + SWR replay.
          Down phase: global hyperpolarization.
        Phase 2 (REM): Forward replay → world model refinement.

        Returns per-experience world model MSE from SWS phase.
        """
        if len(self._buffer) == 0:
            return []

        gamma = gamma if gamma is not None else self.config.gamma
        experiences = list(self._buffer)
        if n_experiences is not None:
            experiences = experiences[-n_experiences:]

        cfg = self.config
        total = len(experiences)
        n_sws = max(1, int(total * cfg.sws_replay_fraction))
        n_rem = max(1, total - n_sws)

        # Enter SWS mode on oscillator if available
        if oscillator is not None and hasattr(oscillator, 'enter_sws'):
            oscillator.enter_sws()

        # Elevate inhibitory gain during SWS (GABA surge)
        sws_pools: list[object] = []
        for obj in (bg.critic, bg.actor):
            if hasattr(obj, 'inh_pool'):
                obj.inh_pool.enter_sws(gain_multiplier=2.5)
                sws_pools.append(obj.inh_pool)
            for attr_name in ('inh_pool_d1', 'inh_pool_d2'):
                pool = getattr(obj, attr_name, None)
                if pool is not None:
                    pool.enter_sws(gain_multiplier=2.5)
                    sws_pools.append(pool)

        # Phase 1: SWS — reverse replay (most recent first)
        sws_exps = experiences[-n_sws:]
        sws_errors = self._sws_phase(
            sws_exps, world_model, bg, gamma, sleep_gain,
            oscillator=oscillator,
        )

        # Exit SWS
        for pool in sws_pools:
            pool.exit_sws()
        if oscillator is not None and hasattr(oscillator, 'exit_sws'):
            oscillator.exit_sws()

        # Phase 2: REM — forward replay
        rem_exps = experiences[:n_rem]
        self._rem_phase(
            rem_exps, world_model, sequence_memories,
        )

        return sws_errors

    # ------------------------------------------------------------------
    # SWS: Reverse replay (sharp-wave ripples)
    # ------------------------------------------------------------------

    def _sws_phase(
        self,
        experiences: list[Experience],
        world_model: "SNNWorldModel",
        bg: object,
        gamma: float,
        sleep_gain: float,
        oscillator: object | None = None,
    ) -> list[float]:
        """Reverse-chronological replay for critic/actor consolidation.

        Gated by slow oscillation Up/Down states (~1 Hz):
          Up phase: SWR replay — world model + BG weight updates.
          Down phase: global hyperpolarization — skip replay, reset LIF.
        Seizure brake: if mean critic activation >3× baseline → force Down.

        Computes Monte Carlo returns backward, normalizes advantages,
        then updates BG and world model. Uses V_trace for baseline.
        """
        wm_saved = world_model.snapshot_encoder()

        # Cumulative returns backward
        cumulative_returns: list[float] = []
        g = 0.0
        for exp in reversed(experiences):
            if exp.done:
                g = 0.0
            sal_eff = max(0.0, exp.salience - 0.5) * 2.0
            eff_gamma = gamma * (1.0 - sal_eff)
            g = exp.reward + eff_gamma * g
            cumulative_returns.append(g)

        # Batch-normalise advantages using V_trace
        intrinsic_weight = 0.1
        all_advantages: list[float] = []
        for i, exp in enumerate(reversed(experiences)):
            g_aug = cumulative_returns[i] + intrinsic_weight * exp.curiosity
            # Use critic forward to get current V estimate (also updates activation)
            aug = exp.aug_state if exp.aug_state is not None else exp.state
            bg.critic.forward(aug)
            vs = bg.last_v
            all_advantages.append(g_aug - vs)

        adv_arr = np.array(all_advantages, dtype=np.float32)
        adv_std = float(np.std(adv_arr)) + 1e-8
        max_abs = float(np.max(np.abs(adv_arr))) + 1e-8
        variance_scale = float(np.clip(adv_std / 0.5, 0.1, 1.0))
        adv_norm = (adv_arr / max_abs) * variance_scale

        # Reverse replay — gated by slow oscillation Up/Down state
        errors: list[float] = []
        for i, exp in enumerate(reversed(experiences)):
            # ── Advance slow oscillation ──────────────────────────────
            if oscillator is not None and hasattr(oscillator, 'tick_sws'):
                _up_onset, _down_onset = oscillator.tick_sws()

                # Down state: global hyperpolarization — skip replay
                if not oscillator.in_up_state:
                    bg.critic.reset_state()
                    bg.actor.reset_state()
                    errors.append(0.0)
                    continue

                # Seizure brake during Up state
                mean_act = float(np.mean(np.abs(bg.critic.activation)))
                if hasattr(oscillator, 'check_seizure') and oscillator.check_seizure(mean_act):
                    bg.critic.reset_state()
                    bg.actor.reset_state()
                    errors.append(0.0)
                    continue

            # ── Up state: SWR replay ──────────────────────────────────
            # Restore synaptic fingerprint if available
            if "encoder_e_bu" in exp.synaptic_fingerprint:
                e_bu = exp.synaptic_fingerprint["encoder_e_bu"]
                if e_bu.shape == world_model.encoder.e_bu.shape:
                    world_model.encoder.e_bu[:] = e_bu
            world_model.reset_state()

            m_t = exp.recorded_da * max(abs(float(adv_norm[i])), 0.1)
            world_error = world_model.update(
                exp.state, exp.action, exp.next_state, m_t=m_t,
            )
            errors.append(float(np.mean(world_error ** 2)))

            # Direct TD update scaled by advantage (no snapshot restore)
            norm_adv = float(adv_norm[i])
            n_exp = min(max(len(experiences), 1), 200)
            pos_ratio = (0.5 * 200) / max(n_exp, 1)
            neg_ratio = (0.1 * 200) / max(n_exp, 1)
            ratio = pos_ratio if norm_adv >= 0 else neg_ratio
            sleep_signal = norm_adv * ratio * sleep_gain
            bg.critic.update(sleep_signal)
            bg.actor.update(sleep_signal)

        world_model.restore_encoder(wm_saved)
        return errors

    # ------------------------------------------------------------------
    # REM: Forward replay (theta sequences)
    # ------------------------------------------------------------------

    def _rem_phase(
        self,
        experiences: list[Experience],
        world_model: "SNNWorldModel",
        sequence_memories: dict[str, SequenceMemory] | None,
    ) -> None:
        """Forward-chronological replay for world model and sequence memory.

        Processes experiences in order (theta sequences) to refine the
        world model's predictive accuracy and feed sequence memories.
        """
        wm_saved = world_model.snapshot_encoder()

        for exp in experiences:
            # Light world model update (forward direction, reduced LR)
            world_model.update(
                exp.state, exp.action, exp.next_state, m_t=0.5,
            )

            # Sequence memory learning from spike trains
            if sequence_memories is not None:
                for seq_mem in sequence_memories.values():
                    # Use first spike train (encoder layer) if available
                    if exp.spike_trains:
                        seq_mem.observe(exp.spike_trains[0])
                    else:
                        seq_mem.observe(exp.state)

        # Reset sequence memories after REM
        if sequence_memories is not None:
            for seq_mem in sequence_memories.values():
                seq_mem.reset_state()

        world_model.restore_encoder(wm_saved)

    # ------------------------------------------------------------------
    # Online sampling
    # ------------------------------------------------------------------

    def sample(self, n: int) -> list[Experience]:
        n = min(n, len(self._buffer))
        indices = np.random.choice(len(self._buffer), size=n, replace=False)
        buf_list = list(self._buffer)
        return [buf_list[i] for i in indices]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_ready(self, min_size: int = 1) -> bool:
        return len(self._buffer) >= min_size

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
