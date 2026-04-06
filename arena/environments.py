"""
arena.environments — Concrete test environments of increasing difficulty.

All environments use discrete action spaces and 1-D float32 state vectors.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import Environment


# =====================================================================
# Level 1: Single deterministic button
# =====================================================================

class SingleButtonEnv(Environment):
    """
    One button.  Press it → +1 reward.  Don't press → 0 reward.
    Episode length = 1 step (bandit).

    Actions:  0 = do nothing,  1 = press button.
    State:    [1.0] (constant — there is only one state).
    """

    def reset(self) -> np.ndarray:
        return np.array([1.0], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        reward = 1.0 if action == 1 else 0.0
        return self.reset(), reward, True, {}

    @property
    def n_actions(self) -> int:
        return 2

    @property
    def state_size(self) -> int:
        return 1


# =====================================================================
# Level 2: Stochastic button (lottery)
# =====================================================================

class StochasticButtonEnv(Environment):
    """
    One button with stochastic payoff:
      - 40% chance: +10 reward
      - 10% chance:  -1 reward
      - 50% chance:   0 reward
    Expected value of pressing = 0.4*10 + 0.1*(-1) + 0.5*0 = +3.9
    Not pressing always gives 0.

    The agent must learn that pressing is beneficial *on average*
    despite occasional negative outcomes.

    Actions:  0 = do nothing,  1 = press button.
    State:    [1.0] (constant).
    """

    def reset(self) -> np.ndarray:
        return np.array([1.0], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if action == 1:
            roll = np.random.random()
            if roll < 0.4:
                reward = 10.0
            elif roll < 0.5:
                reward = -1.0
            else:
                reward = 0.0
        else:
            reward = 0.0
        return self.reset(), reward, True, {}

    @property
    def n_actions(self) -> int:
        return 2

    @property
    def state_size(self) -> int:
        return 1


# =====================================================================
# Level 3: Two buttons — context-dependent choice
# =====================================================================

class TwoButtonEnv(Environment):
    """
    Two buttons, one per context.  The environment alternates randomly
    between two contexts (signalled in the state):

      Context A [1, 0]:  Button 0 → +1,  Button 1 → -1
      Context B [0, 1]:  Button 0 → -1,  Button 1 → +1

    The agent must learn to read the state and pick the correct button.
    Pure exploration won't work — the agent needs state-conditional policy.

    Actions:  0 = press left,  1 = press right.
    State:    2D one-hot encoding of context.
    """

    def __init__(self) -> None:
        self._context: int = 0

    def reset(self) -> np.ndarray:
        self._context = int(np.random.random() < 0.5)
        return self._make_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self._context == 0:
            reward = 1.0 if action == 0 else -1.0
        else:
            reward = 1.0 if action == 1 else -1.0
        return self._make_state(), reward, True, {"context": self._context}

    def _make_state(self) -> np.ndarray:
        s = np.zeros(2, dtype=np.float32)
        s[self._context] = 1.0
        return s

    @property
    def n_actions(self) -> int:
        return 2

    @property
    def state_size(self) -> int:
        return 2


# =====================================================================
# Level 4: Delayed reward — multi-step corridor
# =====================================================================

class CorridorEnv(Environment):
    """
    5-cell corridor.  Agent starts at cell 0, goal is cell 4.

      [0] [1] [2] [3] [4=goal]

    Actions:  0 = stay,  1 = move right.
    Reward:   +10 at goal, -0.1 per step (cost of time).
    Done:     when reaching cell 4 or after max_steps.

    State:    one-hot of position (5D).

    The agent must learn to consistently move right despite getting
    no immediate reward for the first 3 steps.  Tests temporal
    credit assignment (TD learning with γ > 0).
    """

    def __init__(self, corridor_length: int = 5) -> None:
        self._length = corridor_length
        self._pos: int = 0

    def reset(self) -> np.ndarray:
        self._pos = 0
        return self._make_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if action == 1:
            self._pos = min(self._pos + 1, self._length - 1)

        done = self._pos == self._length - 1
        reward = 10.0 if done else -0.1

        return self._make_state(), reward, done, {"position": self._pos}

    def _make_state(self) -> np.ndarray:
        s = np.zeros(self._length, dtype=np.float32)
        s[self._pos] = 1.0
        return s

    @property
    def n_actions(self) -> int:
        return 2

    @property
    def state_size(self) -> int:
        return self._length


# =====================================================================
# Level 5: Multi-armed bandit with shifting payoffs
# =====================================================================

class ShiftingBanditEnv(Environment):
    """
    3-armed bandit where payoff probabilities shift every N episodes.

    Phase A (first `shift_interval` episodes):
      Arm 0: +1 with p=0.8    Arm 1: +1 with p=0.2    Arm 2: +1 with p=0.5
    Phase B (next `shift_interval` episodes):
      Arm 0: +1 with p=0.2    Arm 1: +1 with p=0.8    Arm 2: +1 with p=0.5

    The agent must detect the shift and adapt its policy.
    Tests continual learning / plasticity.

    State:    [1.0, phase_signal]  where phase_signal = 0 or 1.
              (If hide_phase=True, phase_signal is always 0 — harder.)
    """

    def __init__(
        self,
        shift_interval: int = 200,
        hide_phase: bool = False,
    ) -> None:
        self._shift_interval = shift_interval
        self._hide_phase = hide_phase
        self._episode_count: int = 0
        self._phase: int = 0
        self._payoffs = [
            [0.8, 0.2, 0.5],  # Phase A
            [0.2, 0.8, 0.5],  # Phase B
        ]

    def reset(self) -> np.ndarray:
        self._phase = (self._episode_count // self._shift_interval) % 2
        self._episode_count += 1
        return self._make_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action = min(action, 2)
        p = self._payoffs[self._phase][action]
        reward = 1.0 if np.random.random() < p else 0.0
        return self._make_state(), reward, True, {"phase": self._phase}

    def _make_state(self) -> np.ndarray:
        phase_sig = 0.0 if self._hide_phase else float(self._phase)
        return np.array([1.0, phase_sig], dtype=np.float32)

    @property
    def n_actions(self) -> int:
        return 3

    @property
    def state_size(self) -> int:
        return 2


# =====================================================================
# Level 6: Risk vs safety — asymmetric payoffs
# =====================================================================

class RiskRewardEnv(Environment):
    """
    Three choices with different risk profiles:

      Action 0 ("safe"):   always +1
      Action 1 ("risky"):  50% chance +4, 50% chance -2  (EV = +1.0)
      Action 2 ("trap"):   80% chance +0.5, 20% chance -5  (EV = -0.6)

    The state encodes a "market signal" (3 contexts) that shifts EVs:
      Context 0: as above (safe=risky EV, trap bad)
      Context 1: risky becomes 70% +4, 30% -2 (EV=+2.2 → risky is BEST)
      Context 2: risky becomes 30% +4, 70% -2 (EV=-0.2 → safe is BEST)

    The agent must learn:
      1. Trap is always bad.
      2. Context matters for safe-vs-risky choice.
    """

    def __init__(self) -> None:
        self._context: int = 0

    def reset(self) -> np.ndarray:
        self._context = int(np.random.random() * 3) % 3
        return self._make_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action = min(action, 2)
        if action == 0:  # safe
            reward = 1.0
        elif action == 1:  # risky
            p_win = [0.50, 0.70, 0.30][self._context]
            reward = 4.0 if np.random.random() < p_win else -2.0
        else:  # trap
            reward = 0.5 if np.random.random() < 0.8 else -5.0
        return self._make_state(), reward, True, {"context": self._context}

    def _make_state(self) -> np.ndarray:
        s = np.zeros(3, dtype=np.float32)
        s[self._context] = 1.0
        return s

    @property
    def n_actions(self) -> int:
        return 3

    @property
    def state_size(self) -> int:
        return 3


# =====================================================================
# Level 7: T-maze — memory + delayed reward
# =====================================================================

class TMazeEnv(Environment):
    """
    T-maze with cue at start.

    Layout:
      [start] → [corridor1] → [corridor2] → [junction]
                                               ↙       ↘
                                           [left]     [right]

    At reset, a cue is shown (state[0]=0 or 1) indicating which arm
    has the reward (+10).  During corridor traversal the cue disappears.
    At the junction, the agent must remember the cue and choose correctly.

    Actions:  0 = move forward / go left,  1 = move forward / go right
      - In corridor cells: both actions move forward.
      - At junction: 0 = left, 1 = right.

    State:  [cue, position_onehot(5)]  → 6D total.
      Position 0=start (cue visible), 1-2=corridor, 3=junction, 4=terminal.

    Reward: +10 at correct arm, -1 at wrong arm, -0.1 per step.
    """

    def __init__(self) -> None:
        self._cue: int = 0       # 0=left is correct, 1=right is correct
        self._pos: int = 0       # 0=start, 1-2=corridor, 3=junction

    def reset(self) -> np.ndarray:
        self._cue = int(np.random.random() < 0.5)
        self._pos = 0
        return self._make_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self._pos < 3:
            # Corridor: move forward regardless of action
            self._pos += 1
            return self._make_state(), -0.1, False, {"cue": self._cue, "pos": self._pos}
        else:
            # Junction: choose arm
            correct = (action == 0 and self._cue == 0) or (action == 1 and self._cue == 1)
            reward = 10.0 if correct else -1.0
            self._pos = 4  # terminal
            return self._make_state(), reward, True, {"cue": self._cue, "pos": self._pos, "correct": correct}

    def _make_state(self) -> np.ndarray:
        s = np.zeros(6, dtype=np.float32)
        # Cue is only visible at start position
        if self._pos == 0:
            s[0] = float(self._cue)
        pos_idx = min(self._pos, 4)
        s[1 + pos_idx] = 1.0
        return s

    @property
    def n_actions(self) -> int:
        return 2

    @property
    def state_size(self) -> int:
        return 6


# =====================================================================
# Level 8: Punishment avoidance — learn to NOT act
# =====================================================================

class PunishmentAvoidanceEnv(Environment):
    """
    Two actions, two contexts.  One action is always neutral (0 reward).
    The other action is punished in one context but rewarded in another.

      Context A [1,0]: action 0 → 0,   action 1 → -3 (punishment!)
      Context B [0,1]: action 0 → 0,   action 1 → +2 (reward)

    The agent must learn to suppress action 1 in context A
    while executing it in context B.  This tests inhibitory learning
    (NoGo pathway in basal ganglia).
    """

    def __init__(self) -> None:
        self._context: int = 0

    def reset(self) -> np.ndarray:
        self._context = int(np.random.random() < 0.5)
        return self._make_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if action == 0:
            reward = 0.0
        else:
            reward = -3.0 if self._context == 0 else 2.0
        return self._make_state(), reward, True, {"context": self._context}

    def _make_state(self) -> np.ndarray:
        s = np.zeros(2, dtype=np.float32)
        s[self._context] = 1.0
        return s

    @property
    def n_actions(self) -> int:
        return 2

    @property
    def state_size(self) -> int:
        return 2
