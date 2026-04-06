"""
Active Inference — Epistemic Foraging for directed exploration.

Biological grounding:
  Under Active Inference (Friston, 2010), agents do not merely minimize
  reward prediction error (TD-error) — they also seek to *reduce*
  uncertainty about their world model.  This epistemic drive prevents the
  "dark room problem" where an agent avoids novel stimuli to minimize
  surprise.

  Neurally, the anterior cingulate cortex (ACC) encodes expected
  uncertainty, and the frontopolar cortex tracks information gain.
  Dopaminergic projections to prefrontal cortex carry both reward
  prediction errors AND information prediction errors (Bromberg-Martin
  et al., 2010).

Architecture:
  ActiveInferenceModule wraps an SNNWorldModel and a BasalGangliaAGISystem:
    1. For each candidate action, runs mental_rehearsal to get predicted
       next state and novelty.
    2. Computes *epistemic value* = prediction uncertainty for that action
       (high novelty → high epistemic value → explore this direction).
    3. Combines pragmatic value (expected reward from critic) with
       epistemic value (information gain) to produce a total expected
       free energy per action.
    4. The action with lowest expected free energy (= highest combined
       pragmatic + epistemic value) is selected.

  The epistemic_weight parameter controls the explore/exploit tradeoff:
    - High epistemic_weight → curious agent (explores uncertain states).
    - Low epistemic_weight  → greedy agent (exploits known rewards).
    - Epistemic weight is modulated by noradrenaline: high NE → more
      exploration (biological link: locus coeruleus → exploration mode).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .config import ActiveInferenceConfig

if TYPE_CHECKING:
    from .basal_ganglia import BasalGangliaAGISystem
    from .world_model import SNNWorldModel


class ActiveInferenceModule:
    """
    Combines pragmatic (reward-seeking) and epistemic (uncertainty-reducing)
    drives into a unified action selection mechanism.

    This replaces blind Gaussian exploration in the actor with directed
    epistemic foraging: the agent preferentially explores states where
    its world model is most uncertain.
    """

    def __init__(
        self,
        world_model: "SNNWorldModel",
        config: ActiveInferenceConfig | None = None,
    ) -> None:
        self.world_model = world_model
        self.config = config or ActiveInferenceConfig()
        self.action_size = world_model.action_size

        # Diagnostic outputs from last step
        self.last_epistemic_values: dict[int, float] = {}
        self.last_pragmatic_values: dict[int, float] = {}
        self.last_total_values: dict[int, float] = {}
        self.last_selected_action: int = 0

    # ------------------------------------------------------------------
    # Core: epistemic value computation
    # ------------------------------------------------------------------

    def compute_epistemic_values(
        self,
        state_spikes: np.ndarray,
        candidate_actions: list[int],
    ) -> dict[int, float]:
        """
        Compute epistemic value (information gain) for each candidate action.

        Uses mental_rehearsal to simulate each action and measures the
        world model's prediction uncertainty for the resulting transition.

        Args:
            state_spikes:      Current state as spike vector.
            candidate_actions: List of discrete action indices to evaluate.

        Returns:
            Dict mapping action → epistemic value (higher = more uncertain).
        """
        results = self.world_model.mental_rehearsal(
            state_spikes, candidate_actions
        )

        epistemic = {}
        for action in candidate_actions:
            info = results[action]
            if self.config.uncertainty_method == "novelty":
                epistemic[action] = info["novelty"]
            else:
                # Variance method: re-run rehearsal with perturbed state
                # and measure prediction spread
                epistemic[action] = self._variance_uncertainty(
                    state_spikes, action
                )
        return epistemic

    def _variance_uncertainty(
        self,
        state_spikes: np.ndarray,
        action: int,
        n_samples: int = 3,
    ) -> float:
        """
        Estimate prediction uncertainty via variance across perturbed inputs.

        Run mental_rehearsal multiple times with small noise perturbations
        to the state and measure the variance of predicted next states.
        High variance → model is uncertain about this transition.
        """
        predictions = []
        for _ in range(n_samples):
            noise = np.random.normal(0, 0.05, state_spikes.shape).astype(np.float32)
            perturbed = np.clip(state_spikes + noise, 0.0, 1.0)
            result = self.world_model.mental_rehearsal(perturbed, [action])
            predictions.append(result[action]["predicted_state"])

        if len(predictions) < 2:
            return 0.0

        stacked = np.stack(predictions)
        return float(np.mean(np.var(stacked, axis=0)))

    # ------------------------------------------------------------------
    # Action selection: expected free energy
    # ------------------------------------------------------------------

    def select_action(
        self,
        state_spikes: np.ndarray,
        candidate_actions: list[int],
        pragmatic_values: dict[int, float] | None = None,
        ne_level: float = 0.3,
    ) -> int:
        """
        Select an action by minimizing expected free energy.

        Expected free energy G(a) = -pragmatic(a) - epistemic_weight * epistemic(a)
        The action with lowest G (= highest combined value) is selected.

        Args:
            state_spikes:      Current state as spike vector.
            candidate_actions: List of discrete action indices.
            pragmatic_values:  Optional pre-computed pragmatic values per action.
                               If None, all pragmatic values default to 0.
            ne_level:          Current noradrenaline level [0, 1].
                               Modulates epistemic drive.

        Returns:
            Selected action index.
        """
        # Compute epistemic values
        epistemic = self.compute_epistemic_values(state_spikes, candidate_actions)
        self.last_epistemic_values = epistemic

        # NE-modulated epistemic weight
        eff_weight = (
            self.config.epistemic_weight
            + ne_level * self.config.ne_epistemic_boost
        )

        # Combine pragmatic and epistemic
        total = {}
        for action in candidate_actions:
            prag = 0.0 if pragmatic_values is None else pragmatic_values.get(action, 0.0)
            epist = epistemic.get(action, 0.0)
            total[action] = prag + eff_weight * epist

        self.last_pragmatic_values = pragmatic_values or {a: 0.0 for a in candidate_actions}
        self.last_total_values = total

        # Softmax action selection
        actions = list(total.keys())
        values = np.array([total[a] for a in actions], dtype=np.float32)

        shifted = values - np.max(values)
        exp_vals = np.exp(shifted / max(self.config.pragmatic_temperature, 1e-6))
        probs = exp_vals / (np.sum(exp_vals) + 1e-8)

        selected = int(np.random.choice(actions, p=probs))
        self.last_selected_action = selected
        return selected

    def select_action_greedy(
        self,
        state_spikes: np.ndarray,
        candidate_actions: list[int],
        pragmatic_values: dict[int, float] | None = None,
        ne_level: float = 0.3,
    ) -> int:
        """
        Greedy (argmax) variant of select_action for evaluation.

        Returns the action with highest combined pragmatic + epistemic value
        without stochastic sampling.
        """
        epistemic = self.compute_epistemic_values(state_spikes, candidate_actions)
        self.last_epistemic_values = epistemic

        eff_weight = (
            self.config.epistemic_weight
            + ne_level * self.config.ne_epistemic_boost
        )

        total = {}
        for action in candidate_actions:
            prag = 0.0 if pragmatic_values is None else pragmatic_values.get(action, 0.0)
            total[action] = prag + eff_weight * epistemic.get(action, 0.0)

        self.last_pragmatic_values = pragmatic_values or {a: 0.0 for a in candidate_actions}
        self.last_total_values = total
        self.last_selected_action = max(total, key=total.get)
        return self.last_selected_action
