import unittest
import numpy as np

from core.config import NeuromodulatorConfig
from core.neuromodulator import NeuromodulatorSystem


class TestNeuromodulatorSystem(unittest.TestCase):
    """
    Unit tests for NeuromodulatorSystem.

    Validates:
      A) Initialisation at configured baselines.
      B) Directional responses of each modulator to its primary driver.
      C) Hard clamping to [0, 1].
      D) Decay without stimulation.
      E) reset() restores baselines.
      F) All typed properties exist and return plausible floats.
    """

    def setUp(self) -> None:
        self.config = NeuromodulatorConfig(
            da_decay=0.95,
            ach_decay=0.90,
            ne_decay=0.93,
            sero_decay=0.97,
            baseline_da=0.5,
            baseline_ach=0.5,
            baseline_ne=0.3,
            baseline_sero=0.6,
        )
        self.nm = NeuromodulatorSystem(self.config)

    # ──────────────────────────────────────────────────────────────────
    # A. Initialisation
    # ──────────────────────────────────────────────────────────────────

    def test_initial_dopamine_equals_baseline(self) -> None:
        self.assertAlmostEqual(self.nm.dopamine, self.config.baseline_da)

    def test_initial_acetylcholine_equals_baseline(self) -> None:
        self.assertAlmostEqual(self.nm.acetylcholine, self.config.baseline_ach)

    def test_initial_noradrenaline_equals_baseline(self) -> None:
        self.assertAlmostEqual(self.nm.noradrenaline, self.config.baseline_ne)

    def test_initial_serotonin_equals_baseline(self) -> None:
        self.assertAlmostEqual(self.nm.serotonin, self.config.baseline_sero)

    # ──────────────────────────────────────────────────────────────────
    # B. Directional responses
    # ──────────────────────────────────────────────────────────────────

    def test_positive_td_error_raises_dopamine(self) -> None:
        """
        Positive Temporal Difference error from Basal Ganglia should push
        dopamine above its initial baseline (RPE signal).
        """
        zero_error = np.zeros(5)
        # Neutral td_error=0.0 means RPE=0.5 (neutral baseline)
        for _ in range(5):
            self.nm.update(zero_error, td_error=0.0)
        da_before = self.nm.dopamine

        # Positive td_error > 0
        for _ in range(20):
            self.nm.update(zero_error, td_error=0.5)

        self.assertGreater(
            self.nm.dopamine, da_before,
            "Positive TD error must increase dopamine.",
        )

    def test_explicit_high_novelty_raises_acetylcholine(self) -> None:
        """novelty=1.0 must push acetylcholine upward."""
        zero_error = np.zeros(5)
        ach_before = self.nm.acetylcholine

        for _ in range(20):
            self.nm.update(zero_error, td_error=0.0, novelty=1.0)

        self.assertGreater(
            self.nm.acetylcholine, ach_before,
            "Maximum novelty signal must raise acetylcholine.",
        )

    def test_large_prediction_error_raises_noradrenaline(self) -> None:
        """Large |prediction_error| must push noradrenaline upward."""
        large_error = np.ones(5) * 2.0
        ne_before = self.nm.noradrenaline

        for _ in range(20):
            self.nm.update(large_error, td_error=0.0)

        self.assertGreater(
            self.nm.noradrenaline, ne_before,
            "Large prediction error must raise noradrenaline.",
        )

    def test_consistent_zero_error_raises_serotonin(self) -> None:
        """
        Sustained zero prediction error (perfect world model) represents maximum
        stability and must push serotonin above its initial level.
        """
        sero_before = self.nm.serotonin
        zero_error = np.zeros(5)

        for _ in range(50):
            self.nm.update(zero_error, td_error=0.0, novelty=0.0)

        self.assertGreater(
            self.nm.serotonin, sero_before,
            "Zero prediction error (stable world) must raise serotonin.",
        )

    # ──────────────────────────────────────────────────────────────────
    # C. Hard clamping
    # ──────────────────────────────────────────────────────────────────

    def test_all_levels_stay_within_unit_interval_under_extreme_input(self) -> None:
        """No level must escape [0, 1] under extreme or adversarial input."""
        extreme_error = np.ones(5) * 1000.0
        for _ in range(100):
            self.nm.update(extreme_error, td_error=1000.0, novelty=1000.0)

        for name, val in [
            ("dopamine", self.nm.dopamine),
            ("acetylcholine", self.nm.acetylcholine),
            ("noradrenaline", self.nm.noradrenaline),
            ("serotonin", self.nm.serotonin),
        ]:
            self.assertGreaterEqual(val, 0.0, f"{name} fell below 0.")
            self.assertLessEqual(val, 1.0, f"{name} exceeded 1.")

    def test_no_level_goes_negative_under_zero_input(self) -> None:
        """All levels must remain ≥ 0 even with novelty=0 and td_error=-1.0."""
        zero_error = np.zeros(5)
        for _ in range(200):
            self.nm.update(zero_error, td_error=-1.0, novelty=0.0)

        self.assertGreaterEqual(self.nm.dopamine, 0.0)
        self.assertGreaterEqual(self.nm.acetylcholine, 0.0)
        self.assertGreaterEqual(self.nm.noradrenaline, 0.0)
        self.assertGreaterEqual(self.nm.serotonin, 0.0)

    # ──────────────────────────────────────────────────────────────────
    # D. Decay without stimulation
    # ──────────────────────────────────────────────────────────────────

    def test_dopamine_decays_with_zero_td_error(self) -> None:
        """
        Manually set dopamine to maximum, then drive with neutral (0.0) td_error.
        It must decay from 1.0 towards 0.5.
        """
        self.nm.dopamine = 1.0
        zero_error = np.zeros(5)

        for _ in range(50):
            self.nm.update(zero_error, td_error=0.0, novelty=0.0)

        self.assertLess(
            self.nm.dopamine, 1.0,
            "Dopamine must decay from peak when td_error stays neutral.",
        )

    # ──────────────────────────────────────────────────────────────────
    # E. Reset
    # ──────────────────────────────────────────────────────────────────

    def test_reset_restores_all_baselines(self) -> None:
        """reset() must restore every level to its configured baseline."""
        # Perturb all levels
        self.nm.dopamine = 0.0
        self.nm.acetylcholine = 1.0
        self.nm.noradrenaline = 1.0
        self.nm.serotonin = 0.0

        self.nm.reset()

        self.assertAlmostEqual(self.nm.dopamine, self.config.baseline_da)
        self.assertAlmostEqual(self.nm.acetylcholine, self.config.baseline_ach)
        self.assertAlmostEqual(self.nm.noradrenaline, self.config.baseline_ne)
        self.assertAlmostEqual(self.nm.serotonin, self.config.baseline_sero)

    def test_reset_clears_histories(self) -> None:
        """reset() must empty _error_history."""
        zero_error = np.zeros(5)
        for _ in range(10):
            self.nm.update(zero_error, td_error=0.5)
        self.nm.reset()

        self.assertEqual(len(self.nm._error_history), 0)

    # ──────────────────────────────────────────────────────────────────
    # F. Properties
    # ──────────────────────────────────────────────────────────────────

    def test_all_properties_return_float_in_unit_interval(self) -> None:
        """Every public property must return a float in [0, 1]."""
        for prop_name in [
            "learning_rate_modulation",
            "bottom_up_gain",
            "competition_sharpness",
            "planning_horizon",
        ]:
            val = getattr(self.nm, prop_name)
            self.assertIsInstance(val, float, f"{prop_name} must return float.")
            self.assertGreaterEqual(val, 0.0, f"{prop_name} below 0.")
            self.assertLessEqual(val, 1.0, f"{prop_name} above 1.")

    def test_learning_rate_modulation_reflects_dopamine(self) -> None:
        """learning_rate_modulation must equal dopamine."""
        self.nm.dopamine = 0.73
        self.assertAlmostEqual(self.nm.learning_rate_modulation, 0.73)

    def test_bottom_up_gain_reflects_acetylcholine(self) -> None:
        """bottom_up_gain must equal acetylcholine."""
        self.nm.acetylcholine = 0.11
        self.assertAlmostEqual(self.nm.bottom_up_gain, 0.11)


if __name__ == "__main__":
    unittest.main(verbosity=2)