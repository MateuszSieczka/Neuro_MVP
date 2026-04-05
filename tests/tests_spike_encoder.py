import unittest
import numpy as np

from core.spike_encoder import PoissonEncoder


class TestPoissonEncoder(unittest.TestCase):
    """
    Unit tests for PoissonEncoder.

    Validates:
      A) Output shape and dtype.
      B) Deterministic boundaries (rate=0 → never spike, rate=1 → always spike).
      C) Statistical correctness of intermediate rates.
      D) encode_value normalization and edge cases.
    """

    def setUp(self) -> None:
        self.encoder = PoissonEncoder()

    # ──────────────────────────────────────────────────────────────────
    # A. Output shape and dtype
    # ──────────────────────────────────────────────────────────────────

    def test_output_shape_matches_input(self) -> None:
        rates = np.array([0.5, 0.3, 0.8], dtype=np.float32)
        spikes = self.encoder.encode(rates)
        self.assertEqual(spikes.shape, rates.shape)

    def test_output_dtype_is_float32(self) -> None:
        spikes = self.encoder.encode(np.array([0.5]))
        self.assertEqual(spikes.dtype, np.float32)

    def test_output_is_binary(self) -> None:
        """Every element must be exactly 0.0 or 1.0."""
        rates = np.random.rand(100).astype(np.float32)
        spikes = self.encoder.encode(rates)
        unique_vals = set(spikes.tolist())
        self.assertTrue(unique_vals <= {0.0, 1.0}, f"Non-binary values found: {unique_vals}")

    def test_2d_input_shape(self) -> None:
        rates = np.random.rand(4, 5).astype(np.float32)
        spikes = self.encoder.encode(rates)
        self.assertEqual(spikes.shape, (4, 5))

    # ──────────────────────────────────────────────────────────────────
    # B. Deterministic boundaries
    # ──────────────────────────────────────────────────────────────────

    def test_zero_rate_never_spikes(self) -> None:
        rates = np.zeros(1000, dtype=np.float32)
        spikes = self.encoder.encode(rates)
        self.assertEqual(float(np.sum(spikes)), 0.0)

    def test_one_rate_always_spikes(self) -> None:
        rates = np.ones(1000, dtype=np.float32)
        spikes = self.encoder.encode(rates)
        np.testing.assert_array_equal(spikes, np.ones(1000, dtype=np.float32))

    def test_negative_rates_clamped_to_zero(self) -> None:
        rates = np.array([-1.0, -0.5, -100.0], dtype=np.float32)
        spikes = self.encoder.encode(rates)
        np.testing.assert_array_equal(spikes, np.zeros(3, dtype=np.float32))

    def test_rates_above_one_clamped_to_one(self) -> None:
        rates = np.array([2.0, 5.0, 100.0], dtype=np.float32)
        spikes = self.encoder.encode(rates)
        np.testing.assert_array_equal(spikes, np.ones(3, dtype=np.float32))

    # ──────────────────────────────────────────────────────────────────
    # C. Statistical correctness
    # ──────────────────────────────────────────────────────────────────

    def test_mean_spike_rate_approximates_input_rate(self) -> None:
        """Over many samples, the empirical spike rate should be close to the input rate."""
        rate = 0.4
        rates = np.full(10_000, rate, dtype=np.float32)
        spikes = self.encoder.encode(rates)
        empirical = float(np.mean(spikes))
        self.assertAlmostEqual(empirical, rate, delta=0.03)

    def test_higher_rate_produces_more_spikes(self) -> None:
        n = 10_000
        low = float(np.mean(self.encoder.encode(np.full(n, 0.2))))
        high = float(np.mean(self.encoder.encode(np.full(n, 0.8))))
        self.assertGreater(high, low)

    # ──────────────────────────────────────────────────────────────────
    # D. encode_value
    # ──────────────────────────────────────────────────────────────────

    def test_encode_value_output_is_binary(self) -> None:
        values = np.array([10.0, 20.0, 30.0, 40.0])
        spikes = self.encoder.encode_value(values)
        unique_vals = set(spikes.tolist())
        self.assertTrue(unique_vals <= {0.0, 1.0})

    def test_encode_value_empty_array(self) -> None:
        spikes = self.encoder.encode_value(np.array([]))
        self.assertEqual(spikes.shape, (0,))

    def test_encode_value_constant_input(self) -> None:
        """Constant input maps to 0.5*max_rate; should produce some spikes."""
        values = np.full(10_000, 42.0)
        spikes = self.encoder.encode_value(values, max_rate=1.0)
        empirical = float(np.mean(spikes))
        self.assertAlmostEqual(empirical, 0.5, delta=0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
