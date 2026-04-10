"""
Variational Free Energy — unified objective for all subsystems.

Reference: Friston (2010) "The free-energy principle: a unified brain theory?"

Under the Free Energy Principle (FEP), all neural processes reduce to
minimizing variational free energy F.  For a generative model p(o, s)
with recognition density q(s):

    F  =  E_q[ log q(s) - log p(o, s) ]
       =  -log p(o) + KL[ q(s) || p(s|o) ]
       ≥  -log p(o)                           (surprise lower bound)

In practice (Gaussian assumptions, Laplace approximation):

    F  ≈  ½ (ε_s^T Π_s ε_s  +  ε_o^T Π_o ε_o)  +  ½ ln|Σ_s| + const

where:
    ε_s = μ - η              (state prediction error)
    ε_o = o - g(μ)           (observation prediction error)
    Π_s = Σ_s^{-1}           (state precision)
    Π_o = Σ_o^{-1}           (observation precision)
    g(μ)                     (generative model / top-down prediction)

Expected Free Energy for action selection (Active Inference):

    G(a) = E_q[ ln q(s|a) - ln p(o, s|a) ]
         ≈  ambiguity(a)  +  risk(a)  -  info_gain(a)

    ambiguity   = E_q[ H[p(o|s, a)] ]     (expected sensory entropy)
    risk        = KL[ q(o|a) || p(o) ]     (deviation from preferred outcomes)
    info_gain   = H[q(s|a)] - E_q[H[q(s|o,a)]]  (expected uncertainty reduction)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True)
class FreeEnergyConfig:
    """Hyperparameters for free energy computation."""
    # Default precision (inverse variance) when astrocyte not available
    default_precision: float = 1.0
    # Minimum precision to avoid division by zero
    min_precision: float = 0.01
    # Regularization for log-determinant computation
    log_det_eps: float = 1e-6


def variational_free_energy(
    prediction_error: NDArray[np.float32],
    precision: NDArray[np.float32] | float = 1.0,
) -> float:
    """Compute variational free energy F = ½ ε^T Π ε.

    This is the precision-weighted prediction error energy, the core
    quantity minimized by all neural inference processes.

    Args:
        prediction_error: Error vector ε = observation - prediction.
        precision: Per-element precision (inverse variance). Scalar
                   broadcasts to all elements. From AstrocyteField or
                   learned precision heads.

    Returns:
        Scalar free energy (≥ 0).
    """
    eps = prediction_error.astype(np.float32)
    if isinstance(precision, (int, float)):
        pi = float(precision)
        return 0.5 * pi * float(np.sum(eps ** 2))
    else:
        pi = np.asarray(precision, dtype=np.float32)
        # Broadcast precision to match error shape if needed
        if pi.shape[0] != eps.shape[0]:
            pi = _broadcast_precision(pi, eps.shape[0])
        return 0.5 * float(np.sum(pi * eps ** 2))


def expected_free_energy(
    pragmatic_value: float,
    epistemic_value: float,
    ambiguity: float = 0.0,
    epistemic_weight: float = 1.0,
) -> float:
    """Compute expected free energy G(a) for action selection.

    G(a)  =  -pragmatic(a)  +  ambiguity(a)  -  β × epistemic(a)

    The action minimizing G (= maximizing pragmatic + epistemic) is selected.

    Args:
        pragmatic_value: Expected reward / preference satisfaction.
        epistemic_value: Expected information gain (uncertainty reduction).
        ambiguity:       Expected observation entropy (sensory noise).
        epistemic_weight: β — modulated by NE (curiosity drive).

    Returns:
        Scalar expected free energy (lower is better).
    """
    return -pragmatic_value + ambiguity - epistemic_weight * epistemic_value


def precision_weighted_update(
    prediction_error: NDArray[np.float32],
    precision: NDArray[np.float32] | float,
    learning_rate: float,
) -> NDArray[np.float32]:
    """Compute precision-weighted gradient for belief/weight updates.

    Δμ = lr × Π × ε

    Regions with high precision (low uncertainty / high astrocyte confidence)
    drive stronger updates. This replaces ad-hoc error weighting.

    Args:
        prediction_error: Error vector ε.
        precision: Per-element or scalar precision.
        learning_rate: Base learning rate.

    Returns:
        Gradient vector of same shape as prediction_error.
    """
    eps = prediction_error.astype(np.float32)
    if isinstance(precision, (int, float)):
        return learning_rate * float(precision) * eps
    else:
        pi = np.asarray(precision, dtype=np.float32)
        if pi.shape[0] != eps.shape[0]:
            pi = _broadcast_precision(pi, eps.shape[0])
        return learning_rate * pi * eps


def _broadcast_precision(
    precision: NDArray[np.float32],
    target_n: int,
) -> NDArray[np.float32]:
    """Map n_zones precision to target_n elements by nearest zone."""
    if precision.shape[0] == target_n:
        return precision
    indices = np.linspace(0, precision.shape[0] - 1, target_n).astype(int)
    return precision[indices]
