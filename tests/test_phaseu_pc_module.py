"""Faza U — krok U.1: kanoniczny moduł PC z prawdziwą relaksacją.

These are the *hard gate* for the rest of Faza U (``plan_unification.md``
§9.1–§9.2).  ``core.pc_module`` introduces the dynamical core that
``error_neuron`` lacks: an explicit free-energy relaxation
``μ̇ = −∂F/∂μ``.  Two properties must hold before any region is migrated
to the shared rule:

U.1a  relaxation actually descends the free energy it claims to minimise;
U.1b  the resulting *local* weight rule performs backprop-grade credit
      assignment — proven via the fixed-prediction equilibrium
      reproducing ``jax.grad`` to cosine ≈ 1 (Whittington & Bogacz 2017;
      Millidge & Bogacz 2020).

A third test confirms the end-to-end module actually *learns* a
supervised mapping (the rule is not just directionally correct but
usable), and a fourth documents that the biologically-faithful *online*
relaxation (predictions recomputed each step) aligns strongly with
backprop (the expected ~0.9 approximation, not exact — that is what
fixed-prediction is for).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.pc_module import (
    init_pc_net_params, init_pc_net_state,
    pc_feedforward, pc_clamp_bottom, pc_relax, pc_free_energy,
    pc_weight_grads, pc_fixed_prediction_grads, pc_supervised_step,
)
from core.pc_module import _phi


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _cosine(a_tuple, b_tuple) -> float:
    af = jnp.concatenate([g.ravel() for g in a_tuple])
    bf = jnp.concatenate([g.ravel() for g in b_tuple])
    return float(af @ bf / (jnp.linalg.norm(af) * jnp.linalg.norm(bf)))


def _mlp_grad(weights, act, inp, target):
    """Backprop grad of ½‖forward(inp) − target‖² wrt each weight matrix."""
    def forward(w, x):
        a = x
        for l in range(len(w) - 1, -1, -1):
            a = w[l] @ _phi(act, a)
        return a

    def loss(w):
        return 0.5 * jnp.sum((forward(w, inp) - target) ** 2)

    return jax.grad(loss)(list(weights))


def _setup(seed=0, act="tanh", pert=0.5, **pkw):
    sizes = (3, 8, 8, 6)          # bottom (output)=3 … top (input)=6
    params = init_pc_net_params(sizes, act=act, **pkw)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    state = init_pc_net_state(k1, params)
    inp = jax.random.normal(k2, (sizes[-1],))
    ff = pc_feedforward(state, params, inp)
    target = ff.mu[0] + pert * jax.random.normal(k3, (sizes[0],))
    return params, state, inp, target, ff


# ---------------------------------------------------------------------
# U.1a — relaxation descends free energy
# ---------------------------------------------------------------------


def test_relaxation_descends_free_energy():
    """``pc_relax`` must monotonically reduce F and reach a fixed point."""
    params, state, inp, target, ff = _setup(eta_mu=0.1, eta_w=1e-2, n_relax=200)
    clamped = pc_clamp_bottom(ff, target)

    energies = [
        float(pc_free_energy(pc_relax(clamped, params, clamp_bottom=True, n_steps=k), params))
        for k in range(0, 201, 20)
    ]

    # Strictly descends overall, never increases between checkpoints.
    assert energies[-1] < energies[0] * 0.7, (
        f"free energy barely moved: {energies[0]:.4f} → {energies[-1]:.4f}"
    )
    for i in range(len(energies) - 1):
        assert energies[i + 1] <= energies[i] + 1e-5, (
            f"free energy increased at checkpoint {i}: {energies}"
        )

    # Fixed-point residual: max |∂F/∂μ| over internal nodes ≈ 0.
    relaxed = pc_relax(clamped, params, clamp_bottom=True, n_steps=200)
    L = params.n_layers
    preds = [relaxed.weights[l] @ _phi(params.act, relaxed.mu[l + 1]) for l in range(L)]
    eps = [relaxed.mu[l] - preds[l] for l in range(L)]
    from core.pc_module import _phi_prime
    resid = 0.0
    for l in range(1, L):
        g = eps[l] - _phi_prime(params.act, relaxed.mu[l]) * (relaxed.weights[l - 1].T @ eps[l - 1])
        resid = max(resid, float(jnp.max(jnp.abs(g))))
    assert resid < 1e-3, f"relaxation did not reach a fixed point: resid={resid:.2e}"


# ---------------------------------------------------------------------
# U.1b — fixed-prediction equilibrium == backprop  (THE GATE)
# ---------------------------------------------------------------------


def test_fixed_prediction_equals_backprop():
    """Local PC rule reproduces backprop gradients to cosine > 0.99.

    This is the hard gate for Faza U: if PC does not deliver
    backprop-grade credit assignment, the unification has no scaling
    payoff and we stop (plan §9.2).
    """
    for seed in range(6):
        for act in ("linear", "tanh"):
            for pert in (0.3, 1.0):
                params, state, inp, target, _ = _setup(
                    seed=seed, act=act, pert=pert,
                    eta_mu=0.1, eta_w=1e-2, n_relax=400,
                )
                pc_g = pc_fixed_prediction_grads(state, params, inp, target)
                bp_g = _mlp_grad(state.weights, act, inp, target)
                cos = _cosine(pc_g, bp_g)
                assert cos > 0.99, (
                    f"PC≠backprop (seed={seed}, act={act}, pert={pert}): cos={cos:.4f}"
                )


# ---------------------------------------------------------------------
# online relaxed PC aligns with backprop (documented ~0.9 approximation)
# ---------------------------------------------------------------------


def test_online_relaxation_aligns_with_backprop():
    """Standard relaxation (predictions recomputed) tracks backprop direction.

    Vanilla relaxed PC only approximates backprop (it uses relaxed μ as
    both value and presynaptic factor); cosine ≈ 0.9 is the expected,
    literature-documented behaviour — exactness needs fixed-prediction.
    """
    cosines = []
    for seed in range(6):
        params, state, inp, target, ff = _setup(
            seed=seed, act="tanh", pert=0.3, eta_mu=0.1, eta_w=1e-2, n_relax=300,
        )
        clamped = pc_clamp_bottom(ff, target)
        relaxed = pc_relax(clamped, params, clamp_bottom=True, n_steps=300)
        pc_g = pc_weight_grads(relaxed, params)
        bp_g = _mlp_grad(state.weights, "tanh", inp, target)
        cosines.append(_cosine(pc_g, bp_g))
    assert min(cosines) > 0.80, f"online PC misaligned: min cos={min(cosines):.4f}"


# ---------------------------------------------------------------------
# end-to-end: the module actually learns a supervised mapping
# ---------------------------------------------------------------------


def test_supervised_learning_reduces_loss():
    """Repeated ``pc_supervised_step`` on a fixed (x, y) pair cuts the loss.

    Confirms the local rule is usable for learning, not merely
    directionally aligned for one step.
    """
    sizes = (3, 16, 16, 6)
    params = init_pc_net_params(
        sizes, act="tanh", eta_mu=0.1, eta_w=5e-2, n_relax=50,
    )
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(7), 3)
    state = init_pc_net_state(k1, params)
    inp = jax.random.normal(k2, (sizes[-1],))
    target = jax.random.normal(k3, (sizes[0],)) * 0.5

    out0 = pc_feedforward(state, params, inp).mu[0]
    loss0 = float(0.5 * jnp.sum((out0 - target) ** 2))

    for _ in range(300):
        out = pc_supervised_step(state, params, inp, target)
        state = out.state

    outN = pc_feedforward(state, params, inp).mu[0]
    lossN = float(0.5 * jnp.sum((outN - target) ** 2))

    assert lossN < loss0 * 0.1, (
        f"PC learning did not reduce loss enough: {loss0:.4f} → {lossN:.4f}"
    )
