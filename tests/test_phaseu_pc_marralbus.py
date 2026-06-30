"""Faza U — Phase 1: the Marr–Albus forward model does not collapse.

The motor→cerebellum→sensory forward model used to collapse to a
"predict-the-mean" trivial fixed point: a plastic deep edge feeding a free,
over-complete cerebellar latent decays under the one rule
(``ΔW_mc ≈ −η·W_mc·φφᵀ`` → ``W_mc → 0``), the hidden code goes
command-independent (``cb_var → 0``), and the sensory bias absorbs the mean
reafference.  See ``diagnostics/embodied_reach_collapse/FINDINGS.md``.

The fix restores the Marr–Albus design: the ``motor→cerebellum`` granule
expansion is a *frozen* random non-linearity and the cerebellum is a
*feedforward* (deterministic) layer, so there is no deep plastic edge into a
free latent to collapse; only the ``cerebellum→sensory`` Purkinje readout is
plastic.  These tests assert the acceptance criteria:

* the granule edge / its threshold bias are never modified by learning;
* the cerebellar feature variance stays bounded away from zero across babble;
* the cerebellum is a deterministic function of the motor command
  (its belief equals its top-down prediction after relaxation);
* the forward-model readout actually learns (sensory free energy falls).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from core.backend import DTYPE, make_key
from core.pc_brain import init_pc_brain, pc_brain_learn_forward
from core.pc_graph import (
    REGION_INDEX, pc_graph_clamp, pc_graph_relax, pc_graph_predictions,
)
from core.pc_module import _phi
from sensory.population_code import (
    gaussian_population_encode, monotonic_population_encode,
)

# Small synthetic 2-link arm body (CPU, no MuJoCo) — same shape the real
# embodiment uses: proprioception (gaussian) + absolute tip (monotonic).
N = 6
L1 = L2 = 0.25
HALF = 0.5
JR = 2.0
PROP = 2 * 2 * N
TIP = 2 * N
SENS = PROP + TIP


def _fk(c):
    q = JR * c
    return jnp.stack([
        L1 * jnp.cos(q[0]) + L2 * jnp.cos(q[0] + q[1]),
        L1 * jnp.sin(q[0]) + L2 * jnp.sin(q[0] + q[1]),
    ])


def _enc(c):
    q = JR * c
    ang = jnp.concatenate(
        [gaussian_population_encode(q[j], N, x_min=-JR, x_max=JR) for j in range(2)]
    )
    vel = jnp.concatenate(
        [gaussian_population_encode(0.0, N, x_min=-JR, x_max=JR) for _ in range(2)]
    )
    t = _fk(c)
    tip = jnp.concatenate([
        monotonic_population_encode(t[0], N, x_min=-HALF, x_max=HALF),
        monotonic_population_encode(t[1], N, x_min=-HALF, x_max=HALF),
    ])
    return jnp.concatenate([ang, vel, tip]).astype(DTYPE)


def _mc_edge(params):
    m, cb = params.motor_idx, params.cerebellum_idx
    for e, (a, b) in enumerate(params.graph.edges):
        if a == m and b == cb:
            return e
    raise AssertionError("no motor→cerebellum edge")


@eqx.filter_jit
def _granule_feats(params, state, bels):
    """φ(μ_cerebellum) for a batch of motor commands (forward pass)."""
    m, cb = params.motor_idx, params.cerebellum_idx
    hold = (m,) + tuple(params.perceptual_nodes)

    def one(bel):
        cl = pc_graph_clamp(state.graph, {m: bel})
        rl = pc_graph_relax(cl, params.graph, clamp=hold, n_steps=40)
        return _phi(params.graph.act, rl.mu[cb])

    return jax.vmap(one)(bels)


def _cb_var(params, state, n_cmd=24):
    """Forward-pass cerebellar feature variance over a command grid."""
    cs = jax.random.uniform(make_key(7), (n_cmd, 2), DTYPE, -0.9, 0.9)
    bels = jnp.arctanh(jnp.clip(cs, -0.999, 0.999))
    feats = _granule_feats(params, state, bels)
    return float(jnp.mean(jnp.var(feats, axis=0)))


def _babble(params, state, n):
    """OU motor babble, many cycles under one compile (lax.scan)."""
    a = jnp.asarray(np.exp(-1.0 / 20.0), DTYPE)
    g = jnp.asarray(1.5 * np.sqrt(1.0 - float(a) * float(a)), DTYPE)

    @eqx.filter_jit
    def run(state, keys):
        def step(carry, k):
            st, bel = carry
            bel = a * bel + g * jax.random.normal(k, (2,), DTYPE)
            st = pc_brain_learn_forward(
                st, params, bel, _enc(jnp.tanh(bel)), n_relax=None,
            )
            return (st, bel), None
        (state, _), _ = jax.lax.scan(step, (state, jnp.zeros(2, DTYPE)), keys)
        return state

    return run(state, jax.random.split(make_key(1), int(n)))


def _build():
    return init_pc_brain(
        make_key(0), sensory_size=SENS, motor_size=2,
        eta_w=0.05, n_relax=20,
    )


def test_region_graph_marr_albus_wiring():
    params, _ = _build()
    gp = params.graph
    cb = REGION_INDEX["cerebellum"]
    mc = _mc_edge(params)
    # The granule edge is frozen; the cerebellum is feedforward; the readout
    # is NLMS-stabilised; the granule threshold bias is NOT learnable.
    assert mc in gp.frozen_edges
    assert cb in gp.feedforward_nodes
    assert cb not in gp.bias_nodes
    assert REGION_INDEX["sensory"] in gp.bias_nodes
    cs = next(
        e for e, (a, b) in enumerate(gp.edges)
        if a == cb and b == REGION_INDEX["sensory"]
    )
    assert cs in gp.nlms_edges


def test_granule_expansion_frozen():
    """The frozen motor→cerebellum edge and its bias never change under learning."""
    params, state = _build()
    cb = params.cerebellum_idx
    mc = _mc_edge(params)
    W0 = state.graph.weights[mc]
    b0 = state.graph.bias[cb]
    # A non-degenerate random expansion (not the zero/LeCun default).
    assert float(jnp.linalg.norm(W0)) > 1.0
    state = _babble(params, state, 300)
    assert jnp.allclose(state.graph.weights[mc], W0), "granule edge was modified"
    assert jnp.allclose(state.graph.bias[cb], b0), "granule threshold was modified"


def test_cerebellum_does_not_collapse():
    """Cerebellar feature variance stays bounded away from zero across babble."""
    params, state = _build()
    v0 = _cb_var(params, state)
    assert v0 > 1e-2, f"granule code uninformative at init: {v0:.2e}"
    state = _babble(params, state, 400)
    vN = _cb_var(params, state)
    # The legacy free-latent cerebellum collapsed cb_var from ~0.15 to ~2e-5;
    # a frozen feedforward expansion cannot collapse at all.
    assert vN > 0.5 * v0, f"cerebellar code collapsed: {v0:.4f} → {vN:.4f}"


def test_cerebellum_is_feedforward_deterministic():
    """After relaxation the cerebellar belief equals its top-down prediction."""
    params, state = _build()
    state = _babble(params, state, 200)
    m, cb = params.motor_idx, params.cerebellum_idx
    hold = (m,) + tuple(params.perceptual_nodes)
    bel = jnp.array([0.3, -0.4], DTYPE)
    cl = pc_graph_clamp(state.graph, {m: bel})
    rl = pc_graph_relax(cl, params.graph, clamp=hold, n_steps=60)
    pred = pc_graph_predictions(rl, params.graph)[cb]
    assert jnp.allclose(rl.mu[cb], pred, atol=1e-5), (
        "feedforward cerebellum belief diverged from its prediction"
    )


@eqx.filter_jit
def _sensory_mse(params, state, cs, targets):
    """Mean sensory prediction error over a batch of (command, reafference)."""
    m, s = params.motor_idx, params.sensory_idx
    hold = (m,) + tuple(params.perceptual_nodes)

    def one(bel, tgt):
        cl = pc_graph_clamp(state.graph, {m: bel})
        rl = pc_graph_relax(cl, params.graph, clamp=hold, n_steps=40)
        pred = pc_graph_predictions(rl, params.graph)[s]
        return jnp.mean((tgt - pred) ** 2)

    return jnp.mean(jax.vmap(one)(cs, targets))


def test_forward_model_readout_learns():
    """The plastic Purkinje readout lowers sensory prediction error with babble."""
    params, state = _build()
    cs = jax.random.uniform(make_key(11), (24, 2), DTYPE, -0.7, 0.7)
    bels = jnp.arctanh(jnp.clip(cs, -0.999, 0.999))
    targets = jax.vmap(_enc)(cs)

    fe0 = float(_sensory_mse(params, state, bels, targets))
    state = _babble(params, state, 500)
    feN = float(_sensory_mse(params, state, bels, targets))
    assert feN < 0.5 * fe0, f"readout did not learn: sensory MSE {fe0:.4f} → {feN:.4f}"
