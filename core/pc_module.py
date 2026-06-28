"""Canonical predictive-coding module — pure JAX (Faza U, krok U.1).

Rate-mode hierarchical predictive coding with an EXPLICIT free-energy
relaxation, the missing dynamical core identified in
``plan_unification.md`` §1/§U.1.  ``core/error_neuron.py`` already holds
the *fields* of a PC area (state μ, error ε, generative weights) but its
``en_step`` integrates AdEx spikes — it does **not** relax the free
energy ``μ̇ = −∂F/∂μ``.  That relaxation is exactly the condition under
which predictive coding approximates backpropagation (Whittington &
Bogacz 2017); without it the credit-assignment benefit that motivates
the whole of Faza U does not exist.  This module implements it.

Model (Rao & Ballard 1999; Friston 2010; Whittington & Bogacz 2017)
-------------------------------------------------------------------
A stack of ``L`` generative weight matrices over ``L + 1`` value-node
layers ``μ₀ … μ_L`` (index 0 = bottom / output, index L = top / cause).
Each level predicts the level **below** it:

    pred_l = W_l · φ(μ_{l+1})            (l = 0 … L−1)
    ε_l    = μ_l − pred_l                (prediction error at level l)
    ξ_l    = Π_l ⊙ ε_l                   (precision-weighted error)

Free energy (Gaussian generative model, diagonal precision):

    F = ½ Σ_l Π_l · ε_l²

Inference (relaxation) — gradient descent on F w.r.t. the free nodes:

    ∂F/∂μ_l = ξ_l  −  φ'(μ_l) ⊙ (W_{l−1}ᵀ ξ_{l−1})        (1 ≤ l ≤ L−1)
    μ_l ← μ_l − η_μ · ∂F/∂μ_l                              (Incremental PC)

The top node μ_L is always clamped (the cause / input).  The bottom node
μ₀ is clamped to a target during supervised learning, free during pure
perception.

Learning (after relaxation; local Hebbian, no backprop):

    ∂F/∂W_l = −ξ_l ⊗ φ(μ_{l+1})
    W_l ← W_l − η_w · ∂F/∂W_l  =  W_l + η_w · ξ_l ⊗ φ(μ_{l+1})

Precision (Friston 2010 §3.2; FitzGerald 2015):

    pe_var_l ← (1−α) pe_var_l + α ε_l²        Π_l = 1 / (pe_var_l + floor)

At a relaxation fixed point with Π = 1 the errors ``ε_l`` equal the
backpropagation deltas and ``∂F/∂W_l`` equals the backprop weight
gradient (Whittington & Bogacz 2017 Thm.).  ``tests/test_phaseu_pc_*``
verify both the free-energy descent (U.1a) and the gradient equivalence
(U.1b, cosine > 0.9) — the hard gate for the rest of Faza U.

Relation to the rest of ``core``
--------------------------------
This is the unit that U.3 (``core/pc_graph.py``) will wire into an
arbitrary-topology graph; one ``(W_l, Π_l, μ_l)`` triple is the
canonical "PC module" (one cortical level, Bastos 2012).  The chain
implemented here is the linear special case used to validate the
dynamics in isolation before the graph generalisation.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, split_key


# =====================================================================
# Activation
# =====================================================================


def _phi(act: str, x: Array) -> Array:
    if act == "tanh":
        return jnp.tanh(x)
    if act == "linear":
        return x
    if act == "relu":
        return jnp.maximum(x, 0.0)
    raise ValueError(f"unknown activation {act!r}")


def _phi_prime(act: str, x: Array) -> Array:
    if act == "tanh":
        t = jnp.tanh(x)
        return 1.0 - t * t
    if act == "linear":
        return jnp.ones_like(x)
    if act == "relu":
        return (x > 0.0).astype(x.dtype)
    raise ValueError(f"unknown activation {act!r}")


# =====================================================================
# Params / state
# =====================================================================


class PCNetParams(eqx.Module):
    """Static hyper-params for a hierarchical PC chain.

    ``sizes`` is ``(n_0, …, n_L)`` from bottom (output) to top (cause);
    there are ``L = len(sizes) − 1`` generative weight matrices, with
    ``W_l`` of shape ``(sizes[l], sizes[l+1])``.
    """

    eta_mu: Array          # inference (relaxation) step size
    eta_w: Array           # weight learning rate
    leak: Array            # prior leak on free nodes (Rao-Ballard prior)
    pi_alpha: Array        # EMA rate for pe_var → precision
    var_floor: Array       # precision floor (Friston 2010)

    sizes: tuple = eqx.field(static=True)
    act: str = eqx.field(static=True)
    n_relax: int = eqx.field(static=True)

    @property
    def n_layers(self) -> int:
        """Number of generative weight matrices ``L``."""
        return len(self.sizes) - 1


def init_pc_net_params(
    sizes: tuple[int, ...],
    *,
    act: str = "tanh",
    eta_mu: float = 0.1,
    eta_w: float = 1e-2,
    leak: float = 0.0,
    tau_pi_steps: float = 1000.0,
    var_floor: float = 1e-4,
    n_relax: int = 20,
) -> PCNetParams:
    """Build params for a chain with the given layer ``sizes`` (bottom→top)."""
    if len(sizes) < 2:
        raise ValueError("need at least 2 layers (one weight matrix)")
    f = lambda x: jnp.asarray(x, DTYPE)
    pi_alpha = 1.0 - jnp.exp(-1.0 / jnp.asarray(tau_pi_steps, DTYPE))
    return PCNetParams(
        eta_mu=f(eta_mu), eta_w=f(eta_w), leak=f(leak),
        pi_alpha=f(pi_alpha), var_floor=f(var_floor),
        sizes=tuple(int(s) for s in sizes), act=act, n_relax=int(n_relax),
    )


class PCNetState(eqx.Module):
    """Dynamic state: value-node beliefs μ, generative weights, precision.

    ``mu`` are the persistent beliefs (one array per layer, length
    ``L+1``); they are relaxed every inference call and carried across
    cycles.  ``weights`` / ``pi`` / ``pe_var`` are the learnable
    parameters (length ``L`` each, one per generative matrix).
    """

    mu: tuple              # (L+1) arrays, mu[l] shape (sizes[l],)
    weights: tuple         # (L) arrays, W[l] shape (sizes[l], sizes[l+1])
    pi: tuple              # (L) arrays, pi[l] shape (sizes[l],)
    pe_var: tuple          # (L) arrays, pe_var[l] shape (sizes[l],)


def init_pc_net_state(
    key: PRNGKey, params: PCNetParams, *, dtype=DTYPE,
) -> PCNetState:
    """He/LeCun-scaled generative weights; μ = 0; Π = 1 (uniform prior)."""
    sizes = params.sizes
    L = params.n_layers
    keys = split_key(key, L)
    weights = []
    for l in range(L):
        n_out, n_in = sizes[l], sizes[l + 1]
        scale = 1.0 / jnp.sqrt(jnp.asarray(n_in, dtype))
        weights.append(
            jax.random.normal(keys[l], (n_out, n_in), dtype) * scale
        )
    mu = tuple(jnp.zeros(s, dtype) for s in sizes)
    pi = tuple(jnp.ones(sizes[l], dtype) for l in range(L))
    pe_var = tuple(jnp.ones(sizes[l], dtype) for l in range(L))
    return PCNetState(mu=mu, weights=tuple(weights), pi=pi, pe_var=pe_var)


# =====================================================================
# Forward / errors / free energy
# =====================================================================


def pc_predictions(state: PCNetState, params: PCNetParams) -> tuple:
    """Top-down predictions ``pred_l = W_l φ(μ_{l+1})`` for l = 0…L−1."""
    L = params.n_layers
    return tuple(
        state.weights[l] @ _phi(params.act, state.mu[l + 1])
        for l in range(L)
    )


def pc_errors(state: PCNetState, params: PCNetParams) -> tuple:
    """Prediction errors ``ε_l = μ_l − pred_l`` for l = 0…L−1."""
    preds = pc_predictions(state, params)
    return tuple(state.mu[l] - preds[l] for l in range(params.n_layers))


def pc_free_energy(state: PCNetState, params: PCNetParams) -> Array:
    """Scalar free energy ``F = ½ Σ_l Π_l · ε_l²`` (the global objective).

    This is the quantity inference and learning both minimise — the
    consumer U.3 will sum it across graph nodes.  Exposed here so U.1
    can verify relaxation actually descends it.
    """
    eps = pc_errors(state, params)
    total = jnp.asarray(0.0, DTYPE)
    for l in range(params.n_layers):
        total = total + 0.5 * jnp.sum(state.pi[l] * eps[l] ** 2)
    return total


def pc_feedforward(
    state: PCNetState, params: PCNetParams, top_input: Array,
) -> PCNetState:
    """Amortised init: set μ_L = input, μ_l = W_l φ(μ_{l+1}) downward.

    After this every ε_l = 0 (each layer equals its own prediction), so
    relaxation starts from the feedforward pass — the regime in which
    PC tracks backprop tightly (Whittington & Bogacz 2017; Song 2020).
    """
    L = params.n_layers
    mu = [None] * (L + 1)
    mu[L] = top_input.astype(DTYPE)
    for l in range(L - 1, -1, -1):
        mu[l] = state.weights[l] @ _phi(params.act, mu[l + 1])
    return eqx.tree_at(lambda s: s.mu, state, tuple(mu))


# =====================================================================
# Relaxation (inference)
# =====================================================================


def _relax_step(
    mu: tuple, weights: tuple, pi: tuple, params: PCNetParams,
    clamp_bottom: bool,
) -> tuple:
    """One inference step: μ_l ← μ_l − η_μ ∂F/∂μ_l on the free nodes."""
    L = params.n_layers
    act = params.act
    preds = [weights[l] @ _phi(act, mu[l + 1]) for l in range(L)]
    eps = [mu[l] - preds[l] for l in range(L)]
    xi = [pi[l] * eps[l] for l in range(L)]

    new_mu = list(mu)
    for l in range(L + 1):
        if l == L:
            continue                      # top cause: always clamped
        if l == 0 and clamp_bottom:
            continue                      # bottom: clamped to target
        g = jnp.zeros_like(mu[l])
        if l <= L - 1:                    # this node is the value in ε_l
            g = g + xi[l]
        if l >= 1:                        # this node is the source in ε_{l−1}
            g = g - _phi_prime(act, mu[l]) * (weights[l - 1].T @ xi[l - 1])
        g = g + params.leak * mu[l]
        new_mu[l] = mu[l] - params.eta_mu * g
    return tuple(new_mu)


def pc_relax(
    state: PCNetState, params: PCNetParams,
    *,
    clamp_bottom: bool = False,
    n_steps: int | None = None,
) -> PCNetState:
    """Relax free value nodes for ``n_steps`` (default ``params.n_relax``).

    ``clamp_bottom`` fixes μ₀ (supervised learning, where μ₀ = target);
    when ``False`` μ₀ is also inferred (pure perception).  μ_L (the top
    cause / input) is always clamped.
    """
    steps = params.n_relax if n_steps is None else int(n_steps)
    weights, pi = state.weights, state.pi

    def body(_, mu):
        return _relax_step(mu, weights, pi, params, clamp_bottom)

    mu = jax.lax.fori_loop(0, steps, body, state.mu)
    return eqx.tree_at(lambda s: s.mu, state, mu)


# =====================================================================
# Learning
# =====================================================================


def pc_weight_grads(state: PCNetState, params: PCNetParams) -> tuple:
    """Free-energy weight gradients ``∂F/∂W_l = −ξ_l ⊗ φ(μ_{l+1})``.

    Exposed separately from :func:`pc_learn` so the equivalence test
    (U.1b) can compare these directly against the backprop gradient.
    """
    L = params.n_layers
    act = params.act
    preds = pc_predictions(state, params)
    grads = []
    for l in range(L):
        eps_l = state.mu[l] - preds[l]
        xi_l = state.pi[l] * eps_l
        grads.append(-jnp.outer(xi_l, _phi(act, state.mu[l + 1])))
    return tuple(grads)


def pc_learn(
    state: PCNetState, params: PCNetParams,
    *,
    update_precision: bool = True,
) -> PCNetState:
    """One Hebbian weight step + precision update, given the relaxed μ.

    ``ΔW_l = η_w · ξ_l ⊗ φ(μ_{l+1})`` (= −η_w ∂F/∂W_l), purely local.
    Precision tracks the EMA of ε² and is the inverse-variance weight
    (Friston 2010); set ``update_precision=False`` to keep Π fixed (used
    by the backprop-equivalence test, which holds Π = 1).
    """
    L = params.n_layers
    act = params.act
    preds = pc_predictions(state, params)

    new_weights = list(state.weights)
    new_pi = list(state.pi)
    new_pe_var = list(state.pe_var)
    a = params.pi_alpha
    for l in range(L):
        eps_l = state.mu[l] - preds[l]
        xi_l = state.pi[l] * eps_l
        new_weights[l] = state.weights[l] + params.eta_w * jnp.outer(
            xi_l, _phi(act, state.mu[l + 1]),
        )
        if update_precision:
            pe_var_l = (1.0 - a) * state.pe_var[l] + a * eps_l ** 2
            new_pe_var[l] = pe_var_l
            new_pi[l] = 1.0 / (pe_var_l + params.var_floor)

    return PCNetState(
        mu=state.mu,
        weights=tuple(new_weights),
        pi=tuple(new_pi),
        pe_var=tuple(new_pe_var),
    )


def pc_clamp_bottom(state: PCNetState, target: Array) -> PCNetState:
    """Set μ₀ (bottom / output node) to ``target`` (supervised clamp)."""
    mu = (target.astype(DTYPE),) + tuple(state.mu[1:])
    return eqx.tree_at(lambda s: s.mu, state, mu)


# =====================================================================
# Fixed-prediction equilibrium — exact backprop correspondence
# =====================================================================
#
# Standard relaxed PC (``pc_relax`` + ``pc_weight_grads``) only
# *approximates* backprop (cosine ≈ 0.9): it uses the relaxed μ both as
# the value AND as the presynaptic / φ′ factor.  The exact equivalence
# (Whittington & Bogacz 2017; Millidge, Tschantz & Bogacz 2020
# "Predictive Coding Approximates Backprop Along Arbitrary Computation
# Graphs"; Song et al. 2020 Z-IL) holds under the *fixed-prediction*
# assumption: predictions and presynaptic activities are held at their
# feedforward values while the error/value nodes relax.  At equilibrium
# the precision-weighted errors equal the backprop deltas and the local
# Hebbian weight grad equals the backprop weight gradient EXACTLY.
#
# This is a genuine PC inference mode (not a hand-coded backprop): the
# weight gradients fall out of relaxing the value nodes under fixed
# predictions.  ``tests/test_phaseu_pc_module`` gates Faza U on it
# reproducing ``jax.grad`` to cosine > 0.99.


def pc_fixed_prediction_grads(
    state: PCNetState, params: PCNetParams,
    top_input: Array, target: Array,
    *,
    n_steps: int | None = None,
) -> tuple:
    """Weight grads from a fixed-prediction relaxation — equals backprop.

    Runs the value-node relaxation with predictions / presynaptic
    activities pinned to the feedforward pass (Millidge & Bogacz 2020),
    then returns ``∂F/∂W_l`` evaluated at the equilibrium.  At the
    equilibrium these equal ``∂Loss/∂W_l`` of the equivalent
    feedforward network (loss = ½‖output − target‖²) — the formal proof
    that this module's local rule performs backprop-grade credit
    assignment without backprop.
    """
    L = params.n_layers
    act = params.act
    W = state.weights
    steps = params.n_relax if n_steps is None else int(n_steps)

    # Feedforward pass → held predictions.
    a = [None] * (L + 1)
    a[L] = top_input.astype(DTYPE)
    for l in range(L - 1, -1, -1):
        a[l] = W[l] @ _phi(act, a[l + 1])
    phi_a = [_phi(act, a[l]) for l in range(L + 1)]      # fixed presynaptic
    phip_a = [_phi_prime(act, a[l]) for l in range(L + 1)]

    # Value nodes: init to feedforward, clamp output to target + top to input.
    v0 = [a[l] for l in range(L + 1)]
    v0[0] = target.astype(DTYPE)
    v0 = tuple(v0)

    def body(_, v):
        eps = [v[l] - W[l] @ phi_a[l + 1] for l in range(L)]   # fixed pred
        xi = eps                                               # Π = 1
        nv = list(v)
        for l in range(1, L):                                  # internal only
            g = xi[l] - phip_a[l] * (W[l - 1].T @ xi[l - 1])
            nv[l] = v[l] - params.eta_mu * g
        return tuple(nv)

    v = jax.lax.fori_loop(0, steps, body, v0)

    eps = [v[l] - W[l] @ phi_a[l + 1] for l in range(L)]
    xi = eps
    return tuple(-jnp.outer(xi[l], phi_a[l + 1]) for l in range(L))


# =====================================================================
# Convenience: one supervised step (feedforward → clamp → relax → learn)
# =====================================================================


class PCStepOutput(NamedTuple):
    state: PCNetState
    free_energy: Array
    output: Array          # μ₀ before clamping = feedforward prediction


def pc_supervised_step(
    state: PCNetState, params: PCNetParams,
    top_input: Array, target: Array,
) -> PCStepOutput:
    """Full supervised cycle on one (input, target) pair.

    1. feedforward init (μ from input),
    2. read the prediction (μ₀),
    3. clamp μ₀ = target, relax internal nodes,
    4. local Hebbian weight + precision update.
    """
    ff = pc_feedforward(state, params, top_input)
    output = ff.mu[0]
    clamped = pc_clamp_bottom(ff, target)
    relaxed = pc_relax(clamped, params, clamp_bottom=True)
    learned = pc_learn(relaxed, params)
    return PCStepOutput(
        state=learned, free_energy=pc_free_energy(relaxed, params),
        output=output,
    )
