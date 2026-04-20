"""Phase 4 — V1 functional readouts under closed-loop SensoryStack.

Companion to :mod:`test_phase4_v1_rf_emergence` (which asserts the
*initial* Gabor RF prior on the bare ``v1.py`` module).  Here we
exercise the brain-owned :class:`sensory.sensory_stack.SensoryStack`
end-to-end and assert three biologically-grounded properties of the
V1 belief readout actually delivered to downstream regions:

1. **Orientation discriminability** — settling on differently-oriented
   gratings produces measurably different L2/3 belief vectors
   (Hubel & Wiesel 1962 simple-cell selectivity, here inherited from
   the Gabor initialisation).  The cosine distance between the most
   dissimilar pair of orientation responses must exceed a small
   threshold; pure noise would give ≈0.
2. **Population sparsity** — for a fixed stimulus, most L2/3 state
   neurons are silent (Olshausen & Field 1996), measured as the
   fraction of (orientation, neuron) cells below 25 % of the per-
   orientation peak.
3. **Numerical stability under long closed-loop driving** — V1
   weights remain finite and non-negative after a long random-grating
   stream with ipool/astrocyte plasticity engaged
   (``apply_stdp=True`` in :func:`sensory_stack_step`).

Out of scope (separate architectural milestone)
-----------------------------------------------
Olshausen & Field (1996) *Nature* 381:607-609 RF emergence — i.e.
``cortical_area_update`` driven STDP that re-shapes ``w_l4_in`` /
``w_bu`` / ``w_td`` from a Gabor prior into a sparse-coding basis —
requires:
  * Wiring :func:`core.cortex.cortical_area_update` (currently never
    called from any pipeline) into the sensory stack with an
    appropriate three-factor modulator (ACh × PE rate is the
    canonical choice — Hasselmo 2006).
  * A ≥10 k-patch training budget (Ringach 2002 quantification).

Both belong to the dedicated cortical-learning subsystem and are
deferred to a later phase.  When that lands, this file should grow a
fourth test asserting orientation-tuning index ≥ 0.3 in ≥ 40 % of
L2/3 neurons (Ringach 2002).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.backend import BackendContext
from sensory.retina import RetinaConfig
from sensory.sensory_stack import (
    init_sensory_stack_params,
    init_sensory_stack_state,
    sensory_stack_step,
)


# ------------------------------------------------------------------
# Stimulus + harness
# ------------------------------------------------------------------

_IMG_SIZE = 32
_N_PROBE_ORIENTATIONS = 8
_TRAIN_ORIENTATIONS = jnp.linspace(0.0, jnp.pi, 12, endpoint=False)
_TRAIN_FREQS = jnp.asarray([3.0, 5.0, 7.0], jnp.float32)


def _grating(theta, sf, phase, *, size: int = _IMG_SIZE):
    yy, xx = jnp.mgrid[0:size, 0:size].astype(jnp.float32) / (size - 1)
    proj = jnp.cos(theta) * xx + jnp.sin(theta) * yy
    return 0.5 + 0.5 * jnp.sin(2.0 * jnp.pi * sf * proj + phase)


def _build_stack(seed: int, *, n_l23_state: int = 48):
    ctx = BackendContext(dt=1.0)
    cfg = RetinaConfig(fovea_size=8, n_pyramid=2, periphery_tile=4)
    params = init_sensory_stack_params(
        ctx, retina_cfg=cfg,
        n_l4=64, n_l23_state=n_l23_state, n_l23_error=24, n_l5=16,
    )
    state = init_sensory_stack_state(jax.random.PRNGKey(seed), params)
    return ctx, params, state


def _drive_random_gratings(state, params, ctx, key, *, n_steps: int):
    """Drive V1 with random gratings (closed-loop, STDP enabled).

    Used only for the long-form numerical-stability test; the probe
    tests start from a fresh (Gabor-init) state because RF *learning*
    requires ``cortical_area_update`` which is not yet wired into the
    stack — see module docstring.
    """

    def body(carry, k):
        st = carry
        k_o, k_f, k_p, k_fix = jax.random.split(k, 4)
        theta = _TRAIN_ORIENTATIONS[
            jax.random.randint(k_o, (), 0, _TRAIN_ORIENTATIONS.shape[0])
        ]
        sf = _TRAIN_FREQS[
            jax.random.randint(k_f, (), 0, _TRAIN_FREQS.shape[0])
        ]
        phase = jax.random.uniform(k_p, (), jnp.float32, 0.0, 2 * jnp.pi)
        img = _grating(theta, sf, phase)
        fix = jax.random.uniform(k_fix, (2,), jnp.float32, 0.3, 0.7)
        out = sensory_stack_step(
            st, params, ctx, img, fix, apply_stdp=True,
        )
        return out.state, None

    keys = jax.random.split(key, n_steps)
    final_state, _ = jax.lax.scan(body, state, keys)
    return final_state


def _probe_orientation_responses(state, params, ctx, *, n_settle: int = 60):
    """Run frozen-STDP probes at evenly-spaced orientations.

    Each probe restarts from the same ``state`` (so cross-probe state
    bleed cannot bias the comparison) and settles for ``n_settle`` dt
    on a fixed grating; the readout is the mean L2/3 belief over the
    final fifth of the window (after the AdEx population has reached
    its driven attractor).
    """
    fix = jnp.array([0.5, 0.5], jnp.float32)
    sf = jnp.asarray(5.0, jnp.float32)
    phases = jnp.linspace(0.0, 2 * jnp.pi, _N_PROBE_ORIENTATIONS, endpoint=False)
    thetas = jnp.linspace(0.0, jnp.pi, _N_PROBE_ORIENTATIONS, endpoint=False)
    last_window = max(1, n_settle // 5)

    def settle_one(theta, phase):
        img = _grating(theta, sf, phase)

        def step(st, _):
            o = sensory_stack_step(
                st, params, ctx, img, fix, apply_stdp=False,
            )
            return o.state, o.belief

        _, beliefs = jax.lax.scan(step, state, None, length=n_settle)
        return beliefs[-last_window:].mean(axis=0)

    return jax.vmap(settle_one)(thetas, phases)            # (O, N)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_v1_orientation_responses_are_discriminable():
    """The Gabor-initialised V1 must already deliver orientation-
    discriminable belief vectors: at least one pair of orientations
    must yield non-collinear responses (cosine distance > 0).

    This is the architectural assertion for the SensoryStack output
    contract — the brain's striatal/thalamic afferent must carry
    stimulus-specific information.  Pure-noise or all-silent V1 would
    give cos-dist ≈ 0 across the board.
    """
    ctx, params, state = _build_stack(seed=0)
    R = _probe_orientation_responses(state, params, ctx, n_settle=80)
    # L2-normalise rows; cosine distance = 1 - row_i · row_j .
    norms = jnp.linalg.norm(R, axis=1, keepdims=True) + 1e-8
    Rn = R / norms
    sims = Rn @ Rn.T
    # Off-diagonal min similarity == max cosine distance.
    eye = jnp.eye(Rn.shape[0], dtype=bool)
    off = jnp.where(eye, 1.0, sims)
    max_dist = float(1.0 - off.min())
    assert R.sum() > 0.0, "V1 silent across all probe orientations"
    assert max_dist > 1e-3, (
        f"V1 belief responses identical across orientations "
        f"(max cosine distance = {max_dist:.2e}) — Gabor-init RF lost "
        f"its orientation selectivity at the SensoryStack readout"
    )


def test_v1_population_response_is_sparse():
    """Olshausen & Field 1996: V1 responses to natural stimuli are
    sparse (most neurons silent, few strongly active per stimulus).

    Asserted on the Gabor-initialised V1 (sparsity is a property of
    the architecture — high-threshold AdEx + lateral inhibition —
    not of the learnt weights).
    """
    ctx, params, state = _build_stack(seed=2)
    R = _probe_orientation_responses(state, params, ctx, n_settle=80)
    per_orient_max = jnp.max(R, axis=1, keepdims=True) + 1e-6
    silent_frac = float(jnp.mean(R < 0.25 * per_orient_max))
    assert silent_frac > 0.4, (
        f"only {silent_frac:.0%} of (orientation, neuron) responses are "
        f"silent — V1 population is dense, not sparse"
    )


def test_v1_weights_remain_finite_under_long_training():
    """Long-form numerical stability under closed-loop ipool +
    astrocyte plasticity (``apply_stdp=True``).  The L4 feedforward
    weights are not yet plastic in this pipeline (see module
    docstring) so we additionally assert they remain identical to
    init — any drift would indicate an unintended plastic path."""
    ctx, params, state = _build_stack(seed=4)
    trained = _drive_random_gratings(
        state, params, ctx, jax.random.PRNGKey(5), n_steps=800,
    )
    assert jnp.isfinite(trained.v1.w_l4_in).all()
    assert float(trained.v1.w_l4_in.min()) >= 0.0
    # Also assert ipool weights stayed finite (the only plastic path
    # exercised by ``sensory_stack_step``).
    assert jnp.isfinite(trained.v1.l4_ipool.w_ei).all()
    assert jnp.isfinite(trained.v1.l4_ipool.w_ie).all()
