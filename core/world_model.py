"""Active-inference world model — pure JAX.

Friston et al. (2015) Active inference and epistemic value;
Pouget, Dayan & Zemel (2000) Probabilistic population coding;
Markram et al. (1997) Vesicle release probability;
De Pittà et al. (2011) Astrocyte Ca²⁺ precision.

Composition:
- Encoder = ``error_neuron_jax`` (L4 error + L2/3 state populations)
  — takes ``[pop_encoded_state | one-hot action]`` as input and
  provides a belief ``μ`` (state-neuron rate) plus a prediction-error
  rate signal.
- Astrocyte = ``astrocyte_jax`` (zone-level Ca²⁺ / precision).
  Slow channel only; fast epistemic drive stays on the error rate.
- Decoder = single Hebbian readout ``w_decode (n_state, state_dim)``
  with Bernoulli(``vesicle_p``) masking producing ambiguity via
  variance across masked samples.

Rehearsal: ``wm_mental_rehearsal(state, action_id, depth, key)`` runs
``depth`` encoder steps, folding the decoded prediction back as the
next state input. Returns ``(predicted_state, novelty, ambiguity)``.

Differences from legacy:
- Snapshot / restore disappears — pytree state is immutable, so
  callers branch on a copy of the world-model state when imagining
  counterfactual rollouts (``jax.lax.scan`` keeps this free).
- ACh / 5-HT modulation exposed through explicit params rather than
  attribute mutation.
- ``curiosity_signal`` returns a JAX scalar so it can drive JIT-able
  neuromodulator updates downstream.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import DTYPE, Array, PRNGKey, BackendContext, split_key
from .error_neuron import (
    ErrorNeuronParams, ErrorNeuronState,
    init_error_neuron_params, init_error_neuron_state,
    en_step, en_update_weights, en_belief, en_prediction_error_rate,
    en_reset_transient,
)
from .astrocyte import (
    init_astrocyte_params, astrocyte_step, AstrocyteParams,
    aggregate_to_zones, precision as astrocyte_precision,
)
from .state import AstrocyteState, init_astrocyte_state
from .spike_encoder import (
    PopulationEncoderParams, init_population_encoder,
    gaussian_population_encode,
)


# =====================================================================
# Params / state
# =====================================================================


class WorldModelParams(eqx.Module):
    """Static params composing encoder + astrocyte + decoder.

    ``pe_short_decay`` / ``pe_long_decay`` are the per-step EMA decays
    for the short- and long-timescale absolute prediction-error
    traces used to compute learning-progress curiosity (Oudeyer 2007
    IAC; Schmidhuber 1991). Defaults are 100 ms (short) and 10 s
    (long); long/short ratio > 10 ensures the difference is dominated
    by trend, not by individual-trial noise.
    """

    encoder: ErrorNeuronParams
    astro: AstrocyteParams
    pop_enc: PopulationEncoderParams          # gaussian population encoder
    decode_lr: Array
    vesicle_p: Array
    zone_idx: Array                           # (n_error,) int32, error neuron \u2192 zone
    pe_short_decay: Array                     # EMA decay for short-τ |PE|
    pe_long_decay: Array                      # EMA decay for long-τ |PE|
    n_zones: int = eqx.field(static=True)
    max_rehearsal_depth: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    action_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    input_size: int = eqx.field(static=True)   # pop_enc.output_size + action_size


def init_world_model_params(
    ctx: BackendContext,
    state_size: int,
    action_size: int,
    *,
    hidden_size: int = 64,
    n_error: int = 64,
    n_neurons_per_dim: int = 8,
    # Unified LR for encoder + decoder (Phase 0.5). Both pathways run
    # at the same plasticity timescale — differential LRs were
    # empirical with no biophysical basis, and modern STDP data
    # (Bi & Poo 1998; Sjöström 2001) show cortical excitatory
    # synapses share a τ_plasticity ≈ 60 s early-LTP time course
    # regardless of afferent type.
    wm_lr: float = 1e-3,
    vesicle_p: float = 0.8,
    max_rehearsal_depth: int = 5,
    astro_n_zones: int | None = None,
    state_min: float = -1.0,
    state_max: float = 1.0,
    # Learning-progress timescales (Oudeyer 2007; Schmidhuber 1991).
    tau_pe_short_ms: float = 100.0,
    tau_pe_long_ms: float = 10_000.0,
) -> WorldModelParams:
    """Build world-model params for ``state_dim`` continuous states + discrete actions."""
    pop_enc = init_population_encoder(
        n_dims=state_size,
        n_neurons_per_dim=n_neurons_per_dim,
        value_min=state_min, value_max=state_max,
    )
    pop_output_size = state_size * n_neurons_per_dim
    input_size = pop_output_size + action_size
    encoder = init_error_neuron_params(
        ctx, input_size,
        n_state=hidden_size, n_error=min(input_size, n_error),
        w_bu_lr=wm_lr, w_td_lr=wm_lr,
    )
    n_zones = astro_n_zones if astro_n_zones is not None else max(4, hidden_size // 16)
    astro = init_astrocyte_params(ctx, ca_accumulation=0.15)
    n_error = encoder.n_error
    zone_idx = (jnp.arange(n_error) * n_zones // n_error).astype(jnp.int32)
    f = lambda x: jnp.asarray(x, DTYPE)
    return WorldModelParams(
        encoder=encoder, astro=astro, pop_enc=pop_enc,
        decode_lr=f(wm_lr), vesicle_p=f(vesicle_p),
        zone_idx=zone_idx, n_zones=n_zones,
        pe_short_decay=f(ctx.decay(tau_pe_short_ms)),
        pe_long_decay=f(ctx.decay(tau_pe_long_ms)),
        max_rehearsal_depth=max_rehearsal_depth,
        state_size=state_size, action_size=action_size,
        hidden_size=hidden_size, input_size=input_size,
    )


class WorldModelState(eqx.Module):
    """Dynamic state for the world model.

    ``pe_short_abs`` / ``pe_long_abs`` are running EMAs of
    ``mean(|prediction_error|)``, with short and long timescales from
    ``WorldModelParams.pe_short_decay/pe_long_decay``. Their
    difference implements learning-progress curiosity
    (Oudeyer 2007 IAC): if PE is decreasing, the model is *still
    learning* this region, so ``curiosity = relu(pe_long − pe_short) > 0``.
    If PE is *increasing* (task drift / novelty), the long trace lags
    below the short one, so ``boredom = relu(pe_short − pe_long) > 0``
    drives arousal instead. Magnitude is self-calibrating: no magic
    normalisation constant.
    """

    encoder: ErrorNeuronState
    astro: AstrocyteState
    w_decode: Array          # (hidden_size, state_size)
    last_prediction: Array   # (state_size,)
    prediction_error: Array  # (state_size,)
    pe_short_abs: Array      # scalar, short-τ EMA of mean(|PE|)
    pe_long_abs: Array       # scalar, long-τ EMA of mean(|PE|)


def init_world_model_state(
    key: PRNGKey, params: WorldModelParams,
    *,
    decode_init_std: float = 0.01,
    dtype=DTYPE,
) -> WorldModelState:
    k_enc, k_dec = split_key(key, 2)
    encoder = init_error_neuron_state(k_enc, params.encoder)
    astro = init_astrocyte_state(
        params.n_zones, dtype=dtype,
    )
    w_decode = (jax.random.normal(
        k_dec, (params.hidden_size, params.state_size), dtype=dtype,
    ) * decode_init_std).astype(dtype)
    z = jnp.asarray(0.0, dtype)
    return WorldModelState(
        encoder=encoder, astro=astro, w_decode=w_decode,
        last_prediction=jnp.zeros(params.state_size, dtype),
        prediction_error=jnp.zeros(params.state_size, dtype),
        pe_short_abs=z, pe_long_abs=z,
    )


# =====================================================================
# Internal helpers
# =====================================================================


def _build_input(
    params: WorldModelParams,
    state_vec: Array,
    action_onehot: Array,
) -> Array:
    """Pop-encode continuous state and concat with one-hot action."""
    encoded = gaussian_population_encode(params.pop_enc, state_vec)
    return jnp.concatenate([encoded, action_onehot])


def _decode(
    w_decode: Array, belief: Array,
    vesicle_p: Array, key: PRNGKey, *, noise: bool,
) -> Array:
    """Decode ``belief → predicted_state`` with optional Bernoulli masking."""
    if noise:
        mask = (jax.random.uniform(key, belief.shape, dtype=belief.dtype) < vesicle_p).astype(
            belief.dtype,
        )
        belief = belief * mask
    return belief @ w_decode


def _ambiguity(
    w_decode: Array, belief: Array, vesicle_p: Array,
    key: PRNGKey, n_samples: int = 5,
) -> tuple[Array, Array]:
    """Mean prediction and per-dim variance across ``n_samples`` vesicle masks."""
    keys = jax.random.split(key, n_samples)
    preds = jax.vmap(
        lambda k: _decode(w_decode, belief, vesicle_p, k, noise=True),
    )(keys)
    mean_pred = jnp.mean(preds, axis=0)
    ambiguity = jnp.mean(jnp.var(preds, axis=0))
    return mean_pred, ambiguity


# =====================================================================
# Online inference + learning
# =====================================================================


class WorldModelOutput(NamedTuple):
    state: WorldModelState
    predicted_state: Array
    belief: Array
    prediction_error: Array        # (state_size,) actual − predicted


@eqx.filter_jit
def wm_predict(
    state: WorldModelState, params: WorldModelParams, ctx: BackendContext,
    state_spikes: Array, action_onehot: Array,
    *, ach: float | Array = 0.5, receptor_gain: float | Array = 1.0,
) -> WorldModelOutput:
    """Integrate encoder for one dt and decode a noise-free prediction."""
    inp = _build_input(params, state_spikes, action_onehot)
    enc_out = en_step(
        state.encoder, params.encoder, ctx, inp,
        ach=ach, receptor_gain=receptor_gain,
    )
    belief = en_belief(enc_out.state)
    pred = _decode(
        state.w_decode, belief, params.vesicle_p,
        jax.random.PRNGKey(0), noise=False,
    )
    new_state = eqx.tree_at(
        lambda s: (s.encoder, s.last_prediction),
        state, (enc_out.state, pred),
    )
    return WorldModelOutput(
        state=new_state, predicted_state=pred, belief=belief,
        prediction_error=state.prediction_error,  # previous, unchanged
    )


@eqx.filter_jit
def wm_update(
    state: WorldModelState, params: WorldModelParams, ctx: BackendContext,
    state_spikes: Array, action_onehot: Array, actual_next: Array,
    *,
    m_t: float | Array = 1.0,
    ach: float | Array = 0.5,
    receptor_gain: float | Array = 1.0,
    receptor_lr: float | Array = 1.0,
) -> WorldModelOutput:
    """Observe the real transition and update encoder + decoder + astrocyte."""
    inp = _build_input(params, state_spikes, action_onehot)
    enc_out = en_step(
        state.encoder, params.encoder, ctx, inp,
        ach=ach, receptor_gain=receptor_gain,
    )
    belief = en_belief(enc_out.state)
    pred = _decode(
        state.w_decode, belief, params.vesicle_p,
        jax.random.PRNGKey(0), noise=False,
    )
    actual = actual_next.astype(DTYPE)
    pe = actual - pred

    # Learning-progress traces: EMA of mean(|PE|) at two timescales.
    # Oudeyer (2007) IAC; Schmidhuber (1991) formal curiosity. A
    # simple difference ``pe_long − pe_short`` is positive iff the
    # model is improving (PE is decreasing) and negative iff the
    # environment has changed (PE is rising), giving both curiosity
    # and boredom signals from one pair of EMAs.
    pe_abs = jnp.mean(jnp.abs(pe))
    pe_short_abs = (
        params.pe_short_decay * state.pe_short_abs
        + (1.0 - params.pe_short_decay) * pe_abs
    )
    pe_long_abs = (
        params.pe_long_decay * state.pe_long_abs
        + (1.0 - params.pe_long_decay) * pe_abs
    )

    # Astrocyte tracks encoder spike activity (error rate as rate proxy)
    zone_rates = aggregate_to_zones(
        enc_out.state.error_rate, params.zone_idx, params.n_zones,
    )
    astro = astrocyte_step(
        state.astro, params.astro, ctx, zone_rates,
    )
    prec = astrocyte_precision(astro)

    # Encoder three-factor STDP \u00d7 m_t \u00d7 astrocyte precision
    enc_state = en_update_weights(
        enc_out.state, params.encoder, modulation=m_t,
        precision=prec,
        receptor_lr=receptor_lr,
    )

    # Decoder Hebbian update (gated + soft clipped)
    max_belief = jnp.max(jnp.abs(belief))
    scale = jnp.where(max_belief > 0.01, 1.0 / (max_belief + 1e-6), 0.0)
    belief_norm = belief * scale
    dw = params.decode_lr * (belief_norm[:, None] * pe[None, :])
    dw = jnp.clip(dw, -0.1, 0.1) * jnp.asarray(m_t, DTYPE)
    w_decode = jnp.clip(state.w_decode + dw, -1.0, 1.0)

    new_state = WorldModelState(
        encoder=enc_state, astro=astro, w_decode=w_decode,
        last_prediction=pred, prediction_error=pe,
        pe_short_abs=pe_short_abs, pe_long_abs=pe_long_abs,
    )
    return WorldModelOutput(
        state=new_state, predicted_state=pred, belief=belief,
        prediction_error=pe,
    )


# =====================================================================
# Mental rehearsal (active inference)
# =====================================================================


class RehearsalResult(NamedTuple):
    predicted_state: Array    # (state_size,)
    novelty: Array            # scalar [0, 2]
    ambiguity: Array          # scalar
    epistemic_value: Array    # scalar (raw PE + ambiguity accumulation)


def wm_mental_rehearsal(
    state: WorldModelState, params: WorldModelParams, ctx: BackendContext,
    current_state_spikes: Array,
    action_onehot: Array,
    key: PRNGKey,
    *,
    depth: int | None = None,
    n_ambiguity_samples: int = 5,
    baseline_precision: float | Array = 0.5,
    ach: float | Array = 0.5,
) -> RehearsalResult:
    """Roll out ``depth`` encoder steps, accumulating epistemic value.

    For a single action, this is side-effect-free on the passed state
    (the simulated encoder state is a local carry). For multiple
    candidate actions use ``jax.vmap(wm_mental_rehearsal, in_axes=...)``.
    """
    D = depth if depth is not None else params.max_rehearsal_depth
    D = max(1, min(D, params.max_rehearsal_depth))

    enc0 = state.encoder

    def body(carry, step_idx):
        enc, cur_state, key = carry
        k_amb, k_next = jax.random.split(key)
        inp = _build_input(params, cur_state, action_onehot)
        out = en_step(enc, params.encoder, ctx, inp, ach=ach)
        belief = en_belief(out.state)
        mean_pred, amb = _ambiguity(
            state.w_decode, belief, params.vesicle_p, k_amb, n_ambiguity_samples,
        )
        enc_pe = jnp.mean(en_prediction_error_rate(out.state))
        step_eps = enc_pe + amb + (1.0 - jnp.asarray(baseline_precision, DTYPE))
        return (out.state, mean_pred, k_next), (step_eps, amb, mean_pred)

    # Start with zeros as "continuous state" since caller passes spike
    # vector — first body iteration will build input from current_state_spikes.
    (_, final_pred, _), (eps_hist, amb_hist, pred_hist) = jax.lax.scan(
        body,
        (enc0, current_state_spikes.astype(DTYPE), key),
        jnp.arange(D),
        length=D,
    )
    total_epistemic = jnp.sum(eps_hist)
    total_ambiguity = jnp.sum(amb_hist)
    # Novelty normalised by D (bounded monotone mapping).
    novelty = jnp.clip(total_epistemic / jnp.asarray(D, DTYPE), 0.0, 2.0)
    return RehearsalResult(
        predicted_state=final_pred, novelty=novelty,
        ambiguity=total_ambiguity, epistemic_value=total_epistemic,
    )


# =====================================================================
# Curiosity + misc
# =====================================================================


def wm_curiosity_signal(
    state: WorldModelState, params: WorldModelParams,
) -> Array:
    """Smoothed mean-|PE| intrinsic drive (Pathak 2017 ICM; Friston 2010).

    Returns ``pe_short_abs``: a short-timescale EMA of mean(|PE|) in
    the *natural* units of the prediction error — no magic
    ``x / (1 + x)`` normalisation, which previously introduced an
    arbitrary saturation point unrelated to the signal's physical
    scale. Temporal smoothing (τ ≈ 100 ms) takes out single-dt spikes
    so the drive represents "how hard the world is to predict *right
    now*", not one transient. This is the intrinsic-reward curiosity
    used by the body actor.

    For the *sign* of learning (is PE going up or down?) see
    :func:`wm_learning_progress` and :func:`wm_boredom_signal`.
    """
    return state.pe_short_abs


def wm_boredom_signal(
    state: WorldModelState, params: WorldModelParams,
) -> Array:
    """Task-drift alarm: ``relu(pe_short − pe_long)``.

    Positive iff the short-term PE exceeds the long-term baseline →
    the environment has changed (new task, new contingency, or the
    current model has become wrong). Drives NE (arousal) rather than
    ACh (curiosity), per Yu & Dayan (2005) uncertainty dissociation.
    """
    lp = state.pe_short_abs - state.pe_long_abs
    return jnp.clip(lp, 0.0, jnp.inf)


def wm_learning_progress(
    state: WorldModelState, params: WorldModelParams,
) -> Array:
    """Signed learning progress ``pe_long − pe_short`` (Oudeyer 2007).

    Positive iff the model's PE is *decreasing* (actively learning
    this region). Use as a modulatory gate on exploration priority:
    regions with positive LP are being mastered and should be
    re-visited; regions with LP ≈ 0 and high PE are noise (the
    classic "noisy-TV" failure mode of pure-PE curiosity); regions
    with negative LP have *drifted* and should trigger boredom /
    arousal via :func:`wm_boredom_signal`.
    """
    return state.pe_long_abs - state.pe_short_abs


def wm_rehearsal_depth_from_serotonin(
    params: WorldModelParams, serotonin: float | Array,
) -> int:
    """Static rule (Doya 2002): high 5-HT → more patience → deeper rollout."""
    sero = float(jnp.clip(jnp.asarray(serotonin, DTYPE), 0.0, 1.0))
    return max(1, int(1 + sero * (params.max_rehearsal_depth - 1)))


def wm_reset_transient(
    state: WorldModelState, params: WorldModelParams,
) -> WorldModelState:
    """Clear encoder + astrocyte + prediction transients; keep ``w_decode``.

    Preserves ``pe_short_abs`` and ``pe_long_abs``: learning-progress
    estimates are *cross-episode* by design (continual learning must
    know whether a region is mastered regardless of episode
    boundaries; resetting them would make every new episode look like
    initial task drift and drive spurious exploration).
    """
    enc = en_reset_transient(state.encoder, params.encoder)
    astro = init_astrocyte_state(params.n_zones)
    return WorldModelState(
        encoder=enc, astro=astro, w_decode=state.w_decode,
        last_prediction=jnp.zeros(params.state_size, DTYPE),
        prediction_error=jnp.zeros(params.state_size, DTYPE),
        pe_short_abs=state.pe_short_abs,
        pe_long_abs=state.pe_long_abs,
    )
