"""
Phase 3 tests — VTA Dopaminergic Circuit for Neural TD Error.

Tests verify that the VTA circuit correctly implements RPE computation
via biophysical E/I balance, replacing the algebraic TD error.

References:
    Eshel et al. (2015): Arithmetic and local circuitry underlying
        dopamine prediction errors.
    Schultz (1997, 1998): Predictive reward signal of DA neurons.
    Tobler et al. (2005): Adaptive coding of reward value.
    Grace (1991): Phasic vs tonic DA firing.
    Schweighofer et al. (2008): Serotonin modulates temporal discount.

Test structure:
    1. VTA Construction & Initialization
    2. VP Pathway — store_prediction() captures V(s)
    3. PPTg Pathway — temporal discount from τ_ppTg
    4. RPE Computation — E/I balance
    5. D2 Autoreceptor Gain Adaptation
    6. Serotonin Modulation of Temporal Discount
    7. Weight Update — three-factor Hebbian
    8. State Management — reset between episodes
    9. Integration with SNNAgent
   10. Numerical Stability
   11. Old API Removal Verification
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config import BasalGangliaConfig, VTAConfig
from core.basal_ganglia import SNNDeepCritic
from core.simulation_context import SimulationContext
from core.vta import VTACircuit


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def ctx() -> SimulationContext:
    return SimulationContext(dt=1.0)


@pytest.fixture
def vta_cfg(ctx: SimulationContext) -> VTAConfig:
    return VTAConfig(ctx=ctx)


@pytest.fixture
def bg_cfg(ctx: SimulationContext) -> BasalGangliaConfig:
    return BasalGangliaConfig(ctx=ctx, hidden_size=64)


@pytest.fixture
def critic(bg_cfg: BasalGangliaConfig) -> SNNDeepCritic:
    return SNNDeepCritic(state_size=10, config=bg_cfg)


@pytest.fixture
def vta(vta_cfg: VTAConfig) -> VTACircuit:
    return VTACircuit(critic_hidden_size=64, config=vta_cfg)


def _run_critic(critic: SNNDeepCritic, n_steps: int = 15,
                input_level: float = 0.5) -> None:
    """Run critic for n_steps with given input level."""
    state = np.full(critic._state_size, input_level, dtype=np.float32)
    for _ in range(n_steps):
        spikes = (np.random.random(critic._state_size) < state).astype(np.float32)
        critic.forward(spikes)


# =====================================================================
# 1. VTA Construction & Initialization
# =====================================================================

class TestVTAConstruction:
    """VTA circuit initializes with correct structure."""

    def test_w_value_shape(self, vta: VTACircuit) -> None:
        assert vta.w_value.shape == (64,)

    def test_w_value_dtype(self, vta: VTACircuit) -> None:
        assert vta.w_value.dtype == np.float32

    def test_w_value_nonzero(self, vta: VTACircuit) -> None:
        assert np.any(vta.w_value != 0.0), "w_value should be nonzero at init"

    def test_diagnostics_init(self, vta: VTACircuit) -> None:
        assert vta.last_rpe == 0.0
        assert vta.last_v_s == 0.0
        assert vta.last_v_s_prime == 0.0
        assert 0.9 < vta.last_gamma_eff < 1.0

    def test_eligibility_init_zero(self, vta: VTACircuit) -> None:
        assert np.all(vta.e_value == 0.0)


# =====================================================================
# 2. VP Pathway — store_prediction()
# =====================================================================

class TestVPPathway:
    """VP pathway captures V(s) from critic activation."""

    def test_store_prediction_captures_activation(
        self, vta: VTACircuit, critic: SNNDeepCritic,
    ) -> None:
        """After store_prediction, stored activation matches critic."""
        _run_critic(critic)
        vta.store_prediction(critic.activation)
        np.testing.assert_array_equal(
            vta._stored_activation, critic.activation,
        )

    def test_store_prediction_computes_v_s(
        self, vta: VTACircuit, critic: SNNDeepCritic,
    ) -> None:
        """stored V(s) = dot(activation, w_value)."""
        _run_critic(critic)
        vta.store_prediction(critic.activation)
        expected_v = float(np.dot(critic.activation, vta.w_value))
        assert abs(vta.last_v_s - expected_v) < 1e-5

    def test_store_prediction_updates_eligibility(
        self, vta: VTACircuit, critic: SNNDeepCritic,
    ) -> None:
        """Eligibility trace should be non-zero after store_prediction."""
        _run_critic(critic, n_steps=25, input_level=0.8)
        vta.store_prediction(critic.activation)
        assert np.any(vta.e_value != 0.0)

    def test_v_s_different_inputs_different_values(
        self, vta: VTACircuit, critic: SNNDeepCritic,
    ) -> None:
        """Different critic activations → different V(s) estimates."""
        values = []
        for level in [0.1, 0.5, 0.9]:
            critic.reset_state()
            _run_critic(critic, n_steps=20, input_level=level)
            vta.store_prediction(critic.activation)
            values.append(vta.last_v_s)
        assert np.std(values) > 1e-8, f"Same V(s) for all inputs: {values}"


# =====================================================================
# 3. PPTg Pathway — Temporal Discount
# =====================================================================

class TestPPTgPathway:
    """PPTg pathway implements temporal discount via τ_ppTg."""

    def test_gamma_eff_matches_formula(
        self, vta: VTACircuit, critic: SNNDeepCritic,
    ) -> None:
        """γ_eff = exp(-n_substeps × dt / τ_ppTg_eff)."""
        _run_critic(critic)
        vta.store_prediction(critic.activation)
        _run_critic(critic, input_level=0.8)

        n_substeps = 25
        serotonin = 0.5
        vta.compute_rpe(critic.activation, reward=1.0,
                        is_terminal=False, serotonin=serotonin,
                        n_substeps=n_substeps)

        cfg = vta.config
        tau_eff = cfg.tau_ppTg * (1.0 + serotonin)
        expected_gamma = float(np.exp(-n_substeps * cfg.ctx.dt / tau_eff))
        assert abs(vta.last_gamma_eff - expected_gamma) < 1e-6

    def test_default_gamma_near_099(
        self, vta: VTACircuit, critic: SNNDeepCritic,
    ) -> None:
        """Default τ_ppTg=2488ms with n_substeps=25 → γ ≈ 0.99."""
        _run_critic(critic)
        vta.store_prediction(critic.activation)
        _run_critic(critic)
        # serotonin=0 → pure PPTg tau
        vta.compute_rpe(critic.activation, reward=0.0,
                        is_terminal=False, serotonin=0.0,
                        n_substeps=25)
        # γ = exp(-25/2488) ≈ 0.99
        assert 0.989 < vta.last_gamma_eff < 0.991

    def test_terminal_zeroes_future_value(
        self, vta: VTACircuit, critic: SNNDeepCritic,
    ) -> None:
        """At terminal state, V(s') contribution is zero."""
        _run_critic(critic)
        vta.store_prediction(critic.activation)
        _run_critic(critic, input_level=0.9)

        rpe_terminal = vta.compute_rpe(
            critic.activation, reward=1.0, is_terminal=True,
            serotonin=0.5, n_substeps=25,
        )
        assert vta.last_v_s_prime == 0.0


# =====================================================================
# 4. RPE Computation — E/I Balance
# =====================================================================

class TestRPEComputation:
    """VTA E/I balance produces correct RPE."""

    def test_positive_reward_positive_rpe(
        self, vta: VTACircuit, critic: SNNDeepCritic,
    ) -> None:
        """Pure reward with zero V(s) and V(s')=0 at terminal → positive RPE."""
        # Reset to ensure zero activations
        critic.reset_state()
        vta.reset_state()
        vta.store_prediction(np.zeros(64, dtype=np.float32))

        rpe = vta.compute_rpe(
            np.zeros(64, dtype=np.float32), reward=1.0,
            is_terminal=True, serotonin=0.5, n_substeps=25,
        )
        assert rpe > 0.0, f"Positive reward should give positive RPE, got {rpe}"

    def test_no_reward_no_change_near_zero_rpe(
        self, vta: VTACircuit,
    ) -> None:
        """Zero reward, same state → RPE ≈ 0."""
        act = np.random.random(64).astype(np.float32) * 0.1
        vta.store_prediction(act)
        rpe = vta.compute_rpe(
            act, reward=0.0, is_terminal=False,
            serotonin=0.5, n_substeps=25,
        )
        # RPE should be small (near zero, not exactly due to γ < 1)
        assert abs(rpe) < 2.0, f"Same state, no reward → near-zero RPE, got {rpe}"

    def test_reward_surprise_burst(
        self, vta: VTACircuit, critic: SNNDeepCritic,
    ) -> None:
        """Unexpected reward (V(s) ≈ 0) → large positive RPE (DA burst).
        Schultz (1997): unpredicted reward → phasic DA burst."""
        vta.reset_state()
        vta.store_prediction(np.zeros(64, dtype=np.float32))

        rpe = vta.compute_rpe(
            np.zeros(64, dtype=np.float32), reward=5.0,
            is_terminal=True, serotonin=0.5, n_substeps=25,
        )
        assert rpe > 1.0, f"Unexpected reward should cause DA burst, got {rpe}"

    def test_reward_omission_pause(
        self, vta: VTACircuit,
    ) -> None:
        """Expected reward not received → negative RPE (DA pause).
        Schultz (1997): omission of predicted reward → DA pause."""
        # Set up high V(s) expectation via direct weight manipulation
        act_high = np.full(64, 0.1, dtype=np.float32)
        vta.w_value[:] = np.full(64, 0.1, dtype=np.float32)  # V ≈ 0.64
        vta.store_prediction(act_high)

        rpe = vta.compute_rpe(
            np.zeros(64, dtype=np.float32), reward=0.0,
            is_terminal=True, serotonin=0.5, n_substeps=25,
        )
        assert rpe < 0.0, f"Omission of expected reward should cause DA pause, got {rpe}"

    def test_rpe_matches_td_formula(
        self, vta: VTACircuit,
    ) -> None:
        """RPE ∝ r + γ × V(s') − V(s) before gain adaptation."""
        # Set known w_value
        vta.w_value[:] = np.ones(64, dtype=np.float32) * 0.01

        act_s = np.full(64, 0.3, dtype=np.float32)
        act_s_prime = np.full(64, 0.5, dtype=np.float32)

        vta.store_prediction(act_s)
        vta._auto_rms = 1.0  # Normalize gain to ~1

        rpe = vta.compute_rpe(
            act_s_prime, reward=1.0, is_terminal=False,
            serotonin=0.0, n_substeps=25,
        )

        v_s = float(np.dot(act_s, vta.w_value))
        v_sp = float(np.dot(act_s_prime, vta.w_value))
        gamma = vta.last_gamma_eff
        raw_td = 1.0 + gamma * v_sp - v_s

        # RPE = raw_td / auto_rms, so raw_td should match closely
        # (auto_rms has been updated by one step, so approximate)
        assert abs(rpe * vta._auto_rms - raw_td) < 0.1 * abs(raw_td) + 0.01


# =====================================================================
# 5. D2 Autoreceptor Gain Adaptation
# =====================================================================

class TestD2Autoreceptor:
    """D2 autoreceptor adapts coding gain (Tobler et al. 2005)."""

    def test_gain_increases_with_large_rpe(self, vta: VTACircuit) -> None:
        """Repeated large |RPE| → auto_rms grows → gain reduces."""
        vta.w_value[:] = 0.0
        initial_rms = vta._auto_rms
        for _ in range(50):
            vta.store_prediction(np.zeros(64, dtype=np.float32))
            vta.compute_rpe(
                np.zeros(64, dtype=np.float32), reward=10.0,
                is_terminal=True, serotonin=0.5, n_substeps=25,
            )
        assert vta._auto_rms > initial_rms, "Large RPE should increase auto_rms"

    def test_gain_decreases_with_small_rpe(self, vta: VTACircuit) -> None:
        """Repeated small |RPE| → auto_rms shrinks → gain increases."""
        # First inflate the RMS
        vta._auto_rms = 10.0
        for _ in range(100):
            vta.store_prediction(np.zeros(64, dtype=np.float32))
            vta.compute_rpe(
                np.zeros(64, dtype=np.float32), reward=0.01,
                is_terminal=True, serotonin=0.5, n_substeps=25,
            )
        assert vta._auto_rms < 10.0, "Small RPE should decrease inflated auto_rms"

    def test_gain_floor_prevents_division_by_zero(
        self, ctx: SimulationContext,
    ) -> None:
        """min_gain prevents division by zero when RPE is consistently zero."""
        cfg = VTAConfig(ctx=ctx, min_gain=0.05)
        vta = VTACircuit(64, cfg)
        vta.w_value[:] = 0.0
        for _ in range(100):
            vta.store_prediction(np.zeros(64, dtype=np.float32))
            rpe = vta.compute_rpe(
                np.zeros(64, dtype=np.float32), reward=0.0,
                is_terminal=True, serotonin=0.5, n_substeps=25,
            )
            assert np.isfinite(rpe), "RPE should be finite with gain floor"


# =====================================================================
# 6. Serotonin Modulation of Temporal Discount
# =====================================================================

class TestSerotoninModulation:
    """5-HT modulates effective γ via τ_ppTg (Schweighofer 2008)."""

    def test_higher_serotonin_higher_gamma(
        self, vta: VTACircuit, critic: SNNDeepCritic,
    ) -> None:
        """Higher serotonin → longer τ_ppTg → higher γ_eff."""
        _run_critic(critic)
        gammas = []
        for sero in [0.0, 0.5, 1.0, 2.0]:
            vta.store_prediction(critic.activation)
            vta.compute_rpe(
                critic.activation, reward=0.0, is_terminal=False,
                serotonin=sero, n_substeps=25,
            )
            gammas.append(vta.last_gamma_eff)

        # γ should be monotonically increasing with serotonin
        for i in range(len(gammas) - 1):
            assert gammas[i] < gammas[i + 1], (
                f"γ should increase with 5-HT: {gammas}"
            )

    def test_zero_serotonin_gives_base_gamma(
        self, vta: VTACircuit, critic: SNNDeepCritic,
    ) -> None:
        """Serotonin=0 → γ = exp(-25/2488) ≈ 0.99."""
        _run_critic(critic)
        vta.store_prediction(critic.activation)
        vta.compute_rpe(
            critic.activation, reward=0.0, is_terminal=False,
            serotonin=0.0, n_substeps=25,
        )
        expected = float(np.exp(-25.0 / vta.config.tau_ppTg))
        assert abs(vta.last_gamma_eff - expected) < 1e-6


# =====================================================================
# 7. Weight Update — Three-Factor Hebbian
# =====================================================================

class TestVTAWeightUpdate:
    """VTA w_value updated by three-factor STDP."""

    def test_positive_rpe_changes_weights(
        self, vta: VTACircuit, critic: SNNDeepCritic,
    ) -> None:
        """Positive RPE → w_value changes."""
        _run_critic(critic, n_steps=25, input_level=0.7)
        vta.store_prediction(critic.activation)
        w_before = vta.w_value.copy()
        vta.update(rpe=1.0)
        assert np.any(vta.w_value != w_before), "Positive RPE should change w_value"

    def test_negative_rpe_opposite_direction(
        self, vta: VTACircuit, critic: SNNDeepCritic,
    ) -> None:
        """Positive and negative RPE produce weight changes whose
        difference is proportional to eligibility (opposite directions).

        dw_pos - dw_neg = 2 × lr × e_value × (1 - decay), so the
        difference should have the same sign as e_value everywhere."""
        _run_critic(critic, n_steps=25, input_level=0.7)
        vta.store_prediction(critic.activation)

        w_base = vta.w_value.copy()
        vta.update(rpe=+1.0)
        w_pos = vta.w_value.copy()

        vta.w_value[:] = w_base
        vta.update(rpe=-1.0)
        w_neg = vta.w_value.copy()

        # The difference (dw_pos - dw_neg) cancels the common decay term
        # and should be 2 × lr × e × (1-decay) → same sign as e_value
        diff = w_pos - w_neg
        e = vta.e_value
        mask = np.abs(e) > 1e-8
        if np.sum(mask) > 0:
            sign_match = np.sum(np.sign(diff[mask]) == np.sign(e[mask]))
            total = np.sum(mask)
            assert sign_match > total * 0.95, (
                f"diff sign should match e_value sign: {sign_match}/{total}"
            )

    def test_zero_rpe_no_change(
        self, vta: VTACircuit, critic: SNNDeepCritic,
    ) -> None:
        """Zero RPE → no weight change (modulo decay)."""
        _run_critic(critic, n_steps=25, input_level=0.7)
        vta.store_prediction(critic.activation)
        w_before = vta.w_value.copy()
        vta.update(rpe=0.0)
        # Only soft decay should modify weights
        max_change = np.max(np.abs(vta.w_value - w_before))
        assert max_change < 1e-4, "Zero RPE should cause negligible weight change"

    def test_weight_norm_bounded(
        self, vta: VTACircuit, critic: SNNDeepCritic,
    ) -> None:
        """Repeated large updates don't cause weight explosion."""
        for _ in range(1000):
            _run_critic(critic, n_steps=5, input_level=0.8)
            vta.store_prediction(critic.activation)
            vta.update(rpe=5.0)

        w_norm = float(np.linalg.norm(vta.w_value))
        assert w_norm < 1e4, f"w_value exploded: {w_norm}"

    def test_readout_decay_shrinks_weights(
        self, vta: VTACircuit,
    ) -> None:
        """Soft readout decay shrinks weights when no learning signal."""
        vta.w_value[:] = np.ones(64, dtype=np.float32)
        w_norm_before = float(np.linalg.norm(vta.w_value))
        for _ in range(100):
            vta.update(rpe=0.0)
        w_norm_after = float(np.linalg.norm(vta.w_value))
        assert w_norm_after < w_norm_before, "Readout decay should shrink weights"


# =====================================================================
# 8. State Management
# =====================================================================

class TestVTAStateManagement:
    """Reset between episodes preserves weights, clears transient state."""

    def test_reset_clears_transient(self, vta: VTACircuit) -> None:
        """reset_state() clears stored activation and eligibility."""
        vta.store_prediction(np.ones(64, dtype=np.float32))
        vta.reset_state()
        assert np.all(vta._stored_activation == 0.0)
        assert vta._stored_v == 0.0
        assert np.all(vta.e_value == 0.0)
        assert vta.last_rpe == 0.0

    def test_reset_preserves_weights(self, vta: VTACircuit) -> None:
        """Weights survive reset."""
        w_before = vta.w_value.copy()
        vta.store_prediction(np.ones(64, dtype=np.float32))
        vta.reset_state()
        np.testing.assert_array_equal(vta.w_value, w_before)

    def test_reset_preserves_auto_rms(self, vta: VTACircuit) -> None:
        """D2 autoreceptor state survives reset (slow adaptation)."""
        vta._auto_rms = 5.0
        vta.reset_state()
        assert vta._auto_rms == 5.0


# =====================================================================
# 9. Integration with SNNAgent
# =====================================================================

class TestVTAAgentIntegration:
    """VTA circuit integrates correctly into the full agent pipeline."""

    @pytest.fixture
    def agent(self):
        from arena.snn_agent import SNNAgent
        return SNNAgent(
            state_size=4, n_actions=2,
            use_world_model=False, use_working_memory=False,
        )

    def test_agent_has_vta(self, agent) -> None:
        assert hasattr(agent, 'vta')
        assert isinstance(agent.vta, VTACircuit)

    def test_act_stores_prediction(self, agent) -> None:
        """After act(), VTA should have a stored V(s)."""
        state = np.random.uniform(-1, 1, 4).astype(np.float32)
        agent.act(state)
        # VTA should have non-trivial stored activation
        assert np.any(agent.vta._stored_activation != 0.0)

    def test_observe_produces_rpe(self, agent) -> None:
        """After observe(), VTA last_rpe should be non-zero."""
        s = np.random.uniform(-1, 1, 4).astype(np.float32)
        a = agent.act(s)
        ns = np.random.uniform(-1, 1, 4).astype(np.float32)
        agent.observe(s, a, 1.0, ns, False)
        # With reward=1, RPE should be nonzero
        assert agent.vta.last_rpe != 0.0

    def test_agent_no_welford_attributes(self, agent) -> None:
        """Welford TD normalisation removed — attributes must not exist."""
        assert not hasattr(agent, '_td_ema_mean')
        assert not hasattr(agent, '_td_ema_var')
        assert not hasattr(agent, '_TD_EMA_DECAY')

    def test_vta_w_value_finite_after_episodes(self, agent) -> None:
        """Multiple episodes — VTA w_value stays finite."""
        for ep in range(5):
            agent.reset()
            for _ in range(50):
                s = np.random.uniform(-1, 1, 4).astype(np.float32)
                a = agent.act(s)
                ns = np.random.uniform(-1, 1, 4).astype(np.float32)
                done = np.random.random() < 0.1
                agent.observe(s, a, 1.0, ns, done)
                if done:
                    break
        assert np.all(np.isfinite(agent.vta.w_value)), "VTA w_value has NaN"

    def test_reset_clears_vta_transient(self, agent) -> None:
        """Agent reset clears VTA transient state."""
        s = np.random.uniform(-1, 1, 4).astype(np.float32)
        agent.act(s)
        agent.reset()
        assert np.all(agent.vta._stored_activation == 0.0)
        assert np.all(agent.vta.e_value == 0.0)


# =====================================================================
# 10. Numerical Stability
# =====================================================================

class TestVTANumericalStability:
    """Long runs with extreme inputs — no NaN, no explosion."""

    def test_10k_steps_no_nan(self, vta: VTACircuit) -> None:
        """10,000 RPE computations with random inputs — all finite."""
        for t in range(10_000):
            act = np.random.random(64).astype(np.float32) * 0.2
            vta.store_prediction(act)
            act_next = np.random.random(64).astype(np.float32) * 0.2
            r = np.random.uniform(-5, 5)
            rpe = vta.compute_rpe(
                act_next, reward=r, is_terminal=(t % 100 == 99),
                serotonin=np.random.uniform(0, 1), n_substeps=25,
            )
            assert np.isfinite(rpe), f"NaN RPE at step {t}"
            vta.update(rpe)
            assert np.all(np.isfinite(vta.w_value)), f"NaN w_value at step {t}"

    def test_extreme_reward_no_nan(self, vta: VTACircuit) -> None:
        """Extreme rewards (±1000) — RPE stays finite."""
        for r in [1000.0, -1000.0, 0.0, 1e-8, -1e-8]:
            vta.store_prediction(np.zeros(64, dtype=np.float32))
            rpe = vta.compute_rpe(
                np.zeros(64, dtype=np.float32), reward=r,
                is_terminal=True, serotonin=0.5, n_substeps=25,
            )
            assert np.isfinite(rpe), f"NaN RPE for reward={r}"


# =====================================================================
# 11. Old API Removal Verification
# =====================================================================

class TestOldAPIRemoved:
    """Verify that old algebraic TD path is fully removed."""

    def test_critic_no_last_value(self, critic: SNNDeepCritic) -> None:
        """SNNDeepCritic.last_value removed."""
        assert not hasattr(critic, 'last_value')

    def test_critic_no_w_v(self, critic: SNNDeepCritic) -> None:
        """SNNDeepCritic.w_v removed — value readout in VTA."""
        assert not hasattr(critic, 'w_v')

    def test_critic_no_b_v(self, critic: SNNDeepCritic) -> None:
        assert not hasattr(critic, 'b_v')

    def test_critic_no_e_v(self, critic: SNNDeepCritic) -> None:
        assert not hasattr(critic, 'e_v')

    def test_critic_no_spike_count(self, critic: SNNDeepCritic) -> None:
        assert not hasattr(critic, '_spike_count')

    def test_critic_no_v_accum(self, critic: SNNDeepCritic) -> None:
        assert not hasattr(critic, '_v_accum')

    def test_critic_no_n_substeps(self, critic: SNNDeepCritic) -> None:
        assert not hasattr(critic, '_n_substeps')

    def test_critic_no_feature_rms_ema(self, critic: SNNDeepCritic) -> None:
        assert not hasattr(critic, '_feature_rms_ema')

    def test_critic_no_reset_spike_counts(self, critic: SNNDeepCritic) -> None:
        assert not hasattr(critic, 'reset_spike_counts')

    def test_critic_still_has_activation(self, critic: SNNDeepCritic) -> None:
        """activation still exists — VTA reads from it."""
        assert hasattr(critic, 'activation')

    def test_critic_still_has_e_h(self, critic: SNNDeepCritic) -> None:
        """e_h (hidden layer eligibility) still exists."""
        assert hasattr(critic, 'e_h')

    def test_critic_still_has_w_h(self, critic: SNNDeepCritic) -> None:
        """w_h (hidden layer weights) still exists."""
        assert hasattr(critic, 'w_h')
