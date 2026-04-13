"""
VTA Dopaminergic Circuit — emergent RPE from excitatory/inhibitory balance.

Replaces algebraic TD error (r + γV(s') − V(s)) with a biophysical circuit
where DA neuron output emerges from competing synaptic pathways.

References:
    Eshel et al. (2015): Arithmetic and local circuitry underlying
        dopamine prediction errors.  Science 350(6256):1–5.
    Schultz (1997, 1998): Predictive reward signal of DA neurons.
    Tobler et al. (2005): Adaptive coding of reward value by DA neurons.
    Grace (1991): Phasic vs tonic DA release.
    Schweighofer et al. (2008): Serotonin modulates temporal discount.
    Watabe-Uchida et al. (2017): Direct inputs to VTA DA neurons.

Circuit:
    VP pathway  (inhibitory): critic activation(s)  → w_value → I_vp
        Net effect of VS → VP → RMTg → VTA (triple GABAergic).
        Higher V(s) → more VTA inhibition.

    PPTg pathway (excitatory): critic activation(s') → w_value → I_ppTg × γ_eff
        PPTg synaptic dynamics attenuate future-value signal by
        γ_eff = exp(−T_step / τ_ppTg), implementing temporal discount
        without an explicit γ parameter.

    Reward pathway (excitatory): reward → reward_gain → I_reward
        Direct LDTg/PPTg glutamatergic drive.

    VTA DA output: RPE ∝ I_reward + I_ppTg − I_vp
        D2 autoreceptors (Tobler 2005) adapt coding gain to recent
        RPE statistics, replacing Welford running-variance normalisation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import VTAConfig


class VTACircuit:
    """VTA dopaminergic circuit for emergent RPE computation.

    A single shared weight vector ``w_value`` reads the critic's
    population activation to estimate state value.  This is
    mathematically equivalent to semi-gradient TD but implemented
    via competing neural pathways:

        RPE = reward + γ_eff × V̂(s') − V̂(s)

    where V̂(·) = dot(critic_activation, w_value) and
    γ_eff = exp(−n_substeps × dt / τ_ppTg_eff) with serotonin
    modulation of τ_ppTg.

    The D2 autoreceptor provides intrinsic gain adaptation
    (Tobler et al. 2005), replacing external Welford normalisation.
    """

    def __init__(self, critic_hidden_size: int, config: VTAConfig) -> None:
        self.config = config
        self._h = critic_hidden_size

        # ── Value readout weight (shared VP/PPTg pathway) ─────────
        # Analogous to the old w_v but now lives in VTA, read through
        # two pathways with different gating (inhibitory vs excitatory).
        v_std = 1.0 / np.sqrt(critic_hidden_size)
        self.w_value: NDArray[np.float32] = np.random.uniform(
            -v_std, v_std, critic_hidden_size,
        ).astype(np.float32)

        # ── VP trace: stores V(s) from act() ─────────────────────
        # VP neurons capture the critic's value prediction at decision
        # time.  Held constant until observe() (no decay during the
        # decision interval — decay applies through PPTg pathway).
        self._stored_activation: NDArray[np.float32] = np.zeros(
            critic_hidden_size, dtype=np.float32,
        )
        self._stored_v: float = 0.0

        # ── D2 autoreceptor state (Tobler et al. 2005) ──────────
        # Tracks RMS of recent unsigned RPE.  Higher RMS → lower gain
        # → wider coding range (adaptive).  Lower RMS → higher gain →
        # sensitive to small signals.  Replaces Welford normalisation.
        self._auto_rms: float = 1.0   # warm-start (avoids early blow-up)
        self._auto_decay: float = config.ctx.decay(config.tau_autoreceptor)

        # ── Eligibility trace for w_value ─────────────────────────
        # Snapshot of centered critic activation at decision time.
        # Called once per act() — stores WHICH critic neurons were
        # active above/below average, enabling selective credit
        # assignment (Shadlen & Newsome 2001).
        self.e_value: NDArray[np.float32] = np.zeros(
            critic_hidden_size, dtype=np.float32,
        )

        # ── Diagnostics ──────────────────────────────────────────
        self.last_rpe: float = 0.0
        self.last_v_s: float = 0.0
        self.last_v_s_prime: float = 0.0
        self.last_gamma_eff: float = 0.99

    # ------------------------------------------------------------------
    # Store V(s) prediction — called after act()
    # ------------------------------------------------------------------

    def store_prediction(self, critic_activation: NDArray[np.float32]) -> None:
        """Capture V(s) in the VP pathway after the decision integration.

        The VP trace holds a snapshot of the critic's population rates
        at decision time.  This represents "what I expected" and will
        inhibit VTA DA neurons during observe().

        Eligibility = raw activation (uncentered).  VP/PPTg neurons
        receive total striatal firing (Eshel et al. 2015) — there is
        no biological mean-subtraction across the population.  The
        semi-gradient ∂V/∂w = φ(s) where V(s) = dot(φ, w), so the
        eligibility must equal the raw feature vector for gradient
        consistency (Sutton & Barto 2018, §9.4).
        """
        self._stored_activation = critic_activation.copy()
        self._stored_v = float(np.dot(critic_activation, self.w_value))
        self.last_v_s = self._stored_v

        # Raw activation as eligibility — gradient-consistent with
        # V(s) = dot(activation, w_value).
        self.e_value = critic_activation.copy()

    # ------------------------------------------------------------------
    # Compute RPE — called during observe()
    # ------------------------------------------------------------------

    def compute_rpe(
        self,
        critic_activation: NDArray[np.float32],
        reward: float,
        is_terminal: bool,
        serotonin: float,
        n_substeps: int,
    ) -> float:
        """Compute RPE from VTA circuit E/I balance.

        VTA DA neuron output ∝ I_reward + I_ppTg − I_vp:
            I_vp      = V̂(s) from VP trace (inhibitory)
            I_ppTg    = γ_eff × V̂(s') via PPTg pathway (excitatory)
            I_reward  = reward × reward_gain (excitatory)

        Temporal discount γ_eff emerges from PPTg pathway dynamics:
            τ_eff = τ_ppTg × (1 + serotonin)
            γ_eff = exp(−n_substeps × dt / τ_eff)

        Higher serotonin → longer τ → higher γ → longer planning
        horizon (Schweighofer et al. 2008).

        D2 autoreceptor performs RMS-based gain adaptation (Tobler 2005),
        replacing Welford EMA normalisation.

        Parameters
        ----------
        critic_activation : array (hidden_size,)
            Critic's population firing rates after processing next_state.
        reward : float
            Immediate environmental reward (possibly shaped).
        is_terminal : bool
            True if next_state is terminal (no future value).
        serotonin : float
            Current serotonin level [0, 1] from neuromodulator.
        n_substeps : int
            Number of integration substeps in the decision interval.
            Determines the temporal extent over which γ_eff is computed.

        Returns
        -------
        rpe : float
            Gain-adapted RPE signal for broadcast to critic/actor STDP.
        """
        cfg = self.config
        dt = cfg.ctx.dt

        # ── Serotonin-modulated temporal discount ─────────────────
        # Schweighofer et al. (2008): 5-HT shifts the effective
        # discount horizon.  Higher 5-HT = longer pathway τ = higher γ.
        sero = float(np.clip(serotonin, 0.0, 2.0))
        tau_ppTg_eff = cfg.tau_ppTg * (1.0 + sero)
        gamma_eff = float(np.exp(-n_substeps * dt / tau_ppTg_eff))
        self.last_gamma_eff = gamma_eff

        # ── VP inhibitory pathway: V(s) ───────────────────────────
        # Stored during act() — represents the prediction from the
        # current state before the outcome was observed.
        I_vp = self._stored_v

        # ── PPTg excitatory pathway: γ × V(s') ───────────────────
        if is_terminal:
            I_ppTg = 0.0
            v_s_prime = 0.0
        else:
            v_s_prime = float(np.dot(critic_activation, self.w_value))
            I_ppTg = gamma_eff * v_s_prime
        self.last_v_s_prime = v_s_prime

        # ── Reward excitatory pathway (direct) ────────────────────
        I_reward = cfg.reward_gain * float(reward)

        # ── VTA DA neuron E/I balance → raw RPE ──────────────────
        # Eshel et al. (2015): DA ∝ excitatory − inhibitory
        rpe_raw = I_reward + I_ppTg - I_vp

        # ── D2 autoreceptor gain adaptation (Tobler 2005) ─────────
        # RMS tracks recent |RPE| — higher variance → lower gain.
        # sqrt(decay × rms² + (1-decay) × rpe²) — same EMA form
        # as the old Welford variance tracker but intrinsic to VTA.
        self._auto_rms = float(np.sqrt(
            self._auto_decay * self._auto_rms ** 2
            + (1.0 - self._auto_decay) * rpe_raw ** 2
        ))
        auto_gain = max(self._auto_rms, cfg.min_gain)

        rpe = rpe_raw / auto_gain
        self.last_rpe = rpe
        return rpe

    # ------------------------------------------------------------------
    # Weight update — called after compute_rpe()
    # ------------------------------------------------------------------

    def update(self, rpe: float) -> None:
        """Three-factor Hebbian update for the value readout weight.

        dw_value = value_lr × RPE × e_value

        The eligibility trace ``e_value`` was accumulated during store_prediction()
        (act phase), so the update reinforces the prediction made at the
        current state — standard semi-gradient TD.

        Soft readout weight decay models compressed protein turnover
        (Bhatt et al. 2009).
        """
        cfg = self.config

        # Three-factor STDP update
        dw = cfg.value_lr * float(rpe) * self.e_value
        self.w_value += dw

        # Soft weight decay (protein turnover)
        self.w_value *= (1.0 - cfg.readout_decay)

        # Homeostatic norm bound (same derivation as old w_v bound)
        # |V(s)| ≤ ||w|| × max||act_centered|| + margin.
        # For centered [0,1] activations of dim h: max||act|| ≈ √(h/4).
        # Bound ||w|| so max |V| ≤ max_return (100 for γ≈0.99).
        gamma_approx = self.last_gamma_eff
        max_return = max(10.0, 1.0 / max(1e-4, 1.0 - gamma_approx))
        max_feat_norm = np.sqrt(self._h / 4.0)
        w_norm_max = max_return / max(max_feat_norm, 1.0)
        w_norm = float(np.linalg.norm(self.w_value))
        if w_norm > w_norm_max:
            self.w_value *= w_norm_max / w_norm

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset transient state between episodes. Weights preserved."""
        self._stored_activation.fill(0.0)
        self._stored_v = 0.0
        self.e_value.fill(0.0)
        self.last_rpe = 0.0
        self.last_v_s = 0.0
        self.last_v_s_prime = 0.0
