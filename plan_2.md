# PLAN: SNN Architecture Audit & Transformation toward AGI-grade Biophysics

## TL;DR

The codebase has strong biophysical foundations (AdEx, STDP, 4-channel neuromodulation, conductance-based synapses) but retains ~15 RL/ML algorithmic patterns that bypass neural dynamics. These must be replaced with emergent biophysical processes. The existing plan.md Phase 0 and Phase 1 are complete. This plan replaces all remaining phases.

---

## AUDIT RESULTS: What's Already Correct

### Biophysically Sound (Keep as-is)

- AdEx neuron model (Brette & Gerstner 2005) with exponential Euler integrator
- Three-factor STDP with Bi & Poo (2001) asymmetric window
- Conductance-based synapses: AMPA/NMDA/GABA-A/GABA-B with Mg²⁺ block (Jahr & Stevens 1990)
- InhibitoryPool with PV+ FS interneurons, Hebbian E→I / anti-Hebbian I→E STDP (Woodin et al. 2003)
- D1/D2 MSN bistable dynamics (Wilson & Kawaguchi 1996)
- Astrocyte Ca²⁺ dynamics with ATP energy budget (De Pittà et al. 2011)
- Population coding (Pouget et al. 2000) with Gaussian tuning curves
- Oscillator: theta-gamma PAC (Lisman & Jensen 2013)
- Pyramidal neurons with BAC firing + Ca²⁺ spikes (Larkum 2013; Payeur et al. 2021)
- Predictive coding: error/state neuron separation (Rao & Ballard 1999; Bogacz 2017)
- Receptor pharmacology via Hill equation dose-response (Doya 2002)
- Neuromodulator decay constants from pharmacokinetics (DAT, AChE, NET, SERT)
- Dale's law enforced throughout
- OpAL D1/D2 sign-gated plasticity (Frank 2005; Collins & Frank 2014)
- Weight initialization from PSP target scaling
- Homeostatic synaptic scaling (Turrigiano 2004, 2008)
- NMDA slow temporal integration (Wang 2002)

### From plan.md Already Implemented

- Phase 0: All Kroki 0.1–0.7 done
- Phase 1: AdEx (1.1), Exp Euler (1.2), ATP (1.3), tonic DA continuous
- FIX 1-11 from diagnostic section: all completed
- internal_dim=0 when no WM, eligibility preservation, membrane readout, D2 protection

---

## CRITICAL RL/ML HACKS FOUND — Must Replace

### HACK 1: Softmax + np.random.choice for action selection

**File:** `core/basal_ganglia.py`, D1D2Actor.forward() ~line 1150

```
shifted = net_evidence - np.max(net_evidence)
exp_val = np.exp(shifted / _T)
probs = exp_val / (np.sum(exp_val) + 1e-10)
action = int(np.random.choice(len(probs), p=probs))
```

**Problem:** Softmax is a mathematical function, not a neural process. Action selection must EMERGE from competitive spiking dynamics, not be computed externally.
**Replace with:** Winner-take-all via mutual inhibition between action populations. The last-man-standing after the integration window IS the action. Tie-breaking via membrane noise (already present). Reference: Usher & McClelland (2001) "The time course of perceptual choice"; Wang (2002) "Probabilistic decision making by slow reverberation in cortical circuits."

### HACK 2: Explicit TD error computation

**File:** `arena/snn_agent.py`, observe() ~line 575

```
td_error = effective_reward + gamma * current_v - prev_v
```

**Problem:** Algebraic subtraction is not neural. In biology, VTA DA neurons compute RPE by comparing excitatory reward/expected-value inputs with inhibitory current-value inputs. The subtraction emerges from E/I balance in the VTA circuit.
**Replace with:** VTA circuit model where reward input excites DA neurons, current V(s) inhibits via GABAergic VP, and the NET firing rate of DA neurons IS the RPE. Reference: Schultz (1997, 1998); Eshel et al. (2015) "Arithmetic and local circuitry underlying dopamine prediction errors."

### HACK 3: Welford EMA for TD normalization

**File:** `arena/snn_agent.py` ~line 590

```
_td_ema_mean += _alpha * (td_for_welford - _td_ema_mean)
_td_ema_var += _alpha * ((td - _td_ema_mean)² - _td_ema_var)
td_error_normed = td_error / _td_std
```

**Problem:** Online statistics computation is a mathematical operation, not a neural process. DA gain adaptation should emerge from VTA intrinsic properties.
**Replace with:** Adaptive gain via DA autoreceptor feedback and RMS-based intrinsic conductance change (Tobler et al. 2005 mechanism). The VTA neuron's own firing history modulates its gain through D2 autoreceptors. This IS already partially implemented (`_da_rms` in neuromodulator.py) but the Welford normalizer in snn_agent.py is separate and redundant.

### HACK 4: Exponential discounting (gamma=0.99)

**File:** `core/config.py`, BasalGangliaConfig
**Problem:** Biological temporal discounting is hyperbolic, not exponential (Ainslie 1975; Green & Myerson 2004). Exponential discounting is an RL convenience. Hyperbolic discounting emerges naturally from competing neural evaluation systems with different time constants.
**Replace with:** Remove explicit gamma from TD computation. Instead, use the natural membrane decay time constants of ventral striatal neurons as temporal discounting. V(s') is naturally discounted by the membrane integration window τ. This is already implicit — the critic integrates over n_substeps_critic, and the membrane decay IS the discount. Make this explicit and remove the algebraic gamma.

### HACK 5: Linear readout w_v · v_centered + b_v

**File:** `core/basal_ganglia.py`, SNNDeepCritic.forward()
**Problem:** Dot product readout is a linear regression. In biology, the ventral striatal "value signal" is carried by the population firing pattern itself, read out by downstream GPi/VP via synaptic convergence.
**Replace with:** Make the critic's output a population rate code that directly feeds VTA as excitatory input. No separate readout weight vector — the VTA circuit subtracts reward from V(s) using its own synaptic weights. Reference: Takahashi et al. (2011): VP neurons encode expected value and project to VTA.

### HACK 6: gate_eligibility() zeroes non-selected actions

**File:** `core/basal_ganglia.py`, D1D2Actor.gate_eligibility()
**Problem:** Explicit zeroing is an algorithmic operation. Biologically, lateral inhibition from the winning action channel suppresses plasticity in losing channels via GABAergic mechanisms.
**Replace with:** Existing InhibitoryPool already provides lateral inhibition. Strengthen the cross-action inhibitory coupling so that the winning action's interneurons suppress the losing actions' STDP traces through GABA-mediated eligibility suppression. Reference: Wickens et al. (2003): "Paths from corticostriatal input to output."

### HACK 7: Sleep phase advantage estimation

**File:** `core/replay_buffer.py`, \_sws_phase()
**Problem:** Monte Carlo return computation, advantage normalization, pos/neg ratio scaling — all RL algorithms, not biological processes. Sleep replay should be temporal sequence reactivation with existing synaptic dynamics.
**Replace with:** Sharp-wave ripple (SWR) replay that reactivates spike sequences stored in weight matrices, with learning modulated solely by the slow oscillation Up/Down state gating and existing STDP rules. No external return computation. Reference: Buzsáki (2015): SWR replay consolidates via STDP with compressed timescale, not advantage estimation.

### HACK 8: Curiosity z-score normalization

**File:** `core/world_model.py`, curiosity_signal()

```
z = (raw - mu) / sigma
return float(np.clip(1.0 + 0.5 * z, 0.0, 2.0))
```

**Problem:** z-score normalization is a statistical operation. Curiosity should emerge directly from prediction error magnitude relative to expected precision.
**Replace with:** Raw precision-weighted PE from the astrocyte/error neuron pathway. The error neuron rate IS the curiosity signal. Normalization happens naturally via homeostatic adaptation of error neuron thresholds.

### HACK 9: Intrinsic reward heuristic formula

**File:** `arena/snn_agent.py`, observe()

```
intrinsic_weight = acfg.intrinsic_reward_weight * (1.0 - learning_rate_modulation)
effective_reward = reward + intrinsic_r
```

**Problem:** Intrinsic reward as additive bonus is an RL pattern (Pathak et al. 2017). Biological curiosity doesn't add to reward — it modulates exploration through NE/ACh pathways that are already implemented.
**Replace with:** Remove additive intrinsic reward. The world model prediction error already flows through NE (surprise) and ACh (novelty) pathways. These modulate exploration temperature and plasticity directly. No need for explicit reward shaping.

### HACK 10: Internal actions via sigmoid on logits

**File:** `core/basal_ganglia.py`, D1D2Actor.forward() ~line 1190

```
internal_logits = state_f32 @ self.w_d1[:, self._total_motor:]
self.last_internal_action = 1 / (1 + exp(-logits))
```

**Problem:** Sigmoid on matrix-vector product is a standard ML operation.
**Replace with:** Internal actions (WM gating) should use the same bistable MSN dynamics as motor actions. The WM gate neuron fires or doesn't fire based on spiking threshold, not a sigmoid.

---

## MODERATE ISSUES — Should Fix

### ISSUE 11: concat aggregation in NetworkGraph

**File:** `core/network.py`, `arena/snn_agent.py`
**Problem:** `aggregation_mode="concat"` concatenates spike arrays. Biological multi-source convergence is always additive — synaptic currents sum at the dendrite.
**Fix:** Replace all concat with sum. Adjust weight matrices to handle different input sizes via projection layers (biologically: different dendritic branches).

### ISSUE 12: Periodic homeostasis via counter modulo

**File:** `core/basal_ganglia.py`, critic/actor update()

```
if self._homeo_counter % cfg.homeo_interval == 0:
```

**Problem:** Digital clock is not biological. Homeostasis is continuous.
**Fix:** Replace with continuous per-step multiplicative scaling with small alpha. The EMA rate tracker already exists — apply correction continuously.

### ISSUE 13: Column norm clipping (hard bound)

**File:** `core/basal_ganglia.py` — for-loop over columns
**Problem:** Hard L2 norm clipping is a gradient-descent regularization technique.
**Fix:** Continuous synaptic scaling already exists (homeostatic). Remove hard clip; rely on continuous scaling + Dale's law floor.

### ISSUE 14: PSP target 0.15 × gap in neuron.py

**File:** `core/neuron.py`
**Problem:** The 0.15 factor is not derived from conductance physiology.
**Fix:** Derive from expected EPSP amplitude: for a single-synapse conductance g_syn ≈ 0.5-1 nS (Feldmeyer et al. 2002), EPSP ≈ g_syn × (E_rev - V_rest) / g_L ≈ 0.5 × 70 / 30 ≈ 1.2 mV. With 5% active inputs → total PSP ≈ fan_in × 0.05 × 1.2 ÷ √(fan_in × 0.05).

### ISSUE 15: EMA alpha=0.01 for feature RMS

**File:** `core/basal_ganglia.py`, SNNDeepCritic.forward()
**Problem:** Arbitrary smoothing constant.
**Fix:** Derive from neuronal rate adaptation τ. The feature RMS tracks population-average activation at a timescale that should match the homeostatic rate averaging τ. Use cfg.homeo_tau.

### ISSUE 16: CartPole task config is anti-AGI

**File:** `arena/task_config.py`
**Problem:** CartPole gives +1 reward EVERY step, creating persistently positive TD. This is terrible for testing an AGI backbone because it doesn't require exploration, model-based reasoning, or adaptation to change.
**Fix:** Remove CartPole config. Replace with:

- Multi-armed bandit with reversal (tests adaptation + NoGo learning)
- Two-step task (Daw et al. 2011 — tests model-based vs model-free)
- T-Maze with delay (already exists — tests working memory)
- Shifting bandit with contingency degradation

### ISSUE 17: Dead code

- `_step_count` in snn_agent.py — incremented, never read
- `BenchmarkConfig.solve_rate_threshold` — never used
- `TrainResult.learning_curve()`, `.is_improving()`, `.action_distribution()` — never called

### ISSUE 18: World model decoder weight clip ±1.0

**File:** `core/world_model.py`, update()

```
np.clip(self.w_decode, -1.0, 1.0, out=self.w_decode)
```

**Problem:** Arbitrary hard clip.
**Fix:** Use continuous synaptic scaling via homeostatic mechanism (same pattern as critic/actor).

### ISSUE 19: sleep_gain formula

**File:** `arena/snn_agent.py`

```
sleep_gain = 1.0 + acfg.sleep_gain_scale * self.neuromod.tonic_da
```

**Problem:** Heuristic linear scaling with arbitrary multiplier.
**Fix:** Sleep consolidation vigor should be controlled by the serotonin level (promotes sleep) and inversely by NE (suppresses REM). Use the already-existing serotonin and NE levels.

### ISSUE 20: SWS replay sleep_signal formula

**File:** `core/replay_buffer.py`

```
pos_ratio = (0.5 * 200) / max(n_exp, 1)
neg_ratio = (0.1 * 200) / max(n_exp, 1)
sleep_signal = norm_adv * ratio * sleep_gain
```

**Problem:** Magic numbers 0.5, 0.1, 200 with no derivation.
**Fix:** Part of HACK 7 replacement — entire advantage-based sleep to be replaced.

---

## IMPLEMENTATION PLAN

### Phase 2: Emergent Action Selection (replaces softmax)

**Goal:** Action selection emerges from competitive spiking dynamics, not from external computation.

**Step 2.1:** Remove softmax/np.random.choice from D1D2Actor.forward(). Replace with WTA:

- After integration window, the action population with highest mean membrane potential wins
- Ties broken by membrane noise already present
- No probability computation, no sampling
- The MOST DEPOLARIZED action channel IS the action
- Reference: Wang (2002); Usher & McClelland (2001)
- **File:** `core/basal_ganglia.py` — D1D2Actor.forward(), get_action()

**Step 2.2:** Remove gate_eligibility() and replace with inhibition-gated plasticity:

- InhibitoryPool cross-action connections scale inversely with action population distance
- Winning action's interneurons emit stronger GABA → suppress losing channels' eligibility traces via adenosine A1 receptor-like modulation (Pascual et al. 2005)
- The suppression IS the gating — no external zeroing
- **File:** `core/basal_ganglia.py` — D1D2Actor, InhibitoryPool interaction

**Step 2.3:** Remove internal action sigmoid. WM gate uses MSN dynamics:

- Internal neurons compete same as motor neurons
- WM gate = spike/no-spike of gate MSN population
- **File:** `core/basal_ganglia.py` — D1D2Actor.forward() internal action block

**Verification:**

- Test: two-armed bandit → agent selects actions with correct frequency matching reward probabilities
- Test: action entropy responds to NE level (high NE → high entropy, low NE → low entropy)
- Test: no NaN/Inf in 10k steps with stochastic reward

---

### Phase 3: Neural TD Error (replaces algebraic subtraction)

**Goal:** RPE emerges from VTA circuit dynamics, not from Python arithmetic.

**Step 3.1:** Create VTA circuit module (`core/vta.py`):

- VTA DA neurons receive:
  - Excitatory: reward signal (direct) + predicted value via PPTg
  - Inhibitory: current V(s) estimate via VP (ventral pallidum) → RMTg → VTA
- DA neuron firing rate = net E/I balance = reward + γV(s') - V(s) naturally
- DA gain adaptation via D2 autoreceptors on VTA soma
- Reference: Eshel et al. (2015); Tian et al. (2016); Watabe-Uchida et al. (2017)
- **Files:** New `core/vta.py`, modify `arena/snn_agent.py`

**Step 3.2:** Remove critic-as-separate-module pattern:

- Ventral striatal neurons (current critic) → VP → RMTg → VTA (inhibitory)
- This replaces the explicit `prev_v` subtraction
- VP neurons encode negative value (excitatory ventral striatum → inhibitory to VTA)
- The critic still exists but its output doesn't get read via dot product
- Instead, VP reads critic population activity via synaptic integration
- **Files:** `core/basal_ganglia.py` (SNNDeepCritic output routing)

**Step 3.3:** Remove Welford TD normalization:

- VTA gain adaptation is handled by D2 autoreceptors in Step 3.1
- The DA RMS tracking in neuromodulator.py already implements Weber-Fechner adaptive gain
- Merge: VTA circuit receives DA via autoreceptors, gain scales intrinsically
- **File:** `arena/snn_agent.py` — remove \_td_ema_mean/var/decay

**Step 3.4:** Replace gamma with temporal discounting from membrane dynamics:

- V(s') is naturally discounted by membrane integration time → exp(-T/τ) where T = decision interval
- This IS exponential discounting but with τ derived from biology, not a parameter
- For tasks requiring different horizons, different τ values (serotonin-modulated) achieve this
- Remove explicit gamma multiplication
- **File:** `arena/snn_agent.py`, `core/config.py` (BasalGangliaConfig.gamma)

**Verification:**

- Test: constant reward → VTA firing rate converges to baseline (zero RPE)
- Test: reward surprise → VTA burst proportional to surprise magnitude
- Test: reward omission → VTA pause proportional to expected reward
- Test: DA gain adaptation → VTA response scales with reward variance (Tobler 2005)

---

### Phase 4: Biological Sleep Consolidation (replaces advantage estimation)

**Goal:** Sleep replay uses only spike dynamics and STDP, no external return computation.

**Step 4.1:** Replace \_sws_phase() with SWR-driven sequence replay:

- Hippocampal replay: stored spike sequences reactivated in compressed time (Buzsáki 2015)
- Each experience's spike_trains field replayed through BG at 5-20× speed
- STDP rules operate on replayed spikes same as online
- Modulation: slow oscillation Up/Down gating (already implemented)
- No return computation, no advantage normalization
- **File:** `core/replay_buffer.py`

**Step 4.2:** Replace curiosity z-score with raw prediction error signal:

- Remove statistical normalization from world_model.curiosity_signal()
- Return raw precision-weighted PE, clipped only by physiological bounds
- Reference: the error neuron rate IS the novelty signal
- **File:** `core/world_model.py`

**Step 4.3:** Remove additive intrinsic reward:

- Curiosity drives exploration via NE/ACh pathways, not reward shaping
- Delete effective_reward = reward + intrinsic_r
- Keep the NE/ACh modulation of plasticity and exploration temperature
- **File:** `arena/snn_agent.py`

**Step 4.4:** Replace sleep_gain with serotonin/NE-controlled consolidation vigor:

- Consolidation strength modulated by 5-HT (promotes) and inversely by NE
- Remove arbitrary linear formula
- **File:** `arena/snn_agent.py`

**Verification:**

- Test: after sleep, agent's value estimates for recently experienced states improve
- Test: slow oscillation Up phase → replay spikes active; Down phase → silence
- Test: no advantage/return/gamma variables used anywhere in sleep code

---

### Phase 5: Structural Cleanup & Consistency

**Step 5.1:** Replace all concat aggregation with sum:

- Biologically: synaptic currents sum at dendrite
- Resize weight matrices for critic/actor to handle full input
- WM→BG connection uses sum (different weights, same fundamental mechanism)
- **Files:** `arena/snn_agent.py`, `core/network.py`

**Step 5.2:** Make homeostasis continuous (remove counter modulo):

- Replace `if counter % interval == 0` with per-step micro-correction
- α_homeo = 1 / homeo_tau → apply at every step
- **Files:** `core/basal_ganglia.py`

**Step 5.3:** Remove hard weight clipping; rely on continuous scaling + Dale's law:

- Column norm clip → redundant if continuous homeostasis works
- Keep Dale's law floor (w ≥ 0 for excitatory)
- w_v norm bound → derive from max V(s) = 1/(1-exp(-T/τ)) instead of arbitrary formula
- **Files:** `core/basal_ganglia.py`

**Step 5.4:** Derive PSP target from biophysics:

- Single-synapse EPSP ≈ g_syn × (E_rev - V_rest) / g_L
- Feldmeyer et al. (2002): unitary EPSP ≈ 0.4-2.0 mV in cortex
- Use config-level g_syn specification
- **File:** `core/neuron.py`, `core/config.py`

**Step 5.5:** Remove dead code:

- `_step_count` in snn_agent.py
- `BenchmarkConfig.solve_rate_threshold`
- Unused TrainResult methods
- **Files:** `arena/snn_agent.py`, `arena/benchmark.py`, `arena/core.py`

**Step 5.6:** Replace CartPole config with AGI-appropriate test environments:

- Remove CartPole-v1 from task_config REGISTRY
- Add multi-armed bandit with reversal (already ShiftingBanditEnv exists)
- Add two-step task (Daw et al. 2011) for model-based vs model-free
- Use existing T-Maze and PunishmentAvoidanceEnv
- **Files:** `arena/task_config.py`, `arena/environments.py`

**Step 5.7:** Fix feature_rms EMA alpha:

- Derive from homeo_tau: alpha = dt / homeo_tau
- **File:** `core/basal_ganglia.py`

**Verification:**

- Test: no "softmax", "np.random.choice(p=", "gamma \*", "intrinsic_reward" in codebase (grep)
- Test: all weight initializations traceable to biophysical parameters
- Test: all environments non-episodic-reward (no constant +1)

---

### Phase 6: Test Suite for Biophysical Correctness

Replace all existing tests with component-level biophysical verification:

**Test 6.1: AdEx dynamics** (already exists, keep)

- RS, FS, IB patterns
- w_adapt increment/decay
- Numerical stability

**Test 6.2: WTA action selection**

- Two populations → one wins consistently with asymmetric input
- Noise breaks ties when inputs equal
- NE modulates decision boundary

**Test 6.3: VTA circuit RPE**

- Constant reward → DA baseline after convergence
- Positive surprise → DA burst > baseline
- Negative surprise → DA pause < baseline
- Gain scales with reward variance

**Test 6.4: D1/D2 pathway balance**

- Positive TD → D1 weights grow, D2 stable
- Negative TD → D2 weights grow, D1 stable
- Persistent positive reward → D2 doesn't collapse (OpAL protection)

**Test 6.5: Homeostatic equilibrium**

- High input → firing rate converges to target
- Low input → firing rate converges to target
- Dark matter neurons remain silent unless NE recruitment

**Test 6.6: Sleep replay**

- SWR reactivation reproduces stored spike patterns
- STDP during replay strengthens replayed weights
- Down state silences all activity

**Test 6.7: Multi-armed bandit with reversal**

- Agent learns initial contingency
- After reversal, agent adapts within ~50 trials
- D2 pathway activates post-reversal (NoGo for previously good action)

**Test 6.8: T-Maze delayed reward**

- Agent uses working memory to hold cue through delay
- Correct choice rate > 70% after learning

**Test 6.9: Two-step task (if environment added)**

- Model-based agent shows different pattern than model-free
- After transition probability change, agent adapts immediately (model-based signature)

---

## Relevant Files

- `core/basal_ganglia.py` — D1D2Actor softmax (HACK 1), gate_eligibility (HACK 6), linear readout (HACK 5), periodic homeostasis (ISSUE 12), column norm clip (ISSUE 13)
- `arena/snn_agent.py` — TD error computation (HACK 2), Welford normalization (HACK 3), gamma (HACK 4), intrinsic reward (HACK 9), sleep_gain (ISSUE 19), dead \_step_count (ISSUE 17)
- `core/replay_buffer.py` — advantage estimation (HACK 7), sleep signal formula (ISSUE 20)
- `core/world_model.py` — curiosity z-score (HACK 8), decoder weight clip (ISSUE 18)
- `core/config.py` — BasalGangliaConfig.gamma (HACK 4), CartPole-specific comments
- `core/network.py` — concat aggregation (ISSUE 11)
- `arena/task_config.py` — CartPole config (ISSUE 16)
- `arena/core.py` — dead TrainResult methods (ISSUE 17)
- `arena/benchmark.py` — dead solve_rate_threshold (ISSUE 17)
- `core/neuron.py` — PSP target 0.15 (ISSUE 14)
- New: `core/vta.py` — VTA circuit model (Phase 3)

## Decisions

- Pure WTA replaces softmax — no probability computation for action selection
- VTA circuit replaces algebraic TD — architecture change, not parameter tuning
- gamma removed — temporal discounting from membrane τ
- No intrinsic reward — curiosity via NE/ACh modulation only
- Sleep uses SWR + STDP only — no return/advantage computation
- CartPole removed — reversal bandit + T-Maze as primary tests
- Tests verify biophysics, not benchmark scores

## Further Considerations

1. **Sparse connectivity (Phase 7):** Current dense O(N²) matrices won't scale. Plan for Small-World topology via structural plasticity — but this is Phase 7, not blocking.
2. **GPU acceleration:** NumPy CPU is fine for <10k neurons. CuPy/JAX migration planned but not urgent.
3. **Continuous time:** Current dt=1ms discretization is fine for biophysics but could be replaced by event-driven simulation for efficiency at scale.
