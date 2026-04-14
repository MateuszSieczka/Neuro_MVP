# PLAN: SNN Architecture — Final Consolidated Plan

## TL;DR

Codebase has strong biophysical foundations after Phase 0+1 (plan.md) and Phase 2-4 (plan_2.md) implementations. ~12 critical RL/ML hacks remain. This plan eliminates them and completes missing biophysics for AGI-grade continuous learning.

---

## AUDIT: What's Already Correct (Keep as-is)

- AdEx neuron model with exp Euler integrator (Brette & Gerstner 2005)
- Three-factor STDP with Bi & Poo asymmetric window
- Conductance-based synapses AMPA/NMDA/GABA with Mg²⁺ block (Jahr & Stevens 1990)
- InhibitoryPool with PV+ FS, Hebbian E→I / anti-Hebbian I→E STDP (Woodin 2003)
- D1/D2 MSN bistable dynamics with OpAL (Frank 2005, Shen 2008)
- VTA circuit with PPTg/VP/RMTg pathways, D2 autoreceptor gain adaptation (Eshel 2015)
- Emergent γ from PPTg τ × serotonin (Schweighofer 2008) — no explicit gamma parameter
- Astrocyte Ca²⁺ + ATP budget (De Pittà 2011)
- Population coding (Pouget 2000) with Fisher-optimal sigma=0.5×spacing
- Theta-gamma oscillator with NE/5-HT frequency modulation
- Pyramidal BAC firing + Ca²⁺ spikes (Larkum 2013, Payeur 2021)
- Predictive coding error/state separation (Rao & Ballard 1999, Bogacz 2017)
- Receptor Hill equation dose-response (Doya 2002, Surmeier 2007)
- Neuromodulator decay from pharmacokinetics (DAT, AChE, NET, SERT)
- Dale's law enforced
- WTA action selection (no softmax in D1D2Actor)
- SWR sleep replay with VTA RPE (no advantage estimation)
- Precision-weighted curiosity (no z-score)
- No additive intrinsic reward — exploration via NE/ACh
- Continuous homeostasis (per-step multiplicative scaling, Turrigiano 2008)
- Weight initialization from PSP target scaling
- SimulationContext with centralized dt/tau management

## AUDIT: What's Been Implemented from Previous Plans

### From plan.md:

- Phase 0 (all 7 steps): DONE
- Phase 1 (AdEx, exp Euler, ATP, tonic DA): DONE
- FIX 1-11 (diagnostic fixes): ALL DONE

### From plan_2.md:

- Phase 2 (emergent action selection): PARTIAL — softmax removed from D1D2Actor ✓, but gate_eligibility() still explicit zeroing ✗, ActiveInferenceModule still uses softmax ✗
- Phase 3 (neural TD): DONE — VTA circuit, Welford removed, gamma emergent
- Phase 4 (biological sleep): DONE — SWR replay, no advantage, curiosity precision-weighted
- Phase 5 (structural cleanup): PARTIAL — sum aggregation ✓, continuous homeostasis ✓, some dead code remains ✗
- Phase 6 (test suite): NOT DONE

---

## REMAINING RL/ML HACKS — Must Eliminate

### HACK A: ε-greedy exploration in snn_agent.py

**Location:** `_BGFacade.compute_exploration_noise()` + `act()` override
**Formula:** `ε = max(min_exploration, (1-DA)×(1-5HT))` then `if rand() < ε: action = random`
**Problem:** Algorithmic override of neural action selection. Biology doesn't roll dice to decide whether to explore.
**Replace with:** STN-GPe hyperdirect pathway. Low DA → STN tonic activity increases → global inhibition of all action channels via GPe → raises decision threshold → forces longer integration → membrane noise breaks symmetry → exploratory action. Reference: Frank (2006) "Hold your horses"; Bogacz & Gurney (2007).

### HACK B: gate_eligibility() explicit zeroing

**Location:** `core/basal_ganglia.py`, D1D2Actor.gate_eligibility()
**Problem:** Algorithmic zeroing of non-selected actions' eligibility traces. Not neural.
**Replace with:** Lateral inhibition from InhibitoryPool already suppresses losing action channels' activity. Low post-synaptic activity → low eligibility trace (since eligibility ∝ outer(pre, v_normalized)). The voltage-based eligibility naturally decays for suppressed channels. Remove explicit zeroing; InhibitoryPool cross-action inhibition already handles this.

### HACK C: ActiveInferenceModule softmax action selection

**Location:** `core/basal_ganglia.py`, ActiveInferenceModule
**Problem:** Uses `exp(G/T) / sum(exp)` + `np.random.choice(p=probs)` for EFE-based selection.
**Replace with:** EFE scores injected as additional excitatory input to D1D2Actor's corresponding action MSN populations. Higher G(a) → more excitation to action a's D1 channel → natural WTA selection. Reference: Pezzulo et al. (2018) "Hierarchical active inference."

### HACK D: Attention softmax normalization

**Location:** `core/attention.py`, `compute()`
**Formula:** `exp(td_shifted / T) / sum(exp)` for top-down attention
**Problem:** Softmax is a mathematical function not implemented by neurons.
**Replace with:** k-WTA competitive dynamics among attention columns. Highest-activated columns suppress others via lateral inhibition. Use existing CompetitiveLIFLayer mechanism applied to attention weights. Reference: Reynolds & Heeger (2009) normalization model of attention IS divisive normalization, which emerges from lateral inhibition.

### HACK E: Working memory sigmoid dual-gate

**Location:** `core/working_memory.py`, gate()
**Formula:** `σ(8×(ACh-thresh)) × σ(8×(DA-thresh))`
**Problem:** Sigmoid with hardcoded gain=8.0 is a DL activation function. Gate gain undocumented.
**Replace with:** WM gate neurons (separate MSN population). ACh and DA levels modulate WM gate MSN excitability via receptor dynamics (already implemented in receptor.py). Gate fires/doesn't fire based on spiking threshold, not sigmoid. Reference: O'Reilly & Frank (2006) — the gating IS supposed to be MSN-like.

### HACK F: World model mental_rehearsal explicit gamma=0.99

**Location:** `core/world_model.py`, mental_rehearsal()
**Formula:** `total_epistemic += (0.99^step) × step_epistemic`
**Problem:** Explicit exponential discount in imagination is an RL parameter.
**Replace with:** Natural decay from rehearsal integration window. Each rehearsal step uses the encoder's membrane dynamics — information naturally decays with τ. Remove explicit gamma multiplication.

---

## MISSING BIOPHYSICS — Must Add

### BIO 1: Dual-exponential synaptic kinetics

**Location:** `core/synapse.py`, SynapticChannels
**Problem:** Instantaneous rise for all channels. AMPA should have τ_rise≈0.4ms, NMDA τ_rise≈10ms, GABA-A τ_rise≈0.3ms, GABA-B τ_rise≈30ms.
**Fix:** Add rise-time state variables. Formula: `g(t) = g_peak × (exp(-t/τ_decay) - exp(-t/τ_rise)) × norm_factor`. Reference: Destexhe et al. (1998).

### BIO 2: Working memory → AdEx

**Location:** `core/working_memory.py`, forward()
**Problem:** Uses simple LIF: `v = v * mem_decay + (v_rest + I) * (1-mem_decay)`. No spike-triggered adaptation (w_adapt), no exponential spike initiation.
**Fix:** Replace with AdEx integration using ctx.exp_euler_step() as in neuron.py. WM neurons should use PFC-like parameters: τ_w=300ms (slow adaptation for sustained firing), a=2nS, b=20pA (mild adaptation to allow persistent activity). Reference: Durstewitz et al. (2000) "Neurocomputational models of working memory."

### BIO 3: Conductance-based competitive layer inhibition

**Location:** `core/competitive_layer.py`, \_apply_proactive_inhibition()
**Problem:** Inhibition formula `i_inh = gap × N/k × strength` is heuristic, not conductance-based.
**Fix:** Use conductance driving force: `I_inh = g_inh × (V - E_inh)` where g_inh derived from InhibitoryPool-like mechanism. This is self-limiting (as V approaches E_inh, inhibition naturally decreases). Reference: Same as InhibitoryPool — Brunel & Wang (2003).

### BIO 4: PSP target from single-synapse conductance

**Location:** `core/neuron.py`, weight init `gap * 0.15`
**Problem:** The 0.15 factor is not derived from biophysics.
**Fix:** Single cortical synapse: g_syn ≈ 0.5-1.0 nS (Feldmeyer et al. 2002). EPSP = g_syn × (E_exc - V_rest) / g_L ≈ 0.7 × 70 / 30 ≈ 1.6 mV. With gap=15mV: ratio = 1.6/15 ≈ 0.11. So 0.15 is reasonable but should be derived from config parameters, not hardcoded.

### BIO 5: Pyramidal apical electrotonic filtering

**Location:** `core/pyramidal_neuron.py`, forward()
**Problem:** Apical input affects soma instantaneously. Real apical dendrites have τ_propagation ≈ 5-10ms delay and low-pass filtering.
**Fix:** Add delay buffer (5ms) and low-pass filter (τ_cable ≈ 10ms) between apical input and somatic effect. Reference: Stuart & Spruston (1998) "Determinants of voltage attenuation in neocortical pyramidal neuron dendrites."

### BIO 6: Oscillator proper PAC

**Location:** `core/oscillator.py`
**Problem 1:** PAC is amplitude modulation `1 - d + d×(1+cos(θ))/2`, not true cross-frequency coupling.
**Problem 2:** Encoding at theta trough (π/2 to 3π/2) may be reversed — Hasselmo et al. (2002): encoding at theta PEAK in EC, retrieval at trough.
**Fix 1:** Gamma oscillator should phase-reset at each theta trough. γ_phase = mod(γ_phase, 2π) at θ crossing. This creates genuine phase-amplitude coupling where gamma bursts are locked to theta phase. Reference: Lisman & Jensen (2013).
**Fix 2:** Verify encoding/retrieval phase assignment against literature. Adjust to: encoding → theta peak (0 to π), retrieval → theta trough (π to 2π).

### BIO 7: Predictive coding relaxation

**Location:** `core/predictive_coding.py`, forward()
**Problem:** Single-step update. Classic PC requires iterative relaxation to minimize free energy (Rao & Ballard 1999 use multiple update steps per input presentation).
**Fix:** Add configurable n_relax_steps (default 3-5) inner loop within forward(). Each step: error neurons update → state neurons update → predictions regenerate. This converges belief toward posterior. Reference: Bogacz (2017) "A tutorial on the free-energy framework."

---

## MODERATE ISSUES — Should Fix

### MOD 1: Receptor effect aggregation

**Location:** `core/receptor.py`, aggregate_receptor_effects()
**Problem:** Simple element-wise sum of receptor effects (DL-style feature fusion).
**Fix:** Receptor effects should interact non-linearly. D1 and D2 are on different MSN populations (not summed). M1 modulates excitability multiplicatively. Use: `gain = Π(1 + effect_i)` for multiplicative interaction within same cell type.

### MOD 2: Neuromodulator stagnation factor

**Location:** `core/neuromodulator.py`
**Problem:** CV-based stagnation detection (std/mean of TD errors) is a statistical operation.
**Fix:** ACC (anterior cingulate) signals could be derived from sustained high prediction error (error neuron persistent activation). If error rate stays elevated for N theta cycles → increase NE (volatility signal). Reference: Behrens et al. (2007) "Learning the value of information in an uncertain world."

### MOD 3: Seizure brake mechanism

**Location:** `core/network.py`
**Problem:** Hard reset `v[:] = -75.0` is brute-force.
**Fix:** Mimic depolarization block: when too many neurons fire, extracellular K⁺ rises → reduced driving force → natural silencing. Or: astrocyte ATP depletion → threshold rise. Both already partially implemented — strengthen the astrocyte pathway instead of using hard reset.

### MOD 4: Sequence memory fixed DG projection

**Location:** `core/sequence_memory.py`
**Problem:** w_dg is random and fixed (RandomState(42)). Not biologically plastic.
**Fix:** DG granule cells DO learn input patterns. Add Hebbian plasticity to w_dg with competitive k-WTA: `dw_dg = lr × outer(sparse_pattern, input)`. Reference: Rolls (2013) — DG pattern separation uses competitive learning.

---

## CLEANUP ITEMS

### CLN 1: Dead code removal

- `_SEIZURE_DOWN_MS` in oscillator.py — defined, never used
- `working_memory.set_ne_level()` — declared as `pass`
- Concat mode in NetworkGraph — if fully replaced with sum, remove concat path
- `encode_value()` in spike_encoder.py — likely unused

### CLN 2: Magic numbers requiring derivation

- `_gate_gain = 8.0` in neuromodulator.py and working_memory.py — derive from receptor binding kinetics or remove in favor of spiking gate
- `3× baseline` seizure threshold — derive from physiological K⁺ ceiling
- `salience_threshold = 0.5` in sequence_memory — derive from NE level
- `theta_window = 8, episode_window = 50` in sequence_memory — derive from oscillator actual periods

### CLN 3: Astrocyte spike cost fix

**Location:** `core/astrocyte.py`
**Problem:** `sqrt(rates²) = |rates|` — redundant computation. Also `rates²` as energy proxy is a hack.
**Fix:** ATP cost should be proportional to spike count directly (Na⁺/K⁺-ATPase: ~10⁹ ATP per spike). Use `rate × spike_cost` directly.

### CLN 4: Synapse sign convention

**Location:** `core/synapse.py`
**Problem:** Confusing comment about sign of excitatory current. Document clearly: positive current = depolarizing.

---

## IMPLEMENTATION ORDER

### Phase 2: Emergent Exploration & Credit Assignment

_Dependencies: None. Blocks Phase 7._

1. **HACK A** — Remove ε-greedy. Add STN-GPe hyperdirect pathway to D1D2Actor. Low DA → STN activity → global inhibition → longer integration → noise-driven exploration.
   - Files: `core/basal_ganglia.py` (D1D2Actor), `arena/snn_agent.py` (remove compute_exploration_noise, remove if rand() < ε)
2. **HACK B** — Remove gate_eligibility(). Rely on voltage-based eligibility natural decay for suppressed channels.
   - Files: `core/basal_ganglia.py` (remove gate_eligibility method), `arena/snn_agent.py` (remove call)

3. **HACK E** — WM gate → spiking MSN. Remove sigmoid gate, use gate neuron population.
   - Files: `core/working_memory.py` (gate method, forward)

4. **CLN 2 partial** — Remove `_gate_gain = 8.0` from working_memory (replaced by spiking gate).

**Verification:**

- Test: ShiftingBandit — agent still explores after reversal (NE-driven, not ε)
- Test: Action entropy correlates with DA level (low DA → high entropy)
- Test: WM gate opens/closes based on combined ACh+DA (not sigmoid)

### Phase 3: Synaptic & Inhibitory Biophysics

_Dependencies: None. Parallel with Phase 2._

1. **BIO 1** — Add dual-exponential kinetics to SynapticChannels.
   - Files: `core/synapse.py`, `core/config.py` (add tau_rise fields)

2. **BIO 3** — Conductance-based inhibition in CompetitiveLIFLayer.
   - Files: `core/competitive_layer.py`

3. **BIO 4** — Derive PSP target from config conductance parameters.
   - Files: `core/neuron.py`, `core/config.py`

4. **MOD 1** — Multiplicative receptor integration.
   - Files: `core/receptor.py`

**Verification:**

- Test: EPSP waveform matches Feldmeyer et al. (2002) — rise 0.4ms, decay 2ms for AMPA
- Test: NMDA current peaks at 10ms post input (not instantaneous)
- Test: Inhibition self-limits near E_inh (-75mV)

### Phase 4: Neural Module Consistency

_Dependencies: Phase 2 (WM gate change)._

1. **BIO 2** — Working memory forward() → AdEx.
   - Files: `core/working_memory.py`

2. **BIO 7** — PC relaxation loop (n_relax_steps=3).
   - Files: `core/predictive_coding.py`

3. **BIO 5** — Pyramidal apical delay + filtering.
   - Files: `core/pyramidal_neuron.py`, `core/config.py` (add tau_cable, delay_ms)

4. **MOD 4** — Plastic DG projection in sequence memory.
   - Files: `core/sequence_memory.py`

**Verification:**

- Test: WM sustains pattern for >500ms without input (attractor stability)
- Test: PC error converges within 3-5 relaxation steps
- Test: Apical input delayed by ~5ms at soma

### Phase 5: Oscillator & Temporal

_Dependencies: None. Parallel with Phase 3-4._

1. **BIO 6a** — Gamma phase-reset at theta trough for proper PAC.
   - Files: `core/oscillator.py`

2. **BIO 6b** — Verify/correct encoding-retrieval phase assignment.
   - Files: `core/oscillator.py`, `core/sequence_memory.py`

3. **CLN 2 partial** — Derive sequence_memory theta_window from oscillator.
   - Files: `core/sequence_memory.py`

4. Unify SWS/awake oscillator (single mode with parameterized frequency).
   - Files: `core/oscillator.py`

**Verification:**

- Test: Gamma power peaks at theta trough (spectral analysis of 1s simulation)
- Test: Encoding occurs at correct theta phase per literature
- Test: SWS oscillation ~1Hz with Up/Down states

### Phase 6: Active Inference & World Model

_Dependencies: Phase 2 (HACK A done for clean exploration)._

1. **HACK C** — Remove softmax from ActiveInferenceModule. EFE → D1 current injection.
   - Files: `core/basal_ganglia.py` (ActiveInferenceModule)

2. **HACK F** — Remove explicit gamma from mental_rehearsal.
   - Files: `core/world_model.py`

3. **HACK D** — Replace attention softmax with WTA/divisive normalization.
   - Files: `core/attention.py`

4. Complete free_energy.py with minimal generative model.
   - Files: `core/free_energy.py`

**Verification:**

- Test: EFE scores bias action selection toward informative states
- Test: Mental rehearsal epistemic value decays with depth (natural τ)
- Test: Attention focuses on columns with highest prediction error

### Phase 7: Cleanup & Derivation

_Dependencies: All previous phases._

1. **CLN 1** — Remove all dead code
2. **CLN 3** — Fix astrocyte spike cost
3. **CLN 4** — Fix synapse sign convention
4. **MOD 2** — Replace stagnation CV with error neuron persistence
5. **MOD 3** — Replace seizure brake with astrocyte ATP pathway
6. All remaining magic numbers

**Verification:**

- grep: no softmax, no np.random.choice(p=, no gate_eligibility, no explicit gamma
- All constants traceable to biophysical parameters or cited papers

### Phase 8: Comprehensive Test Suite

_Dependencies: All phases complete._

**Component tests:**

1. AdEx dynamics: RS, FS, IB spike patterns match Brette & Gerstner Fig. 1
2. Synaptic kinetics: EPSP/IPSP waveforms match literature
3. WTA action selection: asymmetric input → consistent winner; equal input → noise-broken
4. VTA RPE: constant reward → DA baseline; surprise → burst; omission → pause
5. D1/D2 balance: positive TD → D1 grows; negative TD → D2 grows; D2 stable under constant +reward
6. Homeostasis: drives firing rate to target from above and below
7. WM: sustains pattern >500ms; gate opens on ACh+DA conjunction
8. PC: error converges in relaxation loop
9. Oscillator: PAC spectral verification; phase-encoding test
10. Sleep: SWR reactivation improves value estimates

**Integration tests:** 11. ShiftingBandit reversal: agent adapts within ~50 trials post-reversal 12. TMaze: correct choice >70% after learning (WM-dependent) 13. PunishmentAvoidance: D2/NoGo pathway inhibits harmful action 14. Corridor: delayed reward propagated by critic temporal integration 15. Continuous learning: no catastrophic forgetting across task switches (sequential ShiftingBandit → TMaze)

---

## Decisions & Scope

**IN SCOPE:**

- Remove all ε-greedy, softmax, sigmoid gating, explicit gamma
- Add dual-exponential synaptic kinetics
- AdEx for all neuron types (including WM)
- Proper PAC and phase correction
- PC relaxation loop
- STN-GPe exploration circuit
- Voltage-based eligibility replaces gate_eligibility

**OUT OF SCOPE (future phases):**

- Sparse O(Nk) connectivity (Small-World topology) — future Phase 9
- GPU acceleration (CuPy/JAX) — future when >10k neurons
- Event-driven simulation — optimization, not correctness
- Full variational inference (MCMC/message passing) — free energy skeleton sufficient for now
- VTA spiking conversion — current rate-coded VTA with correct topology is adequate

**PRINCIPLES:**

- P1: Physics over algorithm — behavior emerges from equations, not if/else
- P2: Continuous time — no "episode" concept in core (only in arena harness)
- P3: Thermodynamic constraints — ATP limits computation, not flags
- P4: Every constant traceable to paper or derivation
