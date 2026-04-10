# Plan v3: Biologically Grounded SNN-AGI Backbone

## new_issues.txt Validation

| #   | Claim                                                                  | Verdict   | Key Evidence                                                                                   |
| --- | ---------------------------------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------- |
| 1   | Action-Binding Problem — theta-sweep doesn't bind action to prediction | **VALID** | WM encoder needs action input; without efference copy, theta-sweep predicts "nothing specific" |
| 2   | Astrocyte Clock Paradox — tau_ca=5000ms vs theta~167ms                 | **VALID** | Ca²⁺ changes 3.3% per theta cycle; physically cannot track per-cycle urgency                   |
| 3   | Epilepsy Risk in SWS — no inhibitory brakes                            | **VALID** | STDP + amplified noise + no GABA elevation = positive feedback avalanche                       |
| 4   | peek() synchronous anomaly — save/restore is still time-freeze         | **VALID** | In unified NetworkGraph, V(s) is continuous output; peek() unnecessary                         |
| 5   | (Self-discovered) BG integration_steps=10 is stopped time              | **VALID** | Same pattern as PC relaxation loop; MSN τ_m=25ms stable at dt=1ms                              |

## Self-Critique of v2

| Error                             | What I missed                                                                                                      |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Theta-sweep has no action binding | C4 said "encoder gets perturbation" but didn't specify WHICH action. Need efference copy from sub-threshold D1/D2. |
| Conflated fast/slow epistemic     | C4 step 6: "through astrocyte D-Serine / direct connection" — astrocyte path is 30× too slow for theta decisions   |
| SWS has no safety mechanism       | C3 just says "inject noise" with zero protective mechanisms → seizure risk                                         |
| peek() "fixed" not removed        | B4 proposed save/restore hack instead of removing entirely                                                         |
| BG integration_steps not flagged  | 10-step inner loop is identical anti-pattern to PC relaxation; was never caught                                    |

---

## Phase A: Unified NetworkGraph Architecture

**A1. BG as NetworkGraph layers**

- Register SNNDeepCritic and D1D2Actor as named layers
- Critic gets sensory feedforward; Actor gets state + fast epistemic signal (direct error neuron connection)
- Remove forced_action from D1D2Actor
- Files: core/basal_ganglia.py, core/network.py

**A2. WM as NetworkGraph layer** _(parallel with A1)_

- WorkingMemoryModule as recurrent layer; gate from neuromodulator
- Files: core/working_memory.py, core/network.py

**A3. World Model as NetworkGraph layers** _(parallel with A1)_

- ErrorNeuronLayer encoder as named layer; single decoder (no ensemble)
- Error neuron spikes → two paths: (a) direct fast→D1, (b) spike rates→astrocyte slow
- Files: core/world_model.py, core/error_neuron.py, core/network.py

**A4. Rewrite SNNAgent** _(depends on A1-A3)_

- act(): feed sensory to network.step(), read motor output from BG actor
- observe(): feed reward as spike-encoded input to critic region
- File: arena/snn_agent.py

## Phase B: Fix Core Neural Dynamics

**B1. Remove relaxation loops from PC and Pyramidal**

- Single-step dynamics in BOTH files; reference: core/error_neuron.py
- Files: core/predictive_coding.py, core/pyramidal_neuron.py

**B2. Remove integration_steps loops from BG**

- Both Critic and Actor: 1 LIF step per network.step()
- Remove integration_steps from BasalGangliaConfig
- Files: core/basal_ganglia.py, core/config.py

**B3. Remove peek() entirely**

- TD error: δ = r + γ × V_now - V_trace (EMA, τ_trace ≈ 200ms)
- No save/restore, no hypothetical forward
- File: core/basal_ganglia.py

**B4. Pattern separation in SequenceMemory**

- DG-like random projection + competitive thresholding before Hebbian outer
- File: core/sequence_memory.py

**B5. Region-aware neuromodulator**

- dict[str, NDArray] per-region errors; NE/ACh per-region; DA/5-HT global
- File: core/neuromodulator.py

## Phase C: Replace Non-Biological Patterns

**C1. Astrocyte spike-driven + dual timescale epistemic**

- update() takes spike rates not PE vectors; Ca²⁺ ∝ rate²
- Astrocyte = SLOW channel only (NMDA gain via D-Serine)
- New fast_epistemic: error_neuron.error_rate → direct → D1 excitability (no astrocyte)
- Files: core/astrocyte.py, core/world_model.py, core/basal_ganglia.py

**C2. Single decoder + vesicle noise** _(depends on C1)_

- Remove ensemble w_decode[k]; single decoder + Bernoulli(0.8) masking
- File: core/world_model.py

**C3. Biological SWS with Up/Down states** _(depends on A4)_

- Remove Experience dataclass with raw NDArrays
- Oscillator → slow oscillation ~1Hz; Up phase: noise + SWR; Down phase: global hyperpolarization
- InhibitoryPool gain elevated 2-3× during SWS
- Emergency seizure brake: firing rate > threshold → force Down state
- Files: core/replay_buffer.py, core/oscillator.py, core/interneuron.py

**C4. Theta-sweep planning with efference copy** _(depends on A3, C1)_

- Theta trough: encode current state
- Theta peak: D1/D2 sub-threshold activity → efference copy → world model encoder
- WM predicts outcome of THAT action; error neurons → fast direct → D1/D2
- ~6-7 gamma cycles per theta → temporal multiplexing of competing actions
- Files: core/oscillator.py, core/network.py, core/world_model.py, core/basal_ganglia.py

**C5. Wire receptor dose-response** _(parallel)_

- Hill equation from receptor.py applied per-layer
- Files: core/neuromodulator.py, core/receptor.py

## Phase D: Activate Dormant + Cleanup

**D1. Columnar architecture** _(depends on A4)_

- build_columnar_network() for high-dim; agent_factory selects
- Files: core/columnar.py, arena/agent_factory.py

**D2. Dead code removal**

- Relaxation loops, ensemble decoders, peek(), forced_action, integration_steps, Experience raw fields, ActiveInferenceModule.\_variance_uncertainty(), magic numbers

---

## Verification

1. PC/Pyramidal forward() 100× in sequence → PE decreases, no internal loop
2. BG Critic/Actor: no internal for-loops; spikes stable at dt=1ms
3. grep "peek" in codebase → 0 hits; TD error from V_trace
4. Astrocyte isolation: disable astrocyte → D1 modulation still works (fast channel test)
5. Efference copy: 2 distinct D1/D2 states → world model receives different actions → different predictions
6. SWS safety: 1000 cycles → firing rate never >3× baseline, consolidation measured, Down phase rate ≈ 0
7. Slow oscillation: oscillator SWS mode → ~1Hz with correct duty cycle
8. Pattern separation: 50 overlapping patterns → condition number bounded
9. CartPole reward increases over 100 episodes
10. Region NE: high error region A, low B → NE(A) > NE(B)

## Decisions

- peek() REMOVED (not fixed) — continuous V_trace
- integration_steps REMOVED — 1 step per network.step()
- Astrocyte NEVER in fast decision loop — slow NMDA only
- Fast epistemic: error_neuron.error_rate → direct synapse → D1
- Efference copy: D1/D2 sub-threshold voltages → feedforward → world model
- SWS: Up/Down mandatory, GABA elevated, seizure brake
- DA global (Schultz 1998); NE/ACh region-aware
- No backward compat; tests rewritten post-changes
