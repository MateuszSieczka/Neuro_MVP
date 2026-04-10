# Unified Plan v4: Biologically Grounded SNN-AGI — Complete Fix

## Overview

This plan merges the original plan v3 with all gaps found during full codebase audit.
Six phases (A→F), strict dependency ordering, exact file/symbol references.
Addresses: physics (Ohm's law), mathematics (STDP, FEP), biology (spiking, neuromodulation), philosophy (active inference completeness).

No backward compatibility. All tests rewritten post-changes.

---

## Phase A: Fix Foundational Physics & Math

Everything else depends on synaptic dynamics and plasticity rules being correct.
All steps in this phase are independent of each other (parallel OK).

### A1. Reversal potentials in synaptic current (Ohm's law fix)

**Problem**: `compute_current()` in `core/synapse.py` (L106-128) computes `I = g_exc - g_inh`. Correct physics: `I = g_exc × (V - E_exc) + g_inh × (V - E_inh)`.
`e_exc=0.0` and `e_inh=-75.0` are defined in `core/config.py` (L212-213) but never used.

**Change**:

- `SynapticChannels.compute_current(v_post)` already receives `v_post` — use it:
  - `i_exc = (g_ampa + g_nmda * mg_block) * (v_post - cfg.e_exc)`
  - `i_inh = (g_gaba_a + g_gaba_b) * (v_post - cfg.e_inh)`
  - `return i_exc + i_inh` (note: sign change — `e_inh` is negative so `(V - E_inh)` handles polarity)
- Verify caller in `core/neuron.py` `LIFLayer.forward()` passes current membrane voltage to `compute_current()`

**Files**: `core/synapse.py` (L106-128), `core/config.py` (L212-213 — already correct, just wire)

### A2. Event-based STDP traces (spike events, not rates)

**Problem**: `core/neuron.py` (L140): `self.x_pre += np.clip(pre_f32, 0.0, 1.0)` increments pre-trace by continuous rate value. Bi & Poo (2001) require discrete spike events: `x_pre += δ(t - t_spike)`.

**Change**:

- In `LIFLayer.forward()`:
  - Convert: `self.x_pre += (pre_f32 > 0.5).astype(np.float32)` (binary spike threshold)
  - OR better: use the actual binary spike output from the source layer (0 or 1), not the rate-coded `pre_f32`
- Similarly verify post-trace: `self.x_post` should increment by 1.0 only on actual spike, not continuous value
- Check all STDP trace updates in `core/error_neuron.py` (L159-167) and `core/pyramidal_neuron.py` for the same pattern

**Files**: `core/neuron.py` (L95, L131, L140), `core/error_neuron.py` (L159-167), `core/pyramidal_neuron.py`

### A3. STDP causal timing window (±20ms)

**Problem**: Eligibility traces in BG (`core/basal_ganglia.py` L509-517), error neurons (`core/error_neuron.py` L159-167), and LIF layer accumulate correlations from ANY timestep within decay window. Bi & Poo (2001) require |Δt| ≤ 20ms for LTP/LTD.

**Change**:

- Add a timing mask to eligibility updates: only accumulate outer product when pre-spike and post-spike co-occur within ±20 timesteps (±20ms at dt=1ms)
- Implementation: maintain `time_since_last_spike` per neuron (pre and post). Gate eligibility increment:
  ```
  dt_pre_post = time_since_post_spike - time_since_pre_spike
  ltp_mask = (0 < dt_pre_post) & (dt_pre_post <= 20)  # pre before post
  ltd_mask = (-20 <= dt_pre_post) & (dt_pre_post < 0)   # post before pre
  ```
- Apply to all three-factor STDP sites: LIFLayer, ErrorNeuronLayer, SNNDeepCritic, D1D2Actor

**Files**: `core/neuron.py`, `core/error_neuron.py`, `core/basal_ganglia.py`

### A4. Burst boost synapse-specific (not broadcast)

**Problem**: `core/pyramidal_neuron.py` (L229): `self.e *= boost[np.newaxis, :]` broadcasts burst boost to ALL synapses uniformly. Biologically, burst plasticity is synapse-specific, modulated by local Ca²⁺ influx × presynaptic activity.

**Change**:

- Replace broadcast: `self.e *= (1.0 + (burst_factor - 1.0) * burst_f * self.x_pre[:, np.newaxis])`
- This makes boost proportional to presynaptic trace (active synapses get more boost)

**Files**: `core/pyramidal_neuron.py` (L228-229)

---

## Phase B: Remove Anti-Patterns

All steps independent (parallel OK), except B6 depends on B2.

### B1. Remove relaxation loops from PC and Pyramidal

**Problem**: `core/predictive_coding.py` (L63-79) and `core/pyramidal_neuron.py` (L116-128) have inner convergence loops that break spiking paradigm.

**Change**: Replace iterative gradient descent with single-step dynamics matching `core/error_neuron.py` pattern:

- One dt step: `v += rate * (ach * error_gradient + (1-ach) * top_down)`
- Remove `for _ in range(max_relaxation_steps)` loop
- Remove convergence threshold check
- Remove `max_relaxation_steps` from config

**Files**: `core/predictive_coding.py`, `core/pyramidal_neuron.py`, `core/config.py`

### B2. Remove integration_steps from BG

**Problem**: `core/basal_ganglia.py` (L204, L440): `for _ in range(cfg.integration_steps):` runs 10 LIF steps per network step. MSN τ_m=25ms is stable at dt=1ms.

**Change**:

- Remove loop; single LIF step per `forward()` call
- Remove `integration_steps: int = 10` from `BasalGangliaConfig` in `core/config.py` (L1131)
- Apply to both `SNNDeepCritic.forward()` and `D1D2Actor.forward()`

**Files**: `core/basal_ganglia.py` (L204, L440), `core/config.py` (L1131)

### B3. Remove peek() → V_trace EMA

**Problem**: `SNNDeepCritic.peek()` (L248) freezes time for hypothetical forward pass.

**Change**:

- Delete `peek()` method entirely
- Add EMA value trace: `self.v_trace = self.v_trace * ema_decay + (1 - ema_decay) * current_v` (τ_trace ≈ 200ms → decay = exp(-1/200))
- TD error: `δ = r + γ × V_now - V_trace` (no save/restore)
- Remove `BGSnapshot` dataclass and `snapshot_traces()`/`restore_traces()` from BG
- Update `arena/snn_agent.py` observe() to use `bg.critic.v_trace` instead of `bg.last_v`

**Files**: `core/basal_ganglia.py` (L248, L783-807), `arena/snn_agent.py`

### B4. Remove softmax fallback in BG Actor

**Problem**: `core/basal_ganglia.py` (L491-499): when MSN firing rate < 1e-6, falls back to voltage-based softmax. Bypasses spike competition.

**Change**:

- Remove the fallback branch entirely
- If total rate near zero (network silent), use uniform random action (exploration)
- Long-term fix (Phase C): proper thalamic gating via InhibitoryPool in NetworkGraph handles silence biologically

**Files**: `core/basal_ganglia.py` (L491-499)

### B5. Remove forced_action from D1D2Actor

**Problem**: `D1D2Actor.forward()` (L406, L503-505) accepts `forced_action` parameter — non-biological override.

**Change**: Remove parameter from signature and all usage. If curriculum needed, implement via neuromodulatory bias (elevated DA for specific action), not direct override.

**Files**: `core/basal_ganglia.py` (L406, L503-505)

### B6. BG Actor: REINFORCE → DA-modulated STDP _(depends on B2)_

**Problem**: `D1D2Actor` (L509-517) uses `np.outer(state_f32, grad_log_pi)` — this is REINFORCE policy gradient, not biological STDP. D2 trace: `self.e_d2 = ... - np.outer(state_f32, grad_log_pi)`.

**Change**:

- D1 pathway: standard Hebbian STDP modulated by positive DA (LTP):
  `e_d1 = e_d1 * trace_decay + x_pre[:, None] * spikes_d1[None, :]`
  Weight update: `Δw_d1 = lr × max(td_error, 0) × e_d1`
- D2 pathway: anti-Hebbian STDP modulated by negative DA (LTD):
  `e_d2 = e_d2 * trace_decay + x_pre[:, None] * spikes_d2[None, :]`
  Weight update: `Δw_d2 = lr × max(-td_error, 0) × e_d2`
- Remove `grad_log_pi` computation entirely — this belongs to policy gradient, not SNN
- This matches Frank (2005) Go/NoGo model: D1 strengthens chosen action on reward, D2 strengthens inhibition on punishment

**Files**: `core/basal_ganglia.py` (L509-540)

---

## Phase C: Unified NetworkGraph Architecture

A1-A3 parallel. A4 depends on all three.

### C1. BG as NetworkGraph layers _(from original A1)_

- Register `SNNDeepCritic` and `D1D2Actor` as named layers in NetworkGraph
- Critic receives sensory feedforward from spike encoder layer
- Actor receives state spikes + fast epistemic signal (direct error neuron connection from world model)
- Both now receive **spike-encoded** input, not raw float arrays

**Files**: `core/basal_ganglia.py`, `core/network.py`

### C2. WM as NetworkGraph layer _(from original A2, parallel with C1)_

- `WorkingMemoryModule` registered as recurrent layer
- **Fix hard gate → soft gate**: replace boolean conjunction in `gate()` (L93-102) with multiplicative soft gating:
  `self.gate_signal = sigmoid(ach - ach_thresh) * sigmoid(da - da_thresh)`
  Use `gate_signal` to scale input current, not binary on/off
- Gate signal from neuromodulator bus (Phase D)

**Files**: `core/working_memory.py` (L93-102), `core/network.py`

### C3. World Model as NetworkGraph layers _(from original A3, parallel with C1)_

- `ErrorNeuronLayer` encoder as named layer
- Single decoder (no ensemble — moved to Phase E)
- Error neuron spikes → two paths:
  (a) fast: `error_rate` direct synapse → D1 excitability
  (b) slow: spike rates → astrocyte Ca²⁺ → D-Serine → NMDA gain

**Files**: `core/world_model.py`, `core/error_neuron.py`, `core/network.py`

### C4. Rewrite SNNAgent _(depends on C1-C3)_

- `act()`: spike-encode state via `PoissonEncoder` (currently dead code at `core/network.py` L86 — activate it), feed to `network.step()`, read motor output from BG actor layer
- `observe()`: spike-encode reward signal as firing rate of a "reward neuron" population, feed to critic region through NetworkGraph
- Remove `_augment_state()` concatenation hack — WM content flows through NetworkGraph connections, not manual concatenation
- Remove direct scalar TD error flow — TD error emerges from critic spike dynamics + V_trace

**Files**: `arena/snn_agent.py` (L132-278), `core/network.py` (L86 — activate PoissonEncoder)

---

## Phase D: Fix Neuromodulation, Memory & FEP

All steps independent unless noted.

### D1. Region-aware neuromodulator _(from original B5)_

- Replace scalar `da_level`, `ach_level`, `ne_level`, `serotonin` with `dict[str, float]` keyed by region name
- DA and 5-HT: global (Schultz 1998, Doya 2002)
- NE and ACh: per-region (locus coeruleus and basal forebrain project regionally)
- Remove dead properties: `tau_compression` (L213), `membrane_reactivity` (L217)

**Files**: `core/neuromodulator.py`

### D2. Wire receptor dose-response _(from original C5)_

- `compute_layer_modulation()` in `core/receptor.py` (L164) is defined but **never called**
- Wire it: each NetworkGraph layer calls `compute_layer_modulation(transmitter_levels, layer_receptor_densities)` in its `forward()` pass
- Populate receptor densities per layer using predefined profiles: `CORTICAL_L4_RECEPTORS`, `STRIATUM_D1_RECEPTORS` etc. from `core/config.py` (L838-867) — currently dead, activate them
- Apply Hill equation effects to: membrane gain, STDP learning rate, synaptic conductance

**Files**: `core/receptor.py` (L164), `core/config.py` (L838-867), `core/neuron.py`, `core/network.py`

### D3. Free energy: compute ambiguity from world model

**Problem**: `core/free_energy.py` (L75): `ambiguity=0.0` default makes FEP incomplete.

**Change**:

- Compute ambiguity = expected sensory entropy under action policy:
  `ambiguity(a) = H[p(o|s, a)]` where H is entropy of world model predictions
- Source: world model decoder prediction variance (after single-decoder refactor in E2)
- Pass computed ambiguity to `expected_free_energy()` in `ActiveInferenceModule`
- Also: wire `FreeEnergyConfig` (L50-56, currently dead) to parametrize precision broadcasting

**Files**: `core/free_energy.py` (L50-56, L73-95), `core/world_model.py`, `core/basal_ganglia.py` (ActiveInferenceModule)

### D4. Oscillator ↔ SequenceMemory phase coupling

**Problem**: `core/sequence_memory.py` (L221-222) uses hardcoded `theta_window=8`, `episode_window=50` ticks. Oscillator frequency changes from NE/5-HT don't propagate.

**Change**:

- `SequenceMemory.observe()` receives oscillator phase, not dt counter
- Gate theta-level learning only during encoding window (`π/2 < φ_theta < 3π/2`)
- Compute pooling window dynamically from current theta frequency: `theta_ticks = round(1 / (f_theta * dt))`
- Wire `theta_reset` flag from `core/oscillator.py` (L100-104) — currently produced but **never consumed**
- Also remove dead methods: `get_associated_neurons()` (L75-82), `get_temporal_clusters()` (L84-104)

**Files**: `core/sequence_memory.py` (L75-104, L221-222), `core/oscillator.py`

### D5. Pattern separation in SequenceMemory _(from original B4)_

- DG-like random projection + competitive thresholding before Hebbian outer product
- Prevents attractor collapse in transition matrix

**Files**: `core/sequence_memory.py`

---

## Phase E: Replace Non-Biological Patterns

### E1. Astrocyte spike-driven + fix Neumann BC _(from C1 + new BC fix)_

**Problem 1 (original)**: `update()` takes PE vectors, not spike rates. Ca²⁺ should accumulate from spike rates².
**Problem 2 (new)**: Gap junction diffusion (L123-130) uses one-sided differences at boundaries, creating spurious Ca²⁺ edge accumulation.

**Change**:

- `update()` takes spike rates (not PE vectors): `Ca += accumulation * rate²`
- Fix Neumann BC with ghost-point method:
  `laplacian[0] = 2 * (calcium[1] - calcium[0])`
  `laplacian[-1] = 2 * (calcium[-2] - calcium[-1])`
- Astrocyte = SLOW channel only (NMDA gain via D-Serine)
- New `fast_epistemic`: `error_neuron.error_rate` → direct synapse → D1 excitability (no astrocyte involvement)

**Files**: `core/astrocyte.py` (L123-130), `core/world_model.py`, `core/basal_ganglia.py`

### E2. Single decoder + vesicle noise _(from C2, depends on E1)_

- Remove ensemble `w_decode[k]`; single decoder + Bernoulli(0.8) masking (synaptic vesicle release probability)
- Ambiguity now computed from single decoder prediction variance, not ensemble disagreement

**Files**: `core/world_model.py`

### E3. Biological SWS with Up/Down states _(from C3, depends on C4)_

**Changes**:

- Remove `Experience` dataclass raw NDArray fields (`core/replay_buffer.py` L24-43) — replace with spike-time representation: `spike_trains: list[NDArray]`, `synaptic_fingerprint: dict[str, NDArray]`
- Fix advantage weighting operator precedence (`core/replay_buffer.py` L209-212):
  `pos_ratio = (0.5 * 200) / max(n_exp, 1)` — explicit parentheses
  `neg_ratio = (0.1 * 200) / max(n_exp, 1)`
- Oscillator → slow oscillation ~1Hz mode for SWS:
  Up phase: noise + SWR replay; Down phase: global hyperpolarization
- `InhibitoryPool` gain elevated 2-3× during SWS (GABA surge)
- Emergency seizure brake: if mean firing rate > 3× baseline → force Down state for 200ms
- Add soft weight clipping per update: `dw = clip(dw, -0.1, 0.1)` to prevent single-step divergence

**Files**: `core/replay_buffer.py` (L24-43, L93-117, L209-212), `core/oscillator.py`, `core/interneuron.py`

### E4. Theta-sweep planning with efference copy _(from C4, depends on C3 + E1)_

- Theta trough: encode current state
- Theta peak: D1/D2 sub-threshold activity → efference copy → world model encoder
- WM predicts outcome of THAT specific action; error neurons → fast direct → D1/D2
- ~6-7 gamma cycles per theta → temporal multiplexing of competing actions
- Wire oscillator `theta_reset` and `gamma_reset` to gate planning cycles

**Files**: `core/oscillator.py`, `core/network.py`, `core/world_model.py`, `core/basal_ganglia.py`

---

## Phase F: Activate Dormant + Cleanup

### F1. Columnar architecture _(from D1, depends on C4)_

- `build_columnar_network()` for high-dim environments; `agent_factory` selects based on `env.state_size`
- Wire `SpatialAttentionController` (`core/attention.py`) to columnar layers
- Wire `CompetitiveLIFLayer` (`core/competitive_layer.py`) as k-WTA within columns

**Files**: `core/columnar.py`, `core/attention.py`, `core/competitive_layer.py`, `arena/agent_factory.py`

### F2. Dead code removal

Remove:

- Relaxation loops (PC, Pyramidal)
- Ensemble decoders (world_model)
- `peek()`, `BGSnapshot`, `snapshot_traces()`, `restore_traces()`
- `forced_action` parameter
- `integration_steps` config field
- `Experience` raw NDArray fields
- `ActiveInferenceModule._variance_uncertainty()` if unused
- `PoissonEncoder` dead instance in network.py L86 (after activation in C4)
- `FreeEnergyConfig` (after activation in D3)
- `ReceptorProfile` predefined profiles (after activation in D2)
- Dead neuromod properties: `tau_compression`, `membrane_reactivity`
- Dead sequence memory methods: `get_associated_neurons()`, `get_temporal_clusters()`
- `_last_state` in BG Actor (L388, stored but never read)
- `internal_actions` dead branch (L513-517)
- Hardcoded magic numbers: replace with config fields
  - `_scaling_interval = 1000` in neuron.py (L127) → `NeuronConfig.scaling_interval`
  - Precision clamp `[0.1, 10.0]` in network.py (L259) → config
  - NE factor `3.0` in neuron.py (L250) → config
  - ACh factor `1.0` in neuron.py (L254) → config

### F3. Rewrite all tests

All existing tests assume old API. Rewrite for:

- New spike-encoded BG interface
- Removed peek/snapshot
- Single-step dynamics (no loops)
- Soft WM gating
- Reversal potential physics
- Event-based STDP

**Files**: `tests/` (all files)

---

## Dependency Graph

```
Phase A (all parallel):
  A1 ─┐
  A2 ─┼─→ Phase B (all parallel except B6→B2):
  A3 ─┤     B1 ─┐
  A4 ─┘     B2 ─┼─→ B6
            B3 ─┤
            B4 ─┤
            B5 ─┘─→ Phase C:
                      C1 ─┐
                      C2 ─┼─→ C4 ─→ Phase E:
                      C3 ─┘          E1 ─→ E2
                                     E3 (depends on C4)
                    Phase D           E4 (depends on E1 + C3)
                    (parallel):
                      D1 ─┐
                      D2 ─┤
                      D3 ─┼─→ Phase F:
                      D4 ─┤     F1 (depends on C4)
                      D5 ─┘     F2 (after all)
                                F3 (after all)
```

---

## Verification Checklist

### Physics & Math

1. `synapse.compute_current()`: mock V=-70mV, set g_ampa=1 → I = 1.0 × (−70 − 0) = −70 (driving force correct sign)
2. `synapse.compute_current()`: mock V=-75mV, set g_gaba_a=1 → I = 1.0 × (−75 − (−75)) = 0 (at reversal, no current)
3. STDP traces: feed continuous rate=0.7 → trace increments by 0.0 (not 0.7); feed binary spike=1 → trace increments by 1.0
4. STDP causal window: pre fires at t=0, post fires at t=25ms → NO eligibility increment (outside ±20ms window)
5. STDP causal window: pre fires at t=0, post fires at t=10ms → eligibility increment (within window)
6. Burst boost: neuron i bursts, synapse j active, synapse k silent → e[j,i] boosted, e[k,i] unchanged

### Anti-Pattern Removal

7. `grep -rn "for.*range.*relaxation\|for.*range.*integration_steps\|max_relaxation" core/` → 0 hits
8. `grep -rn "peek\|snapshot_traces\|restore_traces\|forced_action" core/` → 0 hits
9. `grep -rn "grad_log_pi\|REINFORCE" core/` → 0 hits
10. BG Critic/Actor: single `forward()` call = 1 LIF step. Profile: no inner loops

### Architecture

11. `snn_agent.act()`: state goes through PoissonEncoder → spikes → NetworkGraph → BG output. No raw float arrays reach BG.
12. `snn_agent.observe()`: reward spike-encoded as population firing rate, fed through NetworkGraph to critic
13. WM gate: set ACh=threshold−0.01 → gate_signal ≈ 0.27 (sigmoid), NOT 0.0 (hard cutoff)
14. WM output flows through NetworkGraph connections, no `np.concatenate([state, wm_signal])`

### Neuromodulation & FEP

15. Region NE: inject high error in region A, low in B → `ne_levels['A'] > ne_levels['B']`
16. Receptor wiring: set DA=0.5 → striatal D1 layer has Hill response R = 0.5^1.5 / (0.4^1.5 + 0.5^1.5) ≈ 0.58
17. Ambiguity test: world model with high sensory noise → `ambiguity > 0.0` in EFE computation
18. Oscillator coupling: change NE → theta frequency changes → SequenceMemory theta_window changes dynamically

### Biology

19. Astrocyte isolation: disable astrocyte → D1 modulation still works (fast channel via error_neuron.error_rate)
20. Efference copy: 2 distinct D1/D2 sub-threshold states → world model encoder receives different actions → different predictions
21. SWS safety: 1000 replay cycles → mean firing rate never >3× baseline; Down phase rate ≈ 0
22. Slow oscillation: oscillator SWS mode → ~1Hz with correct Up/Down duty cycle
23. Seizure brake: artificially elevate excitatory rates → system forces Down state within 5ms
24. Pattern separation: 50 overlapping sequences → cosine similarity of stored representations < 0.3
25. CartPole reward increases over 100 episodes (end-to-end validation)

---

## Decisions

- peek() REMOVED (not fixed) — continuous V_trace EMA
- integration_steps REMOVED — 1 LIF step per network.step()
- Softmax fallback REMOVED — uniform random on silence, thalamic gating long-term
- REINFORCE in Actor REMOVED — replaced by DA-modulated Hebbian STDP (Frank 2005)
- forced_action REMOVED — curriculum via neuromodulatory bias if needed
- Reversal potentials ACTIVATED — Ohm's law enforced in all synaptic currents
- STDP traces: EVENT-BASED only (binary spike deltas, not continuous rates)
- STDP causal window: ±20ms enforced everywhere
- Burst boost: SYNAPSE-SPECIFIC (pre-trace weighted)
- WM gate: SOFT sigmoid (not binary threshold)
- Astrocyte NEVER in fast decision loop — slow NMDA only; fast epistemic via direct error neuron→D1
- Astrocyte BC: ghost-point Neumann (not one-sided differences)
- Ambiguity COMPUTED from world model decoder variance (not hardcoded 0)
- DA/5-HT global (Schultz 1998, Doya 2002); NE/ACh region-aware
- Oscillator theta_reset CONSUMED by SequenceMemory and planning
- Receptor Hill equation WIRED to all layers via compute_layer_modulation()
- Experience encoded as spike trains (not raw NDArrays)
- Advantage weighting: explicit parentheses to fix operator precedence
- No backward compat; all tests rewritten
