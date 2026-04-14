# T-Maze Deep Analysis & Fix Plan

## TL;DR

Deep numerical trace of the T-maze reveals **5 critical issues** preventing reliable learning: (1) WM→BG dimension mismatch dilutes cue signal, (2) actor eligibility trace too short for 4-step delay, (3) no sleep consolidation occurs, (4) WM content maintenance too weak, (5) WM weight update uses wrong prediction error. Fixes target architecture-level bugs, NOT task-specific tuning.

## Numerical Trace: T-Maze Episode

### Environment

- State: 7D one-hot. Cue at s[0]/s[1], position at s[2]-s[6]
- 4 steps: start(cue) → corridor×2 → junction → terminal(reward)
- Reward: +10 correct arm, -1 wrong, -0.1 per corridor step

### Agent Dimensions

- Population encoder: 7×15 = 105 neurons (Gaussian tuning curves)
- Critic hidden: 128 neurons
- Actor: 2 actions × 32 MSNs = 64 motor + 1 internal = 65 D1/D2 neurons
- Working memory: 8 neurons, 7 inputs (raw state)
- n_substeps = round(25/1) = 25 (tau_m_msn_up / dt)
- n_substeps_critic = round(15/1) = 15

### Issue 1: WM→BG Dimension Mismatch (CRITICAL)

**WM output = 8 neurons. BG input = 105 neurons.**

In `network._aggregate_feedforward_inputs()` with `aggregation_mode="sum"`:

```
result = zeros(105)
result[:8] += wm_output[:8]  # Only first 8 positions!
```

The WM output ONLY adds to the first 8 of 105 BG input dimensions. These correspond to the first 8 tuning curves of state dimension 0 (the cue-left signal). The BG weights for dimensions 8-104 never see WM content. The WM signal competes with population-encoded sensory input in only 8/105 = 7.6% of input neurons.

**Impact**: WM cue information has minimal influence on D1/D2 competition because it only reaches a tiny slice of the weight matrix. The actor's action choice at the junction is dominated by the 105D population-encoded sensory input (position encoding), not the 8D WM trace.

**Fix**: WM should either (a) have its own dedicated projection weights to BG (separate from cortical input), or (b) output should be projected to match BG input dimensionality, or (c) WM should output to its own weight matrix on BG neurons rather than summing into the sensory input.

### Issue 2: Actor Eligibility Trace Too Short (CRITICAL)

**tau_e_actor = 30 ms. Total decision-to-reward delay = 4 steps × 25 substeps = 100 ms.**

Decay per ms: exp(-1/30) ≈ 0.967
After 100 ms: 0.967^100 ≈ 0.035 → **96.5% of eligibility has decayed away**

The cue is seen at step 0. The reward arrives at step 4. By the time the TD error broadcasts from the terminal reward, the actor eligibility for the cue-related inputs at step 0 is essentially zero. The actor cannot learn the cue→action association through STDP.

The critic's tau_e_critic=100ms fares better: 0.99^100 ≈ 0.37, retaining 37% of step-0 eligibility. But the actor's trace is far too short.

**Note**: This is NOT about tuning. 30ms is the biophysical STDP window (Bi & Poo 2001). The issue is architectural — STDP eligibility traces are not designed for multi-second delays. The solution must be architectural (e.g., WM providing credit bridge, replay consolidation).

### Issue 3: No Sleep Consolidation Occurs (CRITICAL)

Sleep is triggered ONLY by ATP depletion (`_needs_sleep()`). With:

- atp_regen_rate = 5e-6 per ms
- atp_spike_cost = 1.5e-5 per ms × rate
- Sparse firing ~5% → cost per zone ≈ 7.5e-7 per ms
- Regen per ms ≈ 5e-6 >> cost

**ATP never depletes enough to trigger sleep.** threshold = 0.3 (ATP < 30%). With 4 steps × 25 substeps = 100 ms per episode, and 600 episodes = 60,000 ms total:

- Net regen per ms ≈ 5e-6 → over 60,000 ms: ATP stays near maximum
- sleep_atp_threshold = 0.3 → NEVER reached

**Impact**: The replay buffer accumulates experiences forever but never consolidates. SWS reverse replay (which could bootstrap value signal across the 4-step delay) and REM forward replay (which trains the world model) NEVER run. This is the most impactful bug for T-maze — sleep replay is specifically designed to solve the credit-assignment-over-delay problem that STDP eligibility traces cannot handle alone.

### Issue 4: WM Content Maintenance Weak (MODERATE)

Content decay uses tau_w = 300 ms → content_decay = exp(-1/300) ≈ 0.997 per ms.

Over 100 ms (4 steps): 0.997^100 ≈ 0.74 remaining. This is OK for content trace.

But the ACTUAL WM firing depends on recurrent lateral weights:

- lateral_strength = 0.5
- lateral PSP = gap/3 = 5 mV → with 3 co-active neurons → 15 mV → threshold
- BUT: with 8 total WM neurons and sparse initial activation (~2 neurons from cue), recurrent excitation is insufficient to maintain firing
- After gate closes (ACh×DA product insufficient in corridor), feedforward drive = 0
- Recurrent-only: 2 neurons × 5mV × lateral_strength(0.5) = 5 mV per neighbor — far below threshold

**Impact**: WM content decays during corridor and is effectively zero at junction.

### Issue 5: WM Weight Update Uses Wrong PE (MODERATE)

In observe():

```python
wm_pe = np.zeros(self.working_memory.num_neurons)  # 8D
pe_len = min(len(pred_error), self.working_memory.num_neurons)  # min(7,8)=7
wm_pe[:pe_len] = pred_error[:pe_len]  # Copy world model PE into WM shape
self.working_memory.update_weights(m_t=wm_lr, pred_error=wm_pe)
```

The WM weight update uses world model prediction error (state transition prediction), not a WM-relevant signal. WM should learn what to REMEMBER (cue→reward association), but it gets gradient signal from world model state prediction. These are orthogonal objectives.

**Impact**: WM weights drift based on world model state-prediction accuracy, not memory utility.

## Additional Observations (Non-Bug)

### VTA Gamma with Serotonin

- tau_ppTg = 2488 ms, serotonin ≈ 0.6 at baseline
- tau_eff = 2488 × (1 + 0.6) = 3981 ms
- gamma_eff = exp(-25 × 1.0 / 3981) = exp(-0.00628) ≈ 0.9937
- Effective horizon: 1/(1-0.9937) ≈ 159 steps
- OK for 4-step episodes

### D1/D2 Balance at Baseline

- DA=0.5: d1_mod ≈ 1.53, d2_mod×tonic ≈ 0.75 × d2_gain_comp
- d2_gain_comp calibrated to match d1 at baseline → balanced
- At tonic_da=0 (initial): d2_tonic = 1.5 → D2 bias → cautious exploration ✓

### WM Gate Calibration

- gap = 15 mV, delta_t = 2 mV
- g_L_eff = 281/25 = 11.24 nS
- i_rheo = 11.24 × (15 - 2) = 146.1 pA
- w_adapt_at_thresh = 4.0 × 15 = 60 pA
- gate_drive = (146.1 + 60) / (0.5 × 0.4) = 1030.5 pA
- At ACh=0.5, DA=0.5: current = 0.5 × 0.5 × 1030.5 = 257.6 pA
- This should exceed i_rheo (146 pA) + adaptation → gate opens
- At ACh=0.3, DA=0.3: current = 0.3 × 0.3 × 1030.5 = 92.7 pA < 146 → gate closed ✓

## Proposed Fixes

### Phase 1: Critical Architecture Fixes

**Step 1: Dedicated WM→BG Projection** (_depends on nothing_)

- Add separate weight matrices `w_wm_to_critic` (8, 128) and `w_wm_to_actor` (8, 65) in the critic/actor
- WM output goes through its own projection, not through sensory input summation
- Initialize with init_weights() using appropriate PSP target
- Use conductance-based I = g × (E_exc - V) like other synapses
- Files: [core/basal_ganglia.py](core/basal_ganglia.py) — add WM input path in `SNNDeepCritic.forward()` and `D1D2Actor.forward()`; [arena/snn_agent.py](arena/snn_agent.py) — pass WM output to BG; [core/network.py](core/network.py) — support typed connections or dedicated WM channel

**Step 2: Enable Sleep Consolidation** (_depends on nothing, parallel with Step 1_)

- ATP depletion is too slow to trigger sleep within T-maze timescale
- Add episode-boundary sleep as COMPLEMENT to ATP-triggered sleep (not replacement)
- Biologically: hippocampal replay happens during rest periods between foraging bouts, not only during deep sleep (Foster & Wilson 2006)
- Post-episode "rest" replay is a micro-consolidation event with reduced learning rate
- Files: [arena/snn_agent.py](arena/snn_agent.py) — add post-episode micro-replay in `observe()` when `done=True`

**Step 3: WM Prediction Error Signal** (_depends on nothing, parallel_)

- Replace world-model PE with a WM-relevant learning signal
- TD error × gate_signal: WM learns when gated information led to TD error
- dw_ff = lr × td_error × gate × eligibility (three-factor with gate modulation)
- Files: [arena/snn_agent.py](arena/snn_agent.py) — pass TD error to WM update; [core/working_memory.py](core/working_memory.py) — accept TD-based learning signal

### Phase 2: WM Maintenance Enhancement

**Step 4: Strengthen Recurrent Attractor** (_depends on Step 1 verification_)

- Current: 2 active neurons × 5mV PSP × 0.5 lateral_strength = 5 mV (insufficient)
- Need: recurrent drive alone should sustain firing above threshold
- Option A: Increase lateral_strength from 0.5 to ~1.0 (still within PFC EPSP range 5-10mV, Cruikshank 2012)
- Option B: Use NMDA-based recurrent with slow decay (Compte 2000: NMDA τ=100ms provides temporal bridge)
- This is biophysically motivated: PFC persistent activity relies on NMDA-dominated recurrent excitation (Wang 2001, Durstewitz 2000)
- Files: [core/working_memory.py](core/working_memory.py) — add NMDA slow trace for recurrent connections

## Relevant Files

- [arena/snn_agent.py](arena/snn_agent.py) — main agent wiring, observe(), act(), reset()
- [core/basal_ganglia.py](core/basal_ganglia.py) — D1D2Actor.forward(), SNNDeepCritic.forward()
- [core/working_memory.py](core/working_memory.py) — WM gate, forward(), lateral learning
- [core/network.py](core/network.py) — \_aggregate_feedforward_inputs(), sum mode dimension handling
- [core/vta.py](core/vta.py) — RPE computation, value weight update
- [core/config.py](core/config.py) — WorkingMemoryConfig, BasalGangliaConfig, AgentConfig

## Verification

1. Run `_test_tmaze_quick.py` — currently expect random-level performance (~4.2 mean)
2. After Step 1+2: expect improvement to ~5-6 (WM signal reaches BG + replay consolidation)
3. After Step 3+4: expect convergence to ~7+ (proper WM learning + content maintenance)
4. Run full test suite `tests/` — no regressions in non-WM tasks
5. Monitor electrode traces: `_diag_trace.py` for membrane potential evolution during T-maze episodes

## Decisions

- WM→BG connection via dedicated projection (not sensory summation) — this is how PFC→striatum projections work anatomically (Haber 2003)
- Episode-boundary micro-replay as rest consolidation (Foster & Wilson 2006), coexisting with ATP sleep
- TD error as WM learning signal (Braver & Cohen 2000: PFC updates gated by DA)
- No tuning of any parameter specifically for T-maze — fixes are architectural

## Further Considerations

1. **\_floor = 0.07 eligibility floor**: Is 0.07 a magic number or derived? It's described as "minimum eligibility from membrane proximity to threshold" but the value seems empirically chosen. → Keep for now, address in a separate cleanup pass.
2. **Curiosity scaling `raw × 2.0`**: The world_model curiosity signal has a hardcoded 2.0× multiplier. Not affecting T-maze directly but should eventually be derived. → Out of scope.
3. **NMDA in WM recurrent**: Adding NMDA-based recurrence is a non-trivial change. Alternative: purely increase lateral_strength. → Recommend NMDA approach (more biophysically grounded, Wang 2001).
