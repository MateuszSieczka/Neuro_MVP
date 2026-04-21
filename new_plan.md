# Plan: Phase 6A → 6B → outline to Phase 11 (superseded; see below)

> **Note**: This file now holds the post-5B plan (April 2026). The
> legacy 5A/5B sections are archived below for history.

---

## Phase 6A — Continuous motor substrate (pre-MJX)

### TL;DR

Pre-MuJoCo infrastructure: complete ventral visual hierarchy (V2/V4),
introduce a real M1 module with continuous readout, encode
proprioception, and close the cerebellar efference-copy loop — **all
while the environment stays discrete**. Gridworld/bandit/visual_grid
keep passing; M1 output is discretised via argmax at the body boundary
so the existing embodiment API is unchanged. This phase is
regression-safe and JIT-stable.

### Why split 6 into 6A + 6B

- MJX is a heavy external dependency with its own JIT-compile cost and
  Windows quirks. Don't couple "new motor architecture" debugging
  with "physics simulator" debugging.
- 6A leaves the brain strictly better at every existing task
  (continuous M1 head degrades gracefully to discrete) — 100+ tests
  stay green.
- 6B then only adds the body and the reaching task; if it explodes,
  we know it's the environment, not the brain.

### 6A.1 V2 + V4 instantiation in sensory_stack

- Wire the dormant builders in `sensory/ventral.py`
  (`init_v2_params`, `init_v4_params`) into `sensory/sensory_stack.py`.
- Pipeline becomes: retina → LGN → V1 → **V2 → V4** → thalamic relay
  afferent. V4 belief replaces V1 belief as the sensory vector handed
  to EC/thalamus/BG.
- V2 receives V1 L2/3 as ff input; V4 receives V2 L2/3. Each is a
  single `CorticalArea` — STDP learns mid-level features
  (Felleman & Van Essen 1991).
- Keeps `bypass_sensory_stack=True` path intact (raw sensory → brain
  for bandit / gridworld fast smoke tests).
- Add `SensoryStackParams.v2 / .v4`; `SensoryStackState.v2 / .v4`.
- Consume the existing `test_phase4_ventral_hierarchy.py` test (which
  currently exercises builders only) to assert V4 belief is finite and
  differs from V1 belief under closed loop.

### 6A.2 M1 module (continuous head over cortical L5)

- New `core/m1.py`: `M1Params(cortex: CorticalAreaParams,
motor_readout: Array (n_l5, motor_dim), readout_lr)`.
- `m1_step(state, params, ctx, ff_input, da_level, ach_level)`
  → `(new_state, joint_command: (motor_dim,), l5_rate)`.
- `joint_command = tanh(motor_readout.T @ l5_rate_normalized)`; scales
  into `[-1, 1]` so downstream (discrete GridWorld adapter OR MJX
  torque) can consume it uniformly.
- Learning: three-factor Hebbian on `motor_readout` gated by VTA RPE
  and cerebellar motor error (Doya 2000, Shadmehr & Krakauer 2008).
  Cortex STDP inside M1 is standard.
- Initialization: `motor_readout` PCA-initialised on identity-like
  muscle-group priors (developmental motor-map prior; corresponds to
  infant "primitive synergies", Dominici 2011). This is **not** a
  gradient-trained init — it's a closed-form analytic init analogous to
  Gabor init in V1. No backprop introduced.

### 6A.3 Discrete body adapter (no environment changes)

- New pure function in `embodiment/body_interface.py`:
  `discretise_joint_command(joint_command: Array, n_actions: int)
-> int`. argmax over sign-split joint channels → existing
  `body.act(body_action_int, ...)`.
- Run-loop: insert `m1_step` between BG body-actor and
  `body.act`. Body-actor output becomes `ff_input` to M1 (BG =
  preparatory; M1 = kinematic). Backward-compat flag
  `bypass_m1: bool` for direct-discrete regression.

### 6A.4 Proprioception encoder (kinematics → cortex afferent)

- New `sensory/proprioception.py`:
  `ProprioceptionParams(n_joints, n_cells_per_joint=16,
angle_sigma_rad, velocity_sigma)`; Gaussian population coding
  (Georgopoulos 1986, Pouget & Sejnowski 1997).
- `proprio_encode(joint_angles, joint_velocities) -> Array (n_pop,)`.
- For Phase 6A, there is no real joint yet — the gridworld supplies a
  synthetic "fake proprioception" derived from last action one-hot +
  position delta. This is _not_ biologically real but is
  mechanistically equivalent (encoder exercised, downstream wiring
  tested) and becomes the drop-in for MJX proprio in 6B.
- Proprio vector concatenated into EC afferents (alongside v4_belief,
  pfc.content, last_motor).

### 6A.5 Cerebellum efference-copy motor learning (close the loop)

- Climbing fibre currently carries **sensory** PE only. Extend to
  motor PE (Wolpert 1998):
  `cb_motor_pe = proprio_encoded_actual - proprio_encoded_predicted`.
- Add `w_motor_pc` Purkinje readout learning on this motor error
  (existing Marr-Albus LTD rule, new input channel).
- Deep-nuclei output gains a motor-correction vector projected to M1
  as an additive term on `joint_command`
  (`joint_command_corrected = joint_command + α · cb_motor_correction`).
- `α` derives from τ_cerebellum timing (Medina & Lisberger 2008 ~ms),
  not tuned.

### 6A.6 Regression of existing embodiments

- Gridworld/bandit/visual_grid: discretised M1 path. Tests must still
  pass (tolerance ≤ 10 % in reward curves; M1 adds one tanh-linear
  layer so a small shift is expected and acceptable).
- If any bandit/gridworld test regresses > 10 %, investigate before
  moving to 6B. Do **not** re-tune bandit thresholds to hide the
  regression.

### Tests Phase 6A (6 new)

1. `test_phase6a_v4_learns` — 5k closed-loop cycles on visual_grid,
   V4 belief orientation/form tuning rises over cycle (mean abs-delta
   ≥ 2× init).
2. `test_phase6a_m1_continuous_output` — tanh-bounded, shape
   `(motor_dim,)`, finite, responds to cortex L5.
3. `test_phase6a_m1_readout_learns` — VTA RPE drives a detectable
   `motor_readout` drift correlated with sign of RPE (ρ > 0.3).
4. `test_phase6a_proprio_encoding` — population code peak follows
   synthetic joint trajectory (argmax tracks true angle).
5. `test_phase6a_cerebellum_motor_pe` — motor PE non-zero on random
   commands, converges to ≤ 60 % of initial after 3k cycles.
6. `test_phase6a_discrete_regression` — gridworld reward curve
   within ±10 % of pre-6A baseline over 3 seeds.

### Exit criteria

- 38 + 6 = 44 tests green (plus whatever tests existed — total ≥ 104).
- JIT compile time ≤ 2× baseline.
- `bypass_m1` flag defaults to `False`; sanity test with `True`
  behaves identically to pre-6A behaviour.
- No new orphans introduced; `free_energy.expected_free_energy` still
  deferred to Phase 9.

---

## Phase 6B — MuJoCo MJX body + reaching + endogenous sleep cycle e2e

### TL;DR

Drop the trained 6A brain into a pure-JAX MuJoCo body. 7-DOF reacher
with proprioception + optional foveal vision. Motor babbling warm-up
exercises M1+cerebellum; target-conditioned reaching then layers on
top. The first end-to-end demonstration where ATP-driven sleep
naturally alternates with wake over a long reaching session.

### 6B.1 MJX body integration

- New `embodiment/mjx_arm_body.py`: 7-DOF arm (mujoco-mjx). Implements
  `BodyInterface`; `sensory_size = n_proprio + n_visual_flat`;
  `n_actions = motor_dim` (continuous, but `act()` signature stays
  discrete — the continuous command is handed in through a new
  optional `act_continuous(joint_command)` method; discrete fallback
  still supported).
- Body exposes an XML scene (desk + target cube) with randomised
  target position per episode.
- Visual sensor: a 64×64 RGB foveated camera on the end-effector OR
  a static-table camera (static is cheaper; start there).

### 6B.2 Motor babbling phase

- `embodiment/babbling_env.py`: no target, no reward. For the first
  30k cycles:
  - Body actor output ignored; M1 receives structured noise as its
    ff_input (OU-process joint-by-joint, τ=200 ms, Schaal & Sternad
    2001 motor primitives).
  - Cerebellum learns forward model (proprio\_{t+1} from
    (proprio_t, joint_command_t)).
  - V4 + EC + HC run normally (self-supervised sensory learning).
  - Curiosity (WM learning progress) rewards the BG body-actor, so BG
    _does_ learn preferences for motor primitives that produce
    interesting sensory change — same mechanism as saccade info-gain.
- After babbling: switch back to BG-driven M1 input (resume normal
  perceive-act loop).
- Biological reference: Oller 1980 canonical babbling; Kuhl 2004
  perceptual magnet (will matter more for speech in Phase 8, but
  babbling is a general development principle).

### 6B.3 Reaching task

- `embodiment/reacher_env.py`: target visible, reward =
  `− ‖end_effector − target‖₂` (shaped).
- Episode length 500 substeps; 2000 episodes total.
- Cerebellar forward model from babbling carries over → M1 corrections
  accurate from cycle 0.
- Saccade loop active on camera (existing Phase 4 pipeline) — fixates
  on hand _and_ target (expected emergent divided attention).

### 6B.4 Endogenous sleep cycle in reaching

- ATP drains over long reaching sessions → `brain_cycle` dispatches
  SWS → REM → WAKE organically (Phase 5B machinery already present).
- Sleep consolidates reaching episodes stored in replay_buffer;
  post-sleep reach success rate must rise measurably.
- No scheduler, no timer — just ATP.

### 6B.5 What's **deferred** out of Phase 6B

- Place cells / grid cells (Hafting 2005 hex grid) — tempting to add
  because MJX gives real 3D trajectories, but pushing to **Phase 7**
  because grid cells only matter when the agent _plans routes_, which
  needs the ATL+goal stack Phase 7+ gives. Wiring grid cells in 6B
  without consumers would re-create the Phase 5B "orphan signal"
  anti-pattern.
- Tool use, multi-object manipulation — Phase 10+.
- Continuous torque control with back-drivable dynamics — Phase 6B
  uses position-servo actuators (simpler, still biologically defensible
  — spinal α-motor pools already abstract this).
- Locomotion (CPG + whole-body balance) — Phase 10+.

### Tests Phase 6B (5 new)

7. `test_phase6b_mjx_jit_speed` — 1000 wake cycles < 30 s after warm-up
   (Windows tolerance: 60 s).
8. `test_phase6b_babbling_coverage` — after 30k babbling cycles, end-
   effector traced positions cover ≥ 40 % of 2D workspace voxel grid.
9. `test_phase6b_cerebellum_forward_mse` — after babbling, proprio
   MSE ≤ 40 % of init over a held-out babbling seed.
10. `test_phase6b_reach_success` — success rate (‖e−t‖ < 0.05 m
    within 500 substeps) ≥ 0.4 after 2000 reaching episodes, baseline
    random ≤ 0.05 (5 seeds, mean of means).
11. `test_phase6b_sleep_cycle_improves_reach` — 200 episodes train
    → forced 3 sleep cycles → 100 held-out episodes; success rate
    post-sleep ≥ 1.15× pre-sleep.

### Exit criteria

- 44 + 5 = 49 new tests green (+ existing 100+).
- Endogenous sleep triggers at least once in a 10 k-cycle reaching run
  without manual intervention.
- Cerebellum Purkinje firing 30–100 Hz during active reaching;
  M1 L5 firing 10–40 Hz (Georgopoulos 1986).
- No regression in Phase 0/3/4/5A/5B tests.

---

## Decision log (6A + 6B)

1. **No surrogate gradients. Anywhere.** See discussion below; short
   version: biologically-principled motor priors (PCA motor_readout
   init) + cerebellar forward-model babbling are sufficient, and
   adding `eqx.filter_grad` through `phi1` would break the project's
   "no-ML-shortcuts" invariant and create a dual code path that STDP
   can't round-trip from. Re-evaluate only if Phase 6B reach success
   rate falls below 0.2 after full spec.
2. **V2/V4 now, not in Phase 7.** Needed for reach (hand-shape + 3D
   selectivity) and already designed; moving them forward removes an
   orphan.
3. **M1 with continuous head, not "BG ignores, M1 takes over".**
   Churchland 2012 preparatory subspace: BG picks motor _strategy_,
   M1 refines _kinematics_. Both learn in parallel.
4. **Proprioception encoder in 6A (pre-MJX).** Even with synthetic
   joint signals, this forces the EC/HC wiring to be right before
   MJX shows up. MJX then just swaps the data source.
5. **Efference-copy extends existing cerebellum**, no new module.
   New input channel + new readout weight; standard Marr-Albus LTD
   rule. Keeps the module count flat.
6. **Babbling before reaching, not interleaved.** Separates
   self-supervised forward-model learning from reward-driven policy
   learning; easier to debug; matches developmental timeline
   (Oller 1980).
7. **Grid cells deferred to Phase 7.** Same rule as Phase 5B for
   place cells: don't wire signals without consumers.
8. **Structural plasticity (`sparse.synaptogenesis`, `sparse.prune_below`)
   NOT in Phase 6.** It activates in Phase 7 alongside ATL, where it
   has an obvious consumer (concept attractors need pruning).
9. **`free_energy.expected_free_energy`** stays orphan until Phase 9
   (active inference). We do NOT pre-wire it into M1 just because M1
   is the new consumer-looking thing; without the EFE control loop,
   it would be meaningless.

---

## On surrogate gradients (user question, considered seriously)

### Recommendation: **do NOT introduce surrogate gradients before MJX.**

### If ever introduced, do so **after Phase 8** and **only for offline pre-training of readout matrices** (not cortical STDP layers), behind a default-off flag.

### Why not now

- **Architectural invariant.** The codebase is predicated on local
  learning rules (STDP, three-factor, Hebbian, Marr-Albus LTD). Every
  module — from receptor pharmacology to ATP-driven sleep — composes
  because the learning signals are local and spike-timing based.
  Surrogate gradients are ε away from backprop; once you allow
  `eqx.filter_grad` through `phi1`, the temptation to use it for the
  "hard" parts (credit assignment, long-horizon RL) becomes strong
  and the project stops being what it is.
- **Round-tripping is unreliable.** Pre-training with surrogates
  and then "switching back to STDP" sounds clean but isn't: the
  gradient solution lives in a specific basin of weight space
  (smooth, often dense, with correlations backprop exploits), and
  STDP's pair-based rule has no mechanism to preserve those
  correlations under ongoing plasticity. Real risk: the first hour
  of STDP fine-tuning undoes the pretraining.
- **The bottleneck isn't learning speed.** With Phase 5B machinery,
  credit assignment is distributed over: BG eligibility traces, HC
  replay, SWS reverse consolidation, REM forward rollouts, cerebellar
  forward model. These _are_ the evolutionary shortcut. Adding
  surrogate gradients on top is redundant.
- **MJX reaching is demonstrably feasible without gradients.** DIVA
  (Guenther 2016) and Jordan's distal teacher (1992) both work on
  babbling + forward-model error without backprop on the spiking
  substrate. The motor_readout PCA init + cerebellar forward model
  from 6A.2 / 6A.5 implement exactly this.

### When it might legitimately fit

- Phase 8 introduces speech + Klatt synthesizer: mapping from
  continuous articulator parameters to formant-matched cochleograms
  is a genuinely hard distal supervision problem. If DIVA-style
  babbling alone hits a wall, an _offline_ surrogate-gradient
  pre-train of _only_ the `m1_speech.motor_readout` matrix on a fixed
  phonetic atlas (no trainable spike dynamics, just the linear
  readout) would be defensible as "developmental prior". Cortical
  layers remain STDP-only.
- Similarly after Phase 10 if compositional sequences need bootstrap.

### Proposed future structure if it becomes necessary

- New module `core/surrogate_bootstrap.py`, imported only under a
  default-off `ActionBrainParams.enable_surrogate_bootstrap = False`.
  Bootstrap only touches readout matrices (`motor_readout`,
  `sensory_decoder`), never `cortex` internals, never STDP weights.
- A single regression test proves that with the flag off, every
  Phase 0–N test still passes bit-identically.
- Strict scope: pre-training is _offline_, done once, checkpointed,
  then the flag flips off and STDP runs unperturbed.

---

## Forward-looking outline (Phases 7 → 11)

### Phase 7 — Auditory in loop + ATL convergence + structural plasticity

- Wire `sensory/auditory.py` (cochlea → MGN → A1) into `ActionBrain`
  as a second sensory port. A1 belief joins V4 belief, proprio, and
  PFC content in EC afferents.
- New `core/atl.py` convergence zone (slow STDP + synaptic scaling +
  k-WTA). Afferents: `[v4_belief, a1_belief, m1_l5, ca1_recall,
vta_value_tag]`.
- **First call-site for `sparse.synaptogenesis` and
  `sparse.prune_below`** on ATL: concepts emerge as sparse attractors
  and structural plasticity consolidates them (Rakic 1988 childhood
  pruning; Chklovskii 2004 wiring-cost minimisation).
- Cross-modal PE drive: `‖atl_predicted_audio − a1_actual‖` → ACh
  bonus + curiosity reward (Rao & Ballard 1999 generative PC).
- Value tagging: VTA RPE windows modulate a 64-neuron "valence"
  subpopulation of ATL.
- Grid cells (Hafting 2005) wired into EC now — consumer is
  path-integration for the ATL hub ("where is the target concept
  located").
- Tests: ATL selectivity, cross-modal recall, concept stability over
  sleep, value-tagged rate boost, auditory JIT, grid→place
  emergence.

### Phase 8 — Speech motor + comprehension + inner speech

- `core/m1_speech.py` (separate from M1 because articulator timescale
  differs; ~30 DOF).
- `embodiment/vocal_tract.py` — Klatt 1980 formant synthesizer.
- Canonical babbling (30k cycles) — curiosity-driven, uses the same
  babbling machinery as 6B.2.
- `core/wernicke.py` (post-A1) — sequence_memory second call-site
  (first was CA3). Word-to-concept Hebbian projection to ATL.
- Third BG actor: auditory-attention window (forces the
  `actors: List[ActorParams]` dict refactor that was deferred).
- Inner speech: `pfc.motor_suppress` flag — PFC sequence replay
  drives M1_speech while cerebellar forward model supplies imagined
  cochleogram; no vocal_tract call (Tian & Poeppel 2010).
- **Optional surrogate-gradient pre-train decision point**: if
  babbling doesn't converge on phonetic categories in 30k cycles,
  pre-train `m1_speech.motor_readout` only, behind the flag.

### Phase 9 — Active inference + goal stack + hierarchical PFC

- **First call-site for `free_energy.expected_free_energy`**: PFC
  evaluates candidate goal sequences by EFE and gates BG action
  selection.
- Goal stack in PFC (Botvinick 2014 hierarchical RL). Extends PFC
  from "one WM slot" to "slot hierarchy".
- PFC-BG interaction: prefrontal goal biases striatal action
  selection (Frank & Badre 2012).
- Sequence memory inside PFC (third call-site): intentional,
  multi-step plans.

### Phase 10 — Theory of Mind + tool use + multi-agent

- Second agent or target-agent in MJX scene.
- ATL learns _agent-type_ concepts (concepts about others' states).
- Tool use: MJX manipulates an object that manipulates another
  (kinematic chaining). Cerebellum learns a longer forward model.
- Affordance concepts in ATL.

### Phase 11 — Compositional language + open-ended reasoning

- Hierarchical sequence memory (words → phrases). Requires the
  existing sequence_memory to accept a two-level hierarchy — a
  significant generalisation.
- Grammar emerges as transition-probability clusters over word
  sequences during replay (no explicit rule programming).
- Possible surrogate-gradient re-evaluation point for grammar
  bootstrap, if needed.

### Dependency graph (revised)

```
Phase 5B DONE (adenosine, HC, sleep_replay, brain_cycle)
   ↓
Phase 6A (V2/V4, M1 continuous, proprio, cerebellum motor PE)     [regression-safe]
   ↓
Phase 6B (MJX arm + babbling + reaching + endogenous sleep e2e)   [first embodiment]
   ↓
Phase 7  (auditory + ATL + sparse structural plasticity)          [concepts emerge]
   ↓
Phase 8  (speech motor + vocal tract + Wernicke + inner speech)   [language begins]
   ↓
Phase 9  (EFE active inference + goal stack + hierarchical PFC)   [planning proper]
   ↓
Phase 10 (ToM + tool use + multi-agent)                           [social substrate]
   ↓
Phase 11 (compositional language + open-ended reasoning)
```

---

## Risks & mitigations

1. **MJX JIT cost on Windows** — cap warm-up ≤ 120 s; if exceeded,
   disable camera rendering first, then downscale arm DOF.
2. **Babbling doesn't cover workspace** — if coverage < 40 %, add
   low-gain target-bias _to curiosity signal_, never to extrinsic
   reward (stays biologically defensible).
3. **Cerebellar motor PE saturates** — usually means the forward
   model is mis-scaled; first remedy is to re-derive α from τ, not
   re-tune. If that fails, consider whether proprio population width
   is too narrow (likely cause).
4. **ATL grid cells orphan in 6B** — mitigated by explicit deferral
   to Phase 7.

---

## What is explicitly NOT in Phase 6A + 6B

- Auditory pipeline, ATL, concepts, speech, language — all Phase 7+.
- Surrogate gradients — out of scope; re-evaluated no earlier than Phase 8.
- `free_energy.expected_free_energy` call-site — Phase 9.
- `sparse.synaptogenesis` / `sparse.prune_below` call-sites — Phase 7.
- Place cells, grid cells — Phase 7 (consumer-first rule).
- Tool use, locomotion, multi-agent — Phase 10+.

---

# Archived

## Archived: legacy Phase 5A → 5B → 6 outline (pre-April 2026)

## Meta — dlaczego nie osobna "Phase 0.5.5"

Audit Phase 0.5 readiness: 100/100 testów zielone, precision_bus wpięty,
learning-progress curiosity działa, γ→BG, soft LGN. Items które zostały:

| Item                                    | Klasyfikacja       | Miejsce                  |
| --------------------------------------- | ------------------ | ------------------------ |
| Replay wiring do ActionBrain            | BLOKER P5          | Phase 5A §5A.1           |
| SleepState + core/sleep.py              | BLOKER P5          | Phase 5A §5A.2           |
| Export episodic/seqmem z core           | BLOKER P5          | Phase 5A §5A.3           |
| sws_mode dead stub w \_perceive_substep | DEAD CODE          | Phase 5A §5A.4           |
| Actor dict refactor                     | BLOKER P8 (nie P5) | Phase 8                  |
| learning_pipeline extraction            | Refactor           | Phase 5A §5A.5 (natural) |
| LR τ-derived                            | Polish             | Phase 5B §5B.7           |
| D1/D2 citation overhaul                 | Polish (docstring) | Phase 6                  |
| ventral/auditory orphans                | Defer              | Phase 7                  |
| sparse.py orphan                        | Defer              | Phase 9                  |

Dedykowana "Phase 0.5.5" byłaby sztuczna. Blokery Phase 5 robimy jako
Phase 5A (infrastruktura, ~1–2 dni); polish wpinamy w naturalne
touchpointy Phase 5B i Phase 6.

---

## Phase 5A — Infrastruktura pamięci + snu (BEZ uczenia offline)

### TL;DR

SleepState + replay_buffer wpięte w ActionBrain, eksporty HC-modułów,
learning_pipeline wyekstrahowany. PO 5A agent myśli i uczy się tak
samo (WAKE-only), ALE ma strukturalne miejsce na Phase 5B.

### Dlaczego OSOBNO od 5B

- Phase 5B (HC+EC+SWS-learning+REM) to ~550 LOC i ~7 testów semantycznych
  zmian — bez checkpointu ryzyko regresji rośnie nieliniowo.
- Phase 5A daje _testable substrate_ dla Phase 5B. W 5B nie debugujemy
  shape-mismatchów w ActionBrainState — tylko HC learning.
- Jeśli 5A coś złamie w 100 testach, wiemy że to infrastruktura.

### 5A.1 Replay buffer wpięty w ActionBrain

- Dodać `replay: ReplayState` do ActionBrainState, `replay_params: ReplayParams`
  do ActionBrainParams.
- W `action_brain_cognitive_step` po `vta_compute_rpe` (mamy rpe, r_ext, done)
  wywołać `replay_store` z Experience =
  (state=last_sensory, action=last_body_action_id, reward=r_ext,
  next_state=sensory, prediction_error=|rpe|, done=done,
  salience=wm_learning_progress(), recorded_da=da).
- Salience = learning_progress (Schmidhuber 1991 — priorytet replay dla
  uczących się obszarów).
- Capacity = 10_000 experiences (≈ 3.3 min wake time; wystarczy na
  jedną sesję HC konsolidacji Stickgold 2013).
- W 5A write-only; read przyjdzie w 5B.
- Koszt: ~30 LOC w brain_graph.

### 5A.2 SleepState + core/sleep.py (no-op poza WAKE)

- Nowy moduł `core/sleep.py`:
  - `SleepPhase(IntEnum)`: WAKE=0, SWS=1, REM=2
  - `SleepParams(atp_to_sws=0.3, atp_to_wake=0.8, tau_rem_onset_ms=5.4e6)`
    (Saper 2010 VLPO flip-flop; Achermann & Borbély 1992 hysteresis;
    Nishida & Walker 2007 REM onset ~ 90 min)
  - `SleepState(phase, phase_duration_ms, rng)`
  - `sleep_step(state, params, ctx, atp_mean) -> SleepState`
- Transitions endogeniczne:
  - WAKE→SWS: atp_mean < atp_to_sws
  - SWS→REM: phase_duration_ms > tau_rem_onset
  - REM→WAKE: atp_mean > atp_to_wake
- ActionBrainState dostaje `sleep: SleepState`. `action_brain_cognitive_step`
  liczy `atp_mean = state.cortex.astrocyte.atp.mean()` i woła
  `sleep_step(...)` JAKO PIERWSZE. Ale uczenie/akcja w 5A niezależne
  od phase — testy weryfikują że state się zmienia.
- Koszt: ~80 LOC.

### 5A.3 Eksport HC-modułów z core/**init**.py

- Dodać do `core/__init__.py`:
  - `episodic_memory`: EpisodicParams/State/StoreOutput/RecallOutput,
    init\_\*, try_store, recall, mark_replayed, episodic_size, dg_encode.
  - `sequence_memory`: SeqMemParams/State/Output, init\_\*, seqmem_step,
    seqmem_novelty.
  - `replay_buffer`: ReplayParams/State, Experience, init_replay_state,
    replay_store, replay_size, replay_sample_indices.
  - `sleep`: SleepPhase, SleepParams/State, init\_\*, sleep_step.
- Dodać init_episodic_params / init_seqmem_params jeśli nie istnieją
  (wzorzec z init_cortical_area_params).
- Koszt: ~30 LOC + ewentualne 2 nowe initery.

### 5A.4 Usunięcie sws_mode stub z \_perceive_substep

- Parametr nigdzie nie używany. Usunąć w czystym commicie.
- Gdy Phase 5B doda SWS-specific perceive gating, będzie to osobny
  commit bez cieniowanego dead-code history.
- Koszt: ~10 LOC.

### 5A.5 learning_pipeline extraction

- Z `action_brain_cognitive_step` (~250 LOC) wyekstrahować do
  `core/learning_pipeline.py` cztery pure funkcje:
  - `critic_learn_step(critic, params, rpe, receptor_lr)`
  - `actors_learn_step(actors_states, params, rpe, bonuses_dict)`
  - `cortex_learn_step(cortex, params, modulator)`
  - `attention_learn_step(attn, params, assoc, columns, gains)`
- Dlaczego TERAZ: w 5B dodamy `replay_learn_step` używający TYCH SAMYCH
  funkcji. Bez extractu 5B duplikuje kod.
- SEMANTYKA: zero zmian — pure refactor; test semantic-equiv.
- Koszt: ~100 LOC.

### Tests Phase 5A (5 nowe)

1. test_phase5a_replay_grows — 100 cycles → replay_size rośnie do 100.
2. test_phase5a_replay_salience — high-learning-progress exp sampled
   częściej (Spearman ρ > 0.5).
3. test_phase5a_sleep_atp_gate — sustained firing → ATP↓ → WAKE→SWS
   w < 5k cycles; cisza → ATP↑ → wraca do WAKE (hysteresis).
4. test_phase5a_sleep_rem_timer — forced SWS + mocked ATP → po
   τ_rem_onset przejście do REM.
5. test_phase5a_learning_pipeline_semantic_equiv — ActionBrainState
   bit-identical po refaktorze (100 cycles).

### Kryteria wyjścia 5A

- 100 + 5 = 105 testów zielone.
- ActionBrainState zawiera `sleep` + `replay`.
- JIT compile time < 2× baseline.
- Baseline firing rates ± 20%.

---

## Phase 5B — Hippocampus + offline konsolidacja

### TL;DR

HC (EC→DG→CA3→CA1 + theta gating) + SWS reverse replay plasticity

- REM forward world-model rollout. Phase 5A dał infrastrukturę; 5B
  dokłada semantykę. Motor ciągły / MJX / place cells → Phase 6.

### 5B.1 EC (entorhinal cortex)

- `core/ec.py` — jedna `CorticalArea` z afferentami
  concat[v4_belief, pfc.content_rate, last_motor_joint_onehot].
- ~256 neuronów L2/3, pojedyncza warstwa (Witter 2007 — EC jako projection
  hub, uproszczona cytoarchitektura).
- `EntorhinalParams(cortex, ec_output_dim=256)`.
- `ec_step(state, params, ctx, v4_belief, pfc_content, last_motor)
-> (new_state, ec_output: (256,))`.
- STDP w cortex już jest; EC uczy się out-of-the-box koreluje modalności.
- Place/grid cells → Phase 6.
- Koszt: ~100 LOC.

### 5B.2 Hippocampus wrapper (core/hippocampus.py)

- Thin composite z `episodic_memory` (DG) + `sequence_memory` (CA3).
- `HippocampusParams(dg: EpisodicParams, ca3: SeqMemParams,
ca1_mismatch_weight)`
- `HippocampusState(dg: EpisodicState, ca3: SeqMemState, ca1_prev_recall)`
- `hippocampus_step(state, params, ctx, ec_in, theta_phase, ne_level, key)
-> (new_state, ca1_recall, novelty, mismatch)`
- Theta gating (Hasselmo 2002):
  - encoding_gate = 0.5·(1 + cos(theta − π/2)) (ascending theta)
  - recall_gate = 1 − encoding_gate
  - DG try_store · encoding_gate; CA3 predict · recall_gate.
- CA1 mismatch = ||ca3_recall − current_ec_in|| → boost ACh
  (McGaughy 2008).
- Storage gate: novelty > dg.ne_threshold (Sara 2009 NE-gated encoding).
- Koszt: ~200 LOC.

### 5B.3 HC wpięte w ActionBrain

- Dodać `ec: EntorhinalParams/State`, `hippocampus: HippocampusParams/State`
  do ActionBrainParams/State.
- W `action_brain_cognitive_step` po perceive:
  `ec_out = ec_step(...)` → `hc_out = hippocampus_step(..., ec_out, ...)`
- mismatch → neuromodulator_step (extra term → ACh boost).
- CA1 recall → pfc_step jako dodatkowy ff (nowe wejście PFC).
- Koszt: ~80 LOC.

### 5B.4 SWS replay (reverse chrono + plasticity)

- `sws_replay_step(state, params, ctx, key, n_replay=32) -> ActionBrainState`:
  - Sample N indices z replay_buffer (prioritised by salience=LP).
  - Dla każdego exp: critic_learn_step + actors_learn_step +
    cortex_learn_step + wm_update (używają learning_pipeline z 5A.5).
  - Reverse chronological (Wilson & McNaughton 1994).
  - Kompresja czasu przez ctx_sws z dt=5·dt_wake (Born 2009).
  - Plasticity gated przez `oscillator.in_up_state` (Steriade & Timofeev
    2003 — konsumpcja już-liczonego sygnału).
- Żadnej SWS-specific plasticity — standardowe funkcje z 5A.5.
- Koszt: ~150 LOC.

### 5B.5 REM rollout (forward + seqmem learning)

- `rem_rollout_step`:
  - Sample start z replay_buffer.
  - `wm_predict` w pętli k=10 kroków (forward dream).
  - Każdy generated transition → seqmem_step (CA3 learning).
  - ACh HIGH, DA LOW (Hasselmo 2006).
- Brak re-store do replay_buffer (generacja, nie zapis).
- Koszt: ~100 LOC.

### 5B.6 Brain cycle dispatch

- `brain_cycle(state, params, ctx, sensory, reward, done, key)`:
  ```
  jax.lax.switch(state.sleep.phase, [
      action_brain_cognitive_step,  # WAKE
      sws_replay_step,               # SWS
      rem_rollout_step,              # REM
  ])
  ```
- run_loop używa brain_cycle zamiast action_brain_cognitive_step.
- Poza WAKE body.act nie wołane.
- Koszt: ~50 LOC.

### 5B.7 LR τ-derived (polish, natural touchpoint)

- Edytujemy learning_pipeline.py → pora wyprowadzić LR:
  - critic_lr, actor_lr → ctx.complement(τ_plasticity=60_000 ms)
    (Malenka & Bear 2004 early-LTP).
  - ltd_ratio → Bi & Poo 1998 STDP asymmetry
    exp(−20/33.7) / exp(20/16.8) ≈ 0.6 (pair_delay=20ms).
- Efekt: parameter-free, reality-anchored.
- Akceptujemy ew. rekalibrację 1-2 testów (nowy baseline).
- Koszt: ~30 LOC.

### Tests Phase 5B (7 nowe)

6. test_phase5b_dg_pattern_separation (cos 0.9 → DG cos ≤ 0.3).
7. test_phase5b_ca3_completion (50%-mask → recall cos ≥ 0.7).
8. test_phase5b_ca1_novelty (A,X po wyuczonym A,B → mismatch ≥ 2×).
9. test_phase5b_sws_consolidation (bandit + 3 SWS → regret ≤ 80%).
10. test_phase5b_rem_wm_refinement (REM → unseen MSE ≤ 80%).
11. test_phase5b_theta_encoding (storage ∝ ascending-theta, ρ > 0.4).
12. test_phase5b_hc_wired_in_actionbrain (JIT cycle shapes OK).

### Kryteria wyjścia 5B

- 105 + 7 = 112 zielone.
- Sleep redukuje WM PE o ≥ 20% vs no-sleep baseline.
- DG pattern separation d' > 1.5.
- Diagnostyka: HC podczas SWS ripples 100-200 Hz; poza SWS < 10 Hz
  (Buzsáki 2015).

---

## Phase 6 — Ciało ciągłe MJX + M1 (outline)

### 6.1 M1 CorticalArea + motor_readout

- `core/m1.py` — `M1Params(cortex, motor_readout: (n_l5, motor_dim))`.
- m1_step: weighted-mean L5 rate × readout → joint_command ∈ ℝ^motor_dim.
- Learning: motor_readout Hebbian modulowany RPE (Doya 2000).

### 6.2 MJX body

- `embodiment/mjx_arm_body.py` — 7-DOF reacher (mujoco-mjx pure JAX JIT-able).
- Proprio 14 DOF; opcjonalnie camera RGB → sensory_stack.
- BG body actor (4 actions) = "grupa mięśni" → M1 ciągłe.
- Reward = −dist(end_effector, target).

### 6.3 Cerebellum continuous efference

- joint_command + proprio → cerebellum_step.
- Climbing = proprio_actual − proprio_predicted (Wolpert 1998).

### 6.4 Place cells / grid cells

- EC z 6.4 rozszerzyć o fixed grid-cell RFs (Hafting 2005, 64 cells
  hex grid).
- Pozycja = MJX kinematics (deterministyczna).
- Place cells emergują w CA3 Hebbian (nie hardcoded).

### 6.5 Polish do Phase 6

- D1/D2 density update (Gerfen & Surmeier 2011 + Surmeier 2007)
  — citation touchup bo i tak dotykamy BG-M1 STDP.
- Orphan ventral cleanup (v2/v4 są w sensory_stack; tylko export hygiene).

### Tests Phase 6 (5 nowe)

- test_phase6_mjx_jit (1000 cycles < 30 s).
- test_phase6_reach_learning (success ≥ 0.5 po 20k).
- test_phase6_m1_smoothness (jerk < 5× min-jerk Flash & Hogan 1985).
- test_phase6_cerebellum_forward (proprio MSE −40% w 5k).
- test_phase6_grid_to_place (≥ 30% CA3 z place-field selectivity).

### Kryteria wyjścia Phase 6

- 112 + 5 = 117 zielone.
- M1 firing 5-30 Hz w ruchu (Georgopoulos 1986).
- Cerebellum Purkinje 30-100 Hz (Häusser & Clark 1997).
- End-to-end 10k cycles: ATP cycles widoczne (WAKE↔SWS).

---

## Dependency graph

```
Phase 0.5 DONE — precision bus, LP curiosity, γ→BG, soft LGN
  └─ Phase 5A (2d, 5 tests) — infrastruktura: replay wired, SleepState,
       HC exports, learning_pipeline extract
       └─ Phase 5B (5d, 7 tests) — HC + EC + SWS + REM + LR τ-derived
            └─ Phase 6 (7d, 5 tests) — MJX + M1 + place/grid + D1/D2
                 └─ Phase 7 — ATL, auditory-in-loop, cross-modal
```

---

## Decyzje

1. **Rozbicie Phase 5 na 5A + 5B** — każdy checkpoint niezależnie
   weryfikowalny, mniejsze ryzyko regresji.
2. **Actor dict refactor NIE w Phase 5** — blokuje tylko Phase 8.
   Odraczamy (gdy dodajemy auditory_attend actor w 8, refactor
   naturalnie się wpina).
3. **learning_pipeline extraction w 5A** — bo i tak dodajemy
   replay_learn callsite; semantic-equiv test gwarantuje bezpieczny
   refactor.
4. **LR τ-derived w 5B §7** — polish naturalnie w learning_pipeline;
   akceptujemy rekalibrację.
5. **D1/D2 density w Phase 6** — dotykamy BG STDP przy M1.
6. **EC jako jeden CorticalArea**, nie 6-warstwowy — MVP.
7. **Place cells w Phase 6** — wymagają MJX topologii.
8. **REM tylko forward WM rollouts + seqmem** w 5B.
   Dreaming z value modulation → Phase 7+.
9. **SWS_replay używa learning_pipeline** — brak SWS-specific
   plasticity; kompresja czasu przez ctx_sws z dt×5.
10. **SleepState.rng dla replay sampling** w SleepState
    (logicznie należy do fazy snu, nie do root state).

---

## Ryzyka

1. ActionBrainState rośnie do ~20 fields — JIT compile time ryzyko.
   Mitygacja: 5A kończy benchmark gate (< 2× compile time baseline).
2. HC+EC+SWS+REM razem = 550 LOC — rozbić na 7 sub-commitów
   (EC, HC stub, HC theta, HC→AB wiring, SWS, REM, dispatch) z
   regresją po każdym.
3. LR τ-derived może wymagać rekalibracji bandit threshold —
   akceptujemy jeśli nowy wynik ≥ chance + 20%.
4. MJX JIT na Windows wolne — Phase 6 benchmark gate 1000 cycles
   < 30 s po warm-up.

---

## Co NIE wchodzi

- Auditory pipeline integration → Phase 7.
- ATL convergence → Phase 7.
- Speech / babbling / vocal tract → Phase 8.
- Inner speech → Phase 8.
- Structural plasticity (sparse.py) → Phase 9.
- EFE active inference → Phase 9.
- Goal stack / hierarchical PFC → Phase 9.
- Theory of mind → Phase 9+.
