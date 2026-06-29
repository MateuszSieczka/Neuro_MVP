# Legacy → PC-substrate integration plan

**Purpose.** After Phase U the live system is a clean predictive-coding
substrate (`core/` = `backend, free_energy, pc_module, pc_graph, pc_active,
pc_brain, pc_structural`). The old spiking system was moved to `legacy/`
**as reference only — it is not importable** (its imports point at deleted
modules on purpose). This document decides, per file, what to **restore &
fix**, what to **rewrite on the substrate**, and what was **discarded**,
and gives an integration order. Use it to start a fresh chat per subsystem.

Goal: a functioning neuro-AGI on a clean **rate predictive-coding**
substrate — local learning, one rule. Everything below is judged against
that. **Direction change (2026-06-29):** the optional spiking-PC /
neuromorphic track (old §6) is **dropped**. The forward path is *deepening*
rate-PC and, later, **dendritic predictive coding** (apical/basal
compartments computing local error; Sacramento 2018, Mikulasch 2023) —
which is still rate-mode, not spikes. Nothing is kept "as reference in
`legacy/`" any more: every file below is either already integrated or
**discarded**, and `legacy/` is deleted in full (§10). Git history is the
spec backup.

---

## 0. The integration contract (how anything joins the substrate)

There are only five ways a capability enters the new system. Every
refactor below maps to one of them:

1. **A graph node** — a region = a value population `μ` with precision `Π`
   on `pc_graph` (add to `REGION_NODES` / `init_region_graph`).
2. **A generative edge** — a projection = `W` learned by the one rule
   `ΔW = η·Π·ε·φ(μ)`. Frozen exception = edge with learning off.
3. **A precision gain** — neuromodulation / attention =
   `scale_node_precision` via `pc_brain_cognitive_step(precision_gains=…)`.
4. **An offline mode** — sleep/replay = relax+learn on the graph with the
   sensory node unclamped, samples drawn from the generative model.
5. **An I/O adapter** — sensory encoder (raw input → flat rate vector that
   clamps the sensory node) or body adapter (consume `joint_command`,
   return reafference). Substrate-agnostic; lives outside `core/`.

If a file does not reduce to one of these, it is discarded.

Legend: 🟢 **RESTORE+FIX** (code largely reusable, adapt I/O to rate
vectors / pytrees) · 🟡 **REWRITE** (idea is right, code is spiking/old-
coupled — use the file as a spec, reimplement on the substrate) · 🔴
**DISCARDED** (not in `legacy/`).

---

## 1. Embodiment — embodied reach on `pc_brain` (PRIORITY 1) — ✅ DONE

The first milestone (plan §12): a body adapter driving `pc_brain`,
validated on reach. **Integrated.** The old `legacy/embodiment/` has been
**deleted** (superseded); the live code is a new top-level `embodiment/`
package plus `sensory/proprioception.py`, driving the substrate through
`pc_brain_act` + `pc_brain_learn_forward`.

Two substrate enhancements landed with it (backward-compatible):

* **Per-dimension clamp masks** (`pc_graph_relax(..., clamp_masks=)`,
  `pc_act_infer(..., preference_mask=)`, `pc_brain_act(..., preference_mask=)`)
  — a *partial* preference, so a goal can pin some sensory channels and
  leave the rest inferred. The reach goal pins the **target-error**
  channels to zero, proprioception free.
* **Canonical motor belief** — `motor_belief` (pre-`tanh`) is exposed on
  `PCBrainOutput` and `PCBrainActOutput`; `tanh` lives only at the body
  boundary, and the same belief is the forward-model input. Babble and
  reach are one cycle (*choose belief → execute `tanh` → learn forward*),
  differing only in the belief source (OU vs active inference).

| legacy file | outcome | landed as |
|------|---------|-----------|
| `embodiment/body_interface.py` | rewritten | `embodiment/body_interface.py` — continuous `BodyInterface` (`act(key, joint_command)`), `SensorySample`, and a named `SensoryLayout` (goals address channels by name, no magic index). |
| `embodiment/mjx_arm_body.py` | restored + stripped | `embodiment/mjx_arm_body.py` — MJX 2-link arm, discrete/saccade/`n_actions` removed; publishes its `SensoryLayout` + `reach_goal()`. |
| `embodiment/babbling_env.py` | rewritten | `embodiment/babbling.py` — OU babble in **belief** space; `run_babbling` scans `pc_brain_learn_forward`. |
| `embodiment/reacher_env.py` | rewritten | `embodiment/reach.py::build_reacher` — pairs `MjxArmBody` with `init_pc_brain`. |
| `embodiment/mjx_run_loop.py` | lesson kept | `embodiment/reach.py::run_reach` — many cycles in one `lax.scan`/`filter_jit`; potential-based reward shaping retained. |
| `embodiment/run_loop.py` | dropped | discrete `act(body_action, saccade)` driver — a different (non-substrate) interface. |
| `embodiment/gridworld.py`, `bandit.py` | dropped | discrete-action probes; off the continuous-reach path. Revisit on the new interface if value/policy needs a discrete testbed. |
| `embodiment/visual_grid.py` | dropped | vision-phase; will be rebuilt on the new interface in §2 (references `legacy/sensory/retina`+`lgn`, kept). |
| `sensory/proprioception.py` | restored | `sensory/proprioception.py` (live) — the encoder feeding the embodiment sensory clamp. |

(Everything dropped is recoverable from git history — see §7.)

---

## 2. Sensory front-end — perception that clamps the sensory node (PRIORITY 2) — ✅ DONE (vision; audio deferred)

In the substrate, "sensory" is the clamped observation node. These turn
raw input into the **flat rate vector** that clamps it. **Vision
integrated.** The encoders are substrate-agnostic I/O adapters living
*outside* `core/` (contract §0.5); the legacy reference files are
**deleted** (git history is the backup). The only substrate-touching
change is the opt-in Gabor edge-init. Audio is deferred to a later
milestone.

Three new `sensory/` modules + one backward-compatible `core/` change:

* **`sensory/retina.py`** — scale-invariant image → fixed-length `[0, 1]`
  rate vector (Gaussian pyramid + DoG + foveal patch + motion). The output
  *is* the sensory clamp (no Poisson stage — the substrate is rate-mode).
  `RetinaConfig.afferent_size` sizes the sensory node.
* **`sensory/lgn.py`** — `lgn_normalize`: contrast gain control + tonic
  floor keeps the clamp in a sane, ~contrast-invariant range. Magic
  defaults promoted to named constants.
* **`sensory/vision.py`** — the composition adapter (retina → LGN →
  sensory clamp) + **active saccades**. There is no V1/V2/V4 here: that is
  the brain's own `cortex_l1→l2→l3` hierarchy, the deep node carries the
  cause. Saccade info-gain = sensory **prediction error against the
  model's standing expectation** (Itti & Baldi 2009 surprise; bottom-up
  saliency against a flat prior, sharpening into novelty as the model
  learns); fixation selection = `efe_select` (argmin EFE). `learn=False`
  probes never mutate the brain.
* **`core/pc_graph.py::FovealGaborInit` + `apply_foveal_gabor_init`** —
  the V1 Gabor prior reimplemented (door 2) as an **opt-in** init of the
  `cortex_l1→sensory` generative edge: each source unit's projective field
  onto the foveal ON/OFF sub-blocks is a Gabor, matched to the LeCun
  column scale. `init_region_graph(..., gabor_foveal_init=)` is additive;
  default `None` ⇒ the original LeCun init, byte-identical (asserted). The
  module stays sensory-agnostic — it takes plain geometry, never imports
  `sensory`.

| file | cat | landed as | notes |
|------|-----|--------------|-------|
| `sensory/proprioception.py` | ✅ | `sensory/proprioception.py` | Integrated with §1 — population-coded joints → `[0,1]` clamp. |
| `sensory/retina.py` | ✅ | `sensory/retina.py` | Pyramid + DoG + fovea → fixed rate vector; Poisson stage dropped. |
| `sensory/lgn.py` | ✅ | `sensory/lgn.py` | `lgn_normalize` contrast/tonic stage; named constants. |
| `sensory/v1.py` | ✅ | `core/pc_graph.py` (Gabor edge-init) | Gabor prior → opt-in `cortex_l1→sensory` weight init; `core.cortex`/STDP dependency dropped. |
| `sensory/sensory_stack.py` | ✅ | `sensory/vision.py` | retina→lgn→clamp; saccade info-gain via sensory ε + `efe_select`; V1/V2/V4 = the live cortical hierarchy. |
| `sensory/ventral.py` | 🔴 discard | — | Empty skeleton (imports deleted `core.cortex`, no code to port). The ventral stream **is** the live `cortex_l1→l2→l3` hierarchy. |
| `sensory/auditory.py` | 🔴 discard | — | Cochlea/MGN/A1 front-end, no current consumer. When the audio milestone lands it is a fresh `sensory/audio.py` I/O adapter (contract §0.5): `mgn_normalize` already reduces to `lgn_normalize`, the cochlea is pure DSP — recover from git `0dec6c4` then. Holding it in `legacy/` now is a bridge to nothing. |

---

## 3. Memory & sleep — capabilities the substrate does NOT yet have (PRIORITY 3) — ✅ DONE

Episodic memory + offline replay are real neuro-AGI capabilities missing
from the substrate. Concept is in the plan (§U.2 HC, §4 sleep = offline
FEP). **Integrated.** The data structures landed as substrate-agnostic
primitives in `core/`; the dynamics are graph operations under the one
rule; the legacy reference files are **deleted** (git history is the
backup). `sequence_memory.py` was the sole hold-back — its bespoke
competitive-Hebbian/Oja transition rule was a *second* learning rule —
and is now **integrated in §5b** as a one-rule temporal `w_dyn` edge (the
`world_model`/sensory-transition edge), not a separate plasticity
mechanism. CA3 auto-associative completion this milestone is covered by
episodic recall.

Two new substrate modules + one node landed (all backward-compatible):

* **`core/pc_memory.py`** — `ReplayBuffer` (SoA ring of `(sensory,
  motor_belief, free_energy)`; **free energy is the replay priority**,
  Schaul 2016) + `EpisodicStore` (generic DG-separated, content-
  addressable one-shot store; novelty + surprise-gated write, least-
  salient eviction).
* **`core/pc_sleep.py`** — the offline mode (contract §0.4). `sws_replay`
  reactivates stored experience most-recent-first (sensory **and** motor
  clamped → relax → learn); `rem_rollout` samples a deep cause from its
  prior, leaves sensory **unclamped** so the generative edges render a
  fantasy, then re-explains + learns. WAKE/SWS/REM FSM reused from
  `sleep.py` with the **ATP trigger replaced by free-energy pressure** (a
  low-pass of the cycle's reported `free_energy`; one objective asleep).
* **`hippocampus` + `entorhinal` graph nodes** — `entorhinal` is a new
  multi-parent convergence node (`cortex_l3→ec`, `motor→ec`, `ec→hc`)
  appended to `init_region_graph`. `core/pc_hippocampus.py` gives the
  existing `hippocampus` node its pattern-sep/completion function via the
  episodic store (encode one-shot, complete from a noisy cue); **CA1
  mismatch = the node's own ε**, no extra machinery.

| file | cat | landed as | notes |
|------|-----|--------------|-------|
| `core/replay_buffer.py` | ✅ | `core/pc_memory.py::ReplayBuffer` | SoA ring; schema reshaped to substrate-native `(sensory, motor_belief, free_energy)` — no discrete action / reward / NE. FE = replay salience. |
| `core/episodic_memory.py` | ✅ | `core/pc_memory.py::EpisodicStore` | Generic DG key/value one-shot store; NE gate → generic novelty/surprise gate. |
| `core/sequence_memory.py` | ✅ | **§5b `w_dyn` edge** | Transition rule = one temporal edge (`world_model` self → static `world_model→sensory`); bespoke Oja/k-WTA dropped (pattern-sep already in EC/HC). Legacy file deleted. |
| `core/sleep.py` | ✅ | `core/pc_sleep.py` (FSM) | WAKE/SWS/REM reused; ATP trigger → free-energy pressure + hysteresis. |
| `core/sleep_replay.py` | ✅ | `core/pc_sleep.py` (`sws_replay`/`rem_rollout`) | Rewritten as relax+learn on the graph; world-model is a node, generative model is the edges — no separate `wm_update`/`seqmem`. |
| `core/hippocampus.py` | ✅ | `core/pc_hippocampus.py` | DG+CA3+CA1 → episodic store coupled to the `hippocampus` node; CA1 = node ε. Theta encode/retrieve gating defers with the oscillator (§4). |
| `core/ec.py` | ✅ | `entorhinal` node in `init_region_graph` | Multi-parent convergence node feeding the HC group. |

---

## 4. Neuromodulation, precision & curiosity (PRIORITY 4) — ✅ DONE

Neuromodulators = precision controllers. **Integrated.** The substrate
hooks were already live (`scale_node_precision` / `precision_gains` /
EFE β / `epistemic_value`); §4 supplies the *signals* that drive them and
folds in richer precision tracking, with **two small additive substrate
changes** (opt-in Welford path; the theta encode/retrieve gate). The
legacy reference files are **deleted** (git history is the backup).

Four new `core/` modules landed (all backward-compatible):

* **`core/pc_precision.py`** — Welford mean-centred EMA
  (`welford_precision_update`) + scalar `PrecisionChannel`
  (compose / standardize). Folded into `pc_graph_learn` as an **opt-in**
  `precision_mode="welford"` (the zero-centred ε² EMA stays the default;
  `pe_mean` is the only new `PCGraphState` field, inert in EMA mode).
* **`core/pc_neuromod.py`** — the 4 channels computed from graph-native
  signals (sensory ε novelty → ACh, free-energy volatility → NE, value-
  belief RPE → DA, global stability → 5-HT) and the **curiosity**
  learning-progress (`world_model` ε short/long EMA, Oudeyer). Read-outs
  feed the existing hooks: `neuromod_precision_gains` → `precision_gains`,
  `neuromod_beta` → EFE β, `neuromod_curiosity` → `epistemic_value`
  (augmented with an opt-in `learning_progress` term).
* **`core/pc_attention.py`** — divisive-normalisation + inhibition-of-
  return producing a `(sensory_dim,)` precision gain over sensory
  sub-fields, fed through `scale_node_precision` (which already accepts an
  array gain; `pc_brain`'s `precision_gains` type widened to `float|Array`).
* **`core/pc_oscillator.py`** — pure-functional theta/gamma + PAC in
  cognitive-step units (no wall clock). Drives the **theta encode/retrieve
  gate** deferred from §3: `hippocampus_encode` / `hippocampus_complete`
  gained a backward-compatible `phase_gate` so storage and recall occupy
  opposite theta phases (Hasselmo 2002).

| file | cat | landed as | notes |
|------|-----|--------------|-------|
| `core/precision_bus.py` | ✅ | `core/pc_precision.py` | Welford EMA + multi-channel compose/standardize, step-unit (no `ctx`). Opt-in `precision_mode="welford"` in `pc_graph_learn`; EMA stays default. |
| `core/neuromodulator.py` | ✅ | `core/pc_neuromod.py` | 4 channels from graph ε / free energy / value ε → `precision_gains` (ACh→sensory Π, DA→value Π) + EFE β (NE). DA RPE is now the value node's §5b temporal-edge ε (the `Δvalue` proxy is gone); no external reward/TD. |
| `core/world_model.py` | ✅ | `core/pc_neuromod.py` (`neuromod_curiosity`) | `wm_learning_progress` salvaged as `pe_long−pe_short` on the `world_model` node ε; augments `epistemic_value` (replaces the noise-blind inverse-Π proxy). |
| `core/attention.py` | ✅ | `core/pc_attention.py` | Divisive norm + IOR → per-slice sensory Π gain via `scale_node_precision`. |
| `core/oscillator.py` | ✅ | `core/pc_oscillator.py` | Theta/gamma + PAC, step-unit; wires the §3-deferred HC theta encode/retrieve gate (`phase_gate`). |

---

## 5. Region detail — internal microcircuit for the load-bearing nodes (PRIORITY 5) — ✅ DONE (laminar cortex + working memory; rest deferred)

Already present as single nodes in `init_region_graph`. These files are
**specs** for giving those nodes more internal structure. **Laminar cortex
and working memory integrated**; together with **multi-cycle eligibility
traces** (the §5b temporal deferral) this milestone gives the load-bearing
nodes proper internal structure under the one rule. The remaining rows
(cerebellum, BG, PFC, thalamus, interneuron, astrocyte) are low marginal
capability now and stay single-node specs.

Every enrichment is **sub-populations + intra-region edges (door 1 + door
2)** under the existing one rule — no new module type, no second plasticity
mechanism — gated behind opt-in flags so the default graph is
byte-identical (the established `gabor_foveal_init` / `temporal_edges`
pattern). The public region names, the `pc_brain` read-out indices, and the
§5b temporal-edge targets (`value`, `world_model`) are all preserved.

* **`core/pc_graph.py::init_region_graph(laminar_cortex=True)`** — each
  cortical region splits into the canonical PC microcircuit (Bastos 2012):
  **L4 = ε** (granular error, its own Π — error precision separable from the
  cause's), **L2/3 = μ** (the cause, *kept as the existing `cortex_lN`
  index* so every read-out resolves), **L5 = prediction** (the descending
  output). Intra-region edges `(L2/3→L4)`, `(L2/3→L5)`; inter-region edges
  re-origin from L5 into the lower region's L4; deep consumers read
  `L5_cortex_l3`. Appended after the 11 base nodes (`[c1_l4, c1_l5, …,
  c3_l5]` at 11–16), so base indices are unchanged.
* **`core/pc_graph.py::init_region_graph(working_memory=True)` +
  `apply_wm_persistence_init`** — a `pfc` persistence node with a **leaky
  temporal self-edge** (`(pfc→pfc)` `w_dyn`, init `wm_persistence_gain·I`)
  holding μ across cycles when input is absent (bump attractor → leaky
  integrator). Fed by / feeding back the deep cortical cause. Reuses the §5b
  `w_dyn` primitive + the existing `leak`; the only new thing is the named
  persistence gain. **PFC folds in here** (its persistence *is* the WM node;
  hierarchical goals remain free via graph depth).
* **Eligibility traces** (`init_pc_graph_params(eligibility=True)`) — closes
  the §5b deferral; see §5b below.

| file | cat | use as spec for |
|------|-----|-----------------|
| `core/cortex.py` | ✅ | **Done** — laminar microcircuit (L4=ε, L2/3=μ, L5=prediction), opt-in `laminar_cortex`. Legacy file deleted. |
| `core/working_memory.py` | ✅ | **Done** — leaky temporal self-edge on a `pfc` node, opt-in `working_memory`. Legacy file deleted. |
| `core/plasticity.py` | ✅ | **Done in §5b/§6** — multi-cycle eligibility traces (rate e-prop) on the `w_dyn` edges, opt-in `eligibility`. Legacy file deleted. |
| `core/cerebellum.py` | 🔴 discard | Rate forward-model (granule kWTA expansion + Purkinje LTD). The forward-model role is the live `motor→sensory` generative edge; sparse-expansion pattern-sep already lives in DG (`pc_hippocampus`, `dg_sparsity`). Recover the kWTA node-nonlinearity only if a task needs cerebellar capacity. |
| `core/vta.py` | ✅ | **Done in §5b** — `value(t−1)→value(t)` `w_dyn` edge; ε_value = TD error, DA = its read-out. D2 / reward-baseline / eligibility-snapshot hydraulics dropped. Legacy file deleted. |
| `core/basal_ganglia.py` | 🔴 discard | Spiking D1/D2 actor+critic. Fully covered: `policy` node + `efe_select` (selection), `value` node + the §5b TD edge (critic). Spiking WTA adds nothing to the rate path. |
| `core/pfc.py` | 🔴 discard | Thin wrapper over the spiking `working_memory`. WM is the `pfc` leaky self-edge node (`working_memory=True`); the theta encode/retrieve gate is in `pc_oscillator`. Nothing left to port. |
| `core/thalamus.py` | 🔴 discard | Spiking relay+TRN. Routing/gating = `scale_node_precision`; afferent attention gain = `pc_attention`. Burst/tonic is spike-only, no rate analogue wanted. |
| `core/interneuron.py` | 🔴 discard | Spiking FS PV+ pool (biophysical k-WTA). Lateral inhibition / competition / divisive normalization is `pc_attention`. |
| `core/astrocyte.py` | 🔴 discard | Ca²⁺/D-Serine/ATP field. Precision = `pc_precision`/`pc_neuromod`; the ATP sleep-pressure trigger was explicitly replaced by free-energy pressure in `pc_sleep`. Energy budgeting was a neuromorphic concern — dropped with the track. |

---

## 5b. Temporal credit (plan_unification §6) — ✅ DONE

Closes the deferrals the earlier milestones parked: §3 `sequence_memory`,
§5 `vta`, and the §4 DA proxy. **One additive substrate primitive does
all three** — a **temporal generative edge** `w_dyn`: a generative edge
(door 2 of §0, *not* a new door) whose source is the **previous cycle's
belief** μ_prev (dynamic / generalized predictive coding, Friston 2008).
It learns by the **same** one rule, `ΔW = η·Π·ε·φ(μ_prev)` — no second
plasticity mechanism (the exact reason `sequence_memory` was held back).

Substrate change (all backward-compatible, additive):

* `PCGraphParams.dyn_edges` + `PCGraphState.w_dyn` (temporal weights) and
  `mu_prev` (the 1-cycle delayed-source carry); `pc_graph_roll` advances
  the carry at each cycle boundary (`pc_graph_step` / `pc_brain_cognitive_step`
  call it). A temporal edge feeds its destination's ε but, its source
  being the frozen carry, contributes no relaxation up-term.
* **Empty by default** ⇒ a static-only graph is byte-identical; the whole
  prior suite stays green. `init_region_graph(temporal_edges=True)` adds
  the two canonical self-edges (named nodes, no magic index):
  * `value(t−1)→value(t)` — the **TD bootstrap** (closes §5 `vta.py`).
    The value node's own ε **is** the temporal-difference error; the
    bootstrap γV(s′) is the edge's prediction, so there is no separate
    critic/TD update and none of the D2 / reward-baseline / eligibility-
    snapshot hydraulics.
  * `world_model(t−1)→world_model(t)` — the **sensory-transition** cause
    (closes §3 `sequence_memory.py`). Composed with the static
    `world_model→sensory` edge it predicts the *next* sensory state; the
    bespoke Oja / k-WTA transition rule is dropped (pattern-separation
    already lives in the EC/HC episodic store).
* `core/pc_neuromod.py`'s **DA** channel now reads the value node's ε
  directly (closes the §4 `Δvalue` proxy): with the value temporal edge
  present that ε is the real RPE.

Tests: `tests/test_phaseu_pc_temporal.py` (7/7) — inert-when-absent,
true 1-cycle delay, a transition edge learns a moving sequence and
ablating it degrades next-state prediction, the value edge produces a
TD-like error (vanishes when reward is predictable, spikes on surprise),
DA tracks that ε, and the full region graph trains with temporal edges
live.

**Both deferrals now closed in §5** (region-detail milestone):
* **Multi-cycle eligibility traces** ✅ — `init_pc_graph_params(eligibility
  =True)` adds a decaying per-`w_dyn`-edge trace `elig` (state) +
  `elig_decay` (param). A temporal edge then commits `ΔW = η·ξ_dst ⊗ elig`
  (trace of `φ(μ_prev[src])`) instead of the 1-cycle value — the rate
  analogue of e-prop (Bellec 2020), the *one rule extended in time*, not a
  second mechanism. Scope: `w_dyn` (slow) edges only (spatial edges already
  get backprop-grade credit from relaxation). Modulator: the destination
  node's own `ξ = Π·ε`, which for the value/policy temporal edges *is* the
  RPE (DA) — no separate global broadcast. Bridges credit across a gap >1
  cycle that a 1-cycle carry cannot (`tests/test_phaseu_pc_eligibility.py`).
* **`working_memory`** ✅ — a leaky temporal self-edge on a `pfc` node
  (`init_region_graph(working_memory=True)` + `apply_wm_persistence_init`);
  see §5.

Legacy `sequence_memory.py`, `vta.py`, `working_memory.py` and
`plasticity.py` are deleted (git history is the backup).

---

## 6. The neuromorphic spiking-PC track — DROPPED, DISCARDED

This set was the seed of an optional spike/event-driven inference layer
(the path to Loihi/Akida/TrueNorth). **That track is dropped** (§0 goal):
the forward path deepens *rate*-PC and, later, *dendritic* PC — both
rate-mode. None of these files reduces to a substrate door (§0) the rate
system lacks; every rate-relevant idea was already absorbed in §3–§5b. They
are therefore **discarded** (deleted with the rest of `legacy/`, §10). Git
history is the spec backup — spiking originals at `8009b22`, the
`legacy/`-preserved copies at `0dec6c4`.

| file | decision | already-covered-by / recover-when |
|------|----------|-----------------------------------|
| `core/error_neuron.py` | 🔴 discard | Spiking PC area (Rao-Ballard ε/μ, anti-Hebbian TD). Its content **is** the substrate: ε=μ−pred under the one rule, and laminar L4=ε / L2-3=μ (`laminar_cortex=True`). Nothing left to port. |
| `core/neuron.py` | 🔴 discard | AdEx/LIF membrane step — spike-domain only; the rate substrate has no membrane. Re-derive from git if a spiking layer is ever built. |
| `core/synapse.py` | 🔴 discard | Conductance channels (AMPA/NMDA/GABA). The NMDA Mg²⁺-block supralinearity is the one **dendritic-PC** nugget — recover it *then* as the apical nonlinearity, not now: dropping it into a rate node today is a bridge to nothing. |
| `core/state.py` | 🔴 discard | Spiking pytree containers. The non-spiking leaves already live elsewhere: `OscillatorState`→`pc_oscillator`, `EligibilityState`→`pc_graph` (`eligibility=True`), homeostatic EMAs→`pc_precision`. |
| `core/spike_encoder.py` | 🔴 discard | `gaussian_population_encode` already live as `sensory/population_code.py`; `poisson_spike` is the rate→spike bridge — only the dropped spiking layer needs it. |
| `core/sparse.py` | 🔴 discard | Real BCOO sparse compute. `pc_structural` already implements the dense-mask version of this exact pre-allocated-budget trick (and credits `sparse.py` in its docstring). True sparse compute is a scale / event-driven concern — recover when the graph outgrows dense masks. |
| `core/brain_graph.py` | 🔴 discard | Old spiking orchestrator — wholly superseded by `pc_brain`. Only `DelayBuffer` (O(1) conduction delay) is novel: the delay>1 generalization of the §5b `mu_prev` 1-cycle carry — recover when multi-step temporal edges land. |
| `core/plasticity.py` | ✅ | Already deleted (integrated §5) — multi-cycle eligibility trace on the `w_dyn` edges (`eligibility=True`). |

---

## 7. Discarded (not restored — no place in the new system)

| file | why |
|------|-----|
| `core/m1.py` | Node-perturbation REINFORCE; plan §4/§U.5 explicitly removes it — action is inference (`pc_active`), not policy gradient. |
| `core/learning_pipeline.py` | Orchestration glue calling the deleted per-region STDP/TD updates; the one rule replaces all of it. |
| `core/receptor.py` | Receptor-level pharmacology (Hill curves). Rate-PC models neuromodulation as precision gains; this granularity adds nothing and no consumer remains. |

(If ever needed, all three are still in git history at commit `8009b22`.)

---

## 8. Suggested order

1. ~~**Embodied reach** (§1 + `proprioception`): body adapter + babble →
   `pc_brain_act` reach on MJX.~~ ✅ **Done** — `embodiment/` +
   `sensory/proprioception.py`; run the babble→reach on the Colab MJX box.
2. ~~**Sleep & memory** (§3): `pc_sleep.py` offline replay + episodic store.~~
   ✅ **Done** — `core/pc_memory.py` (replay buffer + episodic store),
   `core/pc_sleep.py` (SWS replay + REM rollout + free-energy FSM),
   `core/pc_hippocampus.py` (HC node group) + the `entorhinal` node.
   First genuinely new capability beyond the current substrate.
   (`sequence_memory` transition memory now closed in §5b.)
3. ~~**Neuromod & curiosity** (§4): precision drivers + epistemic EFE term.~~
   ✅ **Done** — `core/pc_precision.py` (Welford EMA + multi-channel,
   opt-in `precision_mode="welford"`), `core/pc_neuromod.py` (4 channels
   from graph signals → `precision_gains` + EFE β + learning-progress
   curiosity), `core/pc_attention.py` (divisive-norm + IOR → per-slice
   sensory Π gain), `core/pc_oscillator.py` (theta/gamma + PAC + the
   §3-deferred HC theta encode/retrieve gate). All drive existing hooks;
   `epistemic_value` gained an opt-in learning-progress term.
4. ~~**Vision** (§2): retina→lgn→sensory adapter; then visual tasks.~~
   ✅ **Done** — `sensory/retina.py` + `sensory/lgn.py` + `sensory/vision.py`
   (the vision adapter: encode → sensory clamp, active saccades by
   `efe_select`), plus the opt-in Gabor edge-init in `core/pc_graph.py`
   (`FovealGaborInit`). V1/V2/V4 are the live cortical hierarchy; audio is
   deferred to a later milestone.
5. ~~**Temporal credit** (plan §6): dynamic `w_dyn` edges + temporal
   carry.~~ ✅ **Done** (§5b) — `core/pc_graph.py` temporal generative
   edge (`dyn_edges` / `w_dyn` / `mu_prev` carry / `pc_graph_roll`),
   opt-in `init_region_graph(temporal_edges=True)` (value TD bootstrap +
   world-model sensory transition), DA channel rewired to the value-node
   ε. Closes the §3 `sequence_memory`, §5 `vta` and §4 DA-proxy
   deferrals; one rule, additive, static graphs byte-identical
   (`tests/test_phaseu_pc_temporal.py` 7/7). Multi-cycle eligibility +
   `working_memory` self-edge stay deferred.
6. ~~**Region detail** (§5): internal microcircuit for the load-bearing
   nodes.~~ ✅ **Done** — `init_region_graph(laminar_cortex=True)` (each
   cortical region → L4=ε / L2/3=μ / L5=prediction, Bastos 2012),
   `init_region_graph(working_memory=True)` + `apply_wm_persistence_init`
   (a `pfc` leaky temporal self-edge holding μ across cycles), and
   `init_pc_graph_params(eligibility=True)` (multi-cycle eligibility traces
   on the `w_dyn` edges — the §5b deferral closed). All opt-in, default
   graph byte-identical, one rule, read-outs + §5b temporal targets
   preserved (`tests/test_phaseu_pc_laminar.py`,
   `tests/test_phaseu_pc_working_memory.py`,
   `tests/test_phaseu_pc_eligibility.py` 11/11). Cerebellum / BG / thalamus
   / interneuron / astrocyte stay single-node specs (low marginal capability
   now). **spiking-PC** (§6) remains the parallel neuromorphic track.

## 9. Starting a new chat (per subsystem)

Open a fresh conversation scoped to ONE subsystem, e.g.:

> "Integrate the embodiment layer (§1 of `LEGACY_INTEGRATION.md`) into the
> PC substrate. Reference `legacy/embodiment/*` and `legacy/sensory/
> proprioception.py`. Build a body adapter that drives `pc_brain`
> (`pc_brain_cognitive_step` + `pc_brain_act` + `pc_brain_learn_forward`),
> babble → reach on the MJX arm, validate on reach success. Do not import
> `legacy/`; use it as spec, write new code under `embodiment/`."

Keep each chat to one section so context stays focused. New code lands in
`core/` (substrate-internal), `embodiment/`, `sensory/` (adapters).

---

## 10. Final disposition — `legacy/` deleted in full

Per the §0 contract every file resolves to exactly one of two states:
**already integrated** (§1–§5b) or **discarded** (§2 ventral/auditory; §5
cerebellum/BG/pfc/thalamus/interneuron/astrocyte; §6 the whole spiking set).
Nothing reduces to a new substrate door the rate system lacks. So — per the
project's own rule (*no leftover legacy, no bridges, delete after
integrating*) — the entire tree is removed:

```
git rm -r legacy/
```

This deletes `legacy/core/*`, `legacy/sensory/*` and `legacy/README.md`. No
live module, test, notebook or `pyproject` package references `legacy/`
(verified — packaging is `core` + `sensory` + `embodiment` only), so the
deletion is behaviour-neutral and the suite stays green.

**Recovery map** — the spec backup is git, not a folder:

| want it later | recover from | rebuild as |
|---------------|--------------|------------|
| spiking-PC layer (neuron / synapse / state / error_neuron) | `8009b22` (originals), `0dec6c4` (preserved) | a spike-domain analogue of `pc_module` — same one rule, settled by spikes |
| **dendritic PC** | forward design; NMDA supralinearity ref in `synapse.py` @ `8009b22` | apical/basal compartments per node, local apical error |
| real sparse compute | `0dec6c4 : legacy/core/sparse.py` | swap `pc_structural`'s dense masks for BCOO when the graph outgrows them |
| multi-step temporal edges (delay > 1) | `0dec6c4 : legacy/core/brain_graph.py::DelayBuffer` | generalize the §5b `mu_prev` 1-cycle carry |
| cerebellar / kWTA forward-model capacity | `0dec6c4 : legacy/core/cerebellum.py` | an opt-in kWTA node nonlinearity |
| audio | `0dec6c4 : legacy/sensory/auditory.py` | a fresh `sensory/audio.py` adapter (`mgn` = `lgn`, already free) |

After this `legacy/` does not exist and the integration is closed.
