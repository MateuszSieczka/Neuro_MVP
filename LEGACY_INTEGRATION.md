# Legacy → PC-substrate integration plan

**Purpose.** After Phase U the live system is a clean predictive-coding
substrate (`core/` = `backend, free_energy, pc_module, pc_graph, pc_active,
pc_brain, pc_structural`). The old spiking system was moved to `legacy/`
**as reference only — it is not importable** (its imports point at deleted
modules on purpose). This document decides, per file, what to **restore &
fix**, what to **rewrite on the substrate**, and what was **discarded**,
and gives an integration order. Use it to start a fresh chat per subsystem.

Goal: a functioning, **neuromorphic-friendly** neuro-AGI. Everything below
is judged against that — local learning, rate-PC now, optional spiking-PC
layer later for event-driven chips.

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

## 1. Embodiment — get an embodied reach on `pc_brain` (PRIORITY 1)

This is the first milestone (plan §12): a body adapter driving
`pc_brain`, validated on reach success.

| file | cat | integrate as | notes |
|------|-----|--------------|-------|
| `embodiment/body_interface.py` | 🟢 | I/O adapter (the contract) | Clean brain↔world boundary already (action→sensory/reward/done). Extend to **continuous** `joint_command` in, flat-rate `sensory` out. This becomes the canonical interface `pc_brain` plugs into. |
| `embodiment/mjx_arm_body.py` | 🟢 | body (physics) | MuJoCo-MJX 2-link arm, substrate-agnostic. Strip the `ActionBrain` coupling; keep physics + proprio/sensory coding. Returns reafference for `pc_brain_learn_forward`. |
| `embodiment/babbling_env.py` | 🟢 | training mode | OU-noise babbling to learn the motor→sensory forward model (exactly U.5 babble phase). Decouple from M1; feed random `command` to `pc_brain_learn_forward`. |
| `embodiment/reacher_env.py` | 🟡 | factory | Rebuild: pair `MjxArmBody` with `init_pc_brain` instead of `ActionBrainParams`. Tiny. |
| `embodiment/mjx_run_loop.py` | 🟡 | driver | **Keep the lesson, not the code**: many cycles inside one `jax.lax.scan`/`filter_jit` to avoid per-step retrace (16.5 s/cycle bug). Rewrite the scan body to call `pc_brain_cognitive_step` + `pc_brain_act`. |
| `embodiment/run_loop.py` | 🟡 | driver (simple) | CPU/non-MJX driver for quick tests; rewrite for `pc_brain`. |
| `embodiment/gridworld.py` | 🟢 | env (discrete) | Self-contained discrete task; good cheap probe for value/policy nodes. Implements the interface; just retarget output to the sensory clamp vector. |
| `embodiment/bandit.py` | 🟢 | env (discrete) | Minimal value/EFE testbed. Reusable as-is behind the new interface. |
| `embodiment/visual_grid.py` | 🟡 | env (vision) | Gridworld seen through `retina`; only after the sensory front-end is back (§2). |

---

## 2. Sensory front-end — perception that clamps the sensory node (PRIORITY 2)

In the substrate, "sensory" is the clamped observation node. These turn
raw input into the **flat rate vector** that clamp. Drop Poisson spiking;
keep the (rate-friendly) feature geometry. Vision first, audio later.

| file | cat | integrate as | notes |
|------|-----|--------------|-------|
| `sensory/proprioception.py` | 🟢 | encoder | Population-coded joint angles+velocities → `[0,1]` vector. Already rate, body-agnostic. **Directly feeds the embodiment sensory clamp** — do this with §1. |
| `sensory/retina.py` | 🟢 | encoder | Scale-invariant image→fixed vector (pyramid + DoG + fovea). Math is rate-friendly; replace the Poisson-rate output with the rate vector itself. Highest-value vision piece. |
| `sensory/lgn.py` | 🟢 | encoder stage | Contrast gain control + tonic floor → keeps the clamp in a sane range. Reusable normalization. |
| `sensory/v1.py` | 🟡 | node init | The **Gabor initialisation** is a good generative prior for the first cortical edge `cortex_l1→sensory`; reimplement as edge-weight init, drop `core.cortex`/STDP dependency. |
| `sensory/sensory_stack.py` | 🟡 | composition | Recompose retina→lgn→(V1 node) as the sensory adapter; the saccade info-gain = `epistemic_value` on the sensory node. |
| `sensory/ventral.py` | 🟡 | extra nodes | V2/V4 = just deeper cortical nodes in the graph; skeleton only, low code value. |
| `sensory/auditory.py` | 🟢 | encoder (audio) | Cochlea mel-filterbank → fixed vector; same treatment as retina, lower priority. |

---

## 3. Memory & sleep — capabilities the substrate does NOT yet have (PRIORITY 3)

Episodic memory + offline replay are real neuro-AGI capabilities missing
from the substrate. Concept is in the plan (§U.2 HC, §4 sleep = offline
FEP). Data structures are substrate-agnostic and reusable; the dynamics
get rewritten as graph operations.

| file | cat | integrate as | notes |
|------|-----|--------------|-------|
| `core/replay_buffer.py` | 🟢 | data primitive | Fixed-capacity SoA ring buffer of experiences. JIT-safe, substrate-agnostic — reuse verbatim to store `(sensory, command, …)` for replay. |
| `core/episodic_memory.py` | 🟢 | data primitive | Pattern-separated key/value store (one-shot write, recall). Reusable; the "DG-sparse code" becomes a deep-node belief. |
| `core/sequence_memory.py` | 🟢 | data primitive | Outer-product transition matrix `P(s_{t+1}|s_t)` = temporal credit (§6). Reusable as the seed of dynamic edges `w_dyn`. |
| `core/sleep.py` | 🟢 | offline mode FSM | WAKE/SWS/REM state machine. Reuse the FSM; swap the ATP trigger for a substrate signal (e.g. accumulated free energy / time). |
| `core/sleep_replay.py` | 🟡 | offline mode | **This is the `pc_sleep.py` we discussed.** Rewrite SWS reverse-replay + REM rollout as: unclamp sensory → sample from generative model → `pc_graph_relax` → `pc_graph_learn`. Same one rule, offline. |
| `core/hippocampus.py` | 🟡 | composition | DG+CA3+CA1 wrapper; rewrite as a hippocampal node group (pattern sep/completion) on the graph using the three primitives above. |
| `core/ec.py` | 🟡 | node | Entorhinal hub = a convergence node feeding the HC group; reimplement as a multi-parent node. |

---

## 4. Neuromodulation, precision & curiosity (PRIORITY 4)

Neuromodulators = precision controllers (already hooked via
`scale_node_precision` / `precision_gains`). These files supply the
*signals* that drive those gains, and the curiosity term for EFE.

| file | cat | integrate as | notes |
|------|-----|--------------|-------|
| `core/precision_bus.py` | 🟢 | precision math | Welford-EMA precision + multi-channel compose/standardize. Richer than `pc_module`'s inline EMA — fold into node precision tracking. |
| `core/neuromodulator.py` | 🟡 | precision drivers | The 4-channel formulas (ACh novelty, NE surprise, DA RPE/reward-rate, 5-HT stability) → compute these from graph signals and feed `precision_gains` (ACh→sensory Π, DA→value Π, NE→EFE β). |
| `core/world_model.py` | 🟡 | curiosity | Salvage `wm_learning_progress` / curiosity (Oudeyer) → the **epistemic** term of `efe_select`. The model itself is already the `world_model` graph node. |
| `core/attention.py` | 🟡 | precision gain | Divisive normalization + IOR + NE gain = spatial precision control on sensory sub-fields; reimplement as `precision_gains` over node slices. |
| `core/oscillator.py` | 🟢 | timing | Pure-functional theta/gamma + PAC, substrate-agnostic. Use to phase-gate relaxation / replay (encode vs retrieve), per plan §4 "oscillations stay". |

---

## 5. Region detail — references for enriching graph nodes (PRIORITY 5)

Already present as single nodes in `init_region_graph`. These files are
**specs** for giving those nodes more internal structure later; not
load-bearing now.

| file | cat | use as spec for |
|------|-----|-----------------|
| `core/cortex.py` | 🟡 | Laminar microcircuit (L4=ε, L2/3=μ, L5=prediction) — the canonical PC mapping (Bastos 2012) for turning each cortical node into a proper 3-population module. |
| `core/cerebellum.py` | 🟡 | Forward model: granule kWTA expansion + microzones as the structure of the `motor↔cerebellum` edges. |
| `core/vta.py` | 🟡 | Temporal-PC value: TD target as a temporal edge value(t)→value(t+1) (§6). |
| `core/basal_ganglia.py` | 🟡 | Policy precision: D1/D2 opponent + STN as the policy-node precision/EFE selection mechanics. |
| `core/working_memory.py` | 🟡 | Persistence: WM = a recurrent node holding `μ` across cycles (bump attractor → leaky self-edge). |
| `core/pfc.py` | 🟡 | Hierarchical goals = deep nodes whose preferred state propagates down (clamp a deep node, §U.5). |
| `core/thalamus.py` | 🟡 | Routing/gating = precision-gated relay between nodes (low priority). |
| `core/interneuron.py` | 🟡 | Lateral inhibition / competition / normalization within a node population (sparse codes). |
| `core/astrocyte.py` | 🟡 | (Low) metabolic/energy + sleep-pressure signal; only if you model energy for neuromorphic budgeting. |

---

## 6. Neuromorphic spiking-PC track (PARALLEL, for event-driven chips)

The substrate is rate-mode (iterative relaxation). True neuromorphic
hardware is spike/event-driven. This set is the seed of an **optional
spiking-PC inference layer** that keeps the same local rule but settles
via spikes — the path to Loihi/Akida/TrueNorth deployment.

| file | cat | role in the spiking-PC layer |
|------|-----|------------------------------|
| `core/error_neuron.py` | 🟡 | **The seed.** Spiking PC area (AdEx ε/μ populations). Rewrite as the spike-domain analogue of `pc_module` (relaxation by spiking, not `fori_loop`). |
| `core/neuron.py` | 🟢 | AdEx/LIF step (LIF = `a=b=0`, maps to LIF-only chips). Reusable spiking primitive. |
| `core/synapse.py` | 🟢 | Conductance channels (AMPA/NMDA/GABA). Reusable. |
| `core/plasticity.py` | 🟡 | STDP = spike-domain `Δw=η·Π·ε·φ(μ)`; **eligibility traces here are the temporal-credit mechanism (§6)** — relevant even to the rate path. |
| `core/spike_encoder.py` | 🟢 | `gaussian_population_encode` reusable now (rate); `poisson_spike` is the rate→spike bridge for the spiking layer. |
| `core/sparse.py` | 🟢 | BCOO sparse connectivity + `matvec` + synaptogenesis/prune. **Real sparse compute** vs `pc_structural`'s dense-mask trick — needed for scale and event-driven efficiency. |
| `core/state.py` | 🟢 | Spiking state containers; restore together with `neuron`/`synapse`. |
| `core/brain_graph.py` | 🟡 | (Low) salvage only `DelayBuffer` (conduction delays → §6 temporal edges) and the region topology as documentation; the rest is superseded by `pc_graph`. |

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

1. **Embodied reach** (§1 + `proprioception`): body adapter + babble →
   `pc_brain_act` reach on MJX. Proves U.5 on a real body. **Start here.**
2. **Sleep & memory** (§3): `pc_sleep.py` offline replay + episodic store.
   First genuinely new capability beyond the current substrate.
3. **Neuromod & curiosity** (§4): precision drivers + epistemic EFE term.
4. **Vision** (§2): retina→lgn→sensory adapter; then visual tasks.
5. **Region detail** (§5) and **spiking-PC** (§6): as needed; §6 is the
   neuromorphic-deployment track and can run independently.

## 9. Starting a new chat (per subsystem)

Open a fresh conversation scoped to ONE subsystem, e.g.:

> "Integrate the embodiment layer (§1 of `LEGACY_INTEGRATION.md`) into the
> PC substrate. Reference `legacy/embodiment/*` and `legacy/sensory/
> proprioception.py`. Build a body adapter that drives `pc_brain`
> (`pc_brain_cognitive_step` + `pc_brain_act` + `pc_brain_learn_forward`),
> babble → reach on the MJX arm, validate on reach success. Do not import
> `legacy/`; use it as spec, write new code under `embodiment/`."

Keep each chat to one section so context stays focused. `legacy/` is
read-only reference; new code lands in `core/` (substrate-internal),
`embodiment/`, `sensory/` (adapters).
