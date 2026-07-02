# Phase 3 — temporal depth: precision-gated clocks and the key–door credit problem

**Written:** 2026-07-02. **Depends on:** `FINDINGS_PHASE2.md` (spatial-depth
axis: metric unification §2A, collapse §2B, node-type gating §2C). This
document is the **temporal** axis of the same programme, proposed by the user
and verified/refined here against the actual `core/` substrate. It is written
so another chat can implement it from this file alone.

---

## 0. Problem statement

Two symptoms, one mechanism:

1. **Multi-timescale prediction.** The substrate has one clock: every node's
   `mu_prev` (the temporal-edge carry) advances once per cognitive cycle
   (`pc_graph_roll`, `pc_graph.py:1203`), unconditionally, for every node. A
   node meant to represent "which room I'm in" changes exactly as often as the
   node representing "current retinal frame" — there is no way for a deep
   node to treat a whole corridor traversal as *one* event.
2. **Long-horizon credit (key → door).** The existing temporal-credit
   machinery (`dyn_edges` + `elig`, `pc_graph_learn`, `pc_graph.py:1146-1165`)
   bridges credit across cycles via a decaying eligibility trace,
   `trace ← elig_decay·trace + φ(μ_prev[src])`. Over ~1000 idle cycles between
   picking up a key and reaching a door, this trace either decays to
   ~0 (if `elig_decay` is short) or saturates to a fixed point set by whatever
   `μ_prev[src]` happens to be at each of those 1000 steps (if long) — in
   neither case does it preserve "the key" specifically. Spatial credit
   assignment over long action sequences is an open problem this substrate has
   not yet addressed.

## 1. Relation to Phase 2 — orthogonal axis, same principle

Phase 2 asked: *should Π decide the graph's spatial structure (metric, node
type)?* Phase 3 asks: *should Π decide the graph's temporal structure (which
nodes tick on which clock)?* Same answer, same discipline: no builder tag, a
computed runtime quantity derived from precision/error statistics the
substrate already tracks. This is the temporal counterpart of Phase 2C
(precision-gated amortised↔iterative interpolation) — where 2C gates *how a
node is inferred*, Phase 3 gates *when a node's temporal state advances*.

**Verdict on the user's proposal:** the core idea — bottom-up surprise gates a
higher node's temporal update — is correct and is a well-precedented
computational-neuroscience/RL idea (event segmentation, options, jumpy world
models; §8). But the proposal as written gates only the **temporal carry**
(`mu_prev`). Read literally, that is insufficient to deliver the claimed
effect (see §3.3): the node's *actual* belief `μ_j` still updates every cycle
under ordinary relaxation, driven by bottom-up error, regardless of whether
its temporal copy is frozen. Three concrete corrections turn the idea into a
mechanism that actually delivers what it promises — given in §3.

## 2. Mechanism — the gate itself

### 2.1 Surprise signal: reuse the existing precision statistics, don't invent a threshold

Every node already carries `pe_mean`, `pe_var` (Welford EMA of its own error,
`pc_precision.welford_precision_update`) and, through the static generative
edges, an error stream for each of its children. Define the **bottom-up
surprise of node `j`** as the precision-weighted z-score of the errors of the
nodes `j` directly predicts (its static-edge children `{c : (j,c) ∈ edges}`):

```
surprise_j = mean_c( |eps_c| · sqrt(pi_c) )
```

`eps_c · sqrt(Π_c)` is the error expressed in **standard-deviation units**: at
the EMA's own stationary point `Π_c = 1/Var(eps_c)`, so this quantity is
unit-variance by construction whenever `c`'s local model is behaving normally.
This is the same z-score logic already used implicitly by the Welford EMA —
no new statistic, just a different combination of ones already computed in
`_graph_relax_step` (`xi[c] = pi[c]*(mu[c]-pred_full[c])`; `surprise_j` is
`mean_c |xi[c]| / sqrt(pi[c])`, i.e. `mean_c |eps_c|·sqrt(pi[c])`, both
already-computed quantities).

**Threshold is dimensionless.** Because `surprise_j` is expressed in sigmas,
the gate

```
gate_j = sigmoid(surprise_j − k)
```

uses `k` as "how many standard deviations of surprise count as a genuine
violation" (k ≈ 1–2, Shewhart-control-chart territory) — a representational
choice analogous to `var_floor`, not a value tuned against task success. This
is what makes the gate compatible with the "no test-tuned magic constant"
discipline the rest of the substrate holds itself to.

**Causality.** `surprise_j` for cycle `t`'s gating decision must be computed
from information available *before* cycle `t`'s relaxation — i.e. from the
state carried out of cycle `t−1`. Since `mu` at the start of cycle `t` **is**
the relaxed belief from cycle `t−1`, calling `pc_graph_errors(state, params)`
at the top of the cycle (before the new sensory clamp) gives exactly "how
well did last cycle's weights predict last cycle's settled belief" — a fully
causal proxy, using an already-existing function. No new state is required
beyond what `PCGraphState` already carries (`pi`, and `mu` itself).

### 2.2 What "gated" must mean — three things gated together, not one

Gating only the `mu_prev` copy (the user's literal proposal) leaves two gaps.
Fixing them requires gating three things **with the same `gate_j`**, and two
of the three are direct reuse of existing primitives — not new mechanisms.

**(a) The temporal carry itself** (`mu_prev`) — as proposed:

```
mu_prev_j ← gate_j · mu_j + (1 − gate_j) · mu_prev_j
```

This alone controls what `dyn_edges` read as their source. It is necessary
but, alone, insufficient (§3.3).

**(b) The eligibility trace advance** (per `dyn_edge` whose source is `j`) —
must freeze in lockstep with (a), else the trace still decays/refreshes every
idle cycle even while `mu_prev` is frozen:

```
elig_e ← (1 − gate_src) · elig_e  +  gate_src · (elig_decay · elig_e + φ(mu_prev_src))
```

When `gate_src ≈ 0` the trace is held byte-for-byte across the idle interval
(no decay, no new accumulation); when an event fires it advances exactly once.
This is what turns "1000 idle cycles" into "1 semantic tick" for the
eligibility mechanism specifically (§4).

**(c) The node's own belief `μ_j` during the *current cognitive cycle's*
relaxation** — this is the gap in the literal proposal, and the fix is to
**reuse the existing `hold`/clamp primitive**, not add a new one. `μ_j`
already updates every cycle via ordinary gradient relaxation
(`_graph_relax_step`), driven by whatever bottom-up/top-down error currently
reaches it — this is *independent* of whether its temporal carry is frozen.
If `μ_j` is left free, "L3 stays parked on the key" is false: L3 keeps
drifting toward "corridor" the whole time, and by the time the door arrives it
no longer encodes "holding key" at all, defeating the credit-assignment payoff
(§4).

The fix: when `gate_j = 0` for cycle `t`, add `j` to that cycle's clamp set
with `clamp_value = state.mu[j]` (a **self-clamp** — hold the node at its own
last value, the same `hold[j] = True` codepath `_graph_relax_step` already
executes for any clamped node, just with the value sourced from the node
itself instead of an external observation). This is decided once per cycle,
in the cognitive-cycle orchestration (`pc_brain_cognitive_step`,
`pc_brain.py:144`), from `gate_j` computed per §2.1 — **not** a new field on
`_graph_relax_step`; it is a different (dynamic, per-cycle, precision-derived)
source for the existing `clamp` argument of `pc_graph_relax`.

**Naming caution.** Do not conflate this with `PCBrainParams.perceptual_nodes`
/ `hold_nodes` (`pc_active.pc_act_infer`), which holds nodes for a completely
different reason (perception-fixed-during-action-inference, an
active-inference principle, not a temporal-clock decision). Name the new
per-cycle dynamic clamp set something distinct, e.g. `frozen_this_cycle` /
`temporal_gate_clamp`, to keep the two concepts from being accidentally
merged or from silently interacting when both apply to the same node.

### 2.3 Composing with existing persistence (optional accelerant, not required)

`apply_wm_persistence_init` already gives a node's self `dyn_edge` a
near-identity gain so its temporal prediction pulls `μ_j(t) ≈ gain·φ(μ_prev_j)`
— a soft leaky-integrator persistence. Combined with the self-clamp of §2.2(c),
this is redundant-but-harmonious: while `gate_j = 0` the self-clamp holds `μ_j`
*exactly*, so the persistence edge's pull is moot (nothing to integrate
against); the moment `gate_j` opens, the persistence edge gives the now-free
node a *soft* starting point (the old belief) rather than an unconstrained
jump, which is a reasonable inductive bias but not load-bearing. **The
self-clamp of §2.2(c) is what does the actual work**; the persistence edge is
an optional smoothing, not a required composition (correcting the stronger
claim in the preceding chat that persistence was necessary — it is not,
because self-clamp achieves exact freezing directly).

## 3. What breaks without each correction (why all three matter)

**Without (a) alone (only gating `mu_prev`, i.e. literal user proposal):**
`dyn_edges` correctly read a stale source, but `μ_j` itself still updates every
cycle from ordinary relaxation — so the "slow latent variable" framing (§3 of
the user's message) is only true for the *temporal-edge* contribution to
children, not for the node's own content. The node is not actually a slow
variable; only its self-predicted-next-state target is stale.

**Without (b):** the eligibility trace either fades (`elig_decay<1`,
compounding over ~1000 uncompressed cycles even if `mu_prev` is frozen,
because the *decay* still applies every cycle to whatever value the trace
already holds) or saturates to a fixed point that is not obviously "the key
event" — either way it does not deliver the "1000 steps → 1 semantic step"
claim cleanly. Gating the *trace update itself* (not just its input source) is
what makes the compression exact.

**Without (c):** this is the load-bearing gap identified in this document. The
whole "key survives to the door" story implicitly assumes L3's belief stays
parked on "holding key" throughout the corridor. Under plain relaxation it
does not — the node re-infers every cycle from its bottom-up drive
(`cortex_l2`'s error), which keeps flowing even through a "boring" corridor
(footstep-scale visual change), so `μ_{L3}` keeps drifting even though nothing
"surprising" (by the gate's own definition) happened. Self-clamping `μ_j`
exactly when `gate_j = 0` is what keeps it pinned to "holding key" until a
genuine event (the door) reopens the gate — and it is why the credit-
assignment payoff (§4) does not need any change to the eligibility mechanism
on **static** edges at all (see next section).

## 4. The key–door payoff, precisely

With all three legs of §2.2 in place: while `gate_{L3} = 0` (walking the
"boring" corridor), `μ_{L3}` is self-clamped at "holding key" (frozen by (c)),
its temporal carry is frozen (a), and any `dyn_edge` eligibility sourced from
it is frozen (b). The **static** edge `cortex_l3 → value` (already in
`init_region_graph`'s edge list, `pc_graph.py:1430`) therefore *still reads*
`μ_{L3} = "holding key"` at the moment the door's prediction error arrives —
no eligibility trace on a static edge is needed, because the belief that
should get credit **never moved away** in the first place. The TD error at the
door (the value node's own `ε` against its `value(t−1)→value(t)` self-edge)
updates `cortex_l3 → value` through the ordinary one rule, using the
still-current "key" representation, exactly as if the key and the door had
been adjacent in time. This is the mechanism that turns "1000 spatial steps"
into "1 semantic step" — it falls entirely out of §2.2, no additional
eligibility-trace generalisation to static edges is required.

(A genuine multi-key, multi-door scenario, or a scenario needing credit to
reach *further back* than the single most recent freeze, would need
eligibility traces on static edges too — flagged as a possible Phase 3
extension in §7, not required for the single key→door case.)

## 5. Concrete code changes

**`core/pc_graph.py`:**
- Add `_node_surprise(state, params) -> tuple` (per-node `surprise_j`, §2.1),
  built from `pc_graph_errors` and `state.pi`, restricted to each node's
  static-edge children (`_outgoing`, already defined at `pc_graph.py:215`).
- Add `temporal_gate(state, params, k) -> tuple` (per-node `gate_j`, a
  `sigmoid(surprise_j - k)`); `k` a new scalar param on `PCGraphParams`
  (representational, not tuned — §2.1), default value chosen so `k=0` (always
  gate at the raw sign of the z-score) reproduces the ungated behaviour when
  every node's gate resolves to ≈1 (backward-compatible: a graph with no
  self/temporal edges is unaffected either way).
- `pc_graph_roll`: accept an optional `gate: tuple | None = None`; when given,
  blend `mu_prev` per §2.2(a) instead of the unconditional copy. `None` (the
  default) preserves today's behaviour exactly.
- `pc_graph_learn`: the `elig` update loop (`pc_graph.py:1157-1165`) gains the
  same optional `gate` (indexed by each `dyn_edge`'s **source** node), applying
  §2.2(b)'s frozen-trace blend. `None` preserves today's behaviour exactly.

**`core/pc_brain.py`:**
- `pc_brain_cognitive_step`: before `pc_graph_clamp`, compute `gate =
  temporal_gate(state.graph, params.graph, k)` for the set of nodes opted into
  gating (`gated_nodes`, a tuple on `PCBrainParams` naming which nodes carry a
  self/temporal edge worth compressing — e.g. `cortex_l3`; this is a
  structural fact about *which nodes have a slow-timescale role*, not a
  per-run tuning choice, same status as `perceptual_nodes`). For nodes in
  `gated_nodes` with `gate_j` below a hard cutoff (e.g. `gate_j < 0.5`), add
  `j` to `frozen_this_cycle` with clamp value `state.graph.mu[j]` (§2.2(c)),
  **kept disjoint from** the `sensory` clamp and from any `perceptual_nodes`
  hold used elsewhere. Pass both the sensory clamp and `frozen_this_cycle`
  into `pc_graph_relax`'s `clamp` argument.
- Thread `gate` into `pc_graph_learn`/`pc_graph_roll` calls at the end of the
  cycle.

**New diagnostics:** `diagnostics/embodied_reach_collapse/diag_keydoor.py` — a
minimal gridworld/labyrinth rig (no MuJoCo): an agent walks a long corridor,
picks up a "key" (a one-hot feature flips), then must reach a "door" cell for
reward. Compare credit assignment (does `cortex_l3 → value` weight toward the
key-feature grow after a single door event?) **with vs without** the Phase 3
gate, at corridor lengths sweeping 10 → 1000 cells.

## 6. Acceptance tests

- **Gate correctness:** on a synthetic corridor with an injected single
  "surprising" cell (wall/turn) amid otherwise-predictable steps, `gate_{L3}`
  stays ≈0 through the predictable steps and spikes at the surprising one
  (unit test on `temporal_gate` directly, no need for the full rig).
- **Freeze fidelity:** with the gate closed for `N` synthetic cycles, `μ_{L3}`
  after `N` cycles equals `μ_{L3}` before (bit-for-bit under the self-clamp) —
  this is the test that would have caught the gap in the literal proposal.
- **Eligibility non-dilution:** the `dyn_edge` trace sourced from a frozen node
  is unchanged after `N` idle cycles (vs the ungated version, which visibly
  decays/saturates over the same `N`).
- **Key–door credit:** `diag_keydoor.py` — the `cortex_l3 → value` weight
  toward the key-feature grows after a **single** door-reward event with the
  gate on, at corridor length 1000; compare against the ungated baseline,
  which should show materially weaker (or absent) credit at that length
  (matching the qualitative TD(λ)-with-long-idle-decay failure mode).
- **No regression:** `gate=None` default path byte-identical to pre-Phase-3
  behaviour on every existing test (`tests/test_phaseu_pc_graph.py`,
  `test_phaseu_pc_temporal.py`, `test_phaseu_pc_brain.py`).

## 7. Risks and open edges (honest)

- **Chattering at the gate boundary.** A `surprise_j` hovering near `k` can
  flip the gate every cycle, thrashing between "frozen" and "free" and
  undermining exactly the stability being sought. Standard fix: hysteresis
  (different open/close thresholds) or a slower EMA on `surprise_j` itself
  before thresholding — needed in practice, not optional polish.
- **Where the event boundaries *should* be is not solved by this mechanism.**
  The gate reacts to existing bottom-up error; it does not learn to place
  boundaries better than the underlying generative model already does. If the
  lower layer's model is poor, its error is noisy and the gate is noisy with
  it. This is a real limitation, not merely a caveat.
- **Multi-key / longer-back credit** needs eligibility on static edges (§4,
  parenthetical) — out of scope here, flagged for a future extension only if
  the single-freeze case proves insufficient in practice.
- **Interaction with active inference's `hold_nodes`.** During goal-directed
  action (`pc_brain_act`), `perceptual_nodes` are already held for a different
  reason. If a perceptual node is *also* gated-frozen this cycle, the two
  holds must agree (both hold to the same value) or the self-clamp must simply
  be additional to, not in conflict with, the existing hold set — verify this
  composition explicitly in tests rather than assuming it falls out.
- **This is a genuinely new interaction, not a free lunch.** Even though every
  piece reuses an existing primitive (`hold`/clamp, `dyn_edges`/`elig`,
  Welford Π), wiring a **precision-derived, per-cycle, per-node dynamic clamp
  set** into the cognitive-cycle orchestration is new control flow in
  `pc_brain_cognitive_step`, not merely a parameter default. State this
  plainly rather than overclaiming "zero added mechanism" — the honest claim
  is "no new *primitive*, one new *wiring decision*, derived from statistics
  already computed."

## 8. References

- **Sutton & Precup & Singh (1999), "Between MDPs and semi-MDPs: A framework
  for temporal abstraction in reinforcement learning."** Options / semi-MDP
  formalism — the standard RL grounding for "a higher policy ticks on
  variable-length temporally-extended actions," the same structure as a
  precision-gated node clock.
- **Zacks, Speer, Swallow, Braver, Reynolds (2007); Baldassano et al. (2017),
  "Discovering event structure in continuous narrative perception and
  memory."** Human event-segmentation is prediction-error-driven — boundaries
  occur where a running situation model's predictions fail — directly the
  bottom-up surprise criterion of §2.1.
- **Rouhani & Niv & Frank; Gershman event-segmentation-and-RL line.**
  Prediction-error-triggered event boundaries interacting with reward credit
  assignment — the RL-facing version of the same idea, relevant to §4.
- **Neitz, Parascandolo, Bauer, Schölkopf (2018), "Adaptive Skip Intervals:
  Temporal Abstraction for Recurrent Dynamical Models"** ("jumpy" world
  models) — learning to skip over predictable intervals in a world model,
  the model-based-RL analogue of gating the temporal carry.
- **Sutton & Barto, "Reinforcement Learning" ch. 12 (eligibility traces,
  TD(λ)).** The baseline eligibility-trace mechanism this substrate's `elig`
  already implements on `dyn_edges`; §4 explains why static-edge traces are
  not additionally required for the single-freeze key–door case.

## 9. Summary — where this sits relative to Phase 2

| | Axis | Gated quantity | Retires |
|---|---|---|---|
| 2A | spatial metric | μ/W step (diagonal vs full Hessian) | `action_nodes`, `nlms_edges` |
| 2C | spatial node-type | amortised vs iterative inference | `feedforward_nodes` (if 2B clears its gate) |
| **3** | **temporal clock** | **whether a node's belief/carry/trace advances this cycle** | **nothing existing** — this is new capability (multi-timescale prediction + long-horizon credit), not a tag removed from today's stack |

Phase 3 does not shrink the tag stack Phase 2 targets; it is additive
capability on the same "Π/error decides, not the builder" principle. Implement
after Phase 2A is stable (Phase 3's gate computation reuses the same precision
statistics 2A already generalises) and independently of whether 2B/2C land.
