# Phase 2 — retiring the tag stack by upgrading the node's physics

**Written:** 2026-07-02. **Method:** code analysis of the real `core/` substrate
(`pc_graph.py`, `pc_module.py`, `pc_precision.py`, `pc_active.py`,
`free_energy.py`) + the two prior diagnoses (`FINDINGS.md` collapse,
`FINDINGS_OSCILLATION.md` inversion) + a literature pass (refs in §11).

This document is written so that **another chat can implement Phase 2 from it
alone**. It states the problem, gives the definitive solution, says exactly what
to change and in what order, and — because the user set a *decisive test* — it
is honest about which parts of the tag stack genuinely dissolve and which do
not, and why.

---

## 0. The decisive test (the user's, restated) and the verdict

> Does Phase 2 let us **remove** the static role-tags — Π itself arbitrates
> body/world, deep plastic edges learn without freezing — or does it **add**
> another control mechanism on top?
> * Removes tags → the architecture is real. Converges. AGI-ready.
> * Adds a mechanism → accretion of patches. Doomed to be a glue-pile.

**Verdict, up front, honest:**

The tag stack splits cleanly into **two families with two different fates.**

1. **The metric / rate tags dissolve.** `action_nodes`, `nlms_edges`, and the
   `fixed_pi_nodes` companion are all patches for *the substrate using a
   diagonal metric where the correct object is the full precision (the natural
   metric)*. Phase 2A makes the **one rule** natural-gradient on **both** μ and
   W everywhere; the diagonal step then falls out automatically as the
   *precision-dominated limit*, chosen by a runtime criterion, not a builder
   tag. This is a genuine "remove the tags" result, low-risk, and it closes the
   reach-inversion story as a **principle** rather than a special case. **Do
   this.**

2. **The collapse tags do *not* dissolve for free, and it is important to say
   so.** `frozen_edges` + `feedforward_nodes` prevent the cerebellar
   *representational collapse*. That collapse is **not** a precision-metric
   problem — it is representational degeneracy — and there is a precise,
   provable reason (the log-det sign argument, §6.1) that **covariance /
   precision learning inside the Gaussian free energy cannot by itself prevent
   it**: the generative objective *rewards* a collapsing code. Anti-collapse is
   an intrinsically **non-generative** constraint (VICReg-style variance floor,
   Tang-2022 decorrelation) — i.e. it *is* "a mechanism added on top." So the
   naive hope ("just make Π a covariance and collapse goes away") is false, and
   Phase 2 must not pretend otherwise.

   The honest resolution is a reframe: **`feedforward_nodes` is not a hack, it
   is one of two legitimate node types** (amortised-recognition vs free-cause).
   For the cerebellum the frozen Marr–Albus expansion is *biologically
   correct*, not a patch, and should stay. The real neuro-AGI target — a
   *learnable cortical hierarchy* that grows deep plastic edges without
   collapse — is a genuine open research problem; Phase 2B is the principled,
   **gated** attempt at it (covariance learning + a variance floor + a
   precision-gated amortised↔iterative interpolation), to be adopted **only if**
   it beats the frozen/feedforward baseline on the diagnostics rig, and kept
   behind a flag until it does.

So: Phase 2 **removes half the stack outright** (the metric family) and
**replaces the other half's justification with a real type distinction**, while
being explicit that a fully tag-free deep-plastic hierarchy is a research bet,
not solved engineering. That is the truthful answer to the decisive test, and
it is more useful than a clean-sounding overclaim.

---

## 1. The tag stack — what exists, what each patches, who sets it

Every tag below is a `static` field on `PCGraphParams` (so it is fixed at build
time by `init_region_graph`, never inferred):

| Tag | Set by builder to | What it patches | Family |
|---|---|---|---|
| `fixed_pi_nodes` | `(motor,)` | holds Π flat so precision-EMA doesn't stiffen the action node | metric |
| `action_nodes` | `(motor,)` | full Gauss–Newton μ-step (diagonal metric is degenerate for a flat-prior low-D cause under a high-gain likelihood) | metric |
| `nlms_edges` | `(cerebellum→sensory,)` | input-side natural-gradient weight step (dense granule readout diverges under plain LMS) | metric |
| `frozen_edges` | `(motor→cerebellum,)` | freeze the deep edge so it can't decay to 0 (collapse) | collapse |
| `feedforward_nodes` | `(cerebellum,)` | make the hidden layer deterministic so it can't absorb the clamp off-manifold | collapse |
| `bias_nodes` | `(sensory,)` | learnable DC term for a non-zero-mean target | model |

The user's worry is exactly right: the *trend* is a growing pile of
hand-assigned static role-tags. But the tags are not all the same kind of thing,
and the fix is different for each family.

---

## 2. Unified diagnosis — two failures, one root

Both prior failures are the **same defect seen twice**, and it is a defect in
*what the substrate uses for precision*:

The substrate models every node as a Gaussian with **diagonal, scalar precision
`Π_j`** that is (a) **tracked by a side EMA** (`pe_var ← (1−α)pe_var + αε²`,
`Π = 1/(pe_var+floor)`), not part of the objective; and (b) used as a
**diagonal inference metric**. The objective it actually descends,
`variational_free_energy = ½ Σ Π·ε²` (`free_energy.py:22`), **drops the
`−½log|Π|` term** of the proper Gaussian surprise. This is the standard
"identity/diagonal-precision, fixed-form" approximation — and the PC literature
flags precisely this as the thing that fails for deep/wide/degenerate models
(most PCNs "fix precisions to the identity … reducing the free energy to a sum
of unweighted squared errors", §11 refs).

Two consequences, one per failure:

* **Inversion failure (`FINDINGS_OSCILLATION.md`).** The diagonal of the
  free-energy curvature `L_j` is a *degenerate metric* for a low-dimensional
  flat-prior cause fanning into a high-gain forward model: it drops the
  off-diagonal cross-coupling and scales as the square of the model gain, so the
  inferred command `∝ ε/‖W‖²` freezes as the model sharpens. The *diagonal
  metric* is wrong. (Patched by `action_nodes` + `fixed_pi_nodes`.)

* **Collapse failure (`FINDINGS.md`).** A wide *free* latent between two clamps,
  with a *plastic* deep edge, has no term in the objective penalising its code
  going rank-deficient (predict-the-mean); the local rule then actively decays
  the deep edge (`ΔW_mc ≈ −η·W_mc·φφᵀ`). The *objective is missing a code
  constraint*. (Patched by `frozen_edges` + `feedforward_nodes` +
  `nlms_edges`.)

Same root — an impoverished precision object — but note carefully: the first is a
**metric** deficiency (curable by using the *right* precision as the metric);
the second is a **representational** deficiency (an over-complete free code),
and §6.1 shows the two are *not* curable by the same move.

---

## 3. Applying the decisive test tag-by-tag

| Tag | Dissolves? | By what, and into what |
|---|---|---|
| `action_nodes` | **Yes** | Universal natural-gradient μ-step (block Hessian). Diagonal-Newton becomes the *precision-dominated limit*, selected by a runtime criterion. No builder tag. |
| `nlms_edges` | **Yes** | Universal input-side natural-gradient weight step (`÷(‖φ‖²+δ)`). It is the input-side twin of the `÷Π` output-side preconditioning already universal. No tag. |
| `fixed_pi_nodes` | **Reduces** | From a hard "hold Π" list to a single **structural fact**: which node is *efferent* (drives actuators) carries a fixed weak effort prior. That fact is embodiment, not statistics — it stays, as one honest bit, not a hack. |
| `bias_nodes` | **Yes (minor)** | Give *every* node a bias with a prior pulling it to 0; it self-activates only where the target is non-zero-mean. No tag. |
| `feedforward_nodes` | **Reframed, not removed** | It is a *legitimate node type* (amortised recognition) — biologically correct for the cerebellum. Long-term, Π gates a smooth amortised↔iterative interpolation (§6.4), which is the real "Π arbitrates" milestone — a research bet, not a guaranteed removal. |
| `frozen_edges` | **Kept for the cerebellum (correct); target for cortex** | Marr–Albus freezing is not a hack. The research goal is a *learnable* hierarchy that doesn't need it; Phase 2B is the gated attempt. |

So the metric family (`action_nodes`, `nlms_edges`) **removes outright**;
`fixed_pi_nodes`/`bias_nodes` **collapse into the universal rule or one honest
structural bit**; the collapse family is **reframed and gated**. Half the stack
goes; the other half stops being arbitrary.

---

## 4. Phase 2A — unify the metric: the *one rule*, natural-gradient on μ and W

This is the part that genuinely retires tags, is well-founded (Ofner et al.
2021: precision weighting *is* a local Fisher/natural-gradient metric; the
substrate's own `action_nodes` code is already the special case), and directly
generalises the accepted `FINDINGS_OSCILLATION.md` fix from a patch to a
principle. **Do this first; it is low-risk and self-contained.**

### 4.1 Inference: block Hessian everywhere, diagonal as the automatic limit

Today `_graph_relax_step` (`pc_graph.py:783`) has **two** code paths: a diagonal
step for normal nodes (`g/L`, `L` = diagonal curvature) and a separate full
Gauss–Newton path for `action_set` (`_action_gn`, `pinv(H)@g`). Unify them:

**Every free node** `j` steps by its **block free-energy Hessian**

```
H_j = (Π_j + leak)·I  +  Σ_{cone leaves c} Jc_jᵀ (Π_c ⊙ hold_c) Jc_j
Δμ_j = η_μ · solve(H_j, g_j)
```

where `Jc_j = ∂ŷ_c/∂μ_j` is the prediction-Jacobian pushed down node `j`'s
feedforward cone (exactly what `_action_gn` already assembles). Key points:

* **This is one path, not two.** The current diagonal `L_j` is precisely
  `diag(H_j)` when the cone is one hop deep. Assembling the *full* `H_j` for
  every node is the generalisation; `_action_gn` is the routine — apply it to
  all free nodes, and delete the `action_set` branch.
* **Diagonal-Newton is recovered automatically** as the **precision-dominated
  limit**: when `Π_j` dominates the child-relayed term (a genuine perceptual
  node, whose own precision is large), `H_j ≈ diag(Π_j)` and `solve` ≡ divide.
  So the *criterion* for "use the cheap diagonal step" is a **runtime test on
  the node's own precision vs its child-relay curvature**, not a static
  `action_nodes` list. That is the concrete sense in which **Π self-arbitrates
  the metric**: a flat-prior (efferent) node has small `Π_j`, so its cross-
  coupling is not negligible and it *automatically* takes the full solve; a
  sharp perceptual node takes the diagonal.
* **Scale/cost.** `H_j` is `d_j × d_j`; `d_motor = 8`, `d_cortex = 32` here — a
  tiny dense solve per node per sweep. For large nodes, gate by the criterion
  above (full solve only where own-precision does *not* dominate) so the common
  case stays diagonal-cheap. This gating is the same test that removes the tag,
  so it costs nothing extra.
* **Flat prior → effort prior.** The `(Π_j + leak)·I` floor is the weak effort
  prior (`MOTOR_EFFORT_PRECISION = 0.01`, `pc_brain.py:58`); it keeps `H_j`
  non-singular and pulls the command off the tanh rails. Fold `set_action_prior`
  into "efferent nodes get a fixed small prior precision" — see §4.3.

**Acceptance:** identical to `FINDINGS_OSCILLATION.md` §6 — reach insensitive to
`‖W_cs‖` scale (`diag_motorfix.py` sweep), `mean|cmd|` bounded away from 0,
`|eps_tip|` falls during planning; **plus** a new invariant: removing the
`action_nodes` tag entirely and relying on the runtime criterion reproduces the
tagged behaviour bit-for-bit on the motor node (regression gate).

### 4.2 Learning: input-side natural gradient everywhere (retire `nlms_edges`)

The weight rule is already output-side natural-gradient (`÷Π`, the curvature-
preconditioning documented in `pc_graph_learn`). NLMS is its **input-side twin**
(`÷(‖φ‖²+δ)`). Make it universal:

```
ΔW_(i→j) = η_w · ε_j ⊗ φ(μ_i) / (‖φ(μ_i)‖² + δ)
```

for **every** edge (and the temporal edges, with `‖trace‖²`). One positive
scalar per edge ⇒ the gradient *direction* and the `∂F/∂W = 0` fixed point are
unchanged; only the rate adapts. Choose `δ` so a dense presynaptic population
(`‖φ‖² ≈ n_in`) keeps its current effective rate (backward-compatible), and a
(near-)silent population is guarded. This is the same natural-gradient principle
as §4.1, applied on the weights — the *one rule* becomes fully natural-gradient
(input × output), and `nlms_edges` disappears.

**Acceptance:** the granule readout no longer diverges without a tag
(`diag_phase1b.py` first-run NaN must not recur); a fully-plastic shallow graph
trains identically to the pre-change LMS rate on dense inputs (numerical
regression).

### 4.3 Fold `fixed_pi_nodes` into one structural fact (efferent nodes)

`fixed_pi_nodes` exists only to stop the ε²-EMA from overwriting the flat action
prior. Replace the list with: **a node marked *efferent* carries a fixed prior
precision `π_effort` and its precision is not EMA-tracked.** This is the same one
bit, but it names the *real* distinction (this node drives the body / is read out
as a command) instead of an opaque "fixed pi" flag. It cannot be removed by
statistics — *which node touches the world* is an embodiment fact — so keeping it
is honest, not accretion. `set_action_prior` becomes "set the efferent prior."
(If you later want Π to *infer* efference: that requires the agent to discover
which of its variables are causally coupled to its actuators — a genuine
capability, out of Phase 2 scope; do not fake it with a tag.)

### 4.4 `bias_nodes` → universal biased node (minor)

Give every node a bias that learns by the one rule with a **zero-mean prior**
(a small precision pulling `bias→0`). It self-activates only where the
generative target is non-zero-mean (sensory), stays ~0 elsewhere (value/policy —
the reason `bias_nodes` currently excludes them, to avoid an always-on unit
stealing intermittent reward credit; the zero-mean prior achieves the same
without a list). The frozen granule threshold stays out of the learnable path
(it is set by `apply_granule_expansion_init`, not a learnable DC term) — that
exclusion is a property of a *frozen edge*, so it is already handled by the
frozen-edge machinery, not a bias tag.

**Net after 2A:** `action_nodes`, `nlms_edges`, `fixed_pi_nodes`, `bias_nodes`
are gone as builder tags. Remaining: one *efferent* structural bit, and the
collapse pair (`frozen_edges`/`feedforward_nodes`) — addressed next.

---

## 5. Why 2A is not enough (and what the collapse actually is)

2A fixes the *metric*. It does **nothing** for the collapse, because the collapse
is representational: a wide *free* latent (`cerebellum`, 64-d) squeezed between a
clamped motor node and a clamped sensory node, over-complete relative to the 2-d
command, reconstructs the sensory clamp **off the motor manifold** using its ~62
spare dimensions; its code goes command-independent (`cb_var → 2e-5`), and the
one rule then decays the deep edge (`‖W_mc‖ 7.96→0.48`). No metric change touches
this — the fixed point is representational.

The collapse needs **three** legs simultaneously: (i) an *over-complete* hidden
layer, (ii) that is a *free* latent (co-relaxes to satisfy the clamp), (iii) fed
by a *plastic* deep edge. Remove any one and it is gone. Phase 1 removed (ii)+(iii)
(feedforward + frozen). Phase 2B asks whether a *principled* mechanism can let
(ii) and (iii) stay — i.e. a **learnable** hidden layer with a **plastic** deep
edge that does not collapse — which is what a growable cortical hierarchy needs.

---

## 6. Phase 2B — structured precision for deep plastic edges (the honest frontier)

### 6.1 The log-det sign argument — why generative covariance learning cannot, alone, prevent collapse

This is the single most important technical point in the document, and it
refutes the naive "make Π a full covariance and collapse disappears" hope.

The proper Gaussian surprise of a node with error `ε` and precision `Λ` is

```
F = ½ εᵀΛε − ½ log|Λ| + const.
```

The substrate currently drops the `−½log|Λ|` term. Add it back and learn `Λ` by
descending `F`: the stationary point is `Λ* = (E[εεᵀ])⁻¹` — exactly what the EMA
already estimates (diagonally). Now ask what this does to the collapse:

* **On the *error* precision.** As the model fits, `E[εεᵀ] → 0`, so
  `−½log|Λ*| → −∞`: good fits are rewarded (correct). But **the collapsed
  cerebellum also has `ε_cb → 0`** (the bias absorbs the mean, `μ_cb ≈ bias_cb`,
  `W_mc → 0` so `ε_cb = μ_cb − W_mc φ − bias_cb → 0`). To the error precision,
  **collapse looks like a perfect fit.** The log-det term *rewards* it.

* **On the *activity* prior.** Put a Gaussian prior on the code `μ_cb` with
  learned precision `Λ_prior = (E[μ_cb μ_cbᵀ])⁻¹`. As the code collapses,
  `E[μμᵀ]` goes rank-deficient, and the surprise `½ tr(Λ Cov) − ½log|Λ|` at the
  ML `Λ` equals `½d + ½log|Cov| → −∞`. A Gaussian prior assigns *maximum*
  likelihood to activity concentrated at a point — **it too rewards collapse.**

**Conclusion.** Anti-collapse is **not** obtainable from any surprise-minimising
(generative) objective — generative pressure *is* compression toward the mean.
Preventing collapse requires an **anti-generative** constraint: an explicit
*variance floor* that pushes each unit's across-input variance **up** (VICReg's
`max(0, γ − std(z))²` hinge — note the sign is opposite to a Gaussian prior),
and/or a *decorrelation* term (Tang-2022 lateral covariance, Barlow-Twins
off-diagonal penalty). These **are** "a mechanism added on top." Any honest
Phase 2 must state this rather than dress a variance floor as if it fell out of
free energy. It does not.

### 6.2 What covariance learning *does* buy (and what it doesn't)

Distinguish two different objects that "covariance learning" is loosely used for:

* **Error/activity variance floor (anti-collapse).** Counters `cb_var → 0`.
  Non-generative (§6.1). This is the leg that actually keeps the deep plastic
  edge alive. It is VICReg's *variance* term made local/online.
* **Off-diagonal precision = lateral connections (anti-redundancy /
  decorrelation).** Tang et al. 2023 (covariance learning in recurrent PC):
  lateral weights implement a whitening/decorrelation that lets a code store
  *correlated* patterns and removes redundancy. This *is* derivable as the
  off-diagonal of the precision in the Gaussian model, and it improves capacity
  — but it does **not** floor the variance, so it does **not** by itself stop
  the predict-the-mean collapse.

So the correct Phase 2B mechanism for a **learnable hidden layer** is *both*:
(a) a local variance floor (anti-collapse, non-generative — owned honestly as an
added constraint), and (b) local lateral precision (decorrelation, generative —
the log-det/off-diagonal term of a proper Gaussian, which §4.1's metric already
wants). (b) alone is elegant but insufficient; (a) is the load-bearing, and
honest, addition.

### 6.3 The cerebellum is the wrong place to remove the freeze

For the **cerebellum specifically**, freezing `motor→cerebellum` as a fixed
random expansion is **Marr–Albus** — biologically correct (mossy→granule is a
fixed high-D random non-linearity; only granule→Purkinje is plastic). Removing
the freeze *there* is not a virtue; it would be modelling the cerebellum wrong.
The diagnostics rig uses the cerebellum only as a convenient *testbed* for a deep
plastic edge into a free hidden layer. **Keep the cerebellum frozen.** Evaluate
2B on the rig as a proxy for the real target:

> a **cortical** hierarchy (`cortex_l3 → cortex_l2 → cortex_l1 → sensory`) whose
> deep edges *must* be plastic and *must not* be frozen — the model the agent has
> to grow on its own once perception is allowed to learn.

### 6.4 The real "Π arbitrates" milestone — amortised↔iterative interpolation

`feedforward_nodes` (amortised recognition) and free-latent (iterative
inference) are the two ends of a spectrum, not a binary a builder must choose.
The principled long-term move (hybrid amortised+iterative PC; cf. the
"Divide-and-Conquer PC" structured-inference line, §11) is a **per-node gate
`g_j ∈ [0,1]`** interpolating

```
μ_j ← (1 − g_j)·[iterative relax step]  +  g_j·[feedforward prediction],
```

with `g_j` driven by the node's **recognition precision** — how reliably its
top-down prediction already explains it. A node whose feedforward prediction is
consistently accurate (high recognition precision) *becomes* effectively
feedforward (`g_j → 1`); one that needs settling stays iterative (`g_j → 0`).
**That** is Π arbitrating node *type* — and it *is* achievable, because it is a
metric/gating statement, not the (impossible, §6.1) "generative anti-collapse."
It retires `feedforward_nodes` as a static tag by making it an inferred limit.
This is the honest reading of the user's aspiration; scope it as Phase 2C
(after 2B proves the plastic hidden layer can be kept alive at all).

### 6.5 Phase 2B — concrete mechanism to test

On the rig, with `motor→cerebellum` **plastic** and `cerebellum` a **free**
latent (the collapse configuration), add to each free hidden node:

1. **Local variance floor (anti-collapse, owned as non-generative).** Track a
   slow per-unit activity variance `v_j` (EMA of `μ_j²` about its slow mean).
   Add to the relax gradient a term pushing `μ_j` to *increase* variance where
   it has fallen below a floor `γ`: a local hinge `∂/∂μ_j [½·relu(γ − √v_j)²]`.
   This is VICReg's variance term, online and per-unit. It is a *constraint*,
   not part of `F` — document it as such.
2. **Local lateral precision (decorrelation, generative).** A lateral weight
   `M_j` (symmetric, zero-diagonal) on the hidden node giving the μ-step a
   decorrelation drive `−M_j φ(μ_j)`; learn `M_j` by a local anti-Hebbian rule
   `ΔM_j ∝ φ(μ_j)φ(μ_j)ᵀ − diag` (Tang 2023 covariance learning; Földiák
   anti-Hebb). This is the off-diagonal of the node's precision and rides on
   §4.1's block-Hessian metric.
3. **Keep §4's natural-gradient rate** on the now-plastic deep edge (input-side
   `÷‖φ‖²` prevents the fast-in-fast-out instability the frozen edge sidestepped).

**Gate (this is the whole point of the decisive test).** 2B is adopted **only
if**, on the rig with the deep edge *plastic* and the hidden layer *free*:

* `cb_var` stays above a fixed floor across the full babble budget (no
  collapse), **and**
* forward-model decoded-tip error falls **< 0.1 m** and *tracks* babble, **and**
* it **generalises**: same result on a 3-link arm and a higher-d hidden layer
  (the mechanism scales), **and ideally**
* it works on the *cortical* hierarchy (deep plastic `cortex` edges) — the
  actual target.

If 2B clears the gate, the learnable hierarchy no longer needs
`frozen_edges`/`feedforward_nodes`, and *that* is the neuro-AGI milestone. If it
does **not**, the correct engineering answer is **not** more patches — it is to
keep amortised recognition (`feedforward_nodes`) as a *legitimate node type*
(§6.4) and record that a fully tag-free deep-plastic hierarchy remains open.
Either outcome is reported honestly; neither is a glue-pile, because 2B lives
behind one flag and is deleted if it loses (per the repo's "no leftover legacy,
delete after integrating" rule).

---

## 7. What this does — and does not — do for the hardcoded `hold_nodes` / seeds

`FINDINGS.md` §4 tied the hand-frozen `perceptual_nodes`/`hold_nodes` and the
Gabor seed to the same disease. Precisely:

* **2A** removes the *metric* excuse for hand-holding the action inversion. It
  does not touch `hold_nodes` (the perception-fixed / action-varies clamp in
  `pc_act_infer`), which is a *correct* active-inference move (perception fixed,
  action varies — Friston 2010), **not** a collapse patch. Keep it; it is not on
  trial here.
* **2C** (§6.4), if it lands, is what actually lets Π decide which nodes are
  effectively feedforward — the mechanism that could eventually retire *static*
  `perceptual_nodes` holds in favour of inferred ones. It depends on 2B first
  proving a plastic hidden layer can be kept alive.
* The **Gabor seed** is a *prior*, not a freeze (the `cortex_l1→sensory` edge
  still learns). It is not a collapse patch and is out of scope.

Do not oversell: Phase 2 makes the hardcoded holds *removable in principle* only
along the 2B→2C path, and only if that path clears its gate.

---

## 8. Implementation plan (ordered, mapped to files)

**Stage A1 — block-Hessian inference (retire `action_nodes`).** `pc_graph.py`
`_graph_relax_step`: apply the `_action_gn`-style full-`H` assembly to **all**
free nodes; add the runtime criterion `own-precision-dominates → diagonal
divide, else full solve`; delete the `action_set` special branch and the
`action_nodes` static field + its validation in `init_pc_graph_params`.
Regression-gate against the current tagged motor behaviour.

**Stage A2 — universal input-side natural gradient (retire `nlms_edges`).**
`pc_graph.py` `pc_graph_learn`: divide every edge (and temporal-edge) step by
`(‖φ_src‖² + δ)`; pick `δ` for dense-input backward-compat; delete the
`nlms_edges` field + branch.

**Stage A3 — efferent prior (fold `fixed_pi_nodes`).** Replace the
`fixed_pi_nodes` list with an `efferent_nodes` bit that carries `π_effort` and is
skipped by precision-EMA; rename `set_action_prior` accordingly
(`pc_active.py`). One structural bit remains, by design.

**Stage A4 — universal biased node (retire `bias_nodes`).** `pc_graph_learn`:
every node's bias learns with a small zero-mean prior precision; delete the
`bias_nodes` field. Verify value/policy biases stay ~0 (no reward-credit theft).

**Stage B0 — objective bookkeeping.** `free_energy.py`: add
`gaussian_surprise(precision, error)` including `−½ Σ log(precision)` (needed
once precision is a *learned parameter in the objective*, not a side-EMA), used
by 2B's precision learning; keep `variational_free_energy` as the diagonal-Π
special case for 2A (byte-compat).

**Stage B1 — 2B on the rig (behind a flag).** New diagnostics script
`diag_covlearn.py`: rig with `marr_albus_cerebellum=False` (plastic deep edge,
free hidden layer) **plus** the §6.5 variance floor + lateral precision; log
`cb_var`, forward-tip error, and the gate metrics. New `PCGraphState` fields for
the slow variance `v_j` and lateral weights `M_j` (empty/inert unless the flag
is on, so every existing graph is byte-identical — same discipline as the
current opt-in fields).

**Stage B2 — port iff gated.** Only if `diag_covlearn.py` clears §6.5, wire the
mechanism into `init_region_graph` for the **cortical** edges (not the
cerebellum), behind a `covariance_learning=False` default. Re-measure reach on
the real MJX arm.

**Stage C (later) — precision-gated amortised↔iterative (§6.4).** Only after B.

Each stage is independently testable and independently revertible.

---

## 9. Acceptance tests (tied to existing rigs)

* **A1:** `diag_motorfix.py` — reach insensitive to `‖W_cs‖` scale with **no**
  `action_nodes` tag (runtime criterion only); GN-equivalent step on the motor
  node reproduces the tagged result. `diag_inversion2.py` — `mean|cmd|` bounded
  away from 0, brute-force-grid reachability unchanged.
* **A2:** `diag_phase1b.py` — no NaN on the granule readout without an
  `nlms_edges` tag; dense-input LMS-rate numerical regression on a shallow graph.
* **A3/A4:** unit tests — efferent node keeps `π_effort` across learning;
  value/policy biases stay ≈ 0 over babble.
* **B1 (the decisive gate):** `diag_covlearn.py` — with the deep edge **plastic**
  and hidden layer **free**: `cb_var` above floor across babble; forward
  < 0.1 m and falling; scales to 3-link + wider hidden; ideally holds on the
  cortical hierarchy. Compare head-to-head against the frozen/feedforward
  baseline (`diag_collapse.py` numbers): 2B must *match or beat* it, else 2B is
  dropped.

Reuse the existing collapse signatures as the yardstick: `‖W_mc‖` constancy,
`cb_var` floor (`diag_collapse.py`), reach-vs-`‖W_cs‖` decoupling
(`diag_oscillation.py`), metric discrimination (`diag_motorfix.py`).

---

## 10. Risk register (honest)

* **2A is safe.** It is the accepted `FINDINGS_OSCILLATION.md` fix generalised;
  the only cost is a small dense solve per node, bounded by the diagonal-limit
  gate. Main risk: the "own-precision-dominates" threshold needs to be a
  *dimensionless, un-tuned* ratio (e.g. `Π_j` vs the spectral norm of the
  child-relay), not a magic constant — derive it from the same stability bound
  that motivates the preconditioner, not from reach success.
* **2B may not clear the gate.** Deep-PCN trainability past a few layers is an
  *open* research problem (Qi et al. 2025: exponentially imbalanced inter-layer
  errors; depth degradation past 5–7 layers). Our failure is primarily the
  *width/degeneracy* axis (curable-ish by the variance floor + decorrelation),
  but the *depth* axis may additionally need Qi's inference-step-size schedule +
  surrogate weight objective. Budget for "2B works on width, still needs a depth
  recipe."
* **The variance floor is a real added term.** It is non-generative (§6.1). Do
  not launder it as free energy. Its `γ` is a representational floor (units must
  stay informative), analogous to `var_floor` — set it by a representational
  criterion, not by reach. If keeping it offends the "one objective" principle
  more than keeping `feedforward_nodes`, then **keep the amortised node type**
  (§6.4) — it is the more principled of the two "extras".
* **Do not add tags to rescue 2B.** If 2B needs a third and fourth knob to work,
  that is the decisive test failing — stop, keep amortised recognition, and
  record the negative result.

---

## 11. References (verified 2026-07-02)

Precision as the natural-gradient metric (the basis of §4):
* **Ofner, Ratul, Ghosh, Stober (2021), "Predictive coding, precision and
  natural gradients."** arXiv:2111.06942. Precision weighting in PC is a local
  approximation of the Fisher-information / natural-gradient metric; learnable
  precision matches backprop-with-natural-gradients and is more robust under
  noise. Grounds "the diagonal step is the precision-dominated limit of the
  natural metric."
* **PredProp (Ofner & Stober 2021), arXiv:2111.08792** — bidirectional
  precision-weighted PC optimisation (precision on both inference and weights).

Covariance / precision learning and anti-degeneracy (the basis of §6.2):
* **Tang, Salvatori, Millidge, Song, Lukasiewicz, Bogacz (2023), "Recurrent
  predictive coding models for associative memory employing covariance
  learning."** PLoS Comput Biol 19(4):e1010719. Lateral covariance learning
  (decorrelation/whitening) in recurrent PC — the generative off-diagonal term.
* **VICReg — Bardes, Ponce, LeCun (2021).** Explicit variance (hinge, floor) +
  covariance regularisation to prevent embedding collapse. The variance term is
  the *non-generative* anti-collapse constraint of §6.1/§6.5 — sign opposite to
  a Gaussian prior, hence not derivable from free energy.

Deep-PCN trainability (the basis of §10 risk):
* **Qi, Forasassi, Lukasiewicz, Salvatori (2025), "Towards the Training of
  Deeper Predictive Coding Neural Networks."** arXiv:2506.23800. Depth
  degradation past 5–7 layers from exponentially imbalanced inter-layer errors;
  fixes = inference-step-size schedule + surrogate weight objective.
* **"Tight Stability, Convergence, and Robustness Bounds for Predictive Coding
  Networks" (2024), arXiv:2410.04708.** Formal stability/convergence bounds — use
  to set the §4.1 diagonal-limit criterion and μ-step from a bound, not a knob.
* **"Divide-and-Conquer Predictive Coding" (2024), arXiv:2408.05834** —
  structured Bayesian inference in PC; a reference point for the §6.4
  amortised↔iterative interpolation.

Substrate-internal cross-refs: `FINDINGS.md` (collapse root cause + Phase 1),
`FINDINGS_OSCILLATION.md` (inversion root cause + the metric fix 2A generalises),
`core/free_energy.py:22` (the objective that drops `−½log|Π|`),
`core/pc_graph.py:783` (`_graph_relax_step`, the two metric paths 2A unifies),
`core/pc_graph.py:900` (`_action_gn`, the routine 2A makes universal),
`core/pc_precision.py` (the side-EMA precision that 2B moves into the objective).
