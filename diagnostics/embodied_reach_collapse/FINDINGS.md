# Embodied reach plateau — root cause, and a two-phase plan

**Diagnosed:** 2026-06-30. **Method:** code analysis + a faithful local
reproduction (CPU JAX, no MuJoCo) using a synthetic 2-link forward kinematics
(FK) as the body and the **real** `core/` graph (`init_pc_brain`,
`pc_brain_act`, `pc_brain_learn_forward`). The failure is in the substrate's
learning, so it reproduces without the MJX arm.

This document is written so another model can (Phase 1) implement the immediate,
biologically-correct fix and (Phase 2) pursue a general, neuro-AGI-worthy
solution to the underlying defect. All numbers below come from the scripts in
this folder.

---

## 0. Symptom to explain

`colab/phase_u_embodied_reach.ipynb` reach success oscillates **6–50 %** around
a ~20 % mean (chance), **with no upward trend as the babble budget grows
0→47 500**, mean min-distance ~0.2 m (success threshold 0.05 m), and a probe
free energy that climbs to ~500 and stays there. The recorded "fix → 38 %"
(see the `embodied-reach-rootcause` memory) does **not** match this run.

The question: *what structurally blocks a stable high plateau, in a way that
also matters for neuro-AGI (no test-tuning, real cause)?*

---

## 1. Verdict

**The `motor → cerebellum → sensory` forward model never forms. The single
local predictive-coding (PC) learning rule collapses the cerebellar hidden
layer to a command-independent constant and decays the deep `motor→cerebellum`
edge toward zero; the per-node biases absorb the mean reafference. The result
is a "predict-the-mean" forward model. Active inference then inverts that mean
model, so the inferred command rails toward saturation and lands ~0.2 m off;
the tiny residual command-dependence drifts as online babble continues, which
is exactly the observed oscillation-around-chance with no babble trend.**

This is a *representational collapse to a trivial fixed point* of the local
rule — not a tuning, leak, capacity, or population-code issue.

---

## 2. Mechanism (why it collapses)

During `pc_brain_learn_forward` the motor node **and** the sensory node are both
clamped (command in, reafference in) and the **wide, free cerebellum (64-d)** is
relaxed between them. Because the cerebellum is over-complete relative to the
2-d motor input, it has ~62 spare dimensions to reconstruct the clamped sensory
target **off the motor-reachable manifold** — it never needs its top-down
`motor→cerebellum` drive. So at the relaxed equilibrium

```
μ_cb ≈ bias_cb            (command-independent)
ε_cb  = μ_cb − W_mc·φ(μ_motor) − bias_cb ≈ −W_mc·φ(μ_motor)
```

and the one rule gives

```
ΔW_mc = η · ε_cb ⊗ φ(μ_motor) ≈ −η · W_mc · φφᵀ      → exponential decay of W_mc.
```

The deep edge is **actively pushed to zero**. With the cerebellum stuck at a
constant, the only way left to reduce sensory error is the learnable
`bias_sensory`, which grows to the mean reafference. Net generative model:
"predict the mean", hidden weights ≈ 0. The curvature-preconditioned weight
rule that "bounds ‖W‖" (celebrated as health in the notebook) here *removes the
very drive that would grow the deep edge* — **bounded-and-shrinking is the
disease**, not health.

---

## 3. Evidence

| # | Script | What it tests | Result |
|---|--------|---------------|--------|
| 1 | `diag_inversion.py` | Reproduce reach on synthetic FK | inversion FK err 0.4–0.5 m, success 0 %, **no** improvement 5k→20k babble. Matches the notebook. |
| 2 | `diag_code.py` (A) | Is the map representable? Adam on the same 2→64→72 tanh net | **0.024 m** decoded tip err. Capacity is *not* the limit. |
| 3 | `diag_code.py` (B) | Monotonic vs Gaussian tip code under PC | 0.39 m for **both** — the code choice is not the cause. |
| 4 | `diag_mech.py` | Cerebellar code informativeness | feature variance = **0.00002** (constant across commands); optimal linear readout off the collapsed cerebellum = 0.21 m (= predicting the mean). |
| 5 | `diag_collapse.py` | Track the collapse | ‖W_mc‖ 7.96→**0.48**, ‖bias_sensory‖ 0→**1.78**, cb_var 0.155→**0.00002**. Seeding W_mc ×8 is **decayed back** (strong attractor). |
| 6 | `diag_marralbus.py` | Fixed random expansion + optimal linear readout | **0.010–0.028 m** — a non-plastic granule expansion represents the model fine. |
| 7 | `diag_phase1.py` | Freeze `motor→cb` only, cerebellum still a free latent | **insufficient**: forward 0.4 m, success 0 % (readout trained at a different cerebellar operating point than inference uses). |
| 8 | `diag_phase1b.py` | Feedforward granule + plastic readout | forward **0.05 m** (online) / 0.019 m (least-squares); reach via clean 2-d inversion rises above chance (see §6 caveat). |

Key discriminators:
- **Representable but not learned:** Adam 0.024 m vs PC 0.39 m.
- **The hidden layer is the failure:** cb_var → 0; the deep edge decays even
  when seeded large.
- **Freezing the edge is not enough:** the cerebellum must stop being a *free
  inferential latent* (test 7 vs 8).

---

## 4. Why this is the neuro-AGI blocker (and ties to the two hints)

A substrate whose only plasticity is the local PC rule **cannot bootstrap any
deep *plastic* generative edge that feeds a free hidden layer** — it collapses
to predict-the-mean every time. It can only form:

- **shallow** (single-plastic-layer) models, or
- **hand-seeded-and-frozen** edges. This is exactly why the cortical **Gabor
  seed works**: the cortex is *held* during babble, so its edge never learns, so
  it never decays. The forward model *must* learn — so it collapses.

This explains the user's two observations directly:

- **Free energy stuck at ~500.** No deep structure forms anywhere. The forward
  model collapses to the mean; the *entire cortical hierarchy is frozen at μ=0*
  during babble (held), so it never learns either. Biases carry the only fit.
  FE plateaus high because the generative model is permanently shallow.
- **`hold_nodes` hardcoded as a static tuple.** The static `perceptual_nodes`
  hold and the Gabor seed are band-aids around this same disease: because the
  substrate cannot grow deep models or let precision Π arbitrate body-vs-world,
  the architecture hand-freezes nodes and hand-seeds priors. For AGI, precision
  *should* infer what is body vs background and what should learn — but Π cannot
  do that job until deep models can form **without collapsing**. The hardcoding
  is a symptom, not the cause.

The general lesson — *the one local rule collapses deep plastic edges* — will
recur for **every** other deep generative model the agent must grow on its own
(cortical hierarchies, world models, options). Those cannot all be frozen.
Hence the two phases below.

---

## 5. Phase 1 — immediate, biologically-correct fix

### Principle (not tuning)

The codebase comment already calls the cerebellum a *"Marr–Albus granule
expansion"*. In Marr–Albus, granule cells are a **fixed, high-dimensional,
random non-linear expansion** of mossy-fibre input; **only the granule→Purkinje
synapse is plastic**. The current code violates this by making `motor→cerebellum`
**plastic** *and* treating the cerebellum as a **free inferential latent** — the
two ingredients that produce the collapse. The fix restores the Marr–Albus
design:

1. **`motor → cerebellum` is a frozen random expansion** (granule layer; not
   plastic). Gain set by standard random-feature / ELM scaling so granule
   pre-activations span their informative `tanh` range (e.g. pre-activation std
   ≈ 1 over the babble command distribution) — a representational criterion,
   **not** tuned to reach success. Random `bias_cb` = granule threshold
   diversity (also fixed, part of the expansion).
2. **The cerebellum is a feedforward node, not a free latent.** Its activity is
   a deterministic function of its mossy input, `g(motor) = tanh(W_mc·φ(motor)
   + b_mc)`, during **both** learning and inversion. (Implementation options
   below.) This is the critical half — test 7 shows that freezing the edge
   while leaving the cerebellum a free latent does **not** work.
3. **Only `cerebellum → sensory` (Purkinje) is plastic**, learned by the one
   rule — now a *shallow* map on a fixed rich basis, which PC learns without
   collapse.

### Why it works

With a fixed expansion the hidden code cannot collapse (no plastic deep edge to
decay), and the readout is trained on the **same** features inference uses
(because the cerebellum is feedforward, not pulled by the sensory clamp).
`diag_marralbus.py` and `diag_phase1b.py` confirm the forward model recovers to
**0.02–0.05 m**. The inversion also becomes well-posed: with the cerebellum a
deterministic function of motor, reaching is a clean **2-d** inversion over the
motor command (no 62 extra hidden DOF to absorb the goal).

### Concrete code changes (`core/pc_graph.py`, `core/pc_brain.py`)

- **`PCGraphParams`**: add a static field `frozen_edges: tuple` (edge indices
  whose `W` is never updated) and a static field `feedforward_nodes: tuple`
  (nodes whose μ is set to their top-down prediction each relax step instead of
  being freely inferred).
- **`init_region_graph`**:
  - put the `motor→cerebellum` edge index in `frozen_edges`;
  - initialise that edge as the scaled random expansion (a dedicated init, like
    `apply_foveal_gabor_init`, e.g. `apply_granule_expansion_init`) and set
    `bias_cb` to a fixed random vector;
  - add `cerebellum` to `feedforward_nodes`;
  - **remove `cerebellum` from the learnable `bias_nodes`** (its threshold is
    the fixed random `b_mc`, part of the frozen expansion). Keep `sensory` in
    `bias_nodes` (the readout DC term).
- **`_graph_relax_step` / `pc_graph_relax`**: for `feedforward_nodes`, after
  computing predictions set `μ_node = prediction_node` (deterministic granule
  activity) instead of taking the gradient step. Equivalent and simpler to start
  with: give feedforward nodes a very high fixed prior precision so they stay
  glued to their motor-driven prediction (approximate feedforward) — but the
  exact "set to prediction" is cleaner and avoids a magic precision.
- **`pc_graph_learn`**: skip `frozen_edges` in the weight-update loop.
- **Readout step stability:** the Purkinje update `ΔW = η·ε⊗φ(g)` is an LMS step
  on features with ‖φ‖² up to ~`cb_size`; LMS is stable only for
  `η < 2/‖φ‖²`. With `cb_size=64` and `eta_w=0.05` it can diverge
  (`diag_phase1b.py`, first run, NaN). Use either a smaller readout `η`, or
  normalise the step by the feature energy (NLMS: `η/(‖φ‖²+δ)`) on this edge.
  Note the prior memory warns NLMS "explodes for silent sources" — that risk is
  for sparse/silent presynaptics; the granule layer is densely active, so NLMS
  (or a fixed conservative `η`) is appropriate here. Pick by the LMS stability
  bound, not by reach success.

### Acceptance tests (Phase 1)

- `cb_var` (forward-pass cerebellar feature variance) stays **> 0** across
  babble (no collapse). `‖W_motor→cb‖` stays constant (frozen).
- Forward-model decoded tip err **< 0.1 m** and *falls* with babble.
- Reach success **rises with babble** and clearly exceeds the ~20 % chance
  floor. Add a unit test asserting `cb_var` does not decay below, say, half its
  init over a short babble.

### Honest caveat — what Phase 1 does and does **not** buy

Phase 1 removes the **binding constraint** (the collapse) and makes the forward
model accurate. It does **not** automatically guarantee 80 % reach: with a good
forward model, `diag_phase1b.py` still shows ~0.1 m residual reach error in the
crude synthetic, exposing **secondary precision limits** — population-code
resolution (12 cells over 1 m ⇒ ~0.09 m spacing, comparable to the 0.05 m
success threshold) and inversion precision. After Phase 1, **re-measure on the
real arm**; reaching 80 % may additionally require finer population codes or the
Phase-2 substrate work. Phase 1's success criterion is *the collapse is cured
and reach rises with babble*, not a specific percentage.

---

## 6. Phase 2 — the professional, neuro-AGI-worthy solution

### Problem statement

Phase 1 is the right fix **for the cerebellum specifically**, because a real
granule layer *is* a fixed random expansion. But the diagnosis is general: **the
single local PC rule collapses any deep *plastic* generative edge feeding a free
hidden layer to a predict-the-mean trivial fixed point.** Cortical hierarchies,
world models, and other deep causes the agent must *learn* (and cannot legitimately
freeze) will fail the same way. **Phase 2 = make the local rule itself able to
train deep plastic PC edges without collapse**, so the substrate can grow its own
deep generative models — the precondition for precision Π to later arbitrate
body-vs-world and retire the hardcoded `hold_nodes`/seeds.

### Precise characterisation (for the literature match)

Two related but distinct axes:
- **Depth instability** of PCNs (training degrades past ~5–7 layers).
- **Over-complete free-hidden-layer degeneracy** — present here **even at
  depth 1** (one hidden layer): an over-wide free latent squeezed between
  clamps, with learnable biases, collapses to a constant. Our failure is
  primarily this second axis. Phase 2 must address both, but the variance/
  covariance (anti-degeneracy) direction is the closest match to our mechanism.

### Reading list (starting points — verify exact bibliographic details)

Directly on the phenomenon in PC:
- **Qi, Forasassi, Lukasiewicz, Salvatori (2025), "Towards the Training of
  Deeper Predictive Coding Neural Networks."** Systematically diagnoses and
  partially fixes the degradation of equilibrium-prop-trained PCNs past ~5–7
  layers — the right entry point for the depth axis.
- **"Stable and Scalable Deep Predictive Coding" (OpenReview).** Depth makes PCN
  training increasingly unstable; underlying mechanisms are still poorly
  understood — i.e. an open theory problem, not just engineering.

Closest mechanism to our anti-collapse direction:
- **Tang, Salvatori, Millidge, Song, Lukasiewicz, Bogacz (2022), "Recurrent
  predictive coding models for associative memory employing covariance
  learning."** Introduces **covariance learning** as a mechanism that prevents
  representational degeneracy in recurrent PCNs — directly the "decorrelate /
  constrain the hidden code's covariance" idea.

General anti-collapse mechanisms (outside PC, directly transferable):
- **VICReg — Bardes, Ponce, LeCun (2021).** Explicit **variance** + **covariance**
  regularisation to prevent embedding collapse in joint-embedding predictive
  architectures; eliminates negative pairs by replacing them with an explicit
  variance constraint. The variance term *directly* counters our `cb_var → 0`;
  a near drop-in regulariser to graft onto the PC hidden code.
- **V-JEPA with variance-covariance regularisation — Drozdov, Shwartz-Ziv, LeCun
  (2024).** The same anti-collapse regularisation in a hidden-state *prediction*
  setting — closer to a forward model than to image embeddings.

Theory of why the rule reaches the bad fixed point:
- **"Tight Stability, Convergence, and Robustness Bounds for Predictive Coding
  Networks" (2024).** Formal stability/convergence bounds for PCNs — to
  *understand* the decay fixed point (and pick steps/precisions that avoid it),
  not merely patch it.

### Candidate solution directions (concrete, in priority order)

1. **Local variance/covariance regularisation of the hidden code (VICReg- /
   covariance-learning-style), made online and local.** Add, to the cerebellum
   (and generally any deep hidden node), a term that (a) keeps each unit's
   variance across recent inputs above a floor (anti-collapse) and (b)
   decorrelates units (anti-redundancy). Must be expressed as a **local** update
   compatible with the one-rule philosophy (e.g. a running per-unit variance
   estimate gating the relaxation/learning, à la Tang 2022 covariance learning).
   Success target: the `motor→cerebellum` edge can be **plastic** and still learn
   the forward model to < 0.1 m with `cb_var` bounded away from 0.
2. **Stability-bound-guided inference/learning.** Use the 2024 PCN bounds to set
   the μ-step and precision schedule so the decay fixed point is not an
   attractor — addresses the *why*, possibly removing the need for an explicit
   regulariser, or complementing it.
3. **Deeper-PCN training recipe (Qi 2025).** Adopt whatever initialisation /
   normalisation / skip structure that work shows restores trainability past a
   few layers, for the cortical hierarchy specifically (where depth, not just
   width, will bite once perception is allowed to learn).
4. **(Fallback) amortised recognition for deep latents.** Generalise Phase 1's
   feedforward move: learn a feedforward recogniser for deep hidden causes
   instead of pure iterative inference. This is a retreat from "everything is a
   free PC latent" and should be used only if 1–3 are insufficient — but it is a
   legitimate, well-precedented hybrid (amortised + iterative PC).

### Phase-2 evaluation harness

The scripts in this folder **are** the testbed: a deep plastic edge through a
free hidden layer, with direct readouts of `cb_var`, forward-model fit, and
reach — all on CPU, no MuJoCo. Evaluate each candidate by: (i) does `cb_var`
stay > 0 with the deep edge **plastic**? (ii) does the forward model reach
< 0.1 m? (iii) does it generalise to a 3-link arm / higher-d hidden layer (does
the mechanism scale)? Only then port to `core/pc_graph.py` and the real arm.

### Definition of done (Phase 2)

The substrate learns a deep generative model **with a plastic deep edge and a
free hidden layer**, by a **local** rule, without collapse — demonstrated on the
forward model (and ideally on a small learnable cortical hierarchy). At that
point the hardcoded `hold_nodes` and hand-seeds become removable in favour of
precision-arbitrated inference, which is the actual neuro-AGI milestone.

---

## 7. Reproducing the diagnostics

All scripts are self-contained (synthetic FK body + real `core/` graph), CPU,
no MuJoCo. Run from the repo root:

```bash
cd <repo root>
PYTHONPATH="$PWD" python diagnostics/embodied_reach_collapse/diag_collapse.py
```

Order of the argument: `diag_inversion` (reproduce) → `diag_fwd` (rule out
off-manifold) → `diag_code` (rule out capacity / code) → `diag_mech` (find the
collapse) → `diag_collapse` (confirm the mechanism) → `diag_marralbus`
(representability of the fix) → `diag_phase1` (freeze-only is insufficient) →
`diag_phase1b` (the corrected feedforward-granule fix).

To confirm on the **real MJX arm** (optional; the mechanism is body-independent),
add a notebook cell that, over babble checkpoints, logs `cb_var` (variance of
`tanh(μ_cerebellum)` over a command grid in a motor-clamped forward pass) and
`‖W_motor→cb‖`; expect the same collapse signature.

## 8. Key numbers (appendix)

- Adam capacity ceiling: **0.024 m**. PC online: **0.39 m**.
- Collapse: ‖W_mc‖ 7.96→0.48; cb_var 0.155→2e-5; ‖bias_sensory‖ 0→1.78.
- Seed W_mc ×8 → decayed back to 0.54 / cb_var 3e-5 (strong attractor).
- Fixed expansion + optimal readout: **0.010–0.028 m**.
- Phase-1 corrected (feedforward granule + plastic readout): forward **0.05 m**
  (online) / 0.019 m (least-squares ceiling); reach above chance, ~0.1 m
  residual in the crude synthetic (secondary precision limits, see §5 caveat).
