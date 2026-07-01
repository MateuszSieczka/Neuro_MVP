# Phase-1 reach oscillation — root cause (the *inversion*, not the model)

**Diagnosed:** 2026-07-01. **Method:** real `core/` graph (marr_albus default,
Phase-1 implemented) + synthetic 2-link FK body, CPU JAX, no MuJoCo. Scripts in
this folder: `diag_oscillation.py`, `diag_inversion2.py`, `diag_motorflow.py`,
`diag_motorfix.py`.

## 0. Symptom

After Phase 1, reach success oscillates **6–68 %** across babble checkpoints
with **no upward trend** as the budget grows, while `max|W|` is pinned at 3.18.
The user asked: what drives the wild fluctuation, and how to fix it — without
test-tuning.

## 1. Verdict

**Phase 1 worked. The collapse is cured.** `cb_var` stays at its init (0.14, no
decay), `|W_mc|` is constant (frozen granule edge — *that* is the 3.18 `max|W|`,
not a readout that "won't grow"), and the **forward model is accurate and
stable**: decoded-tip error falls 0.95→0.07 m and settles; `|ΔW_cs|` decays
2.4→0.2. So the forward model is *not* the problem and online misadjustment is
*not* the problem.

**The bottleneck is the INVERSION.** Active inference (`pc_act_infer`) cannot
invert even a *near-perfect* forward model here. With the least-squares-ceiling
readout (forward error **0.028 m**), reach is still **0.59 m / 0 %**, and the
inferred command is **≈ 0** (`mean|cmd| = 0.000`, identical for 80→1500 relax
sweeps). Reach is **decoupled from forward-model quality** — which is exactly why
it neither improves with babble nor holds: it wanders with readout *noise*, not
with model accuracy.

## 2. Mechanism — the diagonal-Newton step starves the flat-prior action node

`diag_motorflow.py` instruments the goal→motor error flow at the inference fixed
point (LS readout, `|W_cs|≈3.6e4`):

```
g_motor (gradient)   ≈ [20, 49]          # healthy, correct sign — the goal DOES reach motor
L_motor (curvature)  ≈ [1.4e9, 5.4e8]    # astronomically large
step = η_μ·g/L       ≈ 1e-9              # motor frozen at 0
|eps_tip|            = 2.328 → 2.329     # error NEVER decreases over 1500 sweeps
```

The motor node carries a **flat prior** (`Π_motor = 0`, by design — action has no
prior preference). So its diagonal-Newton curvature is *entirely* the
child-relayed term

```
L_motor = φ'(μ_m)² · (W_mc²)ᵀ · [ φ'(μ_cb)² · (W_cs²)ᵀ · Π_s ]      ∝ Π_s·W_mc²·W_cs²
g_motor = φ'(μ_m)  ·  W_mcᵀ  · [ φ'(μ_cb)  ·  W_csᵀ  · Π_s · ε_s ]   ∝ Π_s·W_mc·W_cs·ε
⇒  step = η_μ · g/L  ∝  ε / (W_mc · W_cs)        (Π_s cancels)
```

The action step **scales as 1/(W_mc·W_cs)** — inversely with the forward-model
gain. The curvature-preconditioner (diagonal-Newton) that makes *perception and
learning* unconditionally stable (it divides by `Π`, which dominates a
perceptual node) becomes **pathological on the action node**, where there is no
`Π_motor` to dominate and the denominator is pure `W²`-fan-out. Two failure
modes compound:

1. **Wrong scale.** The step vanishes as the readout grows/sharpens, so a
   *better* forward model makes the command *more* frozen.
2. **Wrong metric.** The diagonal of `JᵀΠJ` for a 2-D cause fanning into a 24-D
   high-gain likelihood **overcounts** independent output curvatures and ignores
   the dominant cross-coupling — so it undershoots by a large factor even after
   scale is accounted for (`normalized`-gradient also ≈0 %; only the *full* 2×2
   Gauss-Newton reduces the error — see §3).

## 3. Evidence (discriminating scale vs metric vs code)

| Script | Test | Result |
|---|---|---|
| `diag_oscillation` | babble→checkpoint, fixed 64-target probe | fwd_err 0.95→0.07 m (settles); reach stuck 0.3–0.5 m / ~0–6 %; `cb_var` flat 0.14 (no collapse); EMA readout no better → **not** misadjustment |
| `diag_oscillation` | LS-ceiling readout (fwd 0.028 m) | reach **0.59 m / 0 %** → model perfect, reach fails |
| `diag_inversion2` | `pc_act_infer`, n=80…1500 | `mean|cmd| = 0.000`, reach 0.57 m at every depth → a fixed point at μ_motor≈0, not under-relaxation |
| `diag_inversion2` | brute-force command grid | best reach **0.029 m / 73 %** → targets ARE reachable; fault is the inversion |
| `diag_motorflow` | hand-stepped relax | `g≈[20,49]`, `L≈1e9`, step≈1e-9, `eps_tip` constant → diagonal curvature starves the step |
| `diag_motorfix` | swap motor metric, same model | diag-newton **0 %** vs **gauss-newton 17 %** on the identical (perfect) model |
| `diag_motorfix`/sweep | readout scale | `|W_cs|≈15` → diag-newton **49 %**; `|W_cs|≈3.6e4` → diag-newton **0 %** (step ∝ 1/W² confirmed) |

Key discriminators:
- **Reach is set by `|W_cs|`, not by forward accuracy** (49 % vs 0 % for the same
  rule at two readout scales) → the oscillation is the action step's scale
  tracking the (drifting) readout, not model quality.
- **Full GN salvages the badly-scaled case; normalization does not** → it is a
  metric problem (geometry of a low-D cause under a high-gain likelihood), not
  merely a step-size problem.

## 4. Why this is the oscillation (and the "no trend")

The command is essentially *not inverted*: it sits near the un-driven baseline,
and the little motion it has is `∝ ε/(W_mc·W_cs)` in a direction set by the exact
readout entries. As online babble nudges the readout (norm and direction), the
tiny command rotates/rescales, the arm tip drifts near the workspace edge, and it
crosses the 0.05 m success threshold erratically. Closed-loop (180 steps) and the
16-episode sample (±~12 pp binomial noise) amplify it. Because the command is
metric-starved, **forward-model improvement cannot show up as reach** — hence the
flat, trendless wander around chance.

This is the neuro-AGI-relevant lesson: **a flat-prior action variable cannot be
inferred with the same diagonal-Newton metric that stabilises perception.** The
diagonal curvature of a low-D action under a high-D, high-gain forward model is a
degenerate metric; as the model sharpens (the *goal* of learning), the action
step collapses. Any future motor/option/world-model inversion that inverts a
sharp learned likelihood through a flat-prior cause will hit this same wall.

## 5. Fix (principled, not test-tuned)

Two complementary changes; (A) is the core fix, (B) hardens it.

**(A) Scale-free natural-gradient inference on the (low-D) action node.**
Replace the *diagonal* curvature with the **full Gauss-Newton Hessian** for
action nodes (those with a flat prior and feeding a feedforward forward model).
The action dimension is tiny (`motor_dim = 2·n_joints`), so assembling and
inverting `H = JᵀΠJ + λI` per sweep is cheap, and it is the correct natural
metric for inverting `y = f(u)`: it accounts for cross-coupling and gives a step
whose *direction* solves the local linear inversion regardless of `|W|`.
Perceptual nodes keep the diagonal step (full Hessian infeasible and `Π`
dominates there anyway). Implementation: in `_graph_relax_step`, for a tagged
`action_nodes` set, accumulate the *full* child-relayed Hessian
`H_j = Σ_{j→c} W_{j→c}ᵀ · diag(curv_c) · W_{j→c}` (instead of only its diagonal)
and take `Δμ_j = η_μ · H_j⁻¹ g_j`. λ from the existing `var_floor`/`leak`, no new
magic constant.

**(B) Readout homeostasis so the metric stays well-conditioned.** Keep `|W_cs|`
(and `Π_s`) bounded — Purkinje synaptic normalisation / weight decay /
encoding the reafference targets so the LS solution does not need huge weights.
`diag_motorfix` shows the *existing* diagonal rule already reaches ~49 % once
`|W_cs|≈15`, so bounding the readout scale alone substantially de-oscillates the
current substrate even before (A).

**Secondary cap (separate axis, the FINDINGS §5 caveat).** Even a correct
inversion tops out ~45–50 % in this synthetic because of population-code
resolution (N=12 cells ⇒ ~0.09 m spacing vs the 0.05 m threshold) and the
near-flat monotonic tip code. After (A)/(B), re-measure on the real arm; pushing
past ~50 % will additionally want finer / better-conditioned tip codes. (A)/(B)'s
success criterion is **reach tracks forward-model quality and stops oscillating**,
not a specific percentage.

## 6. Acceptance tests

- `pc_act_infer` produces a command with `mean|cmd|` *bounded away from 0* and
  `|eps_tip|` that *falls* during the planning relax (not constant).
- Reach success on a fixed large target set is **insensitive to `|W_cs|` scale**
  (sweep the readout norm; diag-newton fails this, GN passes).
- Reach **rises with babble** and the checkpoint-to-checkpoint variance shrinks
  as the forward model settles (coupling restored).
