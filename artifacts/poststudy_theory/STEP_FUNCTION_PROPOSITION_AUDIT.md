# Scalar latent-estimation / step-function proposition

## 1. Status and provenance

> **Post-study theoretical analysis.** This proposition and its verification were
> derived **after** completion of the three preregistered FormalCRRC studies.
> They were not part of any preregistration, must not be represented as
> preregistered, modify no frozen experimental outcome, and require no additional
> model inference.

| | |
|---|---|
| Classification | post-study theoretical analysis |
| Preregistered | **no** — no S1/2/3 preregistration contains this proposition |
| Model inference | **none run** |
| Frozen outcomes modified | **none** |
| New empirical quantity | **none** — TRR is the preregistered s-3 quantity, recomputed read-only |
| Verified by | [`tests/test_step_proposition.py`](../../tests/test_step_proposition.py) — 218 tests |
| Produced by | [`scripts/audit_step_proposition.py`](../../scripts/audit_step_proposition.py) |
| Machine record | [`step_proposition_audit.json`](step_proposition_audit.json), [`reachability_by_model.csv`](reachability_by_model.csv) |
| Module | [`src/formalcrrc/step_proposition.py`](../../src/formalcrrc/step_proposition.py) (new, unpinned) |
| Overall status | **PASS** |

This document adds an *interpretation* of an existing frozen result. It
introduces no new measurement. s 3 already reported nontrivial TRR; what is new
is a proof of what TRR = 0 rules out.

## 2. Proposition

Let $\tau_s$ be a strictly increasing threshold sequence after canonicalizing each
FormalCRRC predicate family so that larger $\tau$ means stricter evaluation.

**Tier 1 (labels).** Suppose the judge represents artifact $x$ by a single scalar
estimate $\hat z_x$ and evaluates every strictness level with the same comparison

$$\hat Y_x(s) = \mathbf 1[\hat z_x \ge \tau_s].$$

Then $\hat Y_x$ is a nonincreasing step function of $s$ with at most one
met-to-not-met transition.

**Tier 2 (margins).** Suppose additionally that the canonical decision margin is
an order-preserving readout of the same comparison,

$$M_x(s) = g(\hat z_x - \tau_s), \qquad g \text{ strictly increasing}.$$

Then $M_x$ is strictly decreasing, every level $j \in \{1,\dots,8\}$ is a strict
prefix record low, and by the s-3 reachability theorem every crossing
$0,\dots,9$ is reachable by an additive translation. In particular

$$\text{pure scalar latent-estimation model} \implies TR_x = 1.$$

**The usable direction is the contrapositive:**

$$TR_x = 0 \implies \text{the observed response cannot be explained solely by an erroneous scalar estimate under this order-preserving threshold model.}$$

**The converse is not claimed.** $TR_x = 1$ means only that this explanation
remains available; it does not show that the judge internally represents a scalar
latent quantity.

## 3. Assumptions

| # | Assumption | Where it is used | Verified |
|---|---|---|---|
| A1 | $\tau_s$ strictly increasing after canonicalization | Tier 1 and Tier 2 | §5 — **PASS**, $\tau_s = s$ in all three families |
| A2 | One scalar $\hat z_x$ per artifact, fixed across all nine levels | Tier 1 | assumption under test, not verified |
| A3 | The same comparison $\mathbf 1[\hat z \ge \tau_s]$ at every level | Tier 1 | assumption under test |
| A4 | Margin readout $M_x(s) = g(\hat z_x - \tau_s)$ | Tier 2 | assumption under test |
| A5 | $g$ strictly increasing | Tier 2 | assumption under test |
| A6 | s-3 reachability characterisation is exact | Tier 2 → $TR_x$ | §6 — **PASS**, three independent implementations agree |

A2–A5 are the hypothesis being falsified; A1 and A6 are facts about the
repository, and both were checked rather than assumed.

Note that A4/A5 need **no** assumption about where $g$ crosses zero. Reachability
quantifies over all additive translations, which absorbs any offset in $g$. Only
the strict monotonicity of the margin *sequence* is used.

## 4. Proof

**Tier 1.** $\tau_s$ is strictly increasing, so $s < s'$ implies
$\tau_s < \tau_{s'}$. If $\hat Y_x(s') = 1$ then $\hat z_x \ge \tau_{s'} > \tau_s$,
hence $\hat Y_x(s) = 1$. So $\hat Y_x$ is nonincreasing in $s$: once it leaves the
met state it never returns, giving at most one met-to-not-met transition. $\square$

**Tier 2.** For each $s$, $\hat z_x - \tau_{s+1} < \hat z_x - \tau_s$ because
$\tau$ is strictly increasing. Applying the strictly increasing $g$ preserves the
strict inequality:

$$M_x(s+1) = g(\hat z_x - \tau_{s+1}) < g(\hat z_x - \tau_s) = M_x(s).$$

So $M_x$ is strictly decreasing. For any $j \ge 1$, strict decrease gives
$\min_{t<j} M_x(t) = M_x(j-1) > M_x(j)$, so $j$ is a strict prefix record low.
Every $j \in \{1,\dots,8\}$ therefore qualifies, and with crossings 0 and 9
always reachable the reachable set is all of $\{0,\dots,9\}$. Since
$j^*_x \in \{1,\dots,9\}$ by construction, $j^*_x$ is reachable, i.e.
$TR_x = 1$. $\square$

**Converse of the record-low step (used below).** All of $1..8$ are strict prefix
record lows **iff** the curve is strictly decreasing. ($\Leftarrow$) shown above.
($\Rightarrow$) if every $j$ is a record low then $M(j) < \min_{t<j} M(t) \le
M(j-1)$. This equivalence is asserted on 2,000 random curves in
`test_strictly_decreasing_iff_all_noninitial_are_record_lows`, and full
reachability is likewise equivalent to strict decrease.

**Consequence worth stating precisely.** Tier 2 implies strictly more than
$TR_x = 1$: it implies $RC_x = 10$. $TR_x$ is used as the headline test because it
is the *preregistered* s-3 quantity and the *weakest* consequence of the model,
so the falsification rests on the least the model commits to. The stronger
signature is reported in §8 as a secondary, clearly post-study observation.

## 5. Canonicalization of all three predicate families

The implementation stores each family in its own natural orientation, and two of
the three print a threshold that **decreases** with strictness. Canonicalization
is therefore required, not cosmetic. Verified exhaustively over
3 families × 9 latent levels × 9 strictness levels = 243 cells.

| | coverage (A) | max_violation (B) | numeric_tolerance (C) |
|---|---|---|---|
| Stored latent | $m$ — required items present | $v$ — prohibited items present | $e$ — absolute numeric error |
| Latent range | $0..8$ | $0..8$ | $0..8$ |
| Stored strictness index | $s = 0..8$ | $s = 0..8$ | $s = 0..8$ |
| Formal predicate as implemented | $\mathbf 1[m \ge s]$ | $\mathbf 1[v \le 8-s]$ | $\mathbf 1[e \le 8-s]$ |
| Raw printed threshold | $s$ (increasing) | $8-s$ (**decreasing**) | $8-s$ (**decreasing**) |
| Transformation required | no | **yes** (reflect) | **yes** (reflect) |
| Canonical compliance $z$ | $z = m$ | $z = 8 - v$ | $z = 8 - e$ |
| Canonical threshold $\tau_s$ | $s$ | $s$ | $s$ |
| Canonical predicate | $\mathbf 1[z \ge \tau_s]$ | $\mathbf 1[z \ge \tau_s]$ | $\mathbf 1[z \ge \tau_s]$ |
| $\tau_s$ strictly increasing | **yes** | **yes** | **yes** |
| Direction of increasing strictness | $s$ up = stricter | $s$ up = stricter | $s$ up = stricter |
| True first fail $j^*_x$ | $m+1$ | $9-v$ | $9-e$ |
| $j^*$ in canonical form | $z+1$ | $z+1$ | $z+1$ |
| Status | **PASS** | **PASS** | **PASS** |

Families B and C store a *defect* count whose printed allowance falls as
strictness rises; reflecting to $z = 8 - \text{latent}$ turns both into the same
canonical comparison as family A. After canonicalization all three share one
strictly increasing ladder $\tau_s = s$ and one closed form $j^* = z + 1$.

Verified computationally:

- `canonical_labels(family, latent)` reproduces `predicates.truth_curve` exactly
  for all 243 cells.
- `canonical_first_fail` reproduces `predicates.true_first_fail_index` exactly.
- $\tau_s$ is strictly increasing in every family.
- Every family is satisfied at $s = 0$, so $j^* \ge 1$ and the boundary support
  is $1..9$ — consistent with the frozen design.

**Result: PASS.** No family violates the required assumptions, so the proposition
is valid after canonicalization and nothing had to be forced.

## 6. Relationship to the s-3 reachability theorem

s 3 ([`src/formalcrrc/s3.py`](../../src/formalcrrc/s3.py)) defines
reachability exactly as the proposition requires. Conventions confirmed
individually:

| Convention | s-3 implementation | Compatible |
|---|---|---|
| Indexing | strictness $s = 0..8$; crossings $j = 0..9$ | yes |
| Record enumeration | over $j = 1..8$; level 0 has no prefix and is never a record | yes |
| Zero-crossing rule | first level with margin $< 0$, else 9 (`crossing_index`) | yes |
| Margin exactly 0 | counts as **met** — $P = 0.5$ is not yet crossed | yes |
| Inequality | **strict**: $M(j) < \min_{t<j} M(t)$ | yes |
| Ties | equality never creates a reachable crossing | yes — see below |
| Crossing 0 | always reachable (translate far negative) | yes |
| Crossing 9 | always reachable (translate far positive) | yes |

`prefix_record_lows` walks a running minimum and appends $j$ only when
`curve[j] < running_min`; `reachable_crossings` returns
`{0, 9} ∪ prefix_record_lows`. That is precisely the set the proposition's Tier-2
conclusion fills completely.

Cross-checked three ways on 2,000 random integer curves and on all 1,296 frozen
artifact-judge curves: the theorem as written out from its statement, the
repository's `prefix_record_lows`, and a brute-force intercept scan that uses no
theorem at all all produce identical reachable sets.

### Ties

**Ties are not a weakness of the theorem; they are evidence against the model.**

The proposition assumes $\tau_s$ strictly increasing and $g$ strictly increasing.
Their composition forces $M_x$ to be **strictly** decreasing, so under the model
no two strictness levels can carry the same margin. An observed equality
$M(j) = M(t)$ for some $t < j$ is therefore already outside the model's
assumptions — it falsifies A4/A5 directly, before reachability is even consulted.

s3's strict rule records this rather than accommodating it. Had reachability
used $\le$, a tied level would have been counted reachable, which would have
credited the model with a crossing that a strictly-increasing readout can never
produce. The strict convention is the one that matches the hypothesis being
tested; weakening it would have made TRR easier to pass for exactly the curves
that most clearly violate the model.

Ties are common in the frozen s-3 data, and their frequency is itself
informative (counts are over 324 artifacts per judge):

| Judge | artifacts with an adjacent equality | with a prefix tie | boundary blocked by equality |
|---|---:|---:|---:|
| Qwen2.5-14B | 58 | 46 | 3 |
| Llama-3.1-8B | 264 | 201 | 37 |
| Mistral-7B-v0.3 | 37 | 23 | 2 |
| Gemma-2-9B | 150 | 145 | 29 |

Llama shows an adjacent equality on 264 of 324 curves. Those curves are
incompatible with a strictly order-preserving scalar readout regardless of where
their boundary happens to fall.

**Result: PASS.** The proposition and the s-3 theorem use compatible
definitions, and the tie convention is the one the proposition requires.

## 7. Synthetic verification

Cases A–E, all in [`tests/test_step_proposition.py`](../../tests/test_step_proposition.py).

| Case | Margins | Strictly decreasing | Record lows | RC | Unreachable | Verdict |
|---|---|:--:|---|---:|---|---|
| **A** scalar step | $[5,4,3,2,1,-1,-2,-3,-4]$ | yes | $1..8$ (all) | 10 | none | model consistent |
| **B** mislocated | $[4,3,2,1,0,-1,-2,-3,-4]$ | yes | $1..8$ (all) | 10 | none | model consistent |
| **C** ordering failure | $[5,4,2,3,1,0,-1,-2,-3]$ | no | $1,2,4,5,6,7,8$ | 9 | $\{3\}$ | **model falsified at $j^*=3$** |
| **D** tie | $[5,4,4,3,2,1,0,-1,-2]$ | no | $1,3,4,5,6,7,8$ | 9 | $\{2\}$ | **model falsified at $j^*=2$** |

**A.** Every noninitial index is a strict prefix record low; prefix record rate
1.0; all ten crossings reachable; no formal boundary can falsify this curve.

**B — the key contrast.** The curve is still a coherent scalar readout, but its
zero crossing sits at $\hat\jmath = 5$. Against a boundary at $j^* = 2$ the raw
TCE is 3 — a substantial localization error — yet **every** candidate crossing
remains reachable and the oracle translation TCE is 0 for *every* boundary in
$1..9$. Wrong location, intact ordering: fully correctable by a scalar shift.

**C — ordering failure.** Margin rises from 2 (level 2) to 3 (level 3), so level 3
is not a strict prefix record low: $3 \not< \min(5,4,2) = 2$. Crossing 3 is
unreachable. Confirmed independently by scanning 200,001 intercepts across
$[-20, 20]$ — crossing 3 never appears. The oracle translation TCE at $j^*=3$ is
1, bounded away from zero. No amount of location correction produces this
boundary; only crossing 3 is affected, so the falsification is specific.

**D — tie.** Levels 1 and 2 share margin 4. Level 1 is a record low; level 2 is
not, because the rule is strict. Crossing 2 is unreachable, again confirmed by
brute-force intercept scanning. `tie_diagnostics` reports
`adjacent_equal_margins = 1`, `prefix_tie_levels = 1`,
`target_blocked_by_equality = True`. Separately verified that a scalar-model curve
$g(\hat z - \tau_s)$ **never** contains a tie, for 121 values of $\hat z$.

**E — translation invariance.** For all four curves and translations
$\{-100, -3.5, -1, 0, 0.25, 2, 17, 10^6\}$, the prefix-record set, the reachable
set, the strict-decrease verdict and the falsification verdict are all unchanged.
Positive rescaling ($\times 0.01$ to $\times 250$) also leaves the reachable set
identical, confirming that any strictly increasing affine readout gives the same
answer.

Additionally, scalar-model margins $g(\hat z - \tau_s)$ were generated for
$g \in \{\text{identity}, \tanh, \text{affine}, x^3, \text{logistic}\}$ and
$\hat z \in \{-1, 0, 2.5, 4, 9\}$: **every** instance is strictly decreasing,
fully reachable, and never falsified at any boundary — the proposition's Tier-2
conclusion, confirmed.

## 8. Frozen s-3 empirical counts

Recomputed **read-only** from frozen raw margins
(`artifacts/s3/raw_scores/*.parquet`), with $j^*$ rebuilt from the predicate
definitions rather than read from a stored column. Nontrivial means
$j^* \in 1..8$; crossings 0 and 9 are always reachable and carry no information.

| Judge | nontrivial curves | $TR_x=1$ | $TR_x=0$ | TRR (exact) | TRR | $1-TRR$ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-14B | 288 | 138 | 150 | $23/48$ | **47.92%** | **52.08%** |
| Llama-3.1-8B | 288 | 78 | 210 | $13/48$ | **27.08%** | **72.92%** |
| Mistral-7B-v0.3 | 288 | 96 | 192 | $1/3$ | **33.33%** | **66.67%** |
| Gemma-2-9B | 288 | 174 | 114 | $29/48$ | **60.42%** | **39.58%** |

**Agreement with the frozen report: exact.** `RESULTS_s3.md` reports nontrivial
TRR of 47.9%, 27.1%, 33.3% and 60.4% over $n = 288$ — identical to the
recomputation. Three independent reachability implementations (theorem written
from its statement, `s3.prefix_record_lows`, brute-force intercept scan) agree
on all 1,296 curves. **No discrepancy was found.**

### Secondary observation (post-study, not a s-3 endpoint)

Tier 2 implies the stronger signature $RC_x = 10$, equivalently a strictly
decreasing margin curve. That is rare:

| Judge | strictly decreasing curves | share of 324 |
|---|---:|---:|
| Qwen2.5-14B | 12 | 3.70% |
| Llama-3.1-8B | 0 | 0.00% |
| Mistral-7B-v0.3 | 1 | 0.31% |
| Gemma-2-9B | 12 | 3.70% |
| **Total** | **25 / 1296** | **1.93%** |

Read carefully: this says the *full* Tier-2 model — one scalar compared through a
strictly order-preserving readout at every level — describes about 2% of observed
curves. The TRR-based figure in the main table is the conservative statement,
because $TR_x = 1$ is only the weakest consequence of the model. Both are reported;
the TRR figure is the one tied to a preregistered quantity.

## 9. Interpretation

> For curves with $TR_x = 0$, the observed judge margin ordering is incompatible
> with a model in which the only error is an incorrect scalar estimate of
> compliance followed by an otherwise order-preserving comparison against
> progressively stricter rubric thresholds.

Across the four judges, **39.6%–72.9%** of nontrivial formal boundaries are
unreachable by any additive translation. For those curves the failure is not that
the judge put its boundary in the wrong place — it is that the ordering of its
margins does not contain the required boundary at all. No recalibration, no
threshold shift, and no per-judge or per-family intercept can produce it, because
translation cannot reorder margins.

This also explains the s-2 result from a different direction: intercept
calibration improved TCE for some judges but could not close the gap, and the
proposition says why — a location-only correction is the exact operation that
$TR_x = 0$ curves are immune to.

Case B is the control that gives this content. A judge can be badly mislocated
(raw TCE 3 in the synthetic example) and still be entirely consistent with the
scalar model. Being wrong is not the falsifier; being *disordered* is.

## 10. Limitations and claims that must not be made

This is the falsification of **one specific simple explanation**, nothing more.

**Must not be claimed:**

- ❌ that the LLM has no latent representation;
- ❌ that the LLM does not understand the rubric;
- ❌ that all calibration is impossible;
- ❌ that $TR_x = 1$ proves a scalar representation is used;
- ❌ that this establishes anything about hidden neural representations;
- ❌ that every conceivable latent-variable model is ruled out;
- ❌ that any of this was preregistered.

**Further limitations:**

- $TR_x = 0$ falsifies the **conjunction** of A2–A5. It does not isolate which
  conjunct fails: a judge might hold a scalar estimate but read it out
  non-monotonically, or might not use a single scalar at all. The audit cannot
  distinguish these.
- The result is conditional on the FormalCRRC margin definition
  $M = S_A - S_B$ from next-token logits, the 0.5 decision boundary, and the
  nine-level canonical ladder. A different readout could yield different orderings.
- Multi-dimensional latent models, order-dependent readouts, and models with
  level-dependent noise are **not** addressed. A stochastic scalar model in which
  the estimate is re-drawn per level is outside the proposition entirely, since A2
  fixes one $\hat z_x$ across all nine levels.
- $TR_x$ is a per-artifact binary; TRR is a proportion over a balanced design. No
  new uncertainty quantification is introduced here — the s-3 bootstrap CIs
  remain the interval estimates of record.
- The 1.93% full-reachability figure is a post-study descriptive statistic, not a
  preregistered endpoint, and carries no interval.

## 11. Integrity verification

Fingerprinted before any work began and re-verified afterwards.

| | |
|---|---|
| Frozen files covered | **109** — `artifacts/s{1,2,3}/**`, `figures/s{1,2,3}/**`, `artifacts/poststudy_baselines/**`, all three preregistrations, all three results documents, the s-1 errata / provenance / integrity addendum, and the post-study baseline audit |
| Combined SHA-256 before | `8693acf874a6da41fb88d1c3321d9f9bd3fa1d33e568a64db10fdd62348fadeb` |
| Combined SHA-256 after | `8693acf874a6da41fb88d1c3321d9f9bd3fa1d33e568a64db10fdd62348fadeb` |
| Result | **byte-identical — PASS** |
| Recorded prior manifest (`artifacts/s3/prior_experiments_manifest.json`, 77 files) | re-verifies, **read only, never rewritten** |

No old hash was updated to accommodate a modification, because there was no
modification. New files are confined to additive locations:

```
src/formalcrrc/step_proposition.py          new module, unpinned by any preregistration
scripts/audit_step_proposition.py           new script
tests/test_step_proposition.py              new tests (218)
artifacts/poststudy_theory/                 new directory (this file + JSON + CSV + manifest)
```

`docs/DEVLOG.md` receives a labelled entry. It is an engineering log, states in
its own header that it is not the scientific record, and is pinned by no
integrity manifest.

Full test suite after this work: **1097 passed**.

## 12. Manuscript-facing wording

No manuscript source exists in this repository, so nothing was edited. When one is
written, the following conservative wording is recommended, using the exact
repository-derived percentages:

> Under a scalar latent-estimation model, a judge represents an artifact by a
> single compliance estimate and compares that same estimate against
> progressively stricter thresholds. Such a model necessarily produces an ordered
> step response; with a strictly order-preserving margin readout, every
> nontrivial boundary is reachable by a scalar cutoff. Consequently, $TR_x = 0$
> rules out the hypothesis that the observed error is solely a mislocated scalar
> compliance estimate under this model. Across the four judges, 39.6%–72.9% of
> nontrivial formal boundaries were unreachable, indicating that these errors
> require more than a simple location correction. This result does not establish
> the judge's internal representation, and $TR_x = 1$ does not imply that a scalar
> latent representation is actually used.

Exact per-judge figures for a table: unreachable share 52.08% (Qwen2.5-14B),
72.92% (Llama-3.1-8B), 66.67% (Mistral-7B-v0.3), 39.58% (Gemma-2-9B), each over
$n = 288$ nontrivial boundaries.

Wording requirements:

- Label the proposition **post-study theoretical analysis**; never imply it was
  preregistered.
- State the falsified hypothesis as a conjunction (single scalar estimate **and**
  order-preserving readout), not as "the model has no latent variable".
- Do not present $TR_x = 1$ as positive evidence of a scalar representation.
- If the 1.93% full-reachability figure is used, mark it as a post-study
  descriptive statistic distinct from the preregistered TRR.

## 13. Related documents

| | |
|---|---|
| [`docs/PREREGISTRATION_s3.md`](../../docs/PREREGISTRATION_s3.md) | The frozen protocol defining TRR — unchanged |
| [`docs/RESULTS_s3.md`](../../docs/RESULTS_s3.md) | The frozen s-3 record — unchanged |
| [`docs/POSTSTUDY_BASELINE_AUDIT.md`](../../docs/POSTSTUDY_BASELINE_AUDIT.md) | The other post-study audit — reference baselines |
| [`docs/DEVLOG.md`](../../docs/DEVLOG.md) | Engineering log |
