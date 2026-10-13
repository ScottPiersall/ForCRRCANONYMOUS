# FormalCRRC

FormalCRRC is a new research program motivated by the limitations identified in
pointwise, Judge Reliability Harness, and Threshold Invariance measurement methhods. It
replaces free-form semantic rubric generation with programmatically controlled
evaluation thresholds.

## The question

> When evaluation criteria differ only by a formally controlled decision
> threshold, do LLM judges exhibit the monotone counterfactual response function
> implied by the rubric?

An artifact $x$ is held **byte-identical** while a rubric's numeric decision
threshold is varied across nine ordered levels. The resulting sequence of judge
probabilities

$$C_x(\tau_1), \ldots, C_x(\tau_9), \qquad C_x(\tau) = P_J(\text{criterion met} \mid x, R_\tau)$$

is a **counterfactual rubric response curve**. Because the threshold is set
programmatically, the correct answer at every level is known exactly by
construction — no rubric generator, no validator, no consensus panel, no human
annotation.

## S1 at a glance

| | |
|---|---|
| Predicate families | coverage (A), max violation (B), numeric tolerance (C) |
| Artifacts | 3 families × 9 latent levels × 12 instances = **324** |
| Thresholds per artifact | **9** (strictness $s = 0 \ldots 8$) |
| Rubric-response pairs | **2916** per judge |
| Judges | Qwen2.5-14B, Llama-3.1-8B, Mistral-7B-v0.3, Gemma-2-9B (bf16) |
| Endpoint | $P_\text{met}$ from next-token A/B logits; no sampling |
| Primary metrics | MVR (δ = 0.05), MVM, TCE (0.5 boundary) |
| Statistics | 5,000-replicate artifact-clustered paired bootstrap, seed 42 |
| Human annotation | none |

## Pipeline

Run in order; each stage refuses to proceed if an earlier one is missing.

```bash
python scripts/generate_s1_dataset.py          # A/B  build + freeze the dataset
python -m pytest                               # B    full test suite
python scripts/audit_environment.py            # C    model availability gate (GPU)
python scripts/freeze_s1_preregistration.py    # D    freeze + checksum
python scripts/verify_s1_ready.py              # D    fail-closed readiness gate
python scripts/run_s1_inference.py --model …   # E    raw scores only, one judge
python scripts/verify_s1_integrity.py          # F    completeness + hashes
python scripts/analyze_s1.py                   # G    metrics, bootstrap, figures
python scripts/verify_s1_integrity.py --final  # H  integrity manifest + summary
```

On the GPU cluster, phases C and E run as SLURM jobs
(`slurm/audit_models.sbatch`, `slurm/run_inference.sbatch`);

## Layout

```
src/formalcrrc/     predicates, dataset, prompts, scoring, metrics, bootstrap, provenance
scripts/            one script per pipeline phase
slurm/              cluster jobs
tests/              the pre-inference test suite
artifacts/s1/     frozen dataset, preregistration, raw scores, results
figures/s1/       the five predeclared figures
```

## Integrity rules

- The preregistration is frozen before any study inference and is never silently
  edited; corrections happen as numbered, checksummed amendments that preserve
  the original.
- No outcome-driven tuning after inference begins: no changed thresholds, no
  changed tolerances, no recalibration, no monotonicity enforcement, no
  selective reruns, no post-hoc metrics.
- Truth comes from the formal predicate, never from a generator/validator chain.
- A weak or negative result is a result.

Licensed under the MIT License.

---

## S2 — Separating global judge bias from threshold-response shape

S2 asks how much of the S1 threshold-crossing error is explained by a
**global response bias** of the judge, and how much survives once curve location
is separated from curve shape.

The intervention is the weakest correction that can move a curve's location: a
single additive constant in decision-margin space, $M'(s) = M(s) + \alpha_J$,
with the slope fixed at exactly 1. Because the same constant is added at every
strictness level, adjacent differences are untouched — **location can change,
shape cannot** — and that invariance is the experimental control.

| | |
|---|---|
| Partitions | CALIBRATION (seed 43) and TEST (seed 44), 324 artifacts × 9 thresholds each |
| Disjointness | exact: no shared artifact id, rendered prompt, or family-C target pair with each other or S1 |
| Judges | the S1 panel at the S1 revisions, verified bit-identical on a non-study smoke prompt |
| Calibration | one intercept per judge, fitted on CALIBRATION only; family intercepts secondary |
| Statistics | 5,000-replicate two-stage bootstrap, refitting α inside every replicate |


### S1 immutability

S1 is fingerprinted in `artifacts/s2/s1_baseline_manifest.json` (30 files)
and re-verified at the end of S2. S2 code writes only to `artifacts/s2/`
and `figures/s2/`; the sole addition under `artifacts/s1/` is the additive
integrity addendum. The S1 dataset still reproduces content hash
`4462bedb…`, asserted permanently by the test suite.

---

## S3 — Translation limits of threshold-response curves

S3 asks the last location question available: if every curve receives the
**best possible additive translation** while its shape is held exactly fixed, how
much threshold-localization error is mathematically unavoidable?

The handle is exact. Since translation cannot reorder margins, a crossing
$j \in \{1,\dots,8\}$ is reachable by *some* additive translation **iff** $j$ is a
strict prefix record low, $M(j) < \min_{t<j} M(t)$; crossings 0 and 9 are always
reachable. That turns an open-ended optimisation into an enumerable set and makes
the primary quantity closed-form:

$$OTCE_x = \min_{j \in \mathcal{R}_x} |j - j^*_x|.$$

The oracle uses formal truth deliberately. It is a **diagnostic lower bound**, not
a deployable correction, and its optimism is exactly what makes a surviving
residual meaningful.

| | |
|---|---|
| Partition | S3_TEST (seed 45), 324 artifacts × 9 thresholds |
| Disjointness | exact: no shared artifact id, rendered prompt, or family-C target pair with S1, S2 CAL or S2 TEST |
| Judges | the same panel at the same revisions, bit-identical smoke scores for the third time |
| Oracle classes | global ⊆ family ⊆ artifact, all by exact breakpoint enumeration |
| Statistics | 5,000-replicate artifact-clustered bootstrap, group oracles refit in every replicate |


### Prior-experiment immutability

S1 and S2 are fingerprinted in `artifacts/S3/prior_experiments_manifest.json`
(77 files) and re-verified at the end of S3. S3 code writes only to
`artifacts/S3/` and `figures/S3/`.

---

## Post-study analytic baselines

An additive audit, added **after** all three studies were complete. It supplies
interpretable reference points for TCE and pointwise accuracy, derived in closed
form from the design alone: because the true first-fail boundary is exactly
uniform over 1..9, an artifact-blind strategy has an exact expectation and needs
no simulation.

| Reference strategy | TCE | Accuracy |
|---|---:|---:|
| Fixed center / no input (j = 5) | 2.222 | 75.31% |
| Uniform random crossing | 2.963 expected | 67.08% expected |
| Always met / majority class | 4.000 | 55.56% |
| Always not met | 5.000 | 44.44% |

These are structural reference strategies, **not** preregistered endpoints and not
competitive judge models. No model inference was run and no frozen outcome was
modified. See
[`docs/POSTSTUDY_BASELINE_AUDIT.md`](docs/POSTSTUDY_BASELINE_AUDIT.md).

```bash
python scripts/compute_reference_baselines.py   # additive; writes only artifacts/poststudy_baselines/
```

## Post-study theory: the step-function proposition

A second additive audit, also after the studies closed. It asks what the frozen
S-3 reachability result *rules out*.

A judge that holds one scalar compliance estimate and compares it against
progressively stricter thresholds must produce an ordered step response; with a
strictly order-preserving margin readout its margin curve is strictly decreasing,
so every boundary is reachable by a scalar cutoff. Hence `TR_x = 0` falsifies that
explanation. Across the four judges **39.6%-72.9%** of nontrivial formal
boundaries are unreachable, so those errors need more than a location correction.

This does **not** establish anything about internal representations, and
`TR_x = 1` does not imply a scalar latent is used. See
[`artifacts/poststudy_theory/STEP_FUNCTION_PROPOSITION_AUDIT.md`](artifacts/poststudy_theory/STEP_FUNCTION_PROPOSITION_AUDIT.md).

```bash
python scripts/audit_step_proposition.py        # additive; writes only artifacts/poststudy_theory/
```

---

## Label-swap counterbalance robustness study

A preregistered follow-up experiment, frozen before any inference and run on the
frozen S3 partition with the same four judges at the same revisions. Exactly
one conceptual factor moves: which response token means "criterion met".

| | Original (S1-S3) | Swapped |
|---|---|---|
| Response instruction | `Return A if the criterion is met.` | `Return B if the criterion is met.` |
| Canonical margin | `S_A - S_B` | `S_B - S_A` |

The two prompt templates differ at **exactly two byte positions**. A positive
canonical margin means "met" in both conditions, so every S3 convention
applies unchanged.

**The ordering failure survives the swap in every judge**: 47.9%-68.1% of
nontrivial formal boundaries remain unreachable by additive translation, against
39.6%-72.9% originally. The effect is not uniformly negligible, though -- one
judge of four met all three prespecified robustness criteria, and Llama-3.1-8B
shows a genuine ordering change (median within-curve Spearman rho 0.512).

```bash
python scripts/freeze_labelswap_preregistration.py   # freeze + fingerprint (before inference)
sbatch slurm/run_labelswap_inference.sbatch <model>  # one judge, swapped labels
python scripts/analyze_labelswap.py                  # pair, measure, bootstrap
python scripts/figures_labelswap.py
python scripts/verify_labelswap_integrity.py
python scripts/write_results_labelswap.py
```
