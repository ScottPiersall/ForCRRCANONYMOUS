# FormalCRRC Qwen2.5 Reason-Then-Score Protocol Ablation

**Study type:** Post-study preregistered within-model protocol ablation
**Relationship to prior work:** Follow-up to FormalCRRC Day 3 and the reasoning-capable external-anchor study
**Human subjects:** None
**New model inference:** Yes
**Dataset:** Frozen FormalCRRC Day-3 TEST partition, seed 45
**Ablation model:** `Qwen/Qwen2.5-14B-Instruct`
**Comparator:** Frozen Day-3 results for the exact same Qwen2.5-14B model/revision under immediate next-token A/B scoring
**New protocol:** Elicited reason-then-score
**Quantization:** None
**Precision:** BF16, matching Day 3
**Preregistration freeze timestamp:** `2026-09-09T21:38:03+00:00`
**Preregistration SHA-256:** `43e9175b3717f493b58657e2d88e22d145bb769fd4a3ea51f054880acc6a40b2`
**Exact model revision:** `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`
**Exact tokenizer revision:** `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`
**Amendments at freeze:** 0

---

## 1. Motivation

FormalCRRC Day 3 evaluated `Qwen/Qwen2.5-14B-Instruct` using an immediate two-label next-token margin. Under that protocol, Qwen2.5 had:

$$
TCE = 3.373
$$

and nontrivial:

$$
TRR = 0.4792,
$$

corresponding to:

$$
52.08\%
$$

of nontrivial true boundaries being unreachable under additive translation.

A separately preregistered external-anchor experiment evaluated `Qwen/Qwen3-30B-A3B-Thinking-2507` using a reason-then-score protocol. That anchor achieved:

$$
TCE = 0.401
$$

and

$$
TRR = 1.0,
$$

with 0/288 nontrivial true boundaries unreachable.

The anchor differs from the original Day-3 panel in model family, scale, training, and inference/readout procedure. The anchor therefore cannot determine whether the large difference in reachability arises primarily from model capability or from the reason-then-score protocol.

This experiment attacks that confound directly by holding the model and evaluation dataset fixed.

The narrow question is:

> For the exact same Qwen2.5-14B model and frozen Day-3 artifacts, does inserting an explicit reasoning stage before the final A/B likelihood readout materially change boundary localization and boundary reachability?

---

## 2. Experimental contrast

The study compares two protocols for the same model.

### Condition O — frozen original Day-3 protocol

Model:

`Qwen/Qwen2.5-14B-Instruct`

Protocol:

Immediate next-token A/B scoring at the initial assistant response boundary.

Canonical margin:

$$
M_x^O(s)=S_A-S_B.
$$

These scores already exist in the frozen Day-3 scientific record and will be read only.

No original Day-3 inference will be rerun.

### Condition R — new reason-then-score protocol

The same model and exact revision will first generate an elicited reasoning trace and will then be scored for A versus B conditional on that exact trace.

Canonical margin:

$$
M_x^R(s)=L_A-L_B.
$$

where \(L_A\) and \(L_B\) are full contextual continuation log-likelihoods after reasoning.

The principal scientific contrast is therefore:

$$
\text{same model}
+
\text{same artifact}
+
\text{same formal threshold}
+
\text{same A/B semantics}
$$

with:

$$
\boxed{
\text{immediate readout}
\quad\text{vs}\quad
\text{reason-then-score}
}
$$

The protocol change is intentionally composite: it includes explicit reasoning elicitation plus a final conditional likelihood readout. The study does not claim to identify which subcomponent causes any observed difference.

---

## 3. Dataset

Use the complete frozen Day-3 TEST partition, seed 45.

Expected design:

* predicate families: 3,
* artifacts: 324,
* strictness levels per artifact: 9,
* rows: 2,916,
* artifacts per family: 108,
* artifacts per true boundary: 36,
* nontrivial true-boundary curves: 288.

No subsampling is permitted.

Every new study row must pair exactly to one frozen Day-3 Qwen2.5 row.

---

## 4. Predicate canonicalization

All FormalCRRC families use the previously verified canonical form:

$$
Y_x(s)=\mathbf 1[z_x\ge s],
\qquad
s\in\{0,\ldots,8\}.
$$

### Coverage

$$
z=m,
\qquad
Y=1[m\ge s].
$$

### Maximum violations

$$
z=8-v,
\qquad
Y=1[v\le8-s].
$$

### Numeric tolerance

$$
z=8-e,
\qquad
Y=1[e\le8-s].
$$

For all three:

$$
j_x^*=z_x+1.
$$

Formal labels and true crossings must be independently recomputed from primitive dataset fields before inference.

---

## 5. Model

The exact Day-3 Qwen2.5 model will be reused:

`Qwen/Qwen2.5-14B-Instruct`

The exact:

* model revision,
* tokenizer revision,
* chat template,
* dtype,
* attention implementation where relevant,
* library environment,

must be recovered from frozen Day-3 provenance.

No model substitution is permitted.

No quantization is permitted.

No fine-tuning, LoRA, adapters, or parameter updates are permitted.

Use BF16 exactly as in Day 3.

If the frozen Day-3 revision cannot be loaded exactly, the study stops.

---

## 6. Why the Qwen3 native thinking protocol cannot be copied literally

Qwen2.5-14B-Instruct does not use the Qwen3 thinking-only chat-template mechanism.

Therefore this ablation cannot use a native `<think>...</think>` boundary without inventing model behavior that was not part of Qwen2.5.

Instead, Qwen2.5 receives an explicit reasoning instruction and generates until a fixed preregistered final-answer marker.

The experiment therefore evaluates:

**elicited reason-then-score**

rather than native thinking.

This difference must remain explicit in all reporting.

---

# 7. Reason-then-score prompt

The frozen Day-3 prompt is preserved as closely as possible.

The original Day-3 semantic content is:

```text
You are evaluating exactly one criterion for one candidate response.
Use only the stated criterion. Do not reward or penalize the response for properties outside that criterion.
Criterion: {rubric_text}
Candidate response: {candidate_response}
Return A if the criterion is met.
Return B if the criterion is not met.
```

The new reason-then-score template adds only a fixed reasoning scaffold:

```text
You are evaluating exactly one criterion for one candidate response.
Use only the stated criterion. Do not reward or penalize the response for properties outside that criterion.
Criterion: {rubric_text}
Candidate response: {candidate_response}

Return A if the criterion is met.
Return B if the criterion is not met.

Before giving the final label, reason about whether the stated criterion is met.
Do not give the final A or B label until the reasoning is complete.
When the reasoning is complete, write exactly:
FINAL:
and then give the label.
```

The precise frozen prompt bytes and chat-template rendering must be hashed before inference.

The A/B semantic mapping remains unchanged:

* A = criterion met,
* B = criterion not met.

No label swap is performed.

---

# 8. Final-answer marker

The planned marker is:

```text
FINAL:
```

Stage-R generation stops immediately after the complete marker has been generated.

The marker may tokenize to one or multiple tokens.

A custom sequence-aware stopping rule must detect the complete contextual marker token sequence.

The implementation must stop before the next generated token, so the model does not generate the final A/B label used for primary scoring.

## Pre-freeze validation

Before preregistration freeze, the marker may be validated using synthetic prompts unrelated to FormalCRRC.

If `FINAL:` cannot be reliably identified or stopped on under the pinned tokenizer/generation implementation, the experiment must not proceed until a replacement marker is selected and documented **before freeze**.

No FormalCRRC study-row output may be used to choose the marker.

The final marker and token sequence are immutable after freeze.

---

# 9. Stage R — elicited reasoning generation

For every study row:

1. format the reason-then-score prompt;
2. apply the exact frozen Qwen2.5 chat template;
3. generate autoregressively;
4. stop immediately after the complete `FINAL:` marker;
5. retain the full generated token sequence through the marker.

### Generation configuration

To align the reasoning-generation stage with the previously completed Qwen3 reasoning-anchor protocol:

* `do_sample = true`,
* temperature = `0.6`,
* top-p = `0.95`,
* top-k = `20`,
* min-p = `0.0` where supported,
* reasoning samples per row = `1`,
* maximum generated tokens through marker = `8192`,
* base experiment seed = `42`.

These values are chosen for cross-study procedural comparability with the reasoning-anchor experiment; they are not claimed to be optimized for Qwen2.5.

No generation parameter may be changed after study inference begins.

---

## 9.1 Deterministic per-row seed

For artifact \(x\) and strictness index \(s\):

$$
seed_{x,s}
=
\operatorname{int64}
\left(
SHA256(
42\;|\;\text{artifact\_id}\;|\;s
)
\right)
\bmod (2^{31}-1).
$$

Use the same stable seed-construction convention as the reasoning-anchor study if available.

Do not use Python's built-in `hash()`.

The implementation must prove through tests that the same row always receives the same seed.

---

# 10. Stage D — final A/B decision scoring

After Stage R stops at the complete `FINAL:` marker, do not permit free-form generation of the primary final label.

The exact Stage-D context is:

1. the fully formatted user/chat prompt,
2. the exact generated reasoning token sequence,
3. the exact `FINAL:` marker already generated.

The primary candidate continuations are:

```text
 A
```

and

```text
 B
```

with the leading ASCII space included.

The exact contextual tokenization of both candidates must be recorded before preregistration freeze.

Do not tokenize A or B in isolation.

If either candidate requires multiple tokens, compute full sequence log-likelihood:

$$
L_A
=
\sum_k
\log P(a_k\mid c,a_{<k}),
$$

$$
L_B
=
\sum_k
\log P(b_k\mid c,b_{<k}),
$$

where \(c\) is the exact reasoning context through `FINAL:`.

Define:

$$
M=L_A-L_B.
$$

Then:

* \(M>0\): criterion met,
* \(M<0\): criterion not met,
* \(M=0\): criterion met.

No EOS probability is included.

No length normalization is used.

---

# 11. Pre-freeze mechanical preflight

Before preregistration freeze, using only synthetic non-FormalCRRC prompts, verify:

1. exact model revision;
2. exact tokenizer revision;
3. chat-template hash;
4. contextual token sequence for `FINAL:`;
5. sequence-aware stopping works;
6. generation stops before an A/B token after the marker;
7. contextual tokenization of ` A`;
8. contextual tokenization of ` B`;
9. full-sequence candidate likelihood calculation;
10. generation parameters;
11. row-seed function.

Synthetic preflight must not inspect any Day-3 study result.

---

# 12. Reasoning completion and truncation

A row is considered complete if the full `FINAL:` marker occurs within the 8,192-token generation budget.

If the budget is exhausted first:

* `reasoning_truncated = true`,
* do not fabricate the marker,
* do not compute the primary Stage-D margin,
* retain all generated tokens and the stop reason.

No row may be rerun with a larger token budget.

### Completeness warning

If more than:

$$
1\%
$$

of the 2,916 study rows fail to reach `FINAL:`, mark:

`REASONING_COMPLETENESS_WARNING`.

Do not alter the budget after observing this.

A curve with any missing threshold score is excluded mechanically from TCE/TRR curve analysis and the exclusion count is reported.

---

# 13. Primary outcome 1 — nontrivial TRR

The principal outcome is boundary reachability under reason-then-score.

For nontrivial true boundaries:

$$
j_x^*\in\{1,\ldots,8\},
$$

define:

$$
TR_x^R
=
\mathbf 1
\left[
M_x^R(j_x^*)
<
\min_{t<j_x^*}M_x^R(t)
\right].
$$

Use strict inequality.

Then:

$$
TRR_R
=
\frac{1}{N}
\sum_xTR_x^R.
$$

Report:

$$
U_R=1-TRR_R
$$

as the reason-then-score unreachable fraction.

The frozen immediate-readout comparator is:

$$
TRR_O=0.4792,
$$

$$
U_O=0.5208.
$$

These frozen values must be re-read from machine artifacts rather than manually overwritten.

---

# 14. Primary outcome 2 — paired change in reachability

For every complete nontrivial curve:

$$
\Delta TR_x
=
TR_x^R-TR_x^O.
$$

Report:

$$
\Delta TRR
=
TRR_R-TRR_O.
$$

Use a paired artifact-level bootstrap preserving the original/reasoning pair.

The principal protocol-effect question is whether the reason-then-score procedure changes the probability that the true formal boundary is present in the same model's margin ordering.

---

# 15. Primary outcome 3 — unreachable-failure type

For each reason-then-score curve with:

$$
TR_x^R=0,
$$

classify why the true boundary fails the strict prefix-record condition.

At true crossing \(j^*\), compare \(M(j^*)\) with all earlier margins.

### Strict inversion present

$$
\exists t<j^*:
M(t)<M(j^*).
$$

This means an earlier threshold is scored strictly below the true-boundary margin.

### Tie-only blockage

No strict inversion exists, but:

$$
\exists t<j^*:
M(t)=M(j^*).
$$

Thus the boundary fails only because strict record-low reachability excludes equality.

If both a tie and a strict inversion occur, classify as:

**strict inversion present**

because an actual ordering inversion is sufficient to block reachability independently of the tie.

Report:

* count and percentage unreachable,
* strict-inversion-present count,
* tie-only count.

Apply the same classification read-only to frozen original Qwen2.5 Day-3 margins for contextual comparison.

This is a prespecified diagnostic designed to distinguish resolution/flatness failures from strict ordering inversions.

---

# 16. Secondary outcome — TCE

For complete curves:

$$
\hat j_x
=
\min\{s:M_x^R(s)<0\},
$$

or:

$$
\hat j_x=9
$$

if no margin is negative.

Then:

$$
TCE_x^R
=
|\hat j_x-j_x^*|.
$$

Report:

* reason-then-score mean TCE,
* frozen immediate-readout mean TCE,
* paired mean

  $$
  \Delta TCE=TCE_R-TCE_O.
  $$

TCE is secondary because changing the scoring context may induce location shifts even when margin ordering is preserved.

The within-model comparison is nevertheless more interpretable than the previous cross-model anchor TCE comparison.

---

# 17. Additional secondary metrics

Compute:

1. row-level accuracy;
2. Brier score using

   $$
   P(\text{met})=\sigma(M);
   $$
3. practical MVR at \(\delta=0.05\);
4. zero-tolerance MVR;
5. MVM;
6. mean reachable count \(RC\);
7. prefix-record rate;
8. \(RC=10\) frequency;
9. family-specific TCE;
10. family-specific TRR;
11. family-specific unreachable fraction;
12. family-specific accuracy;
13. family-specific Brier score;
14. reasoning-token count distribution;
15. reasoning-token count versus TCE;
16. reasoning-token count versus TR.

Reasoning-length associations are descriptive only and will not be interpreted causally.

---

# 18. Structural reference baselines

Report for context:

| Strategy                |             TCE |          Accuracy |
| ----------------------- | --------------: | ----------------: |
| Fixed center / no input |          2.2222 |          75.3086% |
| Uniform random crossing | 2.9630 expected | 67.0782% expected |
| Always met / majority   |          4.0000 |          55.5556% |
| Always not met          |          5.0000 |          44.4444% |

These are exact structural references.

They are not competitive learned baselines.

---

# 19. Contextual comparison with the Qwen3 anchor

Report, separately and descriptively:

### Qwen2.5 immediate Day 3

* immediate next-token scoring,
* TCE = 3.373,
* nontrivial TRR = 0.4792,
* unreachable = 52.08%.

### Qwen2.5 reason-then-score

* elicited reasoning,
* results from this experiment.

### Qwen3 reasoning anchor

* native reason-then-score,
* TCE = 0.401,
* TRR = 1.0,
* unreachable = 0%.

The Qwen2.5 immediate-versus-reasoning comparison is the primary within-model ablation.

The Qwen3 comparison is external context only.

Do not perform a pooled significance test across these model/protocol regimes.

---

# 20. Statistical analysis

Use:

* 5,000 bootstrap replicates,
* deterministic bootstrap seed 42,
* artifact-curve clustering.

All nine threshold rows of an artifact must stay together.

For paired original-vs-reasoning analyses, resample the paired artifact as one unit.

Report 95% percentile intervals for:

* reason-then-score TCE,
* \(\Delta TCE\),
* reason-then-score TRR,
* \(\Delta TRR\),
* unreachable fraction,
* accuracy,
* Brier,
* mean RC.

For family analyses, preserve family structure.

Do not bootstrap the 2,916 rows independently.

---

## 20.1 Zero-failure reachability reporting

If:

$$
U_R=0
$$

on all 288 nontrivial curves, do **not** present a bootstrap interval of `[0,0]` as if it quantified population uncertainty.

Instead report:

* observed:

  $$
  0/288
  $$

  unreachable,
* exact two-sided 95% Clopper-Pearson interval for the unreachable probability,
* especially its upper bound.

The bootstrap result may be retained in machine artifacts for preregistration completeness, but manuscript-facing reporting should use the exact binomial bound for a zero-count result.

---

# 21. Prespecified interpretation

No binary "success" criterion is used.

Interpret the paired \(\Delta TRR\) as follows.

### Clear protocol-associated increase

If the 95% CI for:

$$
\Delta TRR
$$

lies entirely above zero:

> Elicited reason-then-score increases true-boundary reachability for the same Qwen2.5 judge.

### Clear protocol-associated decrease

If the 95% CI lies entirely below zero:

> Elicited reason-then-score decreases true-boundary reachability for the same Qwen2.5 judge.

### Uncertain directional change

If the interval includes zero:

> The experiment does not establish a directional protocol-associated change in true-boundary reachability.

Effect magnitude must always be reported.

---

## 21.1 Interpretation relative to the anchor confound

If Qwen2.5 reason-then-score approaches the Qwen3 anchor's reachability:

> The previous panel-versus-anchor difference cannot be attributed primarily to model capability alone; the inference/readout protocol materially contributes.

If Qwen2.5 remains close to its original Day-3 reachability:

> Applying elicited reason-then-score to Qwen2.5 does not reproduce the Qwen3 anchor's complete reachability, strengthening the interpretation that model capability/training differences contribute materially to the anchor result.

If Qwen2.5 improves substantially but remains below the anchor:

> Both protocol and model-regime differences are consistent with the observed panel-versus-anchor contrast.

These are descriptive causal constraints, not identification of a single causal mechanism.

---

# 22. Important limitation

This study does **not** isolate "reasoning" as a single factor.

The protocol intervention jointly changes:

* the evaluation instruction,
* generation of an intermediate trace,
* the position at which A/B is scored,
* and the immediate-next-token versus conditional sequence-likelihood readout.

Therefore the supported causal statement is:

**effect of the reason-then-score protocol**

not:

**effect of reasoning itself**.

---

# 23. Prompt-related limitation

The reason-then-score prompt necessarily differs from the frozen Day-3 prompt because explicit reasoning must be elicited.

This is the intended protocol intervention.

The experiment is not a general prompt-robustness study.

No alternative reasoning prompts will be tested.

No prompt optimization will occur.

---

# 24. Reasoning-sample limitation

Stage R is stochastic.

Exactly one preregistered reasoning trace is generated for each row.

The study does not estimate within-prompt reasoning-sample variance.

The same deterministic row seed allows exact reproduction of the sampled trace under the frozen software/hardware implementation to the extent supported by the generation stack.

All raw reasoning token sequences must be retained.

---

# 25. Reasoning marker failure

If a row generates a final A/B label before `FINAL:`, record a QC flag:

`premature_label_observed`.

Do not exclude the row solely for this reason if the required marker is later reached and a valid Stage-D margin can be computed.

If the model never reaches the marker within budget, apply the truncation rule.

Report the rate of premature-label flags descriptively.

---

# 26. Integrity

Before first study inference:

1. verify the current git branch;
2. record git commit;
3. fingerprint all previously protected scientific artifacts;
4. verify Days 1-3 integrity;
5. verify baseline-audit integrity;
6. verify step-proposition integrity;
7. verify label-swap integrity;
8. verify reasoning-anchor integrity;
9. record the frozen Day-3 dataset hash;
10. record exact Qwen2.5 model/tokenizer revisions;
11. freeze the reasoning prompt and marker;
12. freeze candidate continuations;
13. pass all pre-inference tests;
14. freeze this preregistration;
15. record UTC timestamp;
16. record preregistration SHA-256;
17. confirm amendments = 0.

After the study:

* recompute the protected-file fingerprint;
* all prior scientific artifacts must remain byte-identical;
* new experiment files must be additive.

---

# 27. Required tests before study inference

Implement at minimum:

### A. Qwen2.5 revision continuity

Verify exact Day-3 model and tokenizer revisions.

### B. Frozen dataset identity

Verify exact Day-3 TEST dataset.

### C. Prompt construction

Verify only the preregistered reasoning scaffold differs from the frozen Day-3 semantic prompt.

### D. Marker tokenization

Verify contextual `FINAL:` token sequence.

### E. Sequence stopping

Generation halts immediately after the full marker.

### F. Candidate contextual tokenization

Verify exact token sequences for:

` A`

and

` B`.

### G. Multi-token likelihood

Full candidate sequence likelihood works.

### H. Margin sign

Higher A likelihood gives positive canonical margin.

### I. Deterministic seed

Stable row-to-seed mapping.

### J. Crossing convention

First \(M<0\), otherwise 9.

### K. Tie convention

\(M=0\) counts as met.

### L. Strict prefix record

Day-3 reachability rule reproduced exactly.

### M. Tie-vs-inversion classification

Synthetic examples distinguish tie-only and strict-inversion failure.

### N. Frozen original Qwen2.5 recovery

Original Qwen2.5 Day-3 TCE/TRR reproduce exactly from frozen scores.

### O. Frozen-file integrity

No prior protected file changed.

Run the complete existing test suite plus all new tests before inference.

---

# 28. Infrastructure failures and retries

Cluster instructions must come from repository documentation.

Do not invent paths or SLURM settings.

A failed infrastructure job may be retried only with identical:

* model revision,
* tokenizer revision,
* row,
* prompt,
* row seed,
* generation parameters,
* marker,
* scoring procedure.

Log every retry.

No output may be regenerated merely because its score is surprising.

---

# 29. Required raw outputs

For every reason-then-score row retain:

* experiment ID,
* artifact ID,
* family,
* latent level,
* strictness index,
* true label,
* true crossing,
* model repo,
* model revision,
* tokenizer revision,
* prompt hash,
* formatted prompt hash,
* generation seed,
* reasoning token IDs,
* decoded reasoning text,
* reasoning token count,
* marker reached,
* truncation flag,
* premature-label flag,
* stop reason,
* candidate-A token IDs,
* candidate-B token IDs,
* \(L_A\),
* \(L_B\),
* canonical margin,
* semantic decision,
* runtime,
* status/errors.

All original comparator data remain read-only.

---

# 30. Required result tables

## Table A — within-Qwen2.5 protocol ablation

Columns:

* Protocol
* Complete curves
* Mean TCE
* TCE 95% CI
* Nontrivial TRR
* Unreachable %
* Accuracy
* Brier
* Mean RC

Rows:

* Qwen2.5 immediate Day 3
* Qwen2.5 reason-then-score

## Table B — paired protocol effects

Columns:

* Outcome
* Original
* Reason-then-score
* Paired difference
* 95% CI

Include:

* TCE
* TRR
* accuracy
* Brier

## Table C — unreachable failure mechanism

Rows:

* Immediate Day 3
* Reason-then-score

Columns:

* Unreachable count
* Strict inversion present
* Tie-only
* Other/error if any

## Table D — model/protocol context

Rows:

* Qwen2.5 immediate
* Qwen2.5 reason-then-score
* Qwen3 native reason-then-score

Include:

* inference protocol,
* TCE,
* TRR,
* unreachable fraction.

Clearly label cross-model comparisons descriptive.

## Table E — family analysis

For each family report reason-then-score:

* TCE,
* TRR,
* unreachable,
* accuracy,
* Brier,
* mean reasoning tokens.

---

# 31. Required result document

Create:

`RESULTS_REASON_THEN_SCORE_ABLATION.md`

or the repository-equivalent naming convention.

It must contain:

1. status;
2. preregistration timestamp;
3. preregistration SHA;
4. exact model/tokenizer revision;
5. branch/commit;
6. motivation;
7. protocol contrast;
8. exact prompt difference;
9. marker and stopping procedure;
10. candidate scoring procedure;
11. dataset verification;
12. completeness;
13. primary results;
14. paired \(\Delta TRR\);
15. TCE;
16. tie-vs-inversion decomposition;
17. secondary metrics;
18. family analysis;
19. reasoning-length diagnostics;
20. contextual Qwen3 comparison;
21. limitations;
22. deviations/retries;
23. integrity result;
24. manuscript-facing interpretation.

Every numeric value must be generated from machine-readable artifacts.

---

# 32. Machine-readable output location

Create additive outputs under:

`artifacts/reason_then_score_ablation/`

and figures, if needed, under:

`figures/reason_then_score_ablation/`.

Do not alter existing Day-3 or reasoning-anchor artifacts.

---

# 33. Claims that must not be made

This experiment does not establish that:

* chain-of-thought itself causes improved reachability;
* Qwen2.5 represents all non-reasoning models;
* Qwen3's advantage is solely due to scale;
* Qwen3's advantage is solely due to reasoning training;
* prompt wording is irrelevant;
* one reasoning trace represents all possible traces;
* sequence-likelihood and immediate-logit margins have interchangeable absolute scales;
* a lower TCE alone establishes better ordering;
* TRR=1 proves a scalar internal representation.

The permitted central claim is narrower:

> This ablation measures how the FormalCRRC response geometry of the same Qwen2.5 judge changes when immediate A/B readout is replaced by an elicited reason-then-score protocol.

---

# 34. Stopping rule

The experiment ends after:

* all 2,916 rows have been attempted,
* all preregistered analyses are complete,
* integrity is reverified,
* and the result report is generated.

Do not add:

* Llama,
* another Qwen model,
* another reasoning prompt,
* another token budget,
* prompt variants,
* label swaps,
* repeated reasoning samples

after seeing the result.

Those require separate prospective studies.

---

# PREREGISTRATION STATUS

**FROZEN 2026-09-09T21:38:03+00:00 — amendments 0.**

**Frozen before the first FormalCRRC reason-then-score study row. Only synthetic non-study prompts were used before this point.**
