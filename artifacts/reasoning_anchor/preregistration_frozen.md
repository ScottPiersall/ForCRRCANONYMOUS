# FormalCRRC Reasoning-Capable External Anchor Study

**Study type:** Post-study preregistered external-validity / reasoning-capable anchor experiment
**Relationship to prior work:** Follow-up to FormalCRRC Days 1-3 and the label-swap robustness experiment
**Human subjects:** None
**New model inference:** Yes
**Dataset:** Frozen FormalCRRC Day-3 TEST partition, seed 45
**Anchor model:** `Qwen/Qwen3-30B-A3B-Thinking-2507`
**Model class:** Modern reasoning-capable open-weight external anchor
**Quantization:** None; official BF16 weights unless a preregistration-stopping incompatibility is discovered before freeze
**Preregistration freeze timestamp:** `2026-09-09T18:54:49+00:00`
**Preregistration SHA-256:** `b32c547bf8ba994937e144bed8fec0673af7553c0c7c3aca6838239f3ba6582b`
**Exact model revision:** `144afc2f379b542fdd4e85a1fcd5e1f79112d95d`
**Exact tokenizer revision:** `144afc2f379b542fdd4e85a1fcd5e1f79112d95d`
**Amendments at freeze:** 0

---

## 1. Motivation

The original FormalCRRC experiments evaluate four open-weight instruction-tuned judges ranging from approximately 7B to 14B parameters using direct next-token A/B scoring. Those experiments establish reproducible threshold-localization error, distinguish location from ordering failures, characterize boundary reachability, and show that the principal phenomenon survives reversal of the A/B response-token assignment.

A remaining external-validity question is whether these findings are primarily a consequence of the capabilities or inference style of relatively small non-reasoning instruction models.

This study therefore evaluates one substantially different judge: `Qwen/Qwen3-30B-A3B-Thinking-2507`, an open-weight reasoning-capable model that supports thinking mode by design.

The study is intended as an **external anchor**, not as a fifth member of the original Day-3 panel. Its inference procedure necessarily differs because the model reasons before producing a final judgment.

The principal question is:

> Does a modern reasoning-capable judge also exhibit FormalCRRC boundary-localization and boundary-reachability failures when allowed to reason before its A/B decision?

---

## 2. Scientific hypotheses

This study is descriptive and falsification-oriented rather than a superiority benchmark.

### H1 — Boundary localization

The reasoning-capable anchor may reduce raw Threshold Crossing Error relative to the original four-judge Day-3 panel.

The study does not assume that it will.

### H2 — Boundary reachability

If the reasoning-capable anchor still yields nontrivial true boundaries that are not strict prefix record lows of its canonical margin curves, then reasoning capability does not eliminate the FormalCRRC ordering failure.

For a nontrivial true boundary \(j_x^*\),

$$
TR_x =
\mathbf 1
\left[
M_x(j_x^*) <
\min_{t<j_x^*}M_x(t)
\right].
$$

If \(TR_x=0\), the required formal boundary is absent from the observed score ordering under the established FormalCRRC reachability definition.

### H3 — Scalar-thresholder diagnostic

Under the previously established strict scalar latent-estimation/readout model, all candidate crossings must be reachable and

$$
RC_x=10.
$$

The frequency of \(RC_x=10\) will therefore be reported as a post-study theoretical diagnostic, not as a primary endpoint.

---

## 3. Relationship to prior FormalCRRC experiments

This experiment does not alter or rerun:

* Day 1,
* Day 2,
* Day 3,
* the baseline audit,
* the step-function proposition audit,
* or the label-swap study.

All prior scientific artifacts remain frozen.

The Day-3 TEST partition provides the evaluation dataset so that the reasoning-capable anchor is evaluated on exactly the same artifacts, predicate families, formal thresholds, and true boundaries.

The original Day-3 panel provides contextual comparison values only.

Because the inference mechanism differs, this anchor must not be inserted into the original Day-3 table in a way that implies identical inference procedures.

---

## 4. Dataset

Use the complete frozen Day-3 TEST partition.

Expected design:

* seed: 45,
* predicate families: 3,
* artifacts: 324,
* strictness levels per artifact: 9,
* rows: 2,916,
* nontrivial true-boundary curves: 288.

No subsampling is permitted.

Each artifact must be byte-identical to its frozen Day-3 counterpart.

Formal truth and true crossings must be independently recomputed from primitive dataset fields.

---

## 5. Predicate canonicalization

All three FormalCRRC families use the previously verified canonical form

$$
Y_x(s)=\mathbf 1[z_x\ge s],
\qquad s\in\{0,\ldots,8\}.
$$

### Coverage

Stored latent:

$$
m.
$$

Formal predicate:

$$
Y=1[m\ge s].
$$

Canonical compliance:

$$
z=m.
$$

### Maximum violations

Stored latent:

$$
v.
$$

Formal predicate:

$$
Y=1[v\le8-s].
$$

Canonical compliance:

$$
z=8-v.
$$

### Numeric tolerance

Stored latent:

$$
e.
$$

Formal predicate:

$$
Y=1[e\le8-s].
$$

Canonical compliance:

$$
z=8-e.
$$

For all three,

$$
j_x^*=z_x+1
$$

under the established FormalCRRC crossing convention.

The implementation must verify these identities against all frozen Day-3 cells before inference.

---

## 6. Anchor model

The planned model is:

`Qwen/Qwen3-30B-A3B-Thinking-2507`

The exact Hugging Face revision and tokenizer revision must be resolved and pinned before preregistration freeze.

The official non-quantized model must be used.

### Required configuration

* official model weights,
* BF16,
* no quantization,
* no LoRA,
* no fine-tuning,
* no adapters,
* no system-level prompt optimization,
* native Qwen chat template,
* thinking mode enabled by the model's native template,
* no model substitution.

If the exact model cannot be run with the available cluster hardware using the repository's documented environment, the experiment stops rather than silently using a quantized or smaller replacement.

A quantized variant would constitute a different preregistration.

---

## 7. Why direct Day-3 next-token scoring cannot be reused unchanged

The original Day-3 models were scored at the immediate response boundary:

$$
M=S_A-S_B.
$$

The reasoning anchor differs because its native assistant response begins with a reasoning phase before its final answer.

Scoring A and B at the first token after the user prompt would therefore measure the beginning of reasoning rather than the final judgment.

This study consequently uses a **reason-then-score** protocol.

---

# 8. Primary reason-then-score protocol

For each of the 2,916 artifact-threshold prompts, inference consists of two stages.

## 8.1 Stage R — reasoning generation

Use the exact frozen Day-3 user-message content:

> You are evaluating exactly one criterion for one candidate response.
> Use only the stated criterion. Do not reward or penalize the response for properties outside that criterion.
> Criterion: {rubric_text}
> Candidate response: {candidate_response}
> Return A if the criterion is met.
> Return B if the criterion is not met.

No criterion wording or artifact text may change.

The prompt is passed through the anchor model's native chat template with its normal thinking behavior.

Generate reasoning tokens until the first native end-of-thinking delimiter:

`</think>`

or until the preregistered reasoning-token budget is exhausted.

### Reasoning generation parameters

Primary generation uses the model-author recommended sampling family:

* temperature: `0.6`,
* top-p: `0.95`,
* top-k: `20`,
* min-p: `0`,
* number of reasoning samples per row: `1`,
* reasoning-token budget: `8192`,
* deterministic per-row random seed: defined below.

The 8,192-token budget is used to bound computational cost while permitting substantial reasoning. The rate of reasoning traces that fail to terminate before the budget must be reported.

No outcome-dependent increase in reasoning budget is permitted.

### Deterministic seed rule

For row \(r\), construct a seed deterministically from immutable identifiers:

$$
seed_r =
f(
\text{experiment seed},
\text{artifact ID},
\text{strictness index}
)
$$

using a documented stable hash-to-integer procedure.

The experiment-level base seed will be:

$$
42.
$$

The exact hash function and integer reduction rule must be implemented and frozen before inference.

The same row must always receive the same seed.

### Reasoning output retained

For every row retain:

* generated token IDs,
* generated decoded reasoning text,
* token count,
* whether `</think>` was reached,
* stop reason,
* random seed,
* generation parameters,
* model revision,
* tokenizer revision.

---

## 8.2 Stage D — final decision scoring

After Stage R reaches `</think>`, no free-form final answer is generated for the primary metric.

Instead, construct a standardized decision context consisting of:

1. the exact original formatted user prompt,
2. the exact generated reasoning tokens,
3. the native `</think>` delimiter,
4. a fixed post-thinking separator determined and frozen before study inference.

The post-thinking separator must be resolved using only:

* the official Qwen tokenizer/chat-template implementation,
* model documentation,
* and, if required, synthetic non-FormalCRRC preflight prompts.

It must not be selected after inspecting FormalCRRC outcomes.

Record before freeze:

`POST_THINK_SEPARATOR = "\n\n"` (bytes `0x0a 0x0a`, token ids `[271]`)

Expected candidate is the model-native newline separator; the exact byte sequence must be documented.

At this standardized boundary, compute conditional likelihoods for two candidate final continuations:

$$
C_A = \text{``A''},
\qquad
C_B = \text{``B''}.
$$

Candidate scoring must use the tokenizer in **prompt-continuation context**, not isolated string tokenization.

If A or B requires more than one token in this context, use full sequence log-likelihood:

$$
L_A
=
\log P(C_A\mid \text{reasoning context}),
$$

$$
L_B
=
\log P(C_B\mid \text{reasoning context}).
$$

The canonical margin is

$$
M=L_A-L_B.
$$

Thus:

* \(M>0\): criterion met,
* \(M<0\): criterion not met,
* \(M=0\): criterion met under the existing FormalCRRC tie convention.

Do not require A and B to be single tokens.

Do not use isolated tokenizer encodings.

---

## 9. Pre-freeze scoring validation

Before freezing the preregistration, Claude must determine and record:

1. exact model revision;
2. exact tokenizer revision;
3. `</think>` token ID or exact token sequence;
4. exact post-thinking separator;
5. contextual tokenization of A;
6. contextual tokenization of B;
7. exact candidate sequence-scoring implementation;
8. reasoning generation parameters;
9. exact deterministic row-seed function.

This validation may use synthetic prompts unrelated to FormalCRRC.

It may not inspect reasoning-anchor outputs on the Day-3 study dataset.

If no unambiguous standardized final-answer boundary can be established, the study must stop before FormalCRRC inference.

---

## 10. Handling reasoning-budget exhaustion

If Stage R reaches 8,192 generated tokens without producing the native end-of-thinking delimiter:

* mark the row `reasoning_truncated = true`,
* do not fabricate `</think>`,
* do not use the row in the primary complete-case reason-then-score analysis.

The number and distribution of truncated rows must be reported.

### Completeness rule

If more than 1% of the 2,916 rows are truncated, the primary study is marked:

`REASONING_COMPLETENESS_WARNING`

and results must be reported with that qualification.

The budget must not be increased after observing which rows truncate.

A future higher-budget analysis would be a separate post-hoc sensitivity study.

If any artifact curve has one or more missing threshold scores, that curve is excluded from curve-level TCE/TRR calculations and the exclusion is reported mechanically.

No exclusion may depend on model performance.

---

## 11. Primary outcomes

### 11.1 Threshold Crossing Error

For complete curves,

$$
\hat j_x
=
\min\{s:M_x(s)<0\},
$$

or

$$
\hat j_x=9
$$

if no strictness level has negative margin.

Then

$$
TCE_x
=
|\hat j_x-j_x^*|.
$$

Report mean TCE and 95% artifact-cluster bootstrap interval.

---

### 11.2 Nontrivial Translation Reachability Rate

For true boundaries

$$
j_x^*\in\{1,\ldots,8\},
$$

define

$$
TR_x
=
\mathbf 1
\left[
M_x(j_x^*)<
\min_{t<j_x^*}M_x(t)
\right].
$$

Report:

$$
TRR
=
\frac{1}{N}
\sum_x TR_x,
$$

and

$$
1-TRR.
$$

The unreachable fraction \(1-TRR\) is the principal ordering-based endpoint.

Strict inequality and all Day-3 tie conventions remain unchanged.

---

## 12. Secondary outcomes

Prespecified secondary outcomes are:

1. row-level classification accuracy;
2. Brier score using

   $$
   P(\text{met})=\sigma(M);
   $$
3. practical monotonicity violation rate;
4. zero-tolerance monotonicity violation rate;
5. mean monotonicity violation magnitude;
6. reachable count \(RC_x\);
7. prefix-record rate;
8. \(RC_x=10\) frequency;
9. family-stratified TCE;
10. family-stratified TRR;
11. family-stratified accuracy;
12. family-stratified Brier score;
13. reasoning-token count distribution;
14. reasoning truncation rate;
15. relationship between reasoning length and TCE;
16. relationship between reasoning length and TR status.

Items 15-16 are descriptive associations only and will not be interpreted causally.

---

## 13. Structural reference baselines

Report the already established exact FormalCRRC reference values:

| Reference strategy          |                       TCE |                       Accuracy |
| --------------------------- | ------------------------: | -----------------------------: |
| Fixed center / no input     |           \(20/9=2.2222\) |            \(61/81=75.3086\%\) |
| Uniform random crossing     | \(80/27=2.9630\) expected | \(163/243=67.0782\%\) expected |
| Always met / majority class |                \(4.0000\) |                  \(55.5556\%\) |
| Always not met              |                \(5.0000\) |                  \(44.4444\%\) |

These are post-study structural references, not learned competitors.

---

## 14. Contextual comparisons with original Day-3 judges

The frozen Day-3 raw TCE values are:

* Qwen2.5-14B: 3.373,
* Llama-3.1-8B: 4.904,
* Mistral-7B: 2.938,
* Gemma-2-9B: 3.361.

The frozen nontrivial unreachable fractions are:

* Qwen2.5-14B: 52.08%,
* Llama-3.1-8B: 72.92%,
* Mistral-7B: 66.67%,
* Gemma-2-9B: 39.58%.

These comparisons are descriptive because the anchor uses a reasoning-aware scoring procedure rather than the original immediate next-token procedure.

Do not perform a pooled hypothesis test treating all five models as exchangeable samples.

---

## 15. Prespecified interpretation

### Outcome A — ordering failure persists within the original range

If the reasoning anchor's nontrivial unreachable fraction lies within the original Day-3 range:

$$
39.58\%-72.92\%,
$$

the result may be described as:

> The reasoning-capable anchor exhibits boundary unreachability of the same order as the original FormalCRRC panel.

### Outcome B — improved but nonzero ordering failure

If its unreachable fraction is below 39.58% but above 0:

> The reasoning-capable anchor reduces ordering failure relative to every original Day-3 judge, but does not eliminate the FormalCRRC phenomenon.

The exact percentage must be reported.

### Outcome C — no ordering failures

If

$$
TRR=1
$$

for all nontrivial complete curves:

> The reasoning-capable anchor does not reproduce Day-3 boundary unreachability under this reason-then-score protocol.

This result must be reported even if it weakens the external-validity claim.

### TCE interpretation

TCE must be compared separately with:

* the fixed-center reference \(2.222\),
* random crossing \(2.963\),
* and the original Day-3 range \(2.938-4.904\).

Do not collapse TCE and TRR into one success/failure label.

---

## 16. Statistical analysis

Use 5,000 artifact-clustered bootstrap replicates.

Bootstrap unit:

$$
\text{artifact curve}.
$$

All nine strictness levels belonging to one artifact remain together.

Use deterministic bootstrap seed:

$$
42.
$$

Report 95% percentile intervals for:

* mean TCE,
* TRR,
* unreachable fraction,
* accuracy,
* Brier score,
* mean RC,
* principal family-stratified metrics.

For nontrivial TRR, bootstrap only complete nontrivial curves.

No threshold-row bootstrap is permitted.

---

## 17. Reasoning reproducibility analysis

Because Stage R uses sampling, the generated trace is part of the experimental measurement.

The primary study uses exactly one preregistered reasoning sample per row.

The study does not estimate within-prompt sampling variance.

All generated reasoning token sequences must therefore be retained so that Stage D can be reproduced exactly without regenerating reasoning.

The manuscript must state that:

> uncertainty intervals reflect variation across artifacts, not variation across alternative reasoning samples.

No second reasoning sample may be added after viewing outcomes as part of the preregistered study.

---

## 18. Prompt fidelity

The user-message content must be identical to the frozen Day-3 prompt except for changes that are mechanically required by the reasoning model's native chat template.

No additional rubric explanation may be added.

No chain-of-thought instruction such as "reason step by step" is added: the anchor model is already a thinking-only model and its native template initiates reasoning.

The final semantic task remains:

* A = criterion met,
* B = criterion not met.

This experiment does not use the label-swapped mapping.

---

## 19. Exact model and environment provenance

Before inference record:

* model repository,
* exact model commit/revision,
* tokenizer commit/revision,
* Transformers version,
* PyTorch version,
* CUDA version,
* GPU type,
* dtype,
* attention implementation,
* generation configuration,
* tokenizer/chat-template hash,
* scoring code hash,
* dataset hash,
* preregistration hash.

Cluster paths and job configuration must come from existing repository documentation.

Do not invent infrastructure settings.

---

## 20. Integrity requirements

Before first study inference:

1. fingerprint all frozen scientific artifacts;
2. verify prior integrity manifests;
3. verify the merged label-swap scientific record;
4. record Day-3 TEST dataset hash;
5. record exact model and tokenizer revisions;
6. finalize reason-then-score implementation;
7. pass synthetic scoring tests;
8. freeze this preregistration;
9. record UTC freeze timestamp;
10. record SHA-256;
11. confirm amendments = 0.

After inference:

* recompute the frozen-record fingerprint;
* all prior scientific files must remain byte-identical;
* all new files must be additive;
* raw reasoning tokens and decision scores must be retained.

---

## 21. Required pre-inference tests

At minimum implement tests for:

### A. Native thinking boundary

Confirm the anchor's generated reasoning can be identified using its native `</think>` delimiter.

### B. Contextual A/B scoring

Confirm candidate A and B likelihoods are computed in continuation context and not using isolated token IDs.

### C. Multi-token candidate support

Synthetic test proving full sequence likelihood works if a candidate tokenizes to multiple tokens.

### D. Margin sign

Construct synthetic logits verifying:

$$
L_A>L_B \Rightarrow M>0.
$$

### E. Crossing calculation

Verify existing Day-3 crossing convention.

### F. Prefix-record reachability

Verify strict Day-3 reachability convention.

### G. Ties

Verify equal margins do not count as strict prefix record lows.

### H. Dataset identity

Verify frozen Day-3 identity.

### I. Deterministic seed mapping

Verify the same row identifiers always yield the same generation seed.

### J. Frozen-file immutability

Verify previously protected scientific artifacts have unchanged hashes.

---

## 22. Infrastructure failure and retries

A cluster/job failure may be retried only with identical:

* model revision,
* tokenizer revision,
* prompt,
* artifact,
* strictness level,
* seed,
* sampling configuration,
* scoring configuration.

Retries must be logged.

A failed result may not be replaced because its output appears undesirable.

If the model cannot be loaded in official BF16 form, stop rather than substituting a quantized variant.

---

## 23. Deviations

Any methodological change after the first FormalCRRC reasoning-anchor inference constitutes a post-inference deviation.

It cannot be represented as preregistered.

No reasoning budget, generation parameter, answer boundary, candidate scoring rule, or metric may be changed after outcome inspection.

---

## 24. Claims that must not be made

This study does not establish:

* that Qwen3 represents all frontier judges;
* that it represents proprietary frontier APIs;
* that reasoning generally causes better or worse judge reliability;
* that chain-of-thought is causally responsible for any observed difference;
* that reasoning traces faithfully reveal hidden computation;
* that one sampled reasoning trace characterizes all possible reasoning paths;
* that a lower TCE implies correct score ordering;
* that higher TRR implies a scalar latent representation;
* that failure of this anchor invalidates FormalCRRC;
* that success of this anchor proves FormalCRRC generalizes to semantic rubrics.

The correct term is:

**reasoning-capable open-weight external anchor**

rather than "frontier judge."

---

## 25. Planned manuscript reporting

If completed successfully, report the anchor in a separate subsection such as:

**Reasoning-Capable External Anchor**

The main text should include:

1. why immediate next-token scoring is inappropriate;
2. reason-then-score protocol;
3. model/revision;
4. reasoning budget and generation parameters;
5. completion/truncation rate;
6. TCE;
7. structural-baseline comparisons;
8. nontrivial TRR and unreachable fraction;
9. RC and \(RC=10\);
10. comparison with original Day-3 ranges;
11. family-stratified results;
12. limitations of one sampled reasoning trace.

Do not rewrite the original Day-3 experiment as if this judge had been preregistered there.

---

## 26. Required result artifacts

Create additive outputs under a dedicated location such as:

`artifacts/reasoning_anchor/`

At minimum retain:

* frozen preregistration,
* preregistration hash record,
* model provenance,
* environment provenance,
* prompt/template hashes,
* generation config,
* raw row-level reasoning metadata,
* raw reasoning token IDs,
* decoded reasoning traces,
* candidate A/B log-likelihoods,
* canonical margins,
* curve-level metrics,
* bootstrap outputs,
* family summaries,
* integrity manifest,
* result document,
* machine-readable summary JSON.

Figures, if produced, belong under:

`figures/reasoning_anchor/`

---

## 27. Stopping rule

The experiment ends after:

* all 2,916 planned rows have been attempted,
* all preregistered analyses are completed,
* integrity is reverified,
* and the result report is generated.

Do not add:

* a second reasoning model,
* another reasoning budget,
* alternative decoding,
* alternative answer prefix,
* prompt variants,
* or label swapping

based on the observed result.

Those would be separate experiments.

---

# PREREGISTRATION STATUS

**FROZEN 2026-09-09T18:54:49+00:00 — amendments 0.**

**Frozen before the first FormalCRRC reasoning-anchor evaluation. Only synthetic non-study prompts were used before this point.**
