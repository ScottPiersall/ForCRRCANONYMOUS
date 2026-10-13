# FormalCRRC Label-Swap Counterbalance Robustness Study

**Study type:** Post-study preregistered robustness experiment
**Relationship to prior work:** Follow-up to FormalCRRC ss 1-3
**Human subjects:** None
**New model inference:** Yes
**Dataset:** Frozen FormalCRRC s-3 TEST dataset, seed 45
**Models:** Same four models, exact revisions, inference configuration, tokenizer handling, and scoring procedure used in s 3
**Preregistration freeze timestamp:** 2026-09-09T03:29:27+00:00
**Preregistration SHA-256:** [FILL AUTOMATICALLY BEFORE FIRST INFERENCE]
**Amendments at freeze:** 0

## 1. Motivation

FormalCRRC evaluates LLM judges using the next-token margin between response labels A and B while programmatically varying rubric strictness. In ss 1-3, A denoted "criterion met" and B denoted "criterion not met." A possible alternative explanation for some observed response geometry is therefore label-token asymmetry: a model may intrinsically prefer one response token, or the mapping of semantic classes to A and B may interact with the prompt.

This study tests whether FormalCRRC's boundary-localization and boundary-reachability findings are robust to reversing the semantic meanings assigned to A and B while holding the evaluated artifacts, rubric thresholds, model revisions, scoring procedure, and all other prompt content fixed.

The study is intentionally paired with the frozen s-3 dataset. It is a robustness test, not an independent replication dataset.

## 2. Research question

Does reversing the mapping between semantic judgment and response token materially change:

1. semantic judge decisions,
2. threshold-crossing error,
3. boundary reachability,
4. or the ordering of canonical judge margins across rubric strictness?

The principal scientific question is:

> Do the FormalCRRC boundary-localization and reachability findings persist when "criterion met" and "criterion not met" are assigned to the opposite A/B response tokens?

## 3. Experimental intervention

### 3.1 Original s3 response instruction

The frozen s3 prompt ends with:

> Return A if the criterion is met.
> Return B if the criterion is not met.

For the original condition, the canonical margin is

$$
M_x^{O}(s)=S_A-S_B,
$$

where \(S_A\) and \(S_B\) are the next-token logits for A and B.

### 3.2 Label-swapped response instruction

The counterbalanced prompt will be byte-identical to the s-3 prompt except for reversing the semantic label assignment in the final response instructions:

> Return B if the criterion is met.
> Return A if the criterion is not met.

The canonical semantic margin will therefore be

$$
M_x^{S}(s)=S_B-S_A.
$$

Thus, in both conditions,

$$
M>0
$$

means greater model evidence that the criterion is met, and

$$
M<0
$$

means greater evidence that the criterion is not met.

The zero convention is unchanged from s 3: \(M=0\) counts as "criterion met," and the predicted first-fail crossing is the first strictness level satisfying \(M<0\); if none exists, the predicted crossing is 9.

### 3.3 Prompt-difference restriction

The experiment implementation must verify programmatically that the original and swapped prompt templates differ only in the A/B semantic assignments in the two final response-instruction lines.

No other wording, whitespace, rubric text, candidate response, ordering, system prompt, chat template, or formatting may be intentionally changed.

If the implementation cannot reproduce the exact s-3 prompt construction, inference must not begin.

## 4. Dataset

The complete frozen s-3 TEST dataset will be reused.

* Seed: 45
* Predicate families: 3
* Latent/boundary cells per family: 9
* Artifacts per family-by-latent cell: 12
* Total artifacts: 324
* Strictness levels per artifact: 9
* Prompt-response pairs per model: 2,916
* Models: 4
* Total new swapped-label forward evaluations:

$$
324\times9\times4=11,664.
$$

No artifact subsampling will be performed.

The original s-3 scores are frozen and will be read only. The same exact artifact IDs, candidate responses, rubric thresholds, family assignments, strictness indices, and formal labels must be used in the swapped condition.

## 5. Predicate canonicalization

The previously verified canonical form is:

$$
Y_x(s)=\mathbf 1[z_x\ge s], \qquad s\in\{0,\ldots,8\}.
$$

The three families map as follows:

| Family            | Stored latent | Implemented predicate | Canonical compliance \(z\) | Canonical threshold |
| ----------------- | ------------- | --------------------- | -------------------------- | ------------------- |
| Coverage          | \(m\)         | \(1[m\ge s]\)         | \(m\)                      | \(s\)               |
| Max violations    | \(v\)         | \(1[v\le8-s]\)        | \(8-v\)                    | \(s\)               |
| Numeric tolerance | \(e\)         | \(1[e\le8-s]\)        | \(8-e\)                    | \(s\)               |

For all three,

$$
j_x^*=z_x+1,
$$

with the established endpoint convention.

These definitions must be recomputed from frozen artifact fields rather than trusted solely from stored derived columns.

## 6. Models and inference

The exact four s-3 models will be used:

1. `Qwen/Qwen2.5-14B-Instruct` — revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, tokenizer revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`
2. `meta-llama/Llama-3.1-8B-Instruct` — revision `0e9e39f249a16976918f6564b8830bc894c89659`, tokenizer revision `0e9e39f249a16976918f6564b8830bc894c89659`
3. `mistralai/Mistral-7B-Instruct-v0.3` — revision `c170c708c41dac9275d15a8fff4eca08d52bab71`, tokenizer revision `c170c708c41dac9275d15a8fff4eca08d52bab71`
4. `google/gemma-2-9b-it` — revision `11c9b309abf73637e4b6f9a3fa1e92e615547819`, tokenizer revision `11c9b309abf73637e4b6f9a3fa1e92e615547819`

The exact revision identifiers, tokenizer revisions, BF16 configuration, chat-template handling, token IDs, next-token scoring procedure, and all other inference settings must be read from and matched to the frozen s-3 provenance.

No model substitution is allowed.

No quantization, fine-tuning, LoRA, prompt optimization, temperature manipulation, decoding change, or model adaptation is permitted.

The experiment scores next-token logits; no stochastic generation is required.

Before inference, the implementation must verify that the A and B response tokens are handled identically to s 3 for each tokenizer.

## 7. Experimental unit and pairing

The basic response-level observation is an artifact-threshold-model tuple.

The primary curve-level experimental unit is an artifact-model pair consisting of nine ordered strictness levels.

Every swapped curve is paired with the corresponding frozen s-3 original-label curve.

There are:

$$
324
$$

paired curves per model.

For nontrivial boundary-reachability analysis, true boundaries \(j_x^*\in\{1,\ldots,8\}\) are used, yielding:

$$
288
$$

paired nontrivial curves per model.

## 8. Primary outcomes

### 8.1 Semantic decision agreement

Canonical decisions are

$$
D^O_x(s)=\mathbf 1[M^O_x(s)\ge0],
$$

and

$$
D^S_x(s)=\mathbf 1[M^S_x(s)\ge0].
$$

Semantic agreement is

$$
AGR_J=
\frac{1}{N}
\sum_{x,s}
\mathbf 1[D^O_x(s)=D^S_x(s)].
$$

This measures whether reversing response-token identities changes the semantic binary decision after canonicalization.

### 8.2 Change in Threshold Crossing Error

For each curve,

$$
TCE_x=|\hat j_x-j_x^*|.
$$

The paired model-level change is

$$
\Delta TCE_J
=
\overline{TCE^{S}_x-TCE^{O}_x}.
$$

Positive values indicate worse localization after label swapping; negative values indicate better localization.

### 8.3 Change in nontrivial Translation Reachability Rate

Using the preregistered s-3 strict-prefix-record criterion,

$$
TR_x=
\mathbf 1
\left[
M_x(j_x^*)<
\min_{t<j_x^*}M_x(t)
\right]
$$

for nontrivial \(j_x^*\in\{1,\ldots,8\}\).

For each model,

$$
TRR_J=\frac{1}{288}\sum_x TR_x.
$$

The paired label-swap effect is

$$
\Delta TRR_J=TRR_J^{S}-TRR_J^{O}.
$$

The corresponding unreachable fraction is

$$
1-TRR_J.
$$

This is the principal ordering-based robustness outcome.

## 9. Secondary outcomes

The following are prespecified secondary diagnostics:

1. row-level accuracy under the swapped mapping;
2. Brier score using \(P(\text{met})=\sigma(M)\);
3. swapped-label raw TCE;
4. reachable count \(RC_x\);
5. prefix-record rate;
6. practical and zero-tolerance monotonicity violation rates using the existing definitions;
7. curve-wise Spearman correlation between \(M^O_x(s)\) and \(M^S_x(s)\);
8. sign agreement between canonical margins;
9. reachable-set agreement and Jaccard similarity;
10. family-stratified versions of TCE, TRR, agreement, and margin-ordering statistics.

The previously introduced \(RC_x=10\) diagnostic may be recomputed under label swapping but remains a post-study theoretical diagnostic, not a preregistered s-3 endpoint.

## 10. Primary statistical analysis

All comparisons are paired.

For each model separately, report:

* point estimate,
* paired difference where applicable,
* 95% confidence interval.

Confidence intervals will use 5,000 artifact-clustered bootstrap replicates. Resampling will occur at the artifact-curve level so that all nine thresholds for a curve remain together.

Where family stratification is used, resampling will preserve the three-family structure.

For nontrivial TRR, bootstrap only the nontrivial artifact curves while preserving pairing between original and swapped conditions.

The four models are separate planned analyses. A model-macro-average may be reported descriptively but will not replace per-model results.

No row-level bootstrap that treats the nine thresholds of an artifact as independent is permitted.

No outcome-dependent change to the bootstrap procedure is permitted.

## 11. Prespecified practical robustness bands

These bands are interpretive thresholds, not claims about universal clinical or operational significance.

### Strong label-swap robustness for an individual model

A model will be described as strongly robust to label assignment if all three conditions hold:

$$
CI_{95}(\Delta TCE)\subset[-0.5,+0.5],
$$

$$
CI_{95}(\Delta TRR)\subset[-0.10,+0.10],
$$

and the lower 95% confidence bound for semantic decision agreement is at least

$$
0.90.
$$

### Partial robustness

If the strong criterion is not met but the principal qualitative conclusions remain — in particular, substantial localization error and substantial nontrivial boundary unreachability remain after swapping — the result will be described as partially robust, with the magnitude and direction of label sensitivity reported explicitly.

### Label-sensitive result

If swapping labels materially changes TCE, TRR, or semantic decisions outside the predefined practical bands, the paper will state that FormalCRRC response geometry is sensitive to response-token assignment for that model.

No result will be relabeled "robust" merely because a conventional null-hypothesis test fails to reject equality.

## 12. Principal confirmatory interpretation

The main FormalCRRC ordering conclusion will be considered supported against the simple A/B-token-assignment explanation if the swapped-label condition continues to show substantial nontrivial unreachable boundaries and does not produce a large systematic increase in TRR sufficient to remove the previously observed ordering failure.

The exact swapped unreachable percentages will be reported for every model.

The experiment is not designed to prove that labels have zero effect. A nonzero effect may coexist with persistence of the principal FormalCRRC phenomenon.

## 13. Expected interpretations

### Outcome A: high invariance

If semantic agreement is high, \(\Delta TCE\) and \(\Delta TRR\) are practically small, margin ordering is strongly preserved, and the unreachable-boundary phenomenon persists, the supported conclusion is:

> FormalCRRC's principal boundary-localization and reachability findings are robust to reversing the A/B response-token mapping and therefore are not explained by the original assignment of "met" and "not met" to A and B.

### Outcome B: location changes but ordering persists

If TCE or binary decisions change materially while TRR and prefix-record structure remain similar, the supported conclusion is:

> Response-token assignment affects score location or binary classification, but does not explain the ordering-based reachability failures identified by FormalCRRC.

This outcome would be particularly consistent with a label-associated location bias.

### Outcome C: ordering changes materially

If TRR, reachable sets, or within-curve margin ordering change substantially under label swapping, the supported conclusion is:

> Response-token assignment materially affects the judge's threshold-response geometry, and label mapping is therefore a substantive component of FormalCRRC reliability for that model.

This result must be reported even if it weakens the original robustness claim.

## 14. Tie handling

All s-3 conventions remain unchanged.

A nontrivial level \(j\) is reachable only if

$$
M(j)<\min_{t<j}M(t).
$$

Equality does not qualify.

Margins equal to zero count as "met" for binary classification.

No numeric tolerance will be introduced for exact equality unless required by an identified floating-point serialization defect. Any such issue would require a documented preregistration amendment before analyzing outcomes.

## 15. Missing inference and infrastructure failure

All four models are required.

A failed forward evaluation caused by infrastructure or hardware may be retried using the identical model, revision, prompt, artifact, and inference configuration.

Retries must be logged.

No example may be selectively dropped because its margin, decision, TCE, or TR status is undesirable.

If a model revision cannot be reproduced exactly, the study for that model must stop rather than silently substitute another revision.

If systematic missing data remain, the primary complete-panel analysis will not be reported as complete.

## 16. Exclusions

No artifact will be excluded based on model behavior.

An observation may be excluded only for a verified mechanical corruption such as:

* artifact identity mismatch,
* prompt-construction failure,
* tokenizer mismatch,
* inability to obtain the required A/B logits,
* or corrupted stored output.

Any exclusion must be documented before aggregate outcome inspection where possible and reported by model and reason.

## 17. Stopping rule

There is no outcome-based stopping rule.

The planned experiment consists of the entire frozen s-3 dataset across all four exact s-3 model revisions.

Inference stops only after:

* all planned evaluations are completed,
* or a reproducibility/integrity failure requires termination.

No additional artifacts, models, prompts, or label variants will be added based on observed results.

## 18. Integrity requirements

Before the first swapped-label inference:

1. fingerprint all frozen s-1, s-2, and s-3 scientific artifacts;
2. verify the existing integrity manifests;
3. record the exact s-3 dataset hash;
4. record all four exact model revisions;
5. record the original prompt-template hash;
6. create and hash the swapped prompt template;
7. freeze this preregistration;
8. record its SHA-256 and UTC timestamp;
9. record zero amendments.

After inference and analysis:

* all preexisting frozen scientific artifacts must remain byte-identical;
* new experiment files must be additive;
* raw swapped logits must be retained;
* artifact IDs must permit exact pairing to s 3;
* all analysis must be reproducible from raw frozen scores.

## 19. Preregistration amendment policy

An amendment made before any swapped-label inference may be incorporated if:

* its reason is documented,
* the previous preregistration hash is retained,
* a new hash and timestamp are recorded,
* and the amendment is explicitly listed.

Any methodological change made after any swapped-label inference has occurred must be labeled a post-inference deviation and cannot be represented as preregistered.

Outcome-driven metric changes are prohibited.

## 20. Claims that must not be made

Regardless of outcome, this experiment does not establish that:

* A/B tokens are the only source of judge bias;
* the model internally represents semantic classes in any specific way;
* invariance to A/B swapping implies invariance to arbitrary verbal labels;
* label swapping tests all prompt sensitivity;
* TRR invariance proves the model's representation is stable;
* a change under label swapping invalidates all FormalCRRC findings.

The experiment tests one narrow nuisance factor: semantic assignment to the A and B response tokens.

## 21. Relationship to prior FormalCRRC studies

ss 1-3 remain frozen and unchanged.

This experiment is a **post-study preregistered robustness analysis** motivated by the completed FormalCRRC results.

It is not:

* a s-3 amendment,
* a new independent replication dataset,
* a replacement for s-3 TRR,
* or part of the original three preregistrations.

Its role is to test whether the principal findings survive a controlled response-label counterbalance.

## 22. Planned manuscript reporting

The manuscript will report:

1. the paired counterbalance design;
2. exact original and swapped prompt mappings;
3. semantic agreement;
4. original vs swapped TCE;
5. original vs swapped nontrivial TRR and unreachable fraction;
6. confidence intervals for paired changes;
7. margin-ordering diagnostics;
8. family-stratified results as secondary analysis;
9. any model-specific label sensitivity;
10. integrity/provenance status.

The study will be reported regardless of whether label swapping strengthens, weakens, or leaves unchanged the existing FormalCRRC conclusions.

---

**PREREGISTRATION STATUS: FROZEN.** See Appendix A for the recorded hashes and timestamp.

**NO LABEL-SWAP MODEL INFERENCE MAY BEGIN BEFORE FREEZE.**

---

## Appendix A — Repository-derived values recorded at freeze

Filled automatically by `scripts/freeze_labelswap_preregistration.py`. No
protocol text above was altered.

**Freeze timestamp (UTC):** `2026-09-09T03:29:27+00:00`

### A.1 Dataset (frozen s-3 TEST, reused read-only)

| | |
|---|---|
| Path | `artifacts/s3/test_dataset.parquet` |
| Seed | 45 |
| Dataset file SHA-256 | `2160b662fa527498c6bd245e173f0ead9030a7d69633869233b76e5184647a34` |
| Dataset content SHA-256 | `9700960d6f3aa130fa259500aeadbdfa9f511bcc943dd5fdf3401330ee8009d6` |
| s-3 prompt manifest SHA-256 | `b348872573347fdd14ab2af7c080294052798a16977cf41c0ea8eb6243e35850` |
| Hashes match s-3 manifest | True / True |

### A.2 Model panel — exact s-3 revisions

| Model | Revision | Tokenizer revision | dtype | Chat template SHA-256 |
|---|---|---|---|---|
| `Qwen/Qwen2.5-14B-Instruct` | `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` | `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` | torch.bfloat16 | `cd8e9439f0570856…` |
| `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | `0e9e39f249a16976918f6564b8830bc894c89659` | torch.bfloat16 | `e10ca381b1ccc5cf…` |
| `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | torch.bfloat16 | `e16746b40344d6c5…` |
| `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` | torch.bfloat16 | `ecd6ae513fe103f0…` |

Inference configuration: `bfloat16`, transformers
`5.16.1`, torch
`2.13.0+cu130`, scoring rule
`single_token_next_logit` via prompt-continuation label resolution — all
inherited from s 3 unchanged.

### A.3 Prompt templates

| | |
|---|---|
| Original template SHA-256 | `e5f3c5ba3dfd1ccaca0655307d8ad254598a17184c7ae40dfde63c1f660cae14` |
| Swapped template SHA-256 | `c1edca6bdf4e685f1df9ed65ac1b67f80835cd9d8f6a353abf6abbbb9d89a867` |
| Equal length | True |
| Differing byte positions | 2 |
| All differences are label characters | True |
| Non-label text identical | True |
| Verdict | **PASS** |

Differing positions: index 255 (A→B), index 289 (B→A)

Original response instruction:

```text
Return A if the criterion is met.
Return B if the criterion is not met.
```

Swapped response instruction:

```text
Return B if the criterion is met.
Return A if the criterion is not met.
```

### A.4 Statistics

| | |
|---|---|
| Bootstrap replicates | 5000 |
| Bootstrap seed | 42 |
| Scheme | paired artifact-clustered resampling with replacement; all nine strictness levels of a curve stay together; the original and swapped members of a pair are drawn jointly; family strata preserved for family-stratified analyses |
| Interval | percentile, 95% |

### A.5 Amendments

**0** at freeze.
