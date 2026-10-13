# FormalCRRC Qwen2.5 Same-Family Model-Scale Ablation — Preregistration

**Frozen before any FormalCRRC inference with Qwen2.5-7B or Qwen2.5-32B.**

| | |
|---|---|
| Study type | prospectively frozen post-study same-family model-scale ablation |
| Frozen at (UTC) | `2026-10-11T14:55:19+00:00` |
| Machine form | `artifacts/qwen_scale_ablation/preregistration.json` |
| Machine form SHA-256 | `e943462cf99b992cc742ce0a1572a9011593ebcbe8f9440020de7350f5130209` |
| Amendments | 0 |

## 1. Status of this document

The Qwen2.5-14B S3 results already existed, and had been inspected, before this study was designed. This document is therefore not a preregistration with respect to the 14B outcomes; it is prospective only with respect to the 7B and 32B outcomes and to every comparison involving them.

This is not a rerun, extension or revision of the preregistered Day-1, Day-2 or Day-3 studies and must never be described as one.

No outcome-dependent change to this plan is permitted once either new model has been run. If a genuine implementation defect is found after the freeze, the original freeze is preserved byte-identical and a numbered amendment records the defect, why the change is outcome-independent, exactly what changed, and a new checksum.

## 2. Scientific question

> Under the same immediate-readout FormalCRRC protocol, does increasing Qwen2.5 model scale improve rubric-threshold localization and recoverability?

The study addresses the possible objection that the boundary-localization and boundary-recoverability failures FormalCRRC measures mainly reflect limited model capability. It is a same-family model-scale comparison under a common evaluation protocol. The checkpoints differ in scale and potentially in training dynamics, so parameter count is not claimed to be the causal mechanism under any outcome.

## 3. The scale ladder

| Rung | Model | Role |
|---|---|---|
| 7B | `Qwen/Qwen2.5-7B-Instruct` | new, run under this preregistration |
| 14B | `Qwen/Qwen2.5-14B-Instruct` | existing frozen S3 result; **not rerun** |
| 32B | `Qwen/Qwen2.5-32B-Instruct` | new, run under this preregistration |

Run order is fixed: 7B, then 32B.

## 4. Frozen S3 panel (reused, not regenerated)

Partition `DAY3_TEST`, seed 45: 324 artifacts x 9 thresholds = 2916 rubric conditions per model; 288 nontrivial artifacts. An artifact is nontrivial when its truth curve is nonconstant, i.e. j*_x <= 8. In S3 j* ranges over 1..9, so the 36 artifacts with j* = 9 (all nine thresholds met) are the constant-truth cases.

| Asset | SHA-256 |
|---|---|
| `artifacts/day3/test_dataset.parquet` | `2160b662fa527498c6bd245e173f0ead9030a7d69633869233b76e5184647a34` |
| dataset content | `9700960d6f3aa130fa259500aeadbdfa9f511bcc943dd5fdf3401330ee8009d6` |
| `artifacts/day3/prompt_manifest.parquet` | `b348872573347fdd14ab2af7c080294052798a16977cf41c0ea8eb6243e35850` |
| prompt template | `e5f3c5ba3dfd1ccaca0655307d8ad254598a17184c7ae40dfde63c1f660cae14` |

Panel assertions verified at freeze:

- PASS `n_rows_is_2916`
- PASS `prompt_ids_unique`
- PASS `n_artifacts_is_324`
- PASS `three_families`
- PASS `nine_latent_levels`
- PASS `twelve_per_cell`
- PASS `every_artifact_has_exactly_9_thresholds`
- PASS `n_nontrivial_is_288`
- PASS `j_star_matches_truth_labels`
- PASS `truth_is_a_single_step`
- PASS `seed_is_45`
- PASS `dataset_file_matches_manifest`
- PASS `dataset_content_matches_manifest`
- PASS `prompt_manifest_matches_manifest`
- PASS `prompt_template_matches_manifest`

No artifact, threshold, rubric text, truth label or prompt is regenerated.

## 5. The existing Qwen2.5-14B reference

| | |
|---|---|
| Raw margins | `artifacts/day3/raw_scores/qwen2_5_14b_instruct.parquet` |
| Raw margins SHA-256 | `10420978b17f61d2d6717d4ba1f2f7ab16ee4a445996bea3c2bfb7fc1630dbb6` |
| Model and tokenizer revision | `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` |
| Chat template SHA-256 | `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f` |
| Label tokens | A = [32], B = [33] (single token each) |
| Scoring | `single_token_next_logit` |
| Precision | bfloat16, no quantization, batch size 1 |
| Stack | torch 2.13.0 (CUDA 13.0), transformers 5.16.1, tokenizers 0.23.2, Python 3.11.16 |
| Hardware | NVIDIA H100 PCIe, node `evc41`, SLURM job 797710, 161.42 s |

Reference values recomputed from the saved margins at freeze. The analysis recomputes them again and refuses to proceed unless every count matches.

| Metric | Count | Value |
|---|---:|---:|
| accuracy | 2239/2916 | 0.767833 |
| tce | 1093/324 | 3.373457 |
| trr | 138/288 | 0.479167 |
| fsrr | 136/288 | 0.472222 |
| oracle_errors | 201/288 | 0.697917 |
| oracle_errors_all | 201/324 | 0.620370 |
| mvr_05 | 151/2592 | 0.058256 |
| mvr_0 | 798/2592 | 0.307870 |

Accuracy, TCE and TRR for 14B reproduce the frozen Day-3 report. FSRR (136/288) and E* are confirmed by the author against the manuscript definitions. The S3 MVR values are newly recomputed S3 reference values: the manuscript reports Day-1 MVR, not S3 MVR.

Reference assertions verified at freeze:

- PASS `accuracy_count_matches`
- PASS `tce_count_matches`
- PASS `trr_count_matches`
- PASS `fsrr_count_matches`
- PASS `oracle_errors_count_matches`
- PASS `oracle_errors_all_count_matches`
- PASS `mvr_05_count_matches`
- PASS `mvr_0_count_matches`
- PASS `n_artifacts_is_324`
- PASS `n_nontrivial_is_288`
- PASS `fsrr_is_136_over_288`
- PASS `fsr_iff_zero_oracle_errors_on_nontrivial`
- PASS `full_sequence_recovery_implies_reachability`
- PASS `frozen_day3_artifact_ids_match`
- PASS `frozen_day3_tce_raw_matches`
- PASS `frozen_day3_j_hat_raw_matches`
- PASS `frozen_day3_translation_reachable_matches`
- PASS `frozen_day3_otce_matches`
- PASS `frozen_day3_reachable_count_matches`
- PASS `raw_scores_sha256_matches`
- PASS `revision_matches`
- PASS `chat_template_sha256_matches`
- PASS `scoring_method_is_single_token_next_logit`
- PASS `stack_matches`
- PASS `all_2916_scored`

## 6. Inference protocol

The 7B and 32B conditions reproduce the 14B immediate-readout protocol. Held fixed:

- frozen S3 (Day-3 TEST, seed 45) artifacts
- all nine rubric thresholds per artifact
- rubric text
- user prompt (frozen Day-1 template, single user message)
- answer labels A / B
- A = criterion met, B = criterion not met
- native chat template with generation prompt
- bfloat16 weights
- no quantization
- no LoRA or adapters
- no fine-tuning
- no calibration layer
- no prompt optimization
- no few-shot examples
- no chain-of-thought request
- no generated explanation
- no sampled verdict
- batch size 1, one forward pass per rubric condition
- frozen inference order (artifacts/day3/prompt_manifest.parquet)
- scoring rule and scoring implementation (formalcrrc.scoring)
- truth computation
- analysis code

The prompt template, verbatim:

```text
You are evaluating exactly one criterion for one candidate response.

Use only the stated criterion. Do not reward or penalize the response for properties outside that criterion.

Criterion:
{rubric_text}

Candidate response:
{candidate_response}

Return A if the criterion is met.
Return B if the criterion is not met.
```

Canonical margin: `M_x(s) = S_A - S_B`, where A = met and B = not met. `P_met = exp(S_A) / (exp(S_A) + exp(S_B))`. Label tokens are resolved by `prompt_continuation_difference`.

The scoring path is the existing implementation, called unchanged: the Day-3 model loader, `formalcrrc.scoring.render_chat_prompt` and `formalcrrc.scoring.score_prompt`. No reason-then-score or other generation protocol is used.

## 7. Model provenance and continuity requirements

For each new checkpoint the revision is the full commit hash that the Hugging Face `main` ref resolves to when the pre-flight runs. It is resolved once, recorded in model_revisions.json before any S3 row is scored, and used for both model and tokenizer. Inference loads that commit explicitly and refuses a branch name.

Recorded per checkpoint in `artifacts/qwen_scale_ablation/model_revisions.json`: full commit hash, license, `config.json`, measured parameter count, tokenizer configuration, chat-template hash and label tokenization.

Each new checkpoint must satisfy all of the following before any S3 row is scored:

1. 'A' and 'B' each continue the rendered prompt with exactly one token, so the scoring method resolves to single_token_next_logit as it did for 14B
1. each of those single tokens decodes to exactly 'A' / 'B'
1. the label tokenization is identical on the reference prompt, the smoke prompt and every one of the 2,916 rendered S3 prompts
1. all 2,916 rendered S3 prompts hash to the rendered_prompt_sha256 the frozen 14B run recorded for the same prompt_id (byte-identical rendering)
1. the checkpoint loads in unquantized bfloat16 on one 80 GB H100 with no offloading
1. the stack versions equal the frozen 14B run's
1. the frozen 14B checkpoint reproduces its recorded smoke-prompt scores on this stack

If any continuity requirement fails for either checkpoint, no S3 row is scored with that checkpoint and the incompatibility is reported. No alternative scoring method, dtype, quantization, offloading, engine or prompt is substituted without author approval and a numbered amendment.

Tokenization is not assumed identical because the checkpoints share a family; it is checked on every prompt.

## 8. Pre-flight

The pre-flight loads each checkpoint, validates its tokenizer, renders (but does not score) the 2,916 S3 prompts to check byte-identity and label tokenization, and scores only the permanently designated non-study development prompts. It runs no forward pass on any S3 prompt and inspects no S3 outcome.

scripts/preflight_qwen_scale.py is operational validation and is not among the hash-pinned analysis files: it produces no outcome and its acceptance criteria are the continuity requirements frozen here. The inference runner re-checks label tokenization, chat template, dtype, device placement and byte-identical rendering itself before scoring.

Permanently designated development prompts:

```text
Name one primary colour.
```
```text
Decide whether 91 is a prime number.
Return A if it is prime.
Return B if it is not prime.
```

Hardware: one NVIDIA H100 PCIe 80 GB per model. Required stack: torch 2.13.0, transformers 5.16.1, tokenizers 0.23.2.

## 9. Completeness

Each new model must produce exactly 2,916 scored conditions. Every planned condition is tracked with a terminal status. The analysis refuses to run for a model with fewer than 2,916 scored conditions; no partial analysis is produced and no curve is excluded on the basis of any margin value.

A condition may be resubmitted only if an infrastructure failure left no scientifically usable output for it. A condition that has a stored margin is never scored again: the runner refuses to start when a model's final result file exists, and on resumption skips every prompt_id already present in its partial file. Every invocation appends a receipt.

The inference runner prints operational output only (progress, timing, memory, failures). No metric is computed from 7B outputs until the 32B run is complete; the analysis script refuses to run unless both new result files exist.

Reported per model: planned, completed, failed, missing, retried.

## 10. Metrics

Conventions: j_hat = min{s : M(s) < 0}, else 9. M = 0 counts as 'criterion met'. D(s) = 1[P_met(s) >= 0.5]. Exact stored float64 comparison under the strict relations. No tolerance is introduced anywhere except the stated MVR delta.

### Primary

- **accuracy** — Pointwise accuracy: share of the 2,916 rubric conditions with 1[P_met >= 0.5] = Y. Existing definition (formalcrrc.metrics.accuracy). All 324 artifacts.
- **tce** — Mean threshold crossing error, TCE_x = |j_hat_x - j*_x|. Existing definition. All 324 artifacts.
- **trr** — Translation reachability rate: share of the 288 nontrivial artifacts whose true boundary j* is a strict prefix record low of the margin curve, M(j*) < min_{t<j*} M(t). Existing Day-3 definition (formalcrrc.day3.translation_reachable).
- **fsrr** — Full-sequence recovery rate. With P_x = {s : Y_x(s) = 1} and N_x = {s : Y_x(s) = 0}, FSR_x = 1 iff min_{s in P_x} M_x(s) > max_{s in N_x} M_x(s); equivalently one scalar cut c reproduces all nine verdicts, 1[M_x(s) >= c] = Y_x(s). FSRR is the mean of this strict indicator over the 288 nontrivial artifacts only.
- **oracle_errors** — Oracle minimum errors, E*_x = min_c sum_{s=0..8} 1[1[M_x(s) >= c] != Y_x(s)], minimised over every distinct scalar-cut state including all-pass and all-fail, with tied margins always falling on the same side of a cut. PRIMARY: mean over the 288 nontrivial artifacts. SECONDARY: mean over all 324.
- **mvr_05** — MVR_x(delta) = (1/8) sum_{s=0..7} 1[P_met(s+1) > P_met(s) + delta] with delta = 0.05. Existing Day-1 definition (formalcrrc.metrics.monotonicity_violation_rate). All 324 artifacts.
- **mvr_0** — The same with delta = 0, the existing sensitivity diagnostic.

### Secondary (descriptive)

- **oracle_errors_all** — Mean E*_x over all 324 artifacts.
- **otce** — Mean oracle translation TCE (formalcrrc.day3). All 324.
- **reachable_count** — Mean reachable-set size RC (formalcrrc.day3). All 324.
- **exact_boundary_rate** — Share of artifacts with TCE_x = 0. All 324.
- **brier** — Mean Brier score (formalcrrc.metrics.brier_score). All 324.

No metric is added after outcomes are inspected.

## 11. Primary comparison table

One table, columns Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-32B, rows: accuracy, tce, trr, fsrr, oracle_errors, mvr_05, mvr_0. The 14B column is recomputed from the frozen S3 margins, never typed in.

## 12. Paired uncertainty

artifact-clustered resampling with replacement, stratified by predicate family; the unit is the complete nine-threshold curve; one shared index plan underlies every metric and all three models, so each replicate draws an artifact's 7B, 14B and 32B curves together (paired). Reuses formalcrrc.rts_ablation_bootstrap.build_plan / replicate_means, the repository's existing paired procedure. No row-level resampling.

Replicates 5000, seed 42, interval percentile, 95%.

Primary paired contrasts (first minus second): 7B − 14B; 32B − 14B. Secondary, descriptive: 32B − 7B.

Paired differences with intervals are reported for: tce, accuracy, trr, fsrr, oracle_errors (primary); mvr_05, mvr_0, oracle_errors_all, otce, reachable_count, exact_boundary_rate, brier (secondary).

The artifact is the sampling unit. No test treats the 2,916 rubric conditions as independent.

When a model's unreachable count (or non-recovered count) over the 288 nontrivial artifacts is zero, or its reachable count is zero, the rate is reported with an exact two-sided 95% Clopper-Pearson interval; a degenerate bootstrap interval is not presented as population uncertainty.

### Null-hypothesis tests

Exact two-sided McNemar (binomial on discordant pairs) for the binary artifact-level outcomes TR and FS over the 288 nontrivial artifacts, for 7B versus 14B and 32B versus 14B: four tests in one family, Holm-corrected at family-wise alpha = 0.05. Paired gain/loss counts are reported alongside. Confidence intervals and paired effect estimates are the primary evidence; these four p-values are supplementary.

## 13. Scale trend (descriptive)

For each primary metric the three point estimates are compared in ladder order 7B, 14B, 32B. 'monotonic improvement' when they are weakly ordered in the beneficial direction (lower-is-better: 7B >= 14B >= 32B; higher-is-better: 7B <= 14B <= 32B) and not all equal; 'monotonic decline' when weakly ordered in the adverse direction and not all equal; 'flat' when all three are equal; otherwise 'not monotonic'. Each label is accompanied by whether the two adjacent paired 95% intervals exclude zero.

- No regression is fitted and no parameter-scaling law is claimed from three model sizes.
- Monotonicity across three checkpoints is not described as proof that parameter count causes the change.

## 14. Prespecified interpretation

The qualitative cases are given operational conditions here, before inference, so the wording cannot follow the result. They key on TRR. Cases are not mutually exclusive.

| Case | Condition | Interpretation |
|---|---|---|
| A | 32B unreachable fraction (1 - TRR) >= 0.10 over the 288 nontrivial artifacts | The FormalCRRC phenomenon is not confined to the 7B-14B capability regime, although this experiment does not establish behavior for proprietary frontier models. |
| B | TRR strictly increases 7B < 14B < 32B and both adjacent paired 95% intervals (14B - 7B and 32B - 14B) exclude zero | Threshold reliability appears capability/scale-sensitive within the Qwen2.5 family, while FormalCRRC remains useful because it measures that improvement directly. |
| C | 32B TRR >= 0.95 | FormalCRRC distinguishes a stronger threshold-reliable regime rather than mechanically forcing failure. This is not characterised as weakening the framework. |
| D | the TRR trend label is 'not monotonic' | Parameter scale alone does not explain threshold reliability under this protocol. |

If no case condition holds (for example a monotonic decline, a flat trend, or a weak monotonic improvement with 32B TRR between 0.90 and 0.95), the result is described by its trend label and paired intervals without assigning a case.

Under no outcome will the report claim:

- larger models solve FormalCRRC universally
- frontier models are immune
- parameter count causes recoverability
- reasoning is the causal mechanism
- Qwen2.5-32B represents current frontier capability
- this is a frontier-model experiment
- a parameter-scaling law holds

## 15. Family-level analysis

TCE, TRR, FSRR and accuracy are reported separately for coverage, max_violation and numeric_tolerance for every model, with paired differences against 14B. Family results are secondary and descriptive; opposing family effects are reported, not averaged away.

## 16. Integrity checks before results are accepted

1. dataset IDs exactly match the S3 panel
1. nine thresholds exist for every artifact
1. truth labels and j* reproduce the S3 truth exactly
1. rendered prompts are byte-identical to the 14B run's for every prompt_id
1. A/B semantic mapping is unchanged
1. no result rows are duplicated
1. no row is missing without an explicit terminal status
1. every metric recomputed independently from saved margins agrees with the primary analysis
1. the 14B metrics reproduce exactly
1. none of the 205 fingerprinted frozen files has changed (combined SHA-256 `2fef9550b04e043d725a2e2473748e0f3083de2b24a58236f9cdba2a77589bac`)

## 17. Outputs

- `docs/QWEN_SCALE_ABLATION_PREREGISTRATION.md`
- `docs/QWEN_SCALE_ABLATION_RESULTS.md`
- `docs/QWEN_SCALE_ABLATION_INTEGRITY.md`
- `artifacts/qwen_scale_ablation/model_revisions.json`
- `artifacts/qwen_scale_ablation/run_manifest.json`
- `artifacts/qwen_scale_ablation/results_7b.parquet` (results 7b)
- `artifacts/qwen_scale_ablation/results_32b.parquet` (results 32b)
- `artifacts/qwen_scale_ablation/combined_scale_results.csv`
- `artifacts/qwen_scale_ablation/artifact_level_metrics.csv`
- `artifacts/qwen_scale_ablation/paired_effects.csv`
- `artifacts/qwen_scale_ablation/family_results.csv`
- `artifacts/qwen_scale_ablation/summary.json`
- `artifacts/qwen_scale_ablation/integrity_report.json`
- `artifacts/qwen_scale_ablation/checksums.sha256`
- `tables/qwen_scale_ablation.tex`

The main manuscript is not edited by this study. Manuscript-ready text and the LaTeX table are provided for author approval.

## 18. Analysis code pinned at freeze

Tree SHA-256 `9bb3d49f9b24e876a2c4c417bca7a596289e76e15f11454ee32b6c098c4b5da7`. The inference runner and the analysis refuse to run if any pinned file has changed.

| File | SHA-256 |
|---|---|
| `scripts/analyze_qwen_scale.py` | `302fa75c4732c44ee8d4482a95915ca74b07e1a16e26394dabb845ff36f9094f` |
| `scripts/freeze_qwen_scale_preregistration.py` | `56e5a1d638aec15fb87fb7e1e7a32beaf7a8d9868737c3604eb1dc02eda9779a` |
| `scripts/run_day3_inference.py` | `f5576be477db475fae7328b8bccbb0243f0588d57f78f95912072d2c7475bf79` |
| `scripts/run_qwen_scale_inference.py` | `c0750d9a8189b61dc2cc6829ed26be12d7f9636a8d56b8f96e30ef5421fcf646` |
| `scripts/verify_qwen_scale_integrity.py` | `233396b3cf84fa26ad94e8605376028fa800eb0a7f768348556ebf6f2c7f3227` |
| `src/formalcrrc/config.py` | `4558ccfdcdd55e2ffd7cd30dbd0b06d07fbbcf659dd1bbf5d8dbef7e209efa52` |
| `src/formalcrrc/dataset.py` | `8a6cb27c91c8d05ab3c9ca29ec7808ed067b0c7d215044c892bb3e059c240d2d` |
| `src/formalcrrc/day2.py` | `5b61a03e18610a3cd646fad4028f89e0c3b95056297c33caf1c89782d087de32` |
| `src/formalcrrc/day2_bootstrap.py` | `fd3062c196add2e7af00a227294d95de92c0642ce893f75a26961fa7dfb58993` |
| `src/formalcrrc/day3.py` | `213fcd331a67d73b8ec5cdcd30dafa778da9b7d22107c8cd756b8a6ada713cc5` |
| `src/formalcrrc/day3_config.py` | `1c9c2cb1aa4462ee29fcc0215078f99acc2de518d74994efe3169620494cbb5c` |
| `src/formalcrrc/metrics.py` | `7f9e7909272c7bcbe6f0971d140027088ac8cc7a98a582563c08b8fce570ed39` |
| `src/formalcrrc/predicates.py` | `5c119021ffa69d2dc875bb931a249e5f12811fda791227bc0716cb075fc56b68` |
| `src/formalcrrc/prompts.py` | `a5350ba6687fecb2d7526677bb31ad12373df58064e1096d1a15603a03308dd4` |
| `src/formalcrrc/provenance.py` | `34e1c06f91514642b0bcc1beb946997ad275d2b69030f80d83b6e7df9920bd55` |
| `src/formalcrrc/qwen_scale.py` | `c85f71f61b7c3e30d432fe06da1656ea4cee32af560068ccd523f1f9dc89d00c` |
| `src/formalcrrc/qwen_scale_config.py` | `a2b76107f3e6e543b1560fb789b25025187d7266b7a1f3ac8c8223593cf2a070` |
| `src/formalcrrc/rts_ablation.py` | `aab7694a8564fd5fb8645f3e65552bb07f8f35fe731b0baa4821788b482df2d6` |
| `src/formalcrrc/rts_ablation_bootstrap.py` | `ec4fcb12e3414c6c2eeee189bb634b5b0e2f18e3aab61d45c256c0e2f5902a53` |
| `src/formalcrrc/rts_ablation_config.py` | `df020c365e4e73d1b3d48868d19967f70b60a9f9c987d903aad6537add2e21f8` |
| `src/formalcrrc/scoring.py` | `e9dea77babcb60820ad679226d2d826265fe0d844948163296c71be56510c8d4` |

---

*Rendered by `scripts/freeze_qwen_scale_preregistration.py` from the same object as the machine form.*
