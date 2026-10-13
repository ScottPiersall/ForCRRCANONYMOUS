"""Preregistered constants for the FormalCRRC Qwen2.5 same-family scale ablation.

Kept in its own module so this study adds no constant to any prior configuration
and touches no file a previous preregistration pinned by hash. Everything here
is serialised verbatim into ``artifacts/qwen_scale_ablation/preregistration.json``
and may not change after the freeze except through a numbered, checksummed
amendment.

The design is a **post-study same-family model-scale ablation**. The middle rung
is the frozen s-3 (S3) record for ``Qwen/Qwen2.5-14B-Instruct``, read only and
never rerun. The two new rungs, 7B and 32B, are scored over the same artifacts
under the same immediate-readout protocol by the same scoring functions. Nothing
about the protocol moves; only the checkpoint does.

What this can support is a same-family scale *comparison*. The checkpoints
differ in parameter count and possibly in training dynamics, so parameter count
is never named as the causal mechanism.
"""

from __future__ import annotations

from typing import Final

from formalcrrc.config import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED  # noqa: F401
from formalcrrc.config import (  # noqa: F401
    MODEL_DTYPE,
    MVR_TOLERANCE,
    MVR_TOLERANCE_STRICT,
)

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

STUDY_NAME: Final[str] = "FormalCRRC Qwen2.5 Same-Family Model-Scale Ablation"
STUDY_TYPE: Final[str] = (
    "prospectively frozen post-study same-family model-scale ablation"
)

FREEZE_STATEMENT: Final[str] = (
    "Frozen before any FormalCRRC inference with Qwen2.5-7B or Qwen2.5-32B."
)

NOT_PREREGISTERED_FOR_14B: Final[str] = (
    "The Qwen2.5-14B S3 results already existed, and had been inspected, before "
    "this study was designed. This document is therefore not a preregistration "
    "with respect to the 14B outcomes; it is prospective only with respect to "
    "the 7B and 32B outcomes and to every comparison involving them."
)

NOT_A_RERUN: Final[str] = (
    "This is not a rerun, extension or revision of the preregistered s-1, "
    "s-2 or s-3 studies and must never be described as one."
)

PRIMARY_QUESTION: Final[str] = (
    "Under the same immediate-readout FormalCRRC protocol, does increasing "
    "Qwen2.5 model scale improve rubric-threshold localization and "
    "recoverability?"
)

# --------------------------------------------------------------------------
# The scale ladder
# --------------------------------------------------------------------------

TAG_7B: Final[str] = "7b"
TAG_14B: Final[str] = "14b"
TAG_32B: Final[str] = "32b"

#: Ordered from smallest to largest. The trend checks read this order.
LADDER: Final[tuple[str, ...]] = (TAG_7B, TAG_14B, TAG_32B)

#: The rungs this study runs. 14B is never rerun.
NEW_TAGS: Final[tuple[str, ...]] = (TAG_7B, TAG_32B)

REFERENCE_TAG: Final[str] = TAG_14B

MODEL_IDS: Final[dict[str, str]] = {
    TAG_7B: "Qwen/Qwen2.5-7B-Instruct",
    TAG_14B: "Qwen/Qwen2.5-14B-Instruct",
    TAG_32B: "Qwen/Qwen2.5-32B-Instruct",
}

MODEL_SHORT_NAMES: Final[dict[str, str]] = {
    TAG_7B: "Qwen2.5-7B",
    TAG_14B: "Qwen2.5-14B",
    TAG_32B: "Qwen2.5-32B",
}

#: Nominal sizes, used only as figure axis labels. Exact parameter counts are
#: measured from the loaded checkpoints and recorded in model_revisions.json.
NOMINAL_BILLIONS: Final[dict[str, int]] = {TAG_7B: 7, TAG_14B: 14, TAG_32B: 32}

#: The run order is fixed here, not derived from any outcome.
RUN_ORDER: Final[tuple[str, ...]] = (TAG_7B, TAG_32B)

# --------------------------------------------------------------------------
# The frozen 14B reference -- recovered from s-3 provenance, never floated
# --------------------------------------------------------------------------

REFERENCE_REVISION: Final[str] = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
REFERENCE_TOKENIZER_REVISION: Final[str] = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
REFERENCE_CHAT_TEMPLATE_SHA256: Final[str] = (
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)
REFERENCE_RAW_SCORES_PATH: Final[str] = (
    "artifacts/s3/raw_scores/qwen2_5_14b_instruct.parquet"
)
REFERENCE_RUN_RECORD_PATH: Final[str] = (
    "artifacts/s3/raw_scores/qwen2_5_14b_instruct_run.json"
)
REFERENCE_RAW_SCORES_SHA256: Final[str] = (
    "10420978b17f61d2d6717d4ba1f2f7ab16ee4a445996bea3c2bfb7fc1630dbb6"
)

#: The software stack of the frozen 14B run. The new rungs must run on it.
REFERENCE_STACK: Final[dict[str, str]] = {
    "torch": "2.13.0",
    "transformers": "5.16.1",
    "tokenizers": "0.23.2",
}

#: The non-study smoke prompt and the scores the frozen 14B produced for it.
#: Reproducing them proves the numerical stack is the one s 3 ran on.
SMOKE_PROMPT: Final[str] = "Name one primary colour."
REFERENCE_SMOKE_SCORE_MET: Final[float] = 37.0
REFERENCE_SMOKE_SCORE_NOT_MET: Final[float] = 13.75

# --------------------------------------------------------------------------
# Protocol -- held fixed, restated so the freeze records it
# --------------------------------------------------------------------------

PROTOCOL: Final[str] = "immediate_next_token_readout"

HELD_FIXED: Final[tuple[str, ...]] = (
    "frozen S3 (s-3 TEST, seed 45) artifacts",
    "all nine rubric thresholds per artifact",
    "rubric text",
    "user prompt (frozen s-1 template, single user message)",
    "answer labels A / B",
    "A = criterion met, B = criterion not met",
    "native chat template with generation prompt",
    "bfloat16 weights",
    "no quantization",
    "no LoRA or adapters",
    "no fine-tuning",
    "no calibration layer",
    "no prompt optimization",
    "no few-shot examples",
    "no chain-of-thought request",
    "no generated explanation",
    "no sampled verdict",
    "batch size 1, one forward pass per rubric condition",
    "frozen inference order (artifacts/s3/prompt_manifest.parquet)",
    "scoring rule and scoring implementation (formalcrrc.scoring)",
    "truth computation",
    "analysis code",
)

LABEL_RESOLUTION: Final[str] = "prompt_continuation_difference"
REQUIRED_SCORING_METHOD: Final[str] = "single_token_next_logit"
CANONICAL_MARGIN: Final[str] = "M_x(s) = S_A - S_B"
PROBABILITY: Final[str] = "P_met = exp(S_A) / (exp(S_A) + exp(S_B))"

#: What each new checkpoint must satisfy before a single S3 row is scored.
CONTINUITY_REQUIREMENTS: Final[tuple[str, ...]] = (
    "'A' and 'B' each continue the rendered prompt with exactly one token, so "
    "the scoring method resolves to single_token_next_logit as it did for 14B",
    "each of those single tokens decodes to exactly 'A' / 'B'",
    "the label tokenization is identical on the reference prompt, the smoke "
    "prompt and every one of the 2,916 rendered S3 prompts",
    "all 2,916 rendered S3 prompts hash to the rendered_prompt_sha256 the "
    "frozen 14B run recorded for the same prompt_id (byte-identical rendering)",
    "the checkpoint loads in unquantized bfloat16 on one 80 GB H100 with no "
    "offloading",
    "the stack versions equal the frozen 14B run's",
    "the frozen 14B checkpoint reproduces its recorded smoke-prompt scores on "
    "this stack",
)

STOP_RULE: Final[str] = (
    "If any continuity requirement fails for either checkpoint, no S3 row is "
    "scored with that checkpoint and the incompatibility is reported. No "
    "alternative scoring method, dtype, quantization, offloading, engine or "
    "prompt is substituted without author approval and a numbered amendment."
)

#: Forbidden fallbacks, listed so the runner can refuse them by name.
FORBIDDEN_QUANTIZATION: Final[tuple[str, ...]] = (
    "fp8", "gptq", "awq", "bitsandbytes", "gguf", "int8", "int4",
)

REVISION_RULE: Final[str] = (
    "For each new checkpoint the revision is the full commit hash that the "
    "Hugging Face `main` ref resolves to when the pre-flight runs. It is "
    "resolved once, recorded in model_revisions.json before any S3 row is "
    "scored, and used for both model and tokenizer. Inference loads that "
    "commit explicitly and refuses a branch name."
)

PREFLIGHT_SCOPE: Final[str] = (
    "The pre-flight loads each checkpoint, validates its tokenizer, renders "
    "(but does not score) the 2,916 S3 prompts to check byte-identity and "
    "label tokenization, and scores only the permanently designated non-study "
    "development prompts. It runs no forward pass on any S3 prompt and "
    "inspects no S3 outcome."
)

PREFLIGHT_CODE_STATUS: Final[str] = (
    "scripts/preflight_qwen_scale.py is operational validation and is not among "
    "the hash-pinned analysis files: it produces no outcome and its acceptance "
    "criteria are the continuity requirements frozen here. The inference runner "
    "re-checks label tokenization, chat template, dtype, device placement and "
    "byte-identical rendering itself before scoring."
)

#: Permanently designated development prompts. None is an S3 row.
DEVELOPMENT_PROMPTS: Final[tuple[str, ...]] = (
    SMOKE_PROMPT,
    "Decide whether 91 is a prime number.\n"
    "Return A if it is prime.\n"
    "Return B if it is not prime.",
)

# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------

SOURCE_PARTITION: Final[str] = "s3_TEST"
SOURCE_PARTITION_TAG: Final[str] = "D3T"
SOURCE_SEED: Final[int] = 45
N_ARTIFACTS: Final[int] = 324
N_FAMILIES: Final[int] = 3
N_LATENT_LEVELS: Final[int] = 9
N_PER_CELL: Final[int] = 12
N_STRICTNESS_LEVELS: Final[int] = 9
N_ROWS: Final[int] = 2916
N_NONTRIVIAL: Final[int] = 288

NONTRIVIAL_RULE: Final[str] = (
    "An artifact is nontrivial when its truth curve is nonconstant, i.e. "
    "j*_x <= 8. In S3 j* ranges over 1..9, so the 36 artifacts with j* = 9 "
    "(all nine thresholds met) are the constant-truth cases."
)

# --------------------------------------------------------------------------
# Completeness
# --------------------------------------------------------------------------

COMPLETENESS_RULE: Final[str] = (
    "Each new model must produce exactly 2,916 scored conditions. Every planned "
    "condition is tracked with a terminal status. The analysis refuses to run "
    "for a model with fewer than 2,916 scored conditions; no partial analysis "
    "is produced and no curve is excluded on the basis of any margin value."
)

RETRY_RULE: Final[str] = (
    "A condition may be resubmitted only if an infrastructure failure left no "
    "scientifically usable output for it. A condition that has a stored margin "
    "is never scored again: the runner refuses to start when a model's final "
    "result file exists, and on resumption skips every prompt_id already "
    "present in its partial file. Every invocation appends a receipt."
)

NO_PEEKING_RULE: Final[str] = (
    "The inference runner prints operational output only (progress, timing, "
    "memory, failures). No metric is computed from 7B outputs until the 32B "
    "run is complete; the analysis script refuses to run unless both new "
    "result files exist."
)

# --------------------------------------------------------------------------
# Metric definitions
# --------------------------------------------------------------------------

CROSSING_CONVENTION: Final[str] = (
    "j_hat = min{s : M(s) < 0}, else 9. M = 0 counts as 'criterion met'."
)
DECISION_CONVENTION: Final[str] = "D(s) = 1[P_met(s) >= 0.5]"
TIE_POLICY: Final[str] = (
    "Exact stored float64 comparison under the strict relations. No tolerance "
    "is introduced anywhere except the stated MVR delta."
)

METRIC_DEFINITIONS: Final[dict[str, str]] = {
    "accuracy": (
        "Pointwise accuracy: share of the 2,916 rubric conditions with "
        "1[P_met >= 0.5] = Y. Existing definition (formalcrrc.metrics.accuracy). "
        "All 324 artifacts."
    ),
    "tce": (
        "Mean threshold crossing error, TCE_x = |j_hat_x - j*_x|. Existing "
        "definition. All 324 artifacts."
    ),
    "trr": (
        "Translation reachability rate: share of the 288 nontrivial artifacts "
        "whose true boundary j* is a strict prefix record low of the margin "
        "curve, M(j*) < min_{t<j*} M(t). Existing s-3 definition "
        "(formalcrrc.s3.translation_reachable)."
    ),
    "fsrr": (
        "Full-sequence recovery rate. With P_x = {s : Y_x(s) = 1} and "
        "N_x = {s : Y_x(s) = 0}, FSR_x = 1 iff min_{s in P_x} M_x(s) > "
        "max_{s in N_x} M_x(s); equivalently one scalar cut c reproduces all "
        "nine verdicts, 1[M_x(s) >= c] = Y_x(s). FSRR is the mean of this "
        "strict indicator over the 288 nontrivial artifacts only."
    ),
    "oracle_errors": (
        "Oracle minimum errors, E*_x = min_c sum_{s=0..8} 1[1[M_x(s) >= c] != "
        "Y_x(s)], minimised over every distinct scalar-cut state including "
        "all-pass and all-fail, with tied margins always falling on the same "
        "side of a cut. PRIMARY: mean over the 288 nontrivial artifacts. "
        "SECONDARY: mean over all 324."
    ),
    "mvr_05": (
        "MVR_x(delta) = (1/8) sum_{s=0..7} 1[P_met(s+1) > P_met(s) + delta] "
        "with delta = 0.05. Existing s-1 definition "
        "(formalcrrc.metrics.monotonicity_violation_rate). All 324 artifacts."
    ),
    "mvr_0": "The same with delta = 0, the existing sensitivity diagnostic.",
}

#: The seven rows of the primary comparison table, in order.
PRIMARY_METRICS: Final[tuple[str, ...]] = (
    "accuracy",
    "tce",
    "trr",
    "fsrr",
    "oracle_errors",
    "mvr_05",
    "mvr_0",
)

#: Secondary, descriptive only. All use existing repository definitions.
SECONDARY_METRICS: Final[tuple[str, ...]] = (
    "oracle_errors_all",
    "otce",
    "reachable_count",
    "exact_boundary_rate",
    "brier",
)

SECONDARY_DEFINITIONS: Final[dict[str, str]] = {
    "oracle_errors_all": "Mean E*_x over all 324 artifacts.",
    "otce": "Mean oracle translation TCE (formalcrrc.s3). All 324.",
    "reachable_count": "Mean reachable-set size RC (formalcrrc.s3). All 324.",
    "exact_boundary_rate": "Share of artifacts with TCE_x = 0. All 324.",
    "brier": "Mean Brier score (formalcrrc.metrics.brier_score). All 324.",
}

#: Which artifacts each metric averages over.
METRIC_POPULATION: Final[dict[str, str]] = {
    "accuracy": "all",
    "tce": "all",
    "trr": "nontrivial",
    "fsrr": "nontrivial",
    "oracle_errors": "nontrivial",
    "mvr_05": "all",
    "mvr_0": "all",
    "oracle_errors_all": "all",
    "otce": "all",
    "reachable_count": "all",
    "exact_boundary_rate": "all",
    "brier": "all",
}

#: Column of the artifact-level table each metric is the mean of.
METRIC_COLUMN: Final[dict[str, str]] = {
    "accuracy": "accuracy",
    "tce": "tce",
    "trr": "translation_reachable",
    "fsrr": "full_sequence_recovered",
    "oracle_errors": "oracle_min_errors",
    "mvr_05": "mvr_05",
    "mvr_0": "mvr_0",
    "oracle_errors_all": "oracle_min_errors",
    "otce": "otce",
    "reachable_count": "reachable_count",
    "exact_boundary_rate": "exact_boundary",
    "brier": "brier",
}

HIGHER_IS_BETTER: Final[tuple[str, ...]] = ("accuracy", "trr", "fsrr")
LOWER_IS_BETTER: Final[tuple[str, ...]] = ("tce", "oracle_errors", "mvr_05", "mvr_0")

# --------------------------------------------------------------------------
# The frozen 14B reference values -- asserted, never substituted
# --------------------------------------------------------------------------

#: Exact counts recomputed from the frozen S3 margins. The freeze script and
#: the analysis both recompute them from artifacts/s3 and refuse to proceed
#: unless they match. ``mvr`` counts are violating adjacent steps out of 2,592.
REFERENCE_COUNTS: Final[dict[str, tuple[int, int]]] = {
    "accuracy": (2239, 2916),
    "tce": (1093, 324),
    "trr": (138, 288),
    "fsrr": (136, 288),
    "oracle_errors": (201, 288),
    "oracle_errors_all": (201, 324),
    "mvr_05": (151, 2592),
    "mvr_0": (798, 2592),
}

REFERENCE_PROVENANCE_NOTE: Final[str] = (
    "Accuracy, TCE and TRR for 14B reproduce the frozen s-3 report. FSRR "
    "(136/288) and E* are confirmed by the author against the manuscript "
    "definitions. The S3 MVR values are newly recomputed S3 reference values: "
    "the manuscript reports s-1 MVR, not S3 MVR."
)

# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

BOOTSTRAP_SCHEME: Final[str] = (
    "artifact-clustered resampling with replacement, stratified by predicate "
    "family; the unit is the complete nine-threshold curve; one shared index "
    "plan underlies every metric and all three models, so each replicate draws "
    "an artifact's 7B, 14B and 32B curves together (paired). Reuses "
    "formalcrrc.rts_ablation_bootstrap.build_plan / replicate_means, the "
    "repository's existing paired procedure. No row-level resampling."
)
BOOTSTRAP_CI_DESCRIPTION: Final[str] = "percentile, 95%"

#: Paired contrasts, each reported as (first minus second).
PRIMARY_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    (TAG_7B, TAG_14B),
    (TAG_32B, TAG_14B),
)
SECONDARY_CONTRASTS: Final[tuple[tuple[str, str], ...]] = ((TAG_32B, TAG_7B),)

#: Metrics for which paired differences and intervals are reported.
PAIRED_PRIMARY_METRICS: Final[tuple[str, ...]] = (
    "tce",
    "accuracy",
    "trr",
    "fsrr",
    "oracle_errors",
)
PAIRED_SECONDARY_METRICS: Final[tuple[str, ...]] = (
    "mvr_05",
    "mvr_0",
    "oracle_errors_all",
    "otce",
    "reachable_count",
    "exact_boundary_rate",
    "brier",
)

ZERO_COUNT_RULE: Final[str] = (
    "When a model's unreachable count (or non-recovered count) over the 288 "
    "nontrivial artifacts is zero, or its reachable count is zero, the rate is "
    "reported with an exact two-sided 95% Clopper-Pearson interval; a "
    "degenerate bootstrap interval is not presented as population uncertainty."
)
CLOPPER_PEARSON_LEVEL: Final[float] = 0.95

#: The complete family of null-hypothesis tests. Nothing else is tested.
MCNEMAR_OUTCOMES: Final[tuple[str, ...]] = (
    "translation_reachable",
    "full_sequence_recovered",
)
MCNEMAR_FAMILY: Final[str] = (
    "Exact two-sided McNemar (binomial on discordant pairs) for the binary "
    "artifact-level outcomes TR and FS over the 288 nontrivial artifacts, for "
    "7B versus 14B and 32B versus 14B: four tests in one family, "
    "Holm-corrected at family-wise alpha = 0.05. Paired gain/loss counts are "
    "reported alongside. Confidence intervals and paired effect estimates are "
    "the primary evidence; these four p-values are supplementary."
)
MCNEMAR_ALPHA: Final[float] = 0.05

# --------------------------------------------------------------------------
# Scale trend -- descriptive only
# --------------------------------------------------------------------------

TREND_RULE: Final[str] = (
    "For each primary metric the three point estimates are compared in ladder "
    "order 7B, 14B, 32B. 'monotonic improvement' when they are weakly ordered "
    "in the beneficial direction (lower-is-better: 7B >= 14B >= 32B; "
    "higher-is-better: 7B <= 14B <= 32B) and not all equal; 'monotonic "
    "decline' when weakly ordered in the adverse direction and not all equal; "
    "'flat' when all three are equal; otherwise 'not monotonic'. Each label is "
    "accompanied by whether the two adjacent paired 95% intervals exclude zero."
)

TREND_LABELS: Final[tuple[str, ...]] = (
    "monotonic improvement",
    "monotonic decline",
    "flat",
    "not monotonic",
)

TREND_PROHIBITIONS: Final[tuple[str, ...]] = (
    "No regression is fitted and no parameter-scaling law is claimed from "
    "three model sizes.",
    "Monotonicity across three checkpoints is not described as proof that "
    "parameter count causes the change.",
)

# --------------------------------------------------------------------------
# Prespecified interpretation
# --------------------------------------------------------------------------

#: Operational thresholds for the qualitative cases, fixed before inference so
#: that the wording cannot follow the result. They key on TRR, the headline
#: recoverability metric. Cases are not mutually exclusive.
SUBSTANTIAL_UNREACHABLE_MIN: Final[float] = 0.10
NEAR_PERFECT_TRR_MIN: Final[float] = 0.95

INTERPRETATION_CASES: Final[dict[str, dict[str, str]]] = {
    "A": {
        "condition": (
            "32B unreachable fraction (1 - TRR) >= 0.10 over the 288 nontrivial "
            "artifacts"
        ),
        "interpretation": (
            "The FormalCRRC phenomenon is not confined to the 7B-14B capability "
            "regime, although this experiment does not establish behavior for "
            "proprietary frontier models."
        ),
    },
    "B": {
        "condition": (
            "TRR strictly increases 7B < 14B < 32B and both adjacent paired 95% "
            "intervals (14B - 7B and 32B - 14B) exclude zero"
        ),
        "interpretation": (
            "Threshold reliability appears capability/scale-sensitive within "
            "the Qwen2.5 family, while FormalCRRC remains useful because it "
            "measures that improvement directly."
        ),
    },
    "C": {
        "condition": "32B TRR >= 0.95",
        "interpretation": (
            "FormalCRRC distinguishes a stronger threshold-reliable regime "
            "rather than mechanically forcing failure. This is not "
            "characterised as weakening the framework."
        ),
    },
    "D": {
        "condition": "the TRR trend label is 'not monotonic'",
        "interpretation": (
            "Parameter scale alone does not explain threshold reliability under "
            "this protocol."
        ),
    },
}

INTERPRETATION_FALLBACK: Final[str] = (
    "If no case condition holds (for example a monotonic decline, a flat "
    "trend, or a weak monotonic improvement with 32B TRR between 0.90 and "
    "0.95), the result is described by its trend label and paired intervals "
    "without assigning a case."
)

CLAIMS_NOT_MADE: Final[tuple[str, ...]] = (
    "larger models solve FormalCRRC universally",
    "frontier models are immune",
    "parameter count causes recoverability",
    "reasoning is the causal mechanism",
    "Qwen2.5-32B represents current frontier capability",
    "this is a frontier-model experiment",
    "a parameter-scaling law holds",
)

FAMILY_ANALYSIS: Final[str] = (
    "TCE, TRR, FSRR and accuracy are reported separately for coverage, "
    "max_violation and numeric_tolerance for every model, with paired "
    "differences against 14B. Family results are secondary and descriptive; "
    "opposing family effects are reported, not averaged away."
)

AMENDMENT_RULE: Final[str] = (
    "No outcome-dependent change to this plan is permitted once either new "
    "model has been run. If a genuine implementation defect is found after the "
    "freeze, the original freeze is preserved byte-identical and a numbered "
    "amendment records the defect, why the change is outcome-independent, "
    "exactly what changed, and a new checksum."
)

# --------------------------------------------------------------------------
# Paths -- all additive
# --------------------------------------------------------------------------

ARTIFACT_DIR: Final[str] = "artifacts/qwen_scale_ablation"
FIGURE_DIR: Final[str] = "figures/qwen_scale_ablation"

PREREG_MD_PATH: Final[str] = "docs/QWEN_SCALE_ABLATION_PREREGISTRATION.md"
RESULTS_MD_PATH: Final[str] = "docs/QWEN_SCALE_ABLATION_RESULTS.md"
INTEGRITY_MD_PATH: Final[str] = "docs/QWEN_SCALE_ABLATION_INTEGRITY.md"

PREREG_JSON_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration.json"
PREREG_SHA_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration.sha256"
PREREG_FROZEN_COPY_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration_frozen.md"
FROZEN_FINGERPRINT_PATH: Final[str] = f"{ARTIFACT_DIR}/frozen_record_fingerprint.json"

PREFLIGHT_PATH: Final[str] = f"{ARTIFACT_DIR}/preflight.json"
MODEL_REVISIONS_PATH: Final[str] = f"{ARTIFACT_DIR}/model_revisions.json"
RUN_MANIFEST_PATH: Final[str] = f"{ARTIFACT_DIR}/run_manifest.json"

COMBINED_RESULTS_PATH: Final[str] = f"{ARTIFACT_DIR}/combined_scale_results.csv"
ARTIFACT_METRICS_PATH: Final[str] = f"{ARTIFACT_DIR}/artifact_level_metrics.csv"
PAIRED_EFFECTS_PATH: Final[str] = f"{ARTIFACT_DIR}/paired_effects.csv"
FAMILY_RESULTS_PATH: Final[str] = f"{ARTIFACT_DIR}/family_results.csv"
SUMMARY_PATH: Final[str] = f"{ARTIFACT_DIR}/summary.json"
INTEGRITY_REPORT_PATH: Final[str] = f"{ARTIFACT_DIR}/integrity_report.json"
CHECKSUMS_PATH: Final[str] = f"{ARTIFACT_DIR}/checksums.sha256"

TEX_TABLE_PATH: Final[str] = "tables/qwen_scale_ablation.tex"
FIGURE_PATH: Final[str] = f"{FIGURE_DIR}/qwen_scale_tce_trr.png"


# --------------------------------------------------------------------------
# Amendment 001 -- added before any 7B or 32B inference; see AMENDMENTS below
# --------------------------------------------------------------------------

#: The single primary scaling endpoint. Everything else supports it.
PRIMARY_ENDPOINT: Final[str] = "trr"
SUPPORTING_ENDPOINTS: Final[tuple[str, ...]] = (
    "tce",
    "fsrr",
    "oracle_errors",
    "accuracy",
    "mvr_05",
    "mvr_0",
)
PRIMARY_ENDPOINT_STATEMENT: Final[str] = (
    "The primary scaling endpoint is nontrivial TRR: translation reachability "
    "of the true boundary over the 288 artifacts with j* <= 8. TCE, FSRR, E*, "
    "accuracy and MVR are supporting endpoints. All seven remain rows of the "
    "comparison table; 'primary metrics' in the original freeze names the rows "
    "of that table, not seven co-primary endpoints."
)

#: The two adjacent scale steps, each signed so that positive means the larger
#: checkpoint has the higher TRR.
SCALE_STEP_DEFINITIONS: Final[dict[str, str]] = {
    "Delta_7_to_14": "TRR_14B - TRR_7B",
    "Delta_14_to_32": "TRR_32B - TRR_14B",
}

CASE_STATUS: Final[str] = (
    "Cases A-D are non-exclusive descriptive interpretation flags. Any number "
    "of them, including none, may hold at once. They are not hypothesis-test "
    "categories, and the numeric conditions attached to them (0.10 unreachable, "
    "0.95 TRR) are labelling conventions for this report, not universal "
    "reliability thresholds. In Case B, 'both adjacent paired 95% intervals' "
    "means the paired bootstrap intervals for Delta_7_to_14 and Delta_14_to_32."
)

MODEL_REVISIONS_SHA_PATH: Final[str] = f"{ARTIFACT_DIR}/model_revisions.sha256"
INFRASTRUCTURE_LOG_DIR: Final[str] = f"{ARTIFACT_DIR}/infrastructure_logs"

REVISION_PINNING_RULE: Final[str] = (
    "After the pre-flight resolves each checkpoint's model and tokenizer "
    "revision, the full immutable commit SHAs are recorded in "
    "model_revisions.json and that file is checksummed in "
    "model_revisions.sha256. The inference runner refuses to start unless the "
    "checksum verifies, and loads exactly the recorded commits."
)

#: Hashed into the final provenance manifest. Not execution-blocking: neither
#: file computes an outcome.
PROVENANCE_ONLY_CODE_PATHS: Final[tuple[str, ...]] = (
    "scripts/preflight_qwen_scale.py",
    "scripts/write_results_qwen_scale.py",
    "scripts/amend_qwen_scale_preregistration.py",
)

AMENDMENTS: Final[dict[int, dict[str, str]]] = {
    1: {
        "title": "Primary endpoint, case status, revision checksum, provenance",
        "defect": (
            "Four gaps in the original freeze, none affecting any definition or "
            "computation of a metric. (1) Nontrivial TRR was used as the "
            "headline for the interpretation cases but was never named as the "
            "single primary scaling endpoint; seven table rows were all called "
            "'primary'. (2) Cases A-D were stated to be non-exclusive, but not "
            "that they are descriptive flags rather than hypothesis-test "
            "categories or universal thresholds, and the two adjacent scale "
            "steps were not defined with an explicit sign. (3) "
            "model_revisions.json was not checksummed and the runner did not "
            "verify it. (4) The pre-flight and results-writer scripts were not "
            "hashed into any provenance record, and the code gates had no way "
            "to honour an amendment."
        ),
        "outcome_independence": (
            "Created before any FormalCRRC inference with Qwen2.5-7B or "
            "Qwen2.5-32B: the amendment script refuses to run if any result, "
            "partial result, receipt or run record exists. No 7B or 32B outcome "
            "existed to depend on. Four pre-flight submissions had failed on "
            "nodes with unusable GPUs before loading any model; they produced "
            "no model or scientific output."
        ),
        "what_changed": (
            "Declared nontrivial TRR the primary scaling endpoint and TCE, "
            "FSRR, E*, accuracy and MVR supporting endpoints. Declared Cases "
            "A-D non-exclusive descriptive flags and defined Delta_7_to_14 = "
            "TRR_14B - TRR_7B and Delta_14_to_32 = TRR_32B - TRR_14B, which the "
            "analysis now reports with those signs. The pre-flight writes "
            "model_revisions.sha256 and the runner verifies it before loading. "
            "The analysis hashes the pre-flight, results-writer and amendment "
            "scripts into run_manifest.json. The runner, analysis and verifier "
            "read the pinned code manifest from the latest checksummed "
            "amendment. checksums.sha256 now also covers preserved "
            "infrastructure logs. No metric definition, population, contrast, "
            "test, bootstrap setting, case condition, threshold or protocol "
            "element changed."
        ),
    },
}


def amendment_json_path(number: int) -> str:
    return f"{ARTIFACT_DIR}/amendment_{number:03d}.json"


def amendment_sha_path(number: int) -> str:
    return f"{ARTIFACT_DIR}/amendment_{number:03d}.sha256"


def amendment_md_path(number: int) -> str:
    return f"docs/QWEN_SCALE_ABLATION_AMENDMENT_{number:03d}.md"


def results_path(tag: str) -> str:
    """Raw scored rows for one new rung."""
    return f"{ARTIFACT_DIR}/results_{tag}.parquet"


def partial_results_path(tag: str) -> str:
    return f"{ARTIFACT_DIR}/results_{tag}.partial.parquet"


def run_record_path(tag: str) -> str:
    return f"{ARTIFACT_DIR}/results_{tag}_run.json"


def receipts_path(tag: str) -> str:
    """Append-only log, one line per runner invocation."""
    return f"{ARTIFACT_DIR}/results_{tag}_receipts.jsonl"


# --------------------------------------------------------------------------
# Frozen record that must remain byte-identical
# --------------------------------------------------------------------------

PROTECTED_TREES: Final[tuple[str, ...]] = (
    "artifacts/s1",
    "artifacts/s2",
    "artifacts/s3",
    "artifacts/labelswap",
    "artifacts/reasoning_anchor",
    "artifacts/reason_then_score_ablation",
    "artifacts/poststudy_baselines",
    "artifacts/poststudy_theory",
    "figures/s1",
    "figures/s2",
    "figures/s3",
    "figures/labelswap",
    "figures/reasoning_anchor",
    "figures/reason_then_score_ablation",
)

PROTECTED_DOCS: Final[tuple[str, ...]] = (
    "docs/PREREGISTRATION_s1.md",
    "docs/PREREGISTRATION_s2.md",
    "docs/PREREGISTRATION_s3.md",
    "docs/PREREGISTRATION_LABEL_SWAP.md",
    "docs/PREREGISTRATION_REASONING_ANCHOR.md",
    "docs/PREREGISTRATION_REASON_THEN_SCORE_ABLATION.md",
    "docs/RESULTS_s1.md",
    "docs/RESULTS_s2.md",
    "docs/RESULTS_s3.md",
    "docs/RESULTS_LABEL_SWAP.md",
    "docs/RESULTS_REASONING_ANCHOR.md",
    "docs/RESULTS_REASON_THEN_SCORE_ABLATION.md",
    "docs/s1_ERRATA.md",
    "docs/s1_PROVENANCE_AUDIT.md",
    "docs/s1_INTEGRITY_ADDENDUM.md",
    "docs/s2_RATIONALE.md",
    "docs/s3_RATIONALE.md",
    "docs/NOVELTY_BOUNDARIES.md",
    "docs/NOVELTY_BOUNDARIES_s2.md",
    "docs/NOVELTY_BOUNDARIES_s3.md",
    "docs/LINEAGE.md",
    "docs/FORMALCRRC_s1_SPEC.md",
    "docs/POSTSTUDY_BASELINE_AUDIT.md",
)

#: Read by this study and never written.
READ_ONLY_INPUTS: Final[tuple[str, ...]] = (
    "artifacts/s3/test_dataset.parquet",
    "artifacts/s3/test_manifest.json",
    "artifacts/s3/prompt_manifest.parquet",
    "artifacts/s3/model_provenance.json",
    "artifacts/s3/metrics_by_artifact.parquet",
    REFERENCE_RAW_SCORES_PATH,
    REFERENCE_RUN_RECORD_PATH,
)

#: Source files whose content defines this study's outcomes. Hashed at freeze.
#: The first block is frozen prior-study code, reused unchanged.
ANALYSIS_CODE_PATHS: Final[tuple[str, ...]] = (
    "src/formalcrrc/config.py",
    "src/formalcrrc/predicates.py",
    "src/formalcrrc/dataset.py",
    "src/formalcrrc/prompts.py",
    "src/formalcrrc/scoring.py",
    "src/formalcrrc/metrics.py",
    "src/formalcrrc/provenance.py",
    "src/formalcrrc/s2.py",
    "src/formalcrrc/s2_bootstrap.py",
    "src/formalcrrc/s3.py",
    "src/formalcrrc/s3_config.py",
    "src/formalcrrc/rts_ablation.py",
    "src/formalcrrc/rts_ablation_bootstrap.py",
    "src/formalcrrc/rts_ablation_config.py",
    "scripts/run_s3_inference.py",
    "src/formalcrrc/qwen_scale.py",
    "src/formalcrrc/qwen_scale_config.py",
    "scripts/freeze_qwen_scale_preregistration.py",
    "scripts/run_qwen_scale_inference.py",
    "scripts/analyze_qwen_scale.py",
    "scripts/verify_qwen_scale_integrity.py",
)
