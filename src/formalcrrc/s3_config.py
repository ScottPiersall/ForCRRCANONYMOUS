"""Preregistered constants for FormalCRRC s 3.

Kept in its own module so s 3 adds no constant to the s-1 or s-2
configurations. Everything here is serialised verbatim into
``artifacts/s3/preregistration.json`` and may not change after the freeze
except through a numbered, checksummed amendment.

s 3 reuses the s-1 predicate families, strictness ladder, prompt template,
answer labels, decision threshold, model panel and scoring rules unchanged.
"""

from __future__ import annotations

from typing import Final

from formalcrrc.config import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED  # noqa: F401

# --------------------------------------------------------------------------
# Partition
# --------------------------------------------------------------------------

#: Human-readable name of the single s-3 evaluation partition.
PARTITION_NAME: Final[str] = "s3_TEST"

#: Namespace prefix carried in artifact_id and prompt_id. Never model-facing.
PARTITION_TAG: Final[str] = "D3T"

#: Master seed for the s-3 evaluation set.
SEED: Final[int] = 45

# --------------------------------------------------------------------------
# Inference ordering
# --------------------------------------------------------------------------

INFERENCE_ORDER_SALT: Final[str] = "formalcrrc-s3-inference-order-v1"
MODEL_ORDER_SALT: Final[str] = "formalcrrc-s3-model-order-v1"

# --------------------------------------------------------------------------
# The translation class
# --------------------------------------------------------------------------

#: The one and only transformation s 3 admits. Nothing stronger is tested.
TRANSLATION_CLASS: Final[str] = "M'(s) = M(s) + alpha"

#: Reachability condition for a non-extreme crossing j in 1..8.
REACHABILITY_CONDITION: Final[str] = "M(j) < min_{t < j} M(t)  (strict prefix record low)"

#: Crossings always reachable regardless of curve shape.
ALWAYS_REACHABLE: Final[tuple[int, ...]] = (0, 9)

#: Oracle classes, nested G subset F subset A.
ORACLE_CLASSES: Final[tuple[str, ...]] = ("global", "family", "artifact")

#: Tie-breaking for the enumerated group oracles, applied in this order.
ORACLE_TIE_BREAK: Final[tuple[str, ...]] = (
    "lowest mean TCE",
    "then smallest |alpha|",
    "then numerically smallest alpha",
)

#: How candidate intercepts are enumerated for a group oracle.
ORACLE_ENUMERATION: Final[str] = (
    "collect breakpoints -M_x(s) over the group; sort unique; evaluate every "
    "breakpoint, the midpoint of every adjacent unique pair, one value below "
    "the smallest and one above the largest, plus nextafter(b, +/-inf) for "
    "every breakpoint so that no interval is skipped when adjacent breakpoints "
    "are closer than the midpoint is representable"
)

#: Margin equality is handled by the strict inequality alone; no tolerance.
TIE_POLICY: Final[str] = (
    "Exact stored float64 comparison under the strict relation "
    "M(j) < min_{t<j} M(t). No tolerance is introduced. Tie frequencies are "
    "recorded descriptively and the rule is not changed after observing them."
)

# --------------------------------------------------------------------------
# Descriptive taxonomy thresholds, fixed before inference
# --------------------------------------------------------------------------

TAXONOMY: Final[dict[str, str]] = {
    "translation_solvable": "OTCE = 0",
    "translation_near_solvable": "OTCE = 1",
    "translation_limited": "OTCE > 1",
    "order_rich": "RC >= 8",
    "order_poor": "RC <= 3",
}

ORDER_RICH_MIN_RC: Final[int] = 8
ORDER_POOR_MAX_RC: Final[int] = 3

# --------------------------------------------------------------------------
# Prior-experiment immutability
# --------------------------------------------------------------------------

PRIOR_TREES: Final[tuple[str, ...]] = (
    "artifacts/s1",
    "artifacts/s2",
    "figures/s1",
    "figures/s2",
)

PRIOR_DOCS: Final[tuple[str, ...]] = (
    "docs/PREREGISTRATION_s1.md",
    "docs/RESULTS_s1.md",
    "docs/PREREGISTRATION_s2.md",
    "docs/RESULTS_s2.md",
    "docs/s1_ERRATA.md",
    "docs/s1_PROVENANCE_AUDIT.md",
    "docs/s1_INTEGRITY_ADDENDUM.md",
    "docs/s2_RATIONALE.md",
    "docs/NOVELTY_BOUNDARIES.md",
    "docs/NOVELTY_BOUNDARIES_s2.md",
    "docs/LINEAGE.md",
    "docs/FORMALCRRC_s1_SPEC.md",
)

#: Content hashes that must keep reproducing, asserted by the test suite.
s1_DATASET_CONTENT_SHA256: Final[str] = (
    "4462bedbf5f086daa8bdfa077ad82e1ea7b79b4db59d906fafe6c814cfc85a2e"
)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ARTIFACT_DIR: Final[str] = "artifacts/s3"
FIGURE_DIR: Final[str] = "figures/s3"

PRIOR_MANIFEST_PATH: Final[str] = f"{ARTIFACT_DIR}/prior_experiments_manifest.json"
DATASET_PATH: Final[str] = f"{ARTIFACT_DIR}/test_dataset.parquet"
DATASET_MANIFEST_PATH: Final[str] = f"{ARTIFACT_DIR}/test_manifest.json"
PROMPT_MANIFEST_PATH: Final[str] = f"{ARTIFACT_DIR}/prompt_manifest.parquet"

PREREG_JSON_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration.json"
PREREG_SHA_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration.sha256"
MODEL_PROVENANCE_PATH: Final[str] = f"{ARTIFACT_DIR}/model_provenance.json"

RAW_SCORES_DIR: Final[str] = f"{ARTIFACT_DIR}/raw_scores"
REACHABLE_CROSSINGS_PATH: Final[str] = f"{ARTIFACT_DIR}/reachable_crossings.parquet"
ORACLE_GLOBAL_PATH: Final[str] = f"{ARTIFACT_DIR}/oracle_global.json"
ORACLE_FAMILY_PATH: Final[str] = f"{ARTIFACT_DIR}/oracle_family.json"
ORACLE_ARTIFACT_PATH: Final[str] = f"{ARTIFACT_DIR}/oracle_artifact.parquet"
METRICS_PATH: Final[str] = f"{ARTIFACT_DIR}/metrics_by_artifact.parquet"
BOOTSTRAP_PATH: Final[str] = f"{ARTIFACT_DIR}/bootstrap_results.parquet"
SUMMARY_PATH: Final[str] = f"{ARTIFACT_DIR}/summary.json"
INTEGRITY_MANIFEST_PATH: Final[str] = f"{ARTIFACT_DIR}/integrity_manifest.json"
FINAL_CONSOLE_PATH: Final[str] = f"{ARTIFACT_DIR}/final_console_block.txt"

#: Source files whose content defines s-3 outcomes. The preregistration and
#: the integrity manifest hash exactly this list, so the two digests compare.
ANALYSIS_CODE_PATHS: Final[tuple[str, ...]] = (
    "src/formalcrrc/config.py",
    "src/formalcrrc/predicates.py",
    "src/formalcrrc/dataset.py",
    "src/formalcrrc/prompts.py",
    "src/formalcrrc/scoring.py",
    "src/formalcrrc/provenance.py",
    "src/formalcrrc/s2.py",
    "src/formalcrrc/s3.py",
    "src/formalcrrc/s3_bootstrap.py",
    "src/formalcrrc/s3_config.py",
    "scripts/generate_s3_dataset.py",
    "scripts/run_s3_inference.py",
    "scripts/compute_s3_oracle.py",
    "scripts/analyze_s3.py",
    "scripts/figures_s3.py",
    "scripts/verify_s3_integrity.py",
)
