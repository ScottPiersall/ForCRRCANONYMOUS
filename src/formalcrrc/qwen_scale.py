"""Metrics, paired uncertainty and trend labels for the Qwen2.5 scale ablation.

Nothing here runs a model. Every existing FormalCRRC quantity -- accuracy, TCE,
MVR, Brier, translation reachability, OTCE, RC -- is taken from the module that
already defines it and is not reimplemented. Two quantities had no function in
this repository before this study and are defined here, exactly as the frozen
preregistration states them:

* **full-sequence recovery** ``FSR_x``: one scalar cut reproduces all nine
  verdicts, i.e. ``min_{Y=1} M > max_{Y=0} M``;
* **oracle minimum errors** ``E*_x``: the fewest of the nine verdicts that stay
  wrong under the best truth-assisted scalar cut.

The cut states are enumerated exhaustively rather than searched: a cut ``c``
changes the predicted verdicts only when it passes a margin value, so the
distinct states are one per unique margin (``c`` equal to that value) plus the
all-fail state (``c`` above every margin). Tied margins necessarily fall on the
same side of any cut, which is the tie handling the definition requires.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from formalcrrc import s3, metrics, provenance
from formalcrrc import qwen_scale_config as qc
from formalcrrc import rts_ablation_bootstrap as rb
from formalcrrc.config import FAMILIES, N_STRICTNESS, NO_CROSSING_INDEX
from formalcrrc.s2_bootstrap import crossing_index
from formalcrrc.rts_ablation import clopper_pearson


def _as_pair(
    margins: Sequence[float], truth: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    curve = np.asarray(margins, dtype=float)
    labels = np.asarray(truth, dtype=int)
    if curve.shape != (N_STRICTNESS,) or labels.shape != (N_STRICTNESS,):
        raise ValueError(
            f"expected {N_STRICTNESS} margins and labels, got {curve.shape} "
            f"and {labels.shape}"
        )
    if not np.all(np.isfinite(curve)):
        raise ValueError("margin curve contains non-finite values")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("truth labels must be 0 or 1")
    return curve, labels


# --------------------------------------------------------------------------
# The two quantities defined by this study's preregistration
# --------------------------------------------------------------------------


def scalar_cut_errors(margins: Sequence[float], truth: Sequence[int]) -> np.ndarray:
    """Verdict errors under every distinct scalar-cut state.

    One entry per unique margin value ``v`` (the cut ``c = v``, predicting met
    exactly where ``M >= v``; the smallest ``v`` is the all-pass state), followed
    by the all-fail state.
    """
    curve, labels = _as_pair(margins, truth)
    cuts = np.unique(curve)
    predicted = (curve[None, :] >= cuts[:, None]).astype(int)
    errors = (predicted != labels[None, :]).sum(axis=1)
    all_fail = int(labels.sum())
    return np.concatenate([errors, [all_fail]]).astype(int)


def oracle_min_errors(margins: Sequence[float], truth: Sequence[int]) -> int:
    """``E*_x``: fewest verdicts left wrong by the best truth-assisted cut."""
    return int(scalar_cut_errors(margins, truth).min())


def full_sequence_recovered(margins: Sequence[float], truth: Sequence[int]) -> bool:
    """``FSR_x``: ``min_{Y=1} M > max_{Y=0} M``, strictly.

    Vacuously true for a constant truth curve; such artifacts are excluded from
    FSRR, which averages over nontrivial artifacts only.
    """
    curve, labels = _as_pair(margins, truth)
    met, not_met = curve[labels == 1], curve[labels == 0]
    if met.size == 0 or not_met.size == 0:
        return True
    return bool(met.min() > not_met.max())


# --------------------------------------------------------------------------
# Frozen partition
# --------------------------------------------------------------------------


def validate_panel(dataset: pd.DataFrame) -> dict[str, Any]:
    """Structural assertions on the frozen S3 partition, from primitive fields."""
    per = dataset.drop_duplicates("artifact_id")
    cells = per.groupby(["family", "latent_level"]).size()
    levels = dataset.groupby("artifact_id")["strictness_index"].agg(
        lambda s: tuple(sorted(int(v) for v in s))
    )
    first_fail = dataset.sort_values(["artifact_id", "strictness_index"]).groupby(
        "artifact_id"
    )["formal_truth"].agg(
        lambda s: next(
            (i for i, v in enumerate(s) if int(v) == 0), NO_CROSSING_INDEX
        )
    )
    stored = per.set_index("artifact_id")["true_first_fail_index"].astype(int)
    checks = {
        "n_rows_is_2916": len(dataset) == qc.N_ROWS,
        "prompt_ids_unique": bool(dataset["prompt_id"].is_unique),
        "n_artifacts_is_324": int(per.shape[0]) == qc.N_ARTIFACTS,
        "three_families": sorted(per["family"].unique()) == sorted(FAMILIES),
        "nine_latent_levels": sorted(int(v) for v in per["latent_level"].unique())
        == list(range(qc.N_LATENT_LEVELS)),
        "twelve_per_cell": len(cells) == qc.N_FAMILIES * qc.N_LATENT_LEVELS
        and set(int(v) for v in cells.unique()) == {qc.N_PER_CELL},
        "every_artifact_has_exactly_9_thresholds": bool(
            (levels == tuple(range(N_STRICTNESS))).all()
        ),
        "n_nontrivial_is_288": int((stored <= N_STRICTNESS - 1).sum())
        == qc.N_NONTRIVIAL,
        "j_star_matches_truth_labels": bool(
            (first_fail.reindex(stored.index).astype(int) == stored).all()
        ),
        "truth_is_a_single_step": bool(
            dataset.sort_values(["artifact_id", "strictness_index"])
            .groupby("artifact_id")["formal_truth"]
            .agg(lambda s: bool(np.all(np.diff(np.asarray(s, dtype=int)) <= 0)))
            .all()
        ),
    }
    return {
        "n_rows": int(len(dataset)),
        "n_artifacts": int(per.shape[0]),
        "n_nontrivial": int((stored <= N_STRICTNESS - 1).sum()),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


TRUTH_COLUMNS: tuple[str, ...] = ("prompt_id", "formal_truth", "true_first_fail_index")


def attach_truth(scores: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    """Join formal truth onto scored rows; refuse anything but a 1:1 match."""
    if not scores["prompt_id"].is_unique:
        raise ValueError("scored rows contain duplicate prompt_id values")
    if set(scores["prompt_id"]) != set(dataset["prompt_id"]):
        raise ValueError("scored prompt_ids do not match the frozen S3 panel")
    merged = scores.merge(
        dataset[list(TRUTH_COLUMNS)], on="prompt_id", how="inner", validate="1:1"
    )
    if len(merged) != qc.N_ROWS:
        raise ValueError(f"expected {qc.N_ROWS} joined rows, got {len(merged)}")
    return merged


# --------------------------------------------------------------------------
# Artifact-level table
# --------------------------------------------------------------------------

ARTIFACT_VALUE_COLUMNS: tuple[str, ...] = (
    "accuracy",
    "tce",
    "translation_reachable",
    "full_sequence_recovered",
    "oracle_min_errors",
    "mvr_05",
    "mvr_0",
    "otce",
    "reachable_count",
    "exact_boundary",
    "brier",
)


def artifact_table(scores: pd.DataFrame, tag: str) -> pd.DataFrame:
    """One row per artifact for one rung, in canonical ``artifact_id`` order.

    ``scores`` carries ``margin``, ``p_met``, ``formal_truth`` and
    ``true_first_fail_index`` for all nine thresholds of every artifact.
    """
    ordered = scores.sort_values(["artifact_id", "strictness_index"])
    rows: list[dict[str, Any]] = []
    for artifact_id, group in ordered.groupby("artifact_id", sort=True):
        levels = group["strictness_index"].to_numpy()
        if not np.array_equal(levels, np.arange(N_STRICTNESS)):
            raise ValueError(f"{artifact_id}: expected strictness 0..8 exactly once")
        margins = group["margin"].to_numpy(dtype=float)
        p_met = group["p_met"].to_numpy(dtype=float)
        truth = group["formal_truth"].to_numpy(dtype=int)
        j_star = int(group["true_first_fail_index"].iloc[0])

        j_hat = int(crossing_index(margins[None, :])[0])
        tce = abs(j_hat - j_star)
        n_correct = int((metrics.binary_predictions(p_met) == truth).sum())
        rows.append(
            {
                "model_tag": tag,
                "model_id": qc.MODEL_IDS[tag],
                "artifact_id": artifact_id,
                "family": group["family"].iloc[0],
                "latent_level": int(group["latent_level"].iloc[0]),
                "true_first_fail_index": j_star,
                "nontrivial": j_star <= N_STRICTNESS - 1,
                "j_hat": j_hat,
                "tce": tce,
                "exact_boundary": tce == 0,
                "n_correct": n_correct,
                "accuracy": metrics.accuracy(p_met, truth),
                "brier": metrics.brier_score(p_met, truth),
                "mvr_05": metrics.monotonicity_violation_rate(
                    p_met, qc.MVR_TOLERANCE
                ),
                "mvr_0": metrics.monotonicity_violation_rate(
                    p_met, qc.MVR_TOLERANCE_STRICT
                ),
                "mvr_05_violations": int(
                    np.sum(p_met[1:] > p_met[:-1] + qc.MVR_TOLERANCE)
                ),
                "mvr_0_violations": int(
                    np.sum(p_met[1:] > p_met[:-1] + qc.MVR_TOLERANCE_STRICT)
                ),
                "translation_reachable": s3.translation_reachable(margins, j_star),
                "full_sequence_recovered": full_sequence_recovered(margins, truth),
                "oracle_min_errors": oracle_min_errors(margins, truth),
                "otce": s3.oracle_translation_tce(margins, j_star),
                "reachable_count": s3.reachable_count(margins),
            }
        )
    table = pd.DataFrame(rows)
    if len(table) != qc.N_ARTIFACTS:
        raise ValueError(f"expected {qc.N_ARTIFACTS} artifacts, got {len(table)}")
    return table


def _population(table: pd.DataFrame, metric: str) -> np.ndarray:
    if qc.METRIC_POPULATION[metric] == "nontrivial":
        return table["nontrivial"].to_numpy(dtype=bool)
    return np.ones(len(table), dtype=bool)


def metric_value(table: pd.DataFrame, metric: str) -> float:
    """Point estimate of one named metric from an artifact-level table."""
    mask = _population(table, metric)
    values = table[qc.METRIC_COLUMN[metric]].to_numpy(dtype=float)
    return float(values[mask].mean())


def summarise_model(table: pd.DataFrame) -> dict[str, Any]:
    """Every primary and secondary point estimate for one rung."""
    nontrivial = table["nontrivial"].to_numpy(dtype=bool)
    out: dict[str, Any] = {
        "n_artifacts": int(len(table)),
        "n_nontrivial": int(nontrivial.sum()),
        "metrics": {
            name: metric_value(table, name)
            for name in qc.PRIMARY_METRICS + qc.SECONDARY_METRICS
        },
        "counts": exact_counts(table),
    }
    return out


def exact_counts(table: pd.DataFrame) -> dict[str, list[int]]:
    """Integer numerators and denominators, for exact comparison."""
    nontrivial = table["nontrivial"].to_numpy(dtype=bool)
    n, n_nt = len(table), int(nontrivial.sum())
    steps = n * (N_STRICTNESS - 1)

    def total(column: str, mask: np.ndarray | None = None) -> int:
        values = table[column].to_numpy()
        return int(values[mask].sum() if mask is not None else values.sum())

    return {
        "accuracy": [total("n_correct"), n * N_STRICTNESS],
        "tce": [total("tce"), n],
        "trr": [total("translation_reachable", nontrivial), n_nt],
        "fsrr": [total("full_sequence_recovered", nontrivial), n_nt],
        "oracle_errors": [total("oracle_min_errors", nontrivial), n_nt],
        "oracle_errors_all": [total("oracle_min_errors"), n],
        "mvr_05": [total("mvr_05_violations"), steps],
        "mvr_0": [total("mvr_0_violations"), steps],
    }


def family_summary(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    """TCE, TRR, FSRR and accuracy for each predicate family."""
    out: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        part = table[table["family"] == family]
        out[family] = {
            "n_artifacts": int(len(part)),
            "n_nontrivial": int(part["nontrivial"].sum()),
            **{name: metric_value(part, name) for name in FAMILY_METRICS},
        }
    return out


FAMILY_METRICS: tuple[str, ...] = ("tce", "trr", "fsrr", "accuracy")


# --------------------------------------------------------------------------
# The frozen 14B reference
# --------------------------------------------------------------------------


def verify_reference(
    table: pd.DataFrame, frozen_s3_metrics: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Check the recomputed 14B rung against every frozen reference value."""
    counts = exact_counts(table)
    checks: dict[str, bool] = {
        f"{name}_count_matches": tuple(counts[name]) == expected
        for name, expected in qc.REFERENCE_COUNTS.items()
    }
    nontrivial = table[table["nontrivial"]]
    checks["n_artifacts_is_324"] = len(table) == qc.N_ARTIFACTS
    checks["n_nontrivial_is_288"] = len(nontrivial) == qc.N_NONTRIVIAL
    checks["fsrr_is_136_over_288"] = (
        int(nontrivial["full_sequence_recovered"].sum()) == 136
        and len(nontrivial) == 288
    )
    checks["fsr_iff_zero_oracle_errors_on_nontrivial"] = bool(
        (
            nontrivial["full_sequence_recovered"].astype(bool)
            == (nontrivial["oracle_min_errors"] == 0)
        ).all()
    )
    checks["full_sequence_recovery_implies_reachability"] = bool(
        (
            ~nontrivial["full_sequence_recovered"].astype(bool)
            | nontrivial["translation_reachable"].astype(bool)
        ).all()
    )
    if frozen_s3_metrics is not None:
        frozen = frozen_s3_metrics.sort_values("artifact_id").reset_index(drop=True)
        ours = table.sort_values("artifact_id").reset_index(drop=True)
        same_ids = frozen["artifact_id"].tolist() == ours["artifact_id"].tolist()
        checks["frozen_s3_artifact_ids_match"] = same_ids
        if same_ids:
            for ours_col, frozen_col in (
                ("tce", "tce_raw"),
                ("j_hat", "j_hat_raw"),
                ("translation_reachable", "translation_reachable"),
                ("otce", "otce"),
                ("reachable_count", "reachable_count"),
            ):
                checks[f"frozen_s3_{frozen_col}_matches"] = bool(
                    (
                        ours[ours_col].to_numpy().astype(int)
                        == frozen[frozen_col].to_numpy().astype(int)
                    ).all()
                )
    return {
        "counts": counts,
        "expected_counts": {k: list(v) for k, v in qc.REFERENCE_COUNTS.items()},
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


# --------------------------------------------------------------------------
# Paired artifact-clustered bootstrap
# --------------------------------------------------------------------------


def align_tables(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Sort every rung identically and refuse any mismatch in the pairing."""
    aligned = {
        tag: frame.sort_values("artifact_id").reset_index(drop=True)
        for tag, frame in tables.items()
    }
    reference = next(iter(aligned.values()))
    for tag, frame in aligned.items():
        for column in ("artifact_id", "family", "true_first_fail_index"):
            if frame[column].tolist() != reference[column].tolist():
                raise ValueError(f"{tag}: {column} is not aligned across models")
    return aligned


def _rate_interval(table: pd.DataFrame, metric: str) -> dict[str, Any] | None:
    """Exact interval for a nontrivial-population rate sitting at 0 or 1."""
    if metric not in ("trr", "fsrr"):
        return None
    numerator, denominator = exact_counts(table)[metric]
    if numerator not in (0, denominator):
        return None
    low, high = clopper_pearson(numerator, denominator, qc.CLOPPER_PEARSON_LEVEL)
    return {
        "method": "clopper_pearson",
        "level": qc.CLOPPER_PEARSON_LEVEL,
        "successes": numerator,
        "trials": denominator,
        "ci_low": low,
        "ci_high": high,
    }


def bootstrap_scale(
    tables: Mapping[str, pd.DataFrame],
    *,
    replicates: int = qc.BOOTSTRAP_REPLICATES,
    seed: int = qc.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Every preregistered interval, from one plan shared by all three rungs."""
    aligned = align_tables(tables)
    reference = aligned[qc.REFERENCE_TAG]
    plan = rb.build_plan(
        reference["artifact_id"].tolist(),
        reference["family"].tolist(),
        replicates=replicates,
        seed=seed,
    )

    def values(tag: str, metric: str) -> np.ndarray:
        return aligned[tag][qc.METRIC_COLUMN[metric]].to_numpy(dtype=float)

    def interval(
        tag: str, metric: str, extra_mask: np.ndarray | None = None
    ) -> dict[str, Any]:
        mask = _population(reference, metric)
        if extra_mask is not None:
            mask = mask & extra_mask
        column = values(tag, metric)
        return rb.summarise(
            rb.replicate_means(column, plan, mask=mask), float(column[mask].mean())
        )

    def contrast(
        first: str, second: str, metric: str, extra_mask: np.ndarray | None = None
    ) -> dict[str, Any]:
        mask = _population(reference, metric)
        if extra_mask is not None:
            mask = mask & extra_mask
        difference = values(first, metric) - values(second, metric)
        block = rb.summarise(
            rb.replicate_means(difference, plan, mask=mask),
            float(difference[mask].mean()),
        )
        block["excludes_zero"] = bool(block["ci_low"] > 0.0 or block["ci_high"] < 0.0)
        return block

    all_metrics = qc.PRIMARY_METRICS + qc.SECONDARY_METRICS
    results: dict[str, Any] = {
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "unit": "artifact_curve",
            "pairing": "all three models' curves for one artifact drawn together",
            "stratified_by": "family",
            "ci": qc.BOOTSTRAP_CI_DESCRIPTION,
        },
        "models": {},
        "paired": {},
        "by_family": {},
    }

    for tag in qc.LADDER:
        block: dict[str, Any] = {}
        for metric in all_metrics:
            block[metric] = interval(tag, metric)
            exact = _rate_interval(aligned[tag], metric)
            if exact is not None:
                block[metric]["exact_interval"] = exact
        results["models"][tag] = block

    for kind, contrasts in (
        ("primary", qc.PRIMARY_CONTRASTS),
        ("secondary", qc.SECONDARY_CONTRASTS),
    ):
        for first, second in contrasts:
            results["paired"][f"{first}_minus_{second}"] = {
                "kind": kind,
                "first": first,
                "second": second,
                "metrics": {
                    metric: contrast(first, second, metric)
                    for metric in qc.PAIRED_PRIMARY_METRICS
                    + qc.PAIRED_SECONDARY_METRICS
                },
            }

    for family in FAMILIES:
        in_family = (reference["family"] == family).to_numpy(dtype=bool)
        results["by_family"][family] = {
            "models": {
                tag: {m: interval(tag, m, in_family) for m in FAMILY_METRICS}
                for tag in qc.LADDER
            },
            "paired": {
                f"{first}_minus_{second}": {
                    m: contrast(first, second, m, in_family) for m in FAMILY_METRICS
                }
                for first, second in qc.PRIMARY_CONTRASTS
            },
        }
    return results


# --------------------------------------------------------------------------
# Paired binary outcomes
# --------------------------------------------------------------------------


def mcnemar_exact(new: Sequence[bool], reference: Sequence[bool]) -> dict[str, Any]:
    """Paired gain/loss counts and the exact two-sided McNemar p-value."""
    a = np.asarray(new, dtype=bool)
    b = np.asarray(reference, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired outcome vectors differ in length")
    gains = int(np.sum(a & ~b))
    losses = int(np.sum(~a & b))
    discordant = gains + losses
    p_value = (
        1.0
        if discordant == 0
        else float(stats.binomtest(gains, discordant, 0.5).pvalue)
    )
    return {
        "n_pairs": int(a.size),
        "both_true": int(np.sum(a & b)),
        "both_false": int(np.sum(~a & ~b)),
        "gains": gains,
        "losses": losses,
        "p_value": p_value,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (p.size - rank) * p[index]))
        adjusted[index] = running
    return [float(v) for v in adjusted]


def mcnemar_family(tables: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """The complete preregistered test family, Holm-corrected."""
    aligned = align_tables(tables)
    reference = aligned[qc.REFERENCE_TAG]
    nontrivial = reference["nontrivial"].to_numpy(dtype=bool)
    tests: list[dict[str, Any]] = []
    for first, second in qc.PRIMARY_CONTRASTS:
        for outcome in qc.MCNEMAR_OUTCOMES:
            result = mcnemar_exact(
                aligned[first][outcome].to_numpy(dtype=bool)[nontrivial],
                aligned[second][outcome].to_numpy(dtype=bool)[nontrivial],
            )
            tests.append({"new": first, "reference": second, "outcome": outcome, **result})
    for test, adjusted in zip(tests, holm_adjust([t["p_value"] for t in tests])):
        test["p_holm"] = adjusted
        test["significant_at_family_alpha"] = bool(adjusted < qc.MCNEMAR_ALPHA)
    return tests


# --------------------------------------------------------------------------
# Scale trend and prespecified interpretation
# --------------------------------------------------------------------------


def trend_label(values: Sequence[float], higher_is_better: bool) -> str:
    """Descriptive three-point label, in ladder order smallest to largest."""
    v = [float(x) for x in values]
    if len(v) != len(qc.LADDER):
        raise ValueError("one value per rung is required")
    steps = np.diff(v)
    if np.all(steps == 0):
        return "flat"
    rising, falling = bool(np.all(steps >= 0)), bool(np.all(steps <= 0))
    if not (rising or falling):
        return "not monotonic"
    improving = rising if higher_is_better else falling
    return "monotonic improvement" if improving else "monotonic decline"


def scale_trends(
    summaries: Mapping[str, Mapping[str, Any]], bootstrap: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Trend label for every primary metric, with adjacent-interval context."""
    lower = bootstrap["paired"][f"{qc.TAG_7B}_minus_{qc.TAG_14B}"]["metrics"]
    upper = bootstrap["paired"][f"{qc.TAG_32B}_minus_{qc.TAG_14B}"]["metrics"]
    out: dict[str, dict[str, Any]] = {}
    for metric in qc.PRIMARY_METRICS:
        values = [summaries[tag]["metrics"][metric] for tag in qc.LADDER]
        out[metric] = {
            "values": dict(zip(qc.LADDER, values)),
            "higher_is_better": metric in qc.HIGHER_IS_BETTER,
            "label": trend_label(values, metric in qc.HIGHER_IS_BETTER),
            "adjacent_7b_14b_excludes_zero": lower.get(metric, {}).get("excludes_zero"),
            "adjacent_14b_32b_excludes_zero": upper.get(metric, {}).get(
                "excludes_zero"
            ),
        }
    return out


def interpretation_cases(
    summaries: Mapping[str, Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    trends: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Which prespecified cases hold. Cases are not mutually exclusive."""
    trr = {tag: summaries[tag]["metrics"]["trr"] for tag in qc.LADDER}
    lower = bootstrap["paired"][f"{qc.TAG_7B}_minus_{qc.TAG_14B}"]["metrics"]["trr"]
    upper = bootstrap["paired"][f"{qc.TAG_32B}_minus_{qc.TAG_14B}"]["metrics"]["trr"]
    holds = {
        "A": (1.0 - trr[qc.TAG_32B]) >= qc.SUBSTANTIAL_UNREACHABLE_MIN,
        "B": bool(
            trr[qc.TAG_7B] < trr[qc.TAG_14B] < trr[qc.TAG_32B]
            and lower["excludes_zero"]
            and upper["excludes_zero"]
        ),
        "C": trr[qc.TAG_32B] >= qc.NEAR_PERFECT_TRR_MIN,
        "D": trends["trr"]["label"] == "not monotonic",
    }
    def reversed_step(block: Mapping[str, Any]) -> dict[str, Any]:
        """A (smaller - larger) contrast re-signed as (larger - smaller)."""
        return {
            "point": -block["point"],
            "ci_low": -block["ci_high"],
            "ci_high": -block["ci_low"],
            "excludes_zero": block["excludes_zero"],
        }

    return {
        "primary_endpoint": qc.PRIMARY_ENDPOINT,
        "case_status": qc.CASE_STATUS,
        "scale_steps": {
            "definitions": qc.SCALE_STEP_DEFINITIONS,
            "Delta_7_to_14": reversed_step(lower),
            "Delta_14_to_32": {
                key: upper[key] for key in ("point", "ci_low", "ci_high", "excludes_zero")
            },
        },
        "trr": trr,
        "trr_trend": trends["trr"]["label"],
        "cases": {
            key: {"holds": bool(value), **qc.INTERPRETATION_CASES[key]}
            for key, value in holds.items()
        },
        "any_case_holds": bool(any(holds.values())),
        "fallback": qc.INTERPRETATION_FALLBACK,
    }


# --------------------------------------------------------------------------
# Preregistration and amendments
# --------------------------------------------------------------------------


def load_amendments(root: Path | None = None) -> list[dict[str, Any]]:
    """Every numbered amendment, in order, each verified against its checksum.

    Numbering must be contiguous from 1. A missing or mismatching checksum is
    an error, never skipped.
    """
    import json

    root = Path(root) if root is not None else provenance.repo_root()
    amendments: list[dict[str, Any]] = []
    number = 1
    while (root / qc.amendment_json_path(number)).is_file():
        path = root / qc.amendment_json_path(number)
        sha_path = root / qc.amendment_sha_path(number)
        if not sha_path.is_file() or not provenance.verify_checksum_file(path, sha_path):
            raise ValueError(f"amendment {number:03d} does not match its checksum")
        document = json.loads(path.read_text(encoding="utf-8"))
        if int(document["number"]) != number:
            raise ValueError(f"amendment file {number:03d} is misnumbered")
        amendments.append(document)
        number += 1
    return amendments


def effective_code_manifest(root: Path | None = None) -> dict[str, Any]:
    """The pinned code hashes currently in force.

    The original freeze, superseded by the latest amendment if any. The
    original preregistration file is never rewritten; it must still verify.
    """
    import json

    root = Path(root) if root is not None else provenance.repo_root()
    prereg_path, sha_path = root / qc.PREREG_JSON_PATH, root / qc.PREREG_SHA_PATH
    if not provenance.verify_checksum_file(prereg_path, sha_path):
        raise ValueError("the original preregistration does not match its checksum")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    original_sha = sha_path.read_text(encoding="utf-8").split()[0]
    amendments = load_amendments(root)
    for amendment in amendments:
        if amendment["preregistration_sha256"] != original_sha:
            raise ValueError(
                f"amendment {amendment['number']:03d} amends a different freeze"
            )
    source = amendments[-1] if amendments else prereg
    return {
        "source": f"amendment {amendments[-1]['number']:03d}" if amendments else "freeze",
        "n_amendments": len(amendments),
        "files": dict(source["analysis_code"]["files"]),
        "tree_sha256": source["analysis_code"]["tree_sha256"],
    }


def changed_pinned_files(root: Path | None = None) -> list[str]:
    """Pinned files whose content differs from the manifest in force."""
    root = Path(root) if root is not None else provenance.repo_root()
    manifest = effective_code_manifest(root)
    return sorted(
        relative
        for relative, digest in manifest["files"].items()
        if not (root / relative).is_file()
        or provenance.sha256_file(root / relative) != digest
    )


# --------------------------------------------------------------------------
# Frozen-record immutability
# --------------------------------------------------------------------------


def _combined(files: Mapping[str, str]) -> str:
    if not files:
        raise ValueError("refusing to compute a combined hash over an empty file list")
    payload = "\n".join(f"{name}:{digest}" for name, digest in sorted(files.items()))
    return provenance.sha256_text(payload)


def _protected_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for tree in qc.PROTECTED_TREES:
        base = root / tree
        if base.is_dir():
            for path in sorted(p for p in base.rglob("*") if p.is_file()):
                files[path.relative_to(root).as_posix()] = provenance.sha256_file(path)
    for relative in qc.PROTECTED_DOCS:
        if (root / relative).is_file():
            files[relative] = provenance.sha256_file(root / relative)
    return files


def build_frozen_fingerprint(root: Path | None = None) -> dict[str, Any]:
    """Fingerprint every completed FormalCRRC record before this study runs."""
    root = Path(root) if root is not None else provenance.repo_root()
    files = _protected_files(root)
    return {
        "created_at": provenance.utc_now(),
        "purpose": (
            "Immutable fingerprint of every completed FormalCRRC scientific "
            "artifact prior to the Qwen2.5 scale ablation, including the frozen "
            "S3 panel and the Qwen2.5-14B reference margins."
        ),
        "trees": list(qc.PROTECTED_TREES),
        "documents": list(qc.PROTECTED_DOCS),
        "n_files": len(files),
        "files": files,
        "combined_sha256": _combined(files),
    }


def verify_frozen_fingerprint(
    fingerprint: Mapping[str, Any], root: Path | None = None
) -> dict[str, Any]:
    """Re-check a stored fingerprint against the working tree."""
    root = Path(root) if root is not None else provenance.repo_root()
    recorded = dict(fingerprint["files"])
    current = _protected_files(root)
    changed = sorted(
        name for name, digest in recorded.items()
        if name in current and current[name] != digest
    )
    missing = sorted(name for name in recorded if name not in current)
    added = sorted(name for name in current if name not in recorded)
    return {
        "checked_at": provenance.utc_now(),
        "n_recorded": len(recorded),
        "combined_sha256_before": fingerprint["combined_sha256"],
        "combined_sha256_after": _combined(current),
        "changed": changed,
        "missing": missing,
        "added": added,
        "status": "PASS" if not (changed or missing or added) else "FAIL",
    }
