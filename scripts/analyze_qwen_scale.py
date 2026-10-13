#!/usr/bin/env python
"""Frozen analysis for the Qwen2.5 same-family scale ablation.

Runs once, after both new rungs are complete. It refuses to start unless the
preregistration verifies, the analysis code is byte-identical to what the
preregistration hashed, both new result files hold exactly 2,916 scored
conditions, and the frozen 14B rung reproduces every reference value from its
saved margins. There is no partial mode: 7B cannot be analysed before 32B exists.

The 14B column is always recomputed from ``artifacts/day3`` margins. No 14B
number is typed in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import day3_config as d3c  # noqa: E402
from formalcrrc import provenance  # noqa: E402
from formalcrrc import qwen_scale as qs  # noqa: E402
from formalcrrc import qwen_scale_config as qc  # noqa: E402
from formalcrrc.config import FAMILIES, N_STRICTNESS  # noqa: E402

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "tce": "TCE",
    "trr": "TRR",
    "fsrr": "FSRR",
    "oracle_errors": "Oracle errors E*",
    "mvr_05": "MVR (delta=.05)",
    "mvr_0": "MVR (delta=0)",
    "oracle_errors_all": "Oracle errors E* (all 324)",
    "otce": "OTCE",
    "reachable_count": "Reachable-set size RC",
    "exact_boundary_rate": "Exact-boundary rate",
    "brier": "Brier",
}

TEX_LABELS = {
    "accuracy": "Accuracy",
    "tce": "TCE $\\downarrow$",
    "trr": "TRR $\\uparrow$",
    "fsrr": "FSRR $\\uparrow$",
    "oracle_errors": "Oracle errors $E^{*}$ $\\downarrow$",
    "mvr_05": "MVR ($\\delta=.05$) $\\downarrow$",
    "mvr_0": "MVR ($\\delta=0$) $\\downarrow$",
}


def gate(root: Path) -> dict:
    problems: list[str] = []
    prereg_path, sha_path = root / qc.PREREG_JSON_PATH, root / qc.PREREG_SHA_PATH
    if not (prereg_path.is_file() and sha_path.is_file()):
        print("GATE FAIL: the preregistration is not frozen", flush=True)
        raise SystemExit(2)
    if not provenance.verify_checksum_file(prereg_path, sha_path):
        problems.append("the preregistration checksum does not match")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    try:
        in_force = qs.effective_code_manifest(root)
        for relative in qs.changed_pinned_files(root):
            problems.append(f"{relative} differs from the {in_force['source']} manifest")
    except ValueError as error:
        problems.append(str(error))
    if not provenance.verify_checksum_file(
        root / qc.MODEL_REVISIONS_PATH, root / qc.MODEL_REVISIONS_SHA_PATH
    ):
        problems.append("model_revisions.json does not match its checksum")
    for tag in qc.NEW_TAGS:
        if not (root / qc.results_path(tag)).is_file():
            problems.append(f"{qc.results_path(tag)} is missing; both rungs must be complete")
            continue
        run = json.loads((root / qc.run_record_path(tag)).read_text(encoding="utf-8"))
        if run.get("status") != "COMPLETE":
            problems.append(f"{tag} run status is {run.get('status')!r}")
        if provenance.sha256_file(root / qc.results_path(tag)) != run.get("raw_scores_sha256"):
            problems.append(f"{tag} result file does not match its run record")
    if problems:
        for problem in problems:
            print(f"GATE FAIL: {problem}", flush=True)
        raise SystemExit(2)
    return prereg


def row_integrity(tag: str, raw: pd.DataFrame, dataset: pd.DataFrame, frozen: pd.DataFrame) -> dict:
    """Row-level checks on one new rung against the frozen panel and 14B run."""
    by_id = raw.set_index("prompt_id")
    frozen_by_id = frozen.set_index("prompt_id")
    levels = raw.groupby("artifact_id")["strictness_index"].agg(
        lambda s: tuple(sorted(int(v) for v in s))
    )
    checks = {
        "exactly_2916_rows": len(raw) == qc.N_ROWS,
        "no_duplicate_rows": bool(raw["prompt_id"].is_unique),
        "prompt_ids_match_s3_panel": set(raw["prompt_id"]) == set(dataset["prompt_id"]),
        "artifact_ids_match_s3_panel": set(raw["artifact_id"])
        == set(dataset["artifact_id"]),
        "nine_thresholds_per_artifact": bool(
            (levels == tuple(range(N_STRICTNESS))).all()
        )
        and len(levels) == qc.N_ARTIFACTS,
        "all_scores_finite": bool(
            np.isfinite(raw[["raw_score_met", "raw_score_not_met", "margin", "p_met"]])
            .to_numpy()
            .all()
        ),
        "margin_is_met_minus_not_met": bool(
            (raw["margin"] == raw["raw_score_met"] - raw["raw_score_not_met"]).all()
        ),
        "decision_agrees_with_margin_sign": bool(
            ((raw["p_met"] >= 0.5) == (raw["margin"] >= 0.0)).all()
        ),
        "single_scoring_method": set(raw["scoring_method"])
        == {qc.REQUIRED_SCORING_METHOD},
        "single_revision": raw["revision"].nunique() == 1,
        "rendered_prompts_byte_identical_to_14b": bool(
            (
                by_id["rendered_prompt_sha256"]
                == frozen_by_id["rendered_prompt_sha256"].reindex(by_id.index)
            ).all()
        ),
        "prompt_token_counts_identical_to_14b": bool(
            (
                by_id["n_prompt_tokens"]
                == frozen_by_id["n_prompt_tokens"].reindex(by_id.index)
            ).all()
        ),
        "inference_order_identical_to_14b": bool(
            (by_id["order_index"] == frozen_by_id["order_index"].reindex(by_id.index)).all()
        ),
    }
    return {"checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}


def _fmt(value: float, metric: str) -> str:
    if metric == "accuracy":
        return f"{100 * value:.1f}\\%"
    return f"{value:.3f}"


def write_tex(root: Path, summaries: dict, bootstrap: dict) -> None:
    lines = [
        "% Generated by scripts/analyze_qwen_scale.py from",
        f"% {qc.SUMMARY_PATH}. Do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Same-family scale ablation on the identical S3 panel under the "
        "immediate-readout protocol. The Qwen2.5-14B column is the existing frozen "
        "S3 result; 7B and 32B were run post-study under a prospectively frozen "
        "analysis plan. Brackets give paired artifact-clustered bootstrap 95\\% "
        "intervals (5{,}000 replicates, stratified by predicate family). TRR, FSRR "
        "and $E^{*}$ are over the 288 artifacts with nonconstant truth; the "
        "remaining rows are over all 324.}",
        "\\label{tab:qwen-scale-ablation}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        " & " + " & ".join(qc.MODEL_SHORT_NAMES[t] for t in qc.LADDER) + " \\\\",
        "\\midrule",
    ]
    for metric in qc.PRIMARY_METRICS:
        cells = []
        for tag in qc.LADDER:
            block = bootstrap["models"][tag][metric]
            exact = block.get("exact_interval")
            low = exact["ci_low"] if exact else block["ci_low"]
            high = exact["ci_high"] if exact else block["ci_high"]
            point = summaries[tag]["metrics"][metric]
            cells.append(
                f"{_fmt(point, metric)} {{\\scriptsize[{_fmt(low, metric)}, "
                f"{_fmt(high, metric)}]}}"
            )
        lines.append(f"{TEX_LABELS[metric]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path = root / qc.TEX_TABLE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = Path(args.root)

    prereg = gate(root)
    out_dir = root / qc.ARTIFACT_DIR

    dataset = pd.read_parquet(root / d3c.DATASET_PATH)
    panel = qs.validate_panel(dataset)
    if panel["status"] != "PASS":
        print(f"STOP: S3 panel validation failed: {panel['checks']}", flush=True)
        return 2

    frozen_raw = pd.read_parquet(root / qc.REFERENCE_RAW_SCORES_PATH)
    raw: dict[str, pd.DataFrame] = {qc.REFERENCE_TAG: frozen_raw}
    integrity: dict[str, Any] = {}
    for tag in qc.NEW_TAGS:
        raw[tag] = pd.read_parquet(root / qc.results_path(tag))
        integrity[tag] = row_integrity(tag, raw[tag], dataset, frozen_raw)
        if integrity[tag]["status"] != "PASS":
            print(f"STOP: row integrity failed for {tag}: {integrity[tag]['checks']}", flush=True)
            return 2

    tables = {
        tag: qs.artifact_table(qs.attach_truth(raw[tag], dataset), tag)
        for tag in qc.LADDER
    }

    frozen_metrics = pd.read_parquet(root / d3c.METRICS_PATH)
    frozen_metrics = frozen_metrics[
        frozen_metrics["model_id"] == qc.MODEL_IDS[qc.REFERENCE_TAG]
    ]
    reference = qs.verify_reference(tables[qc.REFERENCE_TAG], frozen_metrics)
    if reference["status"] != "PASS":
        print(f"STOP: 14B reference does not reproduce: {reference['checks']}", flush=True)
        return 2

    summaries = {tag: qs.summarise_model(tables[tag]) for tag in qc.LADDER}
    families = {tag: qs.family_summary(tables[tag]) for tag in qc.LADDER}
    bootstrap = qs.bootstrap_scale(tables)
    tests = qs.mcnemar_family(tables)
    trends = qs.scale_trends(summaries, bootstrap)
    cases = qs.interpretation_cases(summaries, bootstrap, trends)

    identities = {
        tag: {
            "fsr_iff_zero_oracle_errors_on_nontrivial": bool(
                (
                    t.loc[t["nontrivial"], "full_sequence_recovered"].astype(bool)
                    == (t.loc[t["nontrivial"], "oracle_min_errors"] == 0)
                ).all()
            ),
            "full_sequence_recovery_implies_reachability": bool(
                (
                    ~t.loc[t["nontrivial"], "full_sequence_recovered"].astype(bool)
                    | t.loc[t["nontrivial"], "translation_reachable"].astype(bool)
                ).all()
            ),
            "otce_le_tce": bool((t["otce"] <= t["tce"]).all()),
        }
        for tag, t in tables.items()
    }

    # ---- artifact-level table -------------------------------------------
    pd.concat([tables[tag] for tag in qc.LADDER], ignore_index=True).to_csv(
        root / qc.ARTIFACT_METRICS_PATH, index=False, lineterminator="\n"
    )

    # ---- combined primary + secondary table ------------------------------
    rows = []
    for metric in qc.PRIMARY_METRICS + qc.SECONDARY_METRICS:
        row: dict[str, Any] = {
            "metric": metric,
            "label": METRIC_LABELS[metric],
            "role": "primary" if metric in qc.PRIMARY_METRICS else "secondary",
            "population": qc.METRIC_POPULATION[metric],
            "trend": trends[metric]["label"] if metric in trends else "",
        }
        for tag in qc.LADDER:
            block = bootstrap["models"][tag][metric]
            row[f"{tag}"] = summaries[tag]["metrics"][metric]
            row[f"{tag}_ci_low"] = block["ci_low"]
            row[f"{tag}_ci_high"] = block["ci_high"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        root / qc.COMBINED_RESULTS_PATH, index=False, lineterminator="\n"
    )

    # ---- paired effects ---------------------------------------------------
    paired_rows = []
    for name, block in bootstrap["paired"].items():
        for metric, effect in block["metrics"].items():
            paired_rows.append(
                {
                    "contrast": name,
                    "kind": block["kind"],
                    "metric": metric,
                    "role": "primary"
                    if metric in qc.PAIRED_PRIMARY_METRICS
                    else "secondary",
                    "delta": effect["point"],
                    "ci_low": effect["ci_low"],
                    "ci_high": effect["ci_high"],
                    "excludes_zero": effect["excludes_zero"],
                }
            )
    pd.DataFrame(paired_rows).to_csv(
        root / qc.PAIRED_EFFECTS_PATH, index=False, lineterminator="\n"
    )

    # ---- family results ---------------------------------------------------
    family_rows = []
    for family in FAMILIES:
        for metric in qs.FAMILY_METRICS:
            row = {"family": family, "metric": metric}
            for tag in qc.LADDER:
                block = bootstrap["by_family"][family]["models"][tag][metric]
                row[tag] = families[tag][family][metric]
                row[f"{tag}_ci_low"] = block["ci_low"]
                row[f"{tag}_ci_high"] = block["ci_high"]
            for name, effects in bootstrap["by_family"][family]["paired"].items():
                row[f"{name}"] = effects[metric]["point"]
                row[f"{name}_ci_low"] = effects[metric]["ci_low"]
                row[f"{name}_ci_high"] = effects[metric]["ci_high"]
            row["trend"] = qs.trend_label(
                [families[tag][family][metric] for tag in qc.LADDER],
                metric in qc.HIGHER_IS_BETTER,
            )
            family_rows.append(row)
    pd.DataFrame(family_rows).to_csv(
        root / qc.FAMILY_RESULTS_PATH, index=False, lineterminator="\n"
    )

    # ---- run manifest -----------------------------------------------------
    reference_run = json.loads(
        (root / qc.REFERENCE_RUN_RECORD_PATH).read_text(encoding="utf-8")
    )
    runs: dict[str, Any] = {
        qc.REFERENCE_TAG: {
            "role": "existing frozen S3 result; not rerun",
            "source": qc.REFERENCE_RAW_SCORES_PATH,
            "model_id": reference_run["model_id"],
            "revision": reference_run["revision"],
            "status": "COMPLETE",
            "completeness": {
                "planned": reference_run["prompts_requested"],
                "completed": reference_run["prompts_scored"],
                "failed": len(reference_run["failures"]),
                "missing": 0,
                "retried": 0,
            },
            "total_seconds": reference_run["total_seconds"],
            "slurm": reference_run["slurm"],
            "environment": reference_run["environment"],
            "raw_scores_sha256": reference_run["raw_scores_sha256"],
        }
    }
    for tag in qc.NEW_TAGS:
        run = json.loads((root / qc.run_record_path(tag)).read_text(encoding="utf-8"))
        runs[tag] = {
            "role": "new rung, run under this preregistration",
            "source": qc.results_path(tag),
            **{
                key: run[key]
                for key in (
                    "model_id",
                    "revision",
                    "tokenizer_revision",
                    "status",
                    "dtype",
                    "quantization",
                    "batch_size",
                    "scoring_method",
                    "label_tokenization",
                    "chat_template_sha256",
                    "load_seconds",
                    "total_seconds",
                    "peak_gpu_memory_gib",
                    "invocations",
                    "slurm",
                    "environment",
                    "raw_scores_sha256",
                    "failures",
                )
            },
            "completeness": {**run["completeness"], "retried": run["retried"]},
            "row_integrity": integrity[tag],
        }
    provenance.write_json(
        root / qc.RUN_MANIFEST_PATH,
        {
            "study": qc.STUDY_NAME,
            "created_at": provenance.utc_now(),
            "preregistration_sha256": (root / qc.PREREG_SHA_PATH)
            .read_text(encoding="utf-8")
            .split()[0],
            "run_order": list(qc.RUN_ORDER),
            "model_revisions_sha256": provenance.sha256_file(
                root / qc.MODEL_REVISIONS_PATH
            ),
            "pinned_code": qs.effective_code_manifest(root),
            "provenance_only_code": {
                "note": (
                    "Hashed for the record. Not execution-blocking: none of "
                    "these files computes an outcome."
                ),
                **provenance.source_manifest(qc.PROVENANCE_ONLY_CODE_PATHS, root),
            },
            "runs": runs,
        },
    )

    # ---- summary ----------------------------------------------------------
    provenance.write_json(
        root / qc.SUMMARY_PATH,
        {
            "study": qc.STUDY_NAME,
            "study_type": qc.STUDY_TYPE,
            "created_at": provenance.utc_now(),
            "analysis_run": "single frozen run",
            "preregistration_sha256": (root / qc.PREREG_SHA_PATH)
            .read_text(encoding="utf-8")
            .split()[0],
            "preregistration_frozen_at": prereg["frozen_at"],
            "amendments": [
                {
                    "number": a["number"],
                    "title": a["title"],
                    "created_at": a["created_at"],
                    "sha256": provenance.sha256_file(
                        root / qc.amendment_json_path(a["number"])
                    ),
                }
                for a in qs.load_amendments(root)
            ],
            "primary_endpoint": {
                "metric": qc.PRIMARY_ENDPOINT,
                "supporting": list(qc.SUPPORTING_ENDPOINTS),
                "statement": qc.PRIMARY_ENDPOINT_STATEMENT,
            },
            "panel": panel,
            "models": {
                tag: {
                    "model_id": qc.MODEL_IDS[tag],
                    "short_name": qc.MODEL_SHORT_NAMES[tag],
                    "revision": runs[tag]["revision"],
                    **summaries[tag],
                    "by_family": families[tag],
                    "identities": identities[tag],
                }
                for tag in qc.LADDER
            },
            "reference_verification": reference,
            "row_integrity": integrity,
            "bootstrap": bootstrap,
            "mcnemar": {"family": qc.MCNEMAR_FAMILY, "tests": tests},
            "trends": trends,
            "interpretation": cases,
        },
    )

    write_tex(root, summaries, bootstrap)

    print("ANALYSIS COMPLETE", flush=True)
    print(f"wrote {out_dir.relative_to(root).as_posix()}/ and {qc.TEX_TABLE_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
