#!/usr/bin/env python
"""Integrity audit for the Qwen2.5 scale ablation.

Recomputes every artifact-level quantity for all three rungs **without** using
``formalcrrc.qwen_scale``, ``formalcrrc.day3`` or ``formalcrrc.metrics``, straight
from saved margins and the dataset's truth labels, and requires agreement with
the primary analysis output. The recomputation deliberately takes different
routes: reachability by brute-force intercept enumeration rather than the
prefix-record theorem, oracle errors over a redundant superset of cuts rather
than the minimal state list, and decisions from the margin sign rather than the
stored probability.

Also re-verifies that no frozen record changed, and writes ``checksums.sha256``
over every final machine-readable result file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import day3_config as d3c  # noqa: E402
from formalcrrc import provenance  # noqa: E402
from formalcrrc import qwen_scale as qs  # noqa: E402  (fingerprint + panel only)
from formalcrrc import qwen_scale_config as qc  # noqa: E402

N = 9


def _sigmoid(delta: float) -> float:
    if delta >= 0.0:
        return 1.0 / (1.0 + math.exp(-delta))
    e = math.exp(delta)
    return e / (1.0 + e)


def _first_negative(curve: np.ndarray) -> int:
    for s in range(N):
        if curve[s] < 0.0:
            return s
    return N


def _reachable_by_enumeration(curve: np.ndarray) -> set[int]:
    points = sorted(set(-curve))
    candidates = list(points)
    candidates += [0.5 * (a + b) for a, b in zip(points, points[1:])]
    candidates += [float(np.nextafter(p, np.inf)) for p in points]
    candidates += [float(np.nextafter(p, -np.inf)) for p in points]
    candidates += [points[0] - 1.0, points[-1] + 1.0]
    return {_first_negative(curve + alpha) for alpha in candidates}


def _min_cut_errors(curve: np.ndarray, truth: np.ndarray) -> int:
    values = sorted(set(curve))
    cuts = list(values)
    cuts += [0.5 * (a + b) for a, b in zip(values, values[1:])]
    cuts += [values[0] - 1.0, values[-1] + 1.0]
    return min(int(np.sum((curve >= c).astype(int) != truth)) for c in cuts)


def independent_table(raw: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    truth_by_prompt = dataset.set_index("prompt_id")["formal_truth"].astype(int)
    rows = []
    for artifact_id, group in raw.sort_values("strictness_index").groupby(
        "artifact_id", sort=True
    ):
        curve = (group["raw_score_met"] - group["raw_score_not_met"]).to_numpy(float)
        truth = truth_by_prompt.reindex(group["prompt_id"]).to_numpy(int)
        j_star = next((s for s in range(N) if truth[s] == 0), N)
        j_hat = _first_negative(curve)
        p = np.array([_sigmoid(float(m)) for m in curve])
        reachable = _reachable_by_enumeration(curve)
        e_star = _min_cut_errors(curve, truth)
        met, not_met = curve[truth == 1], curve[truth == 0]
        rows.append(
            {
                "artifact_id": artifact_id,
                "true_first_fail_index": j_star,
                "nontrivial": j_star <= N - 1,
                "j_hat": j_hat,
                "tce": abs(j_hat - j_star),
                "n_correct": int(np.sum((curve >= 0.0).astype(int) == truth)),
                "brier": float(np.mean((p - truth) ** 2)),
                "mvr_05_violations": int(sum(p[s + 1] > p[s] + 0.05 for s in range(N - 1))),
                "mvr_0_violations": int(sum(p[s + 1] > p[s] + 0.0 for s in range(N - 1))),
                "translation_reachable": j_star in reachable,
                "full_sequence_recovered": bool(
                    met.size == 0 or not_met.size == 0 or met.min() > not_met.max()
                ),
                "oracle_min_errors": e_star,
                "otce": min(abs(j - j_star) for j in reachable),
                "reachable_count": len(reachable),
            }
        )
    return pd.DataFrame(rows)


INT_COLUMNS = (
    "true_first_fail_index",
    "j_hat",
    "tce",
    "n_correct",
    "mvr_05_violations",
    "mvr_0_violations",
    "oracle_min_errors",
    "otce",
    "reachable_count",
)
BOOL_COLUMNS = ("nontrivial", "translation_reachable", "full_sequence_recovered")


def compare(primary: pd.DataFrame, independent: pd.DataFrame) -> dict[str, bool]:
    a = primary.sort_values("artifact_id").reset_index(drop=True)
    b = independent.sort_values("artifact_id").reset_index(drop=True)
    checks = {"artifact_ids": a["artifact_id"].tolist() == b["artifact_id"].tolist()}
    if not checks["artifact_ids"]:
        return checks
    for column in INT_COLUMNS:
        checks[column] = bool((a[column].astype(int) == b[column].astype(int)).all())
    for column in BOOL_COLUMNS:
        checks[column] = bool((a[column].astype(bool) == b[column].astype(bool)).all())
    checks["brier"] = bool(np.allclose(a["brier"], b["brier"], rtol=0, atol=1e-12))
    checks["fsr_iff_zero_oracle_errors"] = bool(
        (
            b.loc[b["nontrivial"], "full_sequence_recovered"]
            == (b.loc[b["nontrivial"], "oracle_min_errors"] == 0)
        ).all()
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = Path(args.root)

    report: dict[str, Any] = {"study": qc.STUDY_NAME, "checked_at": provenance.utc_now()}
    dataset = pd.read_parquet(root / d3c.DATASET_PATH)
    manifest = json.loads((root / d3c.DATASET_MANIFEST_PATH).read_text(encoding="utf-8"))
    primary = pd.read_csv(root / qc.ARTIFACT_METRICS_PATH)
    summary = json.loads((root / qc.SUMMARY_PATH).read_text(encoding="utf-8"))
    prereg = json.loads((root / qc.PREREG_JSON_PATH).read_text(encoding="utf-8"))

    report["preregistration"] = {
        "checksum_verifies": provenance.verify_checksum_file(
            root / qc.PREREG_JSON_PATH, root / qc.PREREG_SHA_PATH
        ),
        "frozen_at": prereg["frozen_at"],
        "amendments": len(qs.load_amendments(root)),
        "pinned_code_source": qs.effective_code_manifest(root)["source"],
        "analysis_code_unchanged": not qs.changed_pinned_files(root),
        "model_revisions_checksum_verifies": provenance.verify_checksum_file(
            root / qc.MODEL_REVISIONS_PATH, root / qc.MODEL_REVISIONS_SHA_PATH
        ),
        "runs_used_pinned_revisions": all(
            json.loads((root / qc.run_record_path(tag)).read_text(encoding="utf-8"))[
                "revision"
            ]
            == json.loads((root / qc.MODEL_REVISIONS_PATH).read_text(encoding="utf-8"))[
                "models"
            ][tag]["revision"]
            for tag in qc.NEW_TAGS
        ),
        "frozen_copy_matches_document": provenance.sha256_file(root / qc.PREREG_MD_PATH)
        == provenance.sha256_file(root / qc.PREREG_FROZEN_COPY_PATH),
    }

    report["panel"] = qs.validate_panel(dataset)
    report["dataset"] = {
        "dataset_file_matches_manifest": provenance.sha256_file(root / d3c.DATASET_PATH)
        == manifest["hashes"]["dataset_file_sha256"],
        "prompt_manifest_matches_manifest": provenance.sha256_file(
            root / d3c.PROMPT_MANIFEST_PATH
        )
        == manifest["hashes"]["prompt_manifest_file_sha256"],
        "frozen_14b_raw_scores_unchanged": provenance.sha256_file(
            root / qc.REFERENCE_RAW_SCORES_PATH
        )
        == qc.REFERENCE_RAW_SCORES_SHA256,
    }

    sources = {qc.REFERENCE_TAG: qc.REFERENCE_RAW_SCORES_PATH} | {
        tag: qc.results_path(tag) for tag in qc.NEW_TAGS
    }
    report["independent_recomputation"] = {}
    for tag, source in sources.items():
        raw = pd.read_parquet(root / source)
        checks = compare(primary[primary["model_tag"] == tag], independent_table(raw, dataset))
        checks["truth_and_j_star_match_s3"] = bool(
            (
                primary[primary["model_tag"] == tag]
                .set_index("artifact_id")["true_first_fail_index"]
                .sort_index()
                == dataset.drop_duplicates("artifact_id")
                .set_index("artifact_id")["true_first_fail_index"]
                .sort_index()
            ).all()
        )
        report["independent_recomputation"][tag] = {
            "source": source,
            "checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
        }

    report["reference_14b"] = summary["reference_verification"]
    report["row_integrity"] = summary["row_integrity"]
    report["completeness"] = {
        tag: json.loads((root / qc.run_record_path(tag)).read_text(encoding="utf-8"))[
            "completeness"
        ]
        for tag in qc.NEW_TAGS
    }
    report["ab_semantics_unchanged"] = {
        tag: json.loads((root / qc.run_record_path(tag)).read_text(encoding="utf-8"))[
            "label_tokenization"
        ]
        for tag in qc.NEW_TAGS
    }

    fingerprint = json.loads((root / qc.FROZEN_FINGERPRINT_PATH).read_text(encoding="utf-8"))
    report["frozen_record"] = qs.verify_frozen_fingerprint(fingerprint, root)

    statuses = [
        all(report["preregistration"][k] for k in (
            "checksum_verifies", "analysis_code_unchanged", "frozen_copy_matches_document",
            "model_revisions_checksum_verifies", "runs_used_pinned_revisions",
        )),
        report["panel"]["status"] == "PASS",
        all(report["dataset"].values()),
        all(r["status"] == "PASS" for r in report["independent_recomputation"].values()),
        report["reference_14b"]["status"] == "PASS",
        all(r["status"] == "PASS" for r in report["row_integrity"].values()),
        all(c["completed"] == qc.N_ROWS and c["missing"] == 0 and c["failed"] == 0
            for c in report["completeness"].values()),
        report["frozen_record"]["status"] == "PASS",
    ]
    report["status"] = "PASS" if all(statuses) else "FAIL"
    provenance.write_json(root / qc.INTEGRITY_REPORT_PATH, report)

    # checksums over every final machine-readable result file
    targets = sorted(
        p for p in (root / qc.ARTIFACT_DIR).rglob("*")
        if p.is_file() and p.name != Path(qc.CHECKSUMS_PATH).name
    )
    targets.append(root / qc.TEX_TABLE_PATH)
    lines = [
        f"{provenance.sha256_file(p)}  {p.relative_to(root).as_posix()}"
        for p in targets if p.is_file()
    ]
    (root / qc.CHECKSUMS_PATH).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    print(f"INTEGRITY STATUS: {report['status']}")
    for tag, block in report["independent_recomputation"].items():
        print(f"  independent recomputation {tag}: {block['status']}")
    print(f"  frozen record: {report['frozen_record']['status']} "
          f"({report['frozen_record']['n_recorded']} files)")
    print(f"  wrote {qc.INTEGRITY_REPORT_PATH} and {qc.CHECKSUMS_PATH}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
