#!/usr/bin/env python
"""Write the results and integrity documents for the Qwen2.5 scale ablation.

Every number is read from a machine artifact; none is typed. The interpretive
sentences are the ones the preregistration froze, selected by the prespecified
case conditions and trend labels, so the wording cannot follow the result.

This script was written before any 7B or 32B outcome existed. It is reporting
only: it computes nothing that the frozen analysis did not already write.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import provenance  # noqa: E402
from formalcrrc import qwen_scale_config as qc  # noqa: E402
from formalcrrc.config import FAMILIES  # noqa: E402

LABELS = {
    "accuracy": "Accuracy",
    "tce": "TCE",
    "trr": "TRR",
    "fsrr": "FSRR",
    "oracle_errors": "Oracle errors E*",
    "mvr_05": "MVR (δ=.05)",
    "mvr_0": "MVR (δ=0)",
    "oracle_errors_all": "Oracle errors E* (all 324)",
    "otce": "OTCE",
    "reachable_count": "Reachable-set size RC",
    "exact_boundary_rate": "Exact-boundary rate",
    "brier": "Brier",
}
PERCENT = ("accuracy",)
FAMILY_LABELS = {
    "coverage": "Coverage",
    "max_violation": "Maximum violations",
    "numeric_tolerance": "Numeric tolerance",
}


def fmt(value: float, metric: str, signed: bool = False) -> str:
    sign = "+" if signed else ""
    if metric in PERCENT:
        return f"{100 * value:{sign}.2f}%" if not signed else f"{100 * value:+.2f} pp"
    return f"{value:{sign}.4f}" if metric in ("trr", "fsrr", "mvr_05", "mvr_0", "brier",
                                              "exact_boundary_rate") else f"{value:{sign}.3f}"


def interval(block: dict, metric: str, signed: bool = False) -> str:
    exact = block.get("exact_interval")
    low = exact["ci_low"] if exact else block["ci_low"]
    high = exact["ci_high"] if exact else block["ci_high"]
    if metric in PERCENT:
        unit = " pp" if signed else "%"
        s = "+" if signed else ""
        return f"[{100 * low:{s}.2f}, {100 * high:{s}.2f}]{unit}"
    return f"[{fmt(low, metric, signed)}, {fmt(high, metric, signed)}]"


def load(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def results_document(root: Path) -> str:
    summary = load(root, qc.SUMMARY_PATH)
    manifest = load(root, qc.RUN_MANIFEST_PATH)
    revisions = load(root, qc.MODEL_REVISIONS_PATH)
    integrity = load(root, qc.INTEGRITY_REPORT_PATH)
    models = summary["models"]
    boot = summary["bootstrap"]
    trends = summary["trends"]
    interp = summary["interpretation"]
    runs = manifest["runs"]
    names = qc.MODEL_SHORT_NAMES
    out: list[str] = []
    add = out.append

    complete = all(
        runs[t]["completeness"]["completed"] == qc.N_ROWS for t in qc.NEW_TAGS
    )
    add(f"# {qc.STUDY_NAME} — Results")
    add("")
    add(f"**{qc.STUDY_TYPE.capitalize()}.** {qc.FREEZE_STATEMENT} "
        f"Amendments: {len(summary['amendments'])}. {qc.NOT_A_RERUN}")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Status | **{'COMPLETE' if complete else 'INCOMPLETE'}** |")
    add(f"| Preregistration frozen | `{summary['preregistration_frozen_at']}` |")
    add(f"| Preregistration SHA-256 | `{summary['preregistration_sha256']}` |")
    add(f"| Integrity audit | **{integrity['status']}** |")
    for a in summary["amendments"]:
        add(f"| Amendment {a['number']:03d} | {a['title']} — created `{a['created_at']}`, "
            f"before any 7B or 32B inference; `{qc.amendment_md_path(a['number'])}` |")
    add("")

    add("## A. Experimental question")
    add("")
    add(f"> {qc.PRIMARY_QUESTION}")
    add("")
    add(qc.NOT_PREREGISTERED_FOR_14B)
    add("")
    add(f"**Primary endpoint.** {summary['primary_endpoint']['statement']}")
    add("")

    add("## B. What was held fixed")
    add("")
    for item in qc.HELD_FIXED:
        add(f"- {item}")
    add("")
    add("Rendered prompts were byte-identical to the frozen 14B run's for all "
        f"{qc.N_ROWS} conditions in both new models, and the A/B continuation "
        "resolved to the same single-token rule.")
    add("")

    add("## C. Exact model revisions")
    add("")
    add("| Model | Revision | Parameters | License | Label tokens (A, B) | Chat template = 14B |")
    add("|---|---|---:|---|---|---|")
    for tag in qc.LADDER:
        rec = revisions["models"][tag]
        tok = rec["label_tokenization"]
        params = rec.get("parameter_count")
        add(
            f"| `{qc.MODEL_IDS[tag]}` | `{rec['revision']}` | "
            f"{f'{params:,}' if params else '— (not reloaded)'} | "
            f"{rec.get('license') or '—'} | "
            f"{tok['met_token_ids']}, {tok['not_met_token_ids']} | "
            f"{'reference' if tag == qc.REFERENCE_TAG else rec.get('chat_template_identical_to_14b')} |"
        )
    add("")

    add("## D. Completion status")
    add("")
    add("| Model | Planned | Completed | Failed | Missing | Retried | Runtime (s) | "
        "Peak GPU (GiB) | Node | SLURM job |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for tag in qc.LADDER:
        run = runs[tag]
        c = run["completeness"]
        add(
            f"| {names[tag]} | {c['planned']} | {c['completed']} | {c['failed']} | "
            f"{c['missing']} | {c['retried']} | {run['total_seconds']} | "
            f"{run.get('peak_gpu_memory_gib', '—')} | {run['slurm']['node']} | "
            f"{run['slurm']['job_id']} |"
        )
    add("")
    add("The 14B row is the existing frozen S3 run; it was not rerun.")
    add("")
    for tag in qc.NEW_TAGS:
        env = runs[tag]["environment"]
        gpu = env["gpu"]["devices"][0]["name"]
        pk = env["packages"]
        add(f"- {names[tag]}: {gpu}, CUDA {env['gpu']['torch_cuda_version']}, "
            f"torch {pk['torch']}, transformers {pk['transformers']}, "
            f"tokenizers {pk['tokenizers']}, {runs[tag]['dtype']}, quantization "
            f"{runs[tag]['quantization']}, batch size {runs[tag]['batch_size']}.")
    add("")

    add("## E. Primary metric table")
    add("")
    add("| Metric | " + " | ".join(names[t] for t in qc.LADDER) + " | Trend |")
    add("|---|" + "---:|" * len(qc.LADDER) + "---|")
    for metric in qc.PRIMARY_METRICS:
        cells = [
            f"{fmt(models[t]['metrics'][metric], metric)} "
            f"{interval(boot['models'][t][metric], metric)}"
            for t in qc.LADDER
        ]
        add(f"| {LABELS[metric]} | " + " | ".join(cells) + f" | {trends[metric]['label']} |")
    add("")
    add("Brackets are paired artifact-clustered bootstrap 95% intervals "
        f"({boot['bootstrap']['replicates']} replicates, seed "
        f"{boot['bootstrap']['seed']}, stratified by family). TRR, FSRR and E* "
        "average over the 288 nontrivial artifacts; the other rows over all 324. "
        "The 14B column is recomputed from the frozen S3 margins. "
        + qc.REFERENCE_PROVENANCE_NOTE)
    add("")
    add("Exact counts:")
    add("")
    add("| Metric | " + " | ".join(names[t] for t in qc.LADDER) + " |")
    add("|---|" + "---:|" * len(qc.LADDER))
    for metric in ("accuracy", "trr", "fsrr", "oracle_errors"):
        add(f"| {LABELS[metric]} | " + " | ".join(
            f"{models[t]['counts'][metric][0]}/{models[t]['counts'][metric][1]}"
            for t in qc.LADDER) + " |")
    add("")
    add("### Secondary (descriptive)")
    add("")
    add("| Metric | " + " | ".join(names[t] for t in qc.LADDER) + " |")
    add("|---|" + "---:|" * len(qc.LADDER))
    for metric in qc.SECONDARY_METRICS:
        add(f"| {LABELS[metric]} | " + " | ".join(
            f"{fmt(models[t]['metrics'][metric], metric)} "
            f"{interval(boot['models'][t][metric], metric)}" for t in qc.LADDER) + " |")
    add("")

    add("## F. Paired effects against 14B")
    add("")
    add("Each difference is (new model − 14B) on the same artifacts.")
    add("")
    add("| Metric | 7B − 14B | 95% CI | 32B − 14B | 95% CI |")
    add("|---|---:|---|---:|---|")
    lower = boot["paired"]["7b_minus_14b"]["metrics"]
    upper = boot["paired"]["32b_minus_14b"]["metrics"]
    for metric in qc.PAIRED_PRIMARY_METRICS + qc.PAIRED_SECONDARY_METRICS:
        tag = "" if metric in qc.PAIRED_PRIMARY_METRICS else " *(secondary)*"
        add(
            f"| Δ {LABELS[metric]}{tag} | {fmt(lower[metric]['point'], metric, True)} | "
            f"{interval(lower[metric], metric, True)} | "
            f"{fmt(upper[metric]['point'], metric, True)} | "
            f"{interval(upper[metric], metric, True)} |"
        )
    add("")
    span = boot["paired"]["32b_minus_7b"]["metrics"]
    add("Secondary contrast, 32B − 7B: " + "; ".join(
        f"Δ {LABELS[m]} {fmt(span[m]['point'], m, True)} {interval(span[m], m, True)}"
        for m in qc.PAIRED_PRIMARY_METRICS) + ".")
    add("")
    add("### Paired binary outcomes (288 nontrivial artifacts)")
    add("")
    add(summary["mcnemar"]["family"])
    add("")
    add("| Comparison | Outcome | Gains | Losses | Exact p | Holm p |")
    add("|---|---|---:|---:|---:|---:|")
    for test in summary["mcnemar"]["tests"]:
        outcome = "TR" if test["outcome"] == "translation_reachable" else "FS"
        add(
            f"| {names[test['new']]} vs {names[test['reference']]} | {outcome} | "
            f"{test['gains']} | {test['losses']} | {test['p_value']:.4g} | "
            f"{test['p_holm']:.4g} |"
        )
    add("")
    add("A gain is an artifact where the new model has the property and 14B does "
        "not; a loss is the reverse.")
    add("")

    add("## G. Scale trend across the three checkpoints")
    add("")
    add(qc.TREND_RULE)
    add("")
    add("| Metric | " + " | ".join(names[t] for t in qc.LADDER)
        + " | Label | 7B→14B CI excludes 0 | 14B→32B CI excludes 0 |")
    add("|---|" + "---:|" * len(qc.LADDER) + "---|---|---|")
    for metric in qc.PRIMARY_METRICS:
        t = trends[metric]
        add(
            f"| {LABELS[metric]} | "
            + " | ".join(fmt(t["values"][tag], metric) for tag in qc.LADDER)
            + f" | {t['label']} | {t['adjacent_7b_14b_excludes_zero']} | "
            f"{t['adjacent_14b_32b_excludes_zero']} |"
        )
    add("")
    for item in qc.TREND_PROHIBITIONS:
        add(f"- {item}")
    add("")

    add("## H. Family-level heterogeneity")
    add("")
    family_csv = pd.read_csv(root / qc.FAMILY_RESULTS_PATH)
    add("| Family | Metric | " + " | ".join(names[t] for t in qc.LADDER)
        + " | 7B − 14B [95% CI] | 32B − 14B [95% CI] | Trend |")
    add("|---|---|" + "---:|" * len(qc.LADDER) + "---|---|---|")
    for family in FAMILIES:
        for metric in ("tce", "trr", "fsrr", "accuracy"):
            row = family_csv[
                (family_csv["family"] == family) & (family_csv["metric"] == metric)
            ].iloc[0]
            fam = boot["by_family"][family]["paired"]
            add(
                f"| {FAMILY_LABELS[family]} | {LABELS[metric]} | "
                + " | ".join(fmt(row[tag], metric) for tag in qc.LADDER)
                + f" | {fmt(fam['7b_minus_14b'][metric]['point'], metric, True)} "
                f"{interval(fam['7b_minus_14b'][metric], metric, True)} | "
                f"{fmt(fam['32b_minus_14b'][metric]['point'], metric, True)} "
                f"{interval(fam['32b_minus_14b'][metric], metric, True)} | {row['trend']} |"
            )
    add("")
    add(qc.FAMILY_ANALYSIS)
    add("")

    add("## I. Prespecified interpretation")
    add("")
    add(interp["case_status"])
    add("")
    add("| Scale step | Definition | Estimate | 95% CI | Excludes 0 |")
    add("|---|---|---:|---|---|")
    for name in ("Delta_7_to_14", "Delta_14_to_32"):
        definition = interp["scale_steps"]["definitions"][name]
        step = interp["scale_steps"][name]
        add(f"| {name} | {definition} | {fmt(step['point'], 'trr', True)} | "
            f"{interval(step, 'trr', True)} | {step['excludes_zero']} |")
    add("")
    add("| Case | Condition | Holds |")
    add("|---|---|---|")
    for key, case in interp["cases"].items():
        add(f"| {key} | {case['condition']} | **{case['holds']}** |")
    add("")
    held = [(k, c) for k, c in interp["cases"].items() if c["holds"]]
    if held:
        for key, case in held:
            add(f"- **Case {key}.** {case['interpretation']}")
    else:
        add(f"No case condition holds. {interp['fallback']}")
    add("")
    add(f"TRR trend label: **{interp['trr_trend']}**.")
    add("")
    add("This is a same-family model-scale comparison under a common evaluation "
        "protocol. The checkpoints differ in scale and potentially in training "
        "dynamics. The report does not claim:")
    add("")
    for claim in qc.CLAIMS_NOT_MADE:
        add(f"- {claim}")
    add("")

    add("## J. Limitations")
    add("")
    for item in (
        "Three checkpoints from one model family. No scaling law is fitted or "
        "implied, and nothing here speaks to other families or to proprietary "
        "frontier models.",
        "The checkpoints differ in parameter count and may differ in training "
        "dynamics; scale is a label for the checkpoint, not an isolated cause.",
        "One protocol: immediate next-token A/B readout. The reason-then-score "
        "ablation showed the readout protocol itself moves these metrics, so "
        "these results do not transfer to reasoning protocols.",
        "One synthetic panel (S3) with three predicate families; intervals "
        "reflect variation across its 324 artifacts.",
        "The 14B rung was observed before this study was designed. The study is "
        "prospective only for 7B, 32B and the comparisons involving them.",
        "The operational thresholds attached to the interpretation cases (0.10 "
        "unreachable, 0.95 TRR) are labelling conventions fixed before inference, "
        "not inferential cut-offs.",
    ):
        add(f"- {item}")
    add("")

    add("## Manuscript-ready material (for author approval; manuscript not edited)")
    add("")
    add("### 1. Results paragraph")
    add("")
    m = {t: models[t]["metrics"] for t in qc.LADDER}
    c = {t: models[t]["counts"] for t in qc.LADDER}
    add(
        "> To test whether FormalCRRC failures are confined to smaller judges, we "
        "conducted a prospectively frozen post-study same-family scale ablation "
        "using Qwen2.5-7B, the existing Qwen2.5-14B condition, and Qwen2.5-32B on "
        "the identical S3 panel under the same immediate-readout protocol. "
        f"True-boundary reachability (TRR) was {fmt(m['7b']['trr'], 'trr')} "
        f"({c['7b']['trr'][0]}/288), {fmt(m['14b']['trr'], 'trr')} "
        f"({c['14b']['trr'][0]}/288) and {fmt(m['32b']['trr'], 'trr')} "
        f"({c['32b']['trr'][0]}/288) for 7B, 14B and 32B respectively "
        f"(paired change 7B to 14B: "
        f"{fmt(interp['scale_steps']['Delta_7_to_14']['point'], 'trr', True)} "
        f"{interval(interp['scale_steps']['Delta_7_to_14'], 'trr', True)}; 14B to 32B: "
        f"{fmt(interp['scale_steps']['Delta_14_to_32']['point'], 'trr', True)} "
        f"{interval(interp['scale_steps']['Delta_14_to_32'], 'trr', True)}); full-sequence recovery (FSRR) was "
        f"{fmt(m['7b']['fsrr'], 'fsrr')}, {fmt(m['14b']['fsrr'], 'fsrr')} and "
        f"{fmt(m['32b']['fsrr'], 'fsrr')}; mean threshold crossing error was "
        f"{fmt(m['7b']['tce'], 'tce')}, {fmt(m['14b']['tce'], 'tce')} and "
        f"{fmt(m['32b']['tce'], 'tce')}; and pointwise accuracy was "
        f"{fmt(m['7b']['accuracy'], 'accuracy')}, "
        f"{fmt(m['14b']['accuracy'], 'accuracy')} and "
        f"{fmt(m['32b']['accuracy'], 'accuracy')}. Across the three checkpoints "
        f"the same-family scale trend was {trends['trr']['label']} for TRR, "
        f"{trends['fsrr']['label']} for FSRR, {trends['tce']['label']} for TCE and "
        f"{trends['accuracy']['label']} for accuracy. "
        + " ".join(case["interpretation"] for _, case in held)
        + (" " if held else "")
        + "Intervals are paired artifact-clustered bootstrap 95% intervals."
    )
    add("")
    add("### 2. Limitations sentence")
    add("")
    add("> The scale ablation compares three checkpoints of a single open-weight "
        "family under one immediate-readout protocol; the checkpoints differ in "
        "scale and potentially in training dynamics, so the comparison does not "
        "identify parameter count as a causal mechanism, does not establish a "
        "scaling law, and does not characterise proprietary frontier models.")
    add("")
    add("### 3. LaTeX table")
    add("")
    add(f"`{qc.TEX_TABLE_PATH}` (label `tab:qwen-scale-ablation`).")
    add("")
    add("### 4. Recommended insertion location")
    add("")
    add("No manuscript source exists in this repository, so the location is given "
        "by content: place the paragraph and table in the Results section "
        "immediately after the S3 (s-3) translation-reachability results and "
        "alongside the other post-study robustness analyses (label-swap, reasoning "
        "anchor, reason-then-score ablation), since it reuses the S3 panel and the "
        "14B S3 condition. Place the limitations sentence with the existing "
        "statement on model coverage.")
    add("")
    add("---")
    add("")
    add(f"*Generated by `scripts/write_results_qwen_scale.py` from "
        f"`{qc.SUMMARY_PATH}`, `{qc.RUN_MANIFEST_PATH}`, `{qc.MODEL_REVISIONS_PATH}` "
        f"and `{qc.INTEGRITY_REPORT_PATH}`. Every number above is read from a "
        "machine artifact.*")
    add("")
    return "\n".join(out)


CHECK_LABELS = (
    ("Dataset IDs exactly match the S3 panel",
     lambda r: all(v["checks"]["prompt_ids_match_s3_panel"]
                   and v["checks"]["artifact_ids_match_s3_panel"]
                   for v in r["row_integrity"].values())),
    ("Nine thresholds exist for every artifact",
     lambda r: r["panel"]["checks"]["every_artifact_has_exactly_9_thresholds"]
     and all(v["checks"]["nine_thresholds_per_artifact"]
             for v in r["row_integrity"].values())),
    ("Truth labels and j* reproduce the S3 truth exactly",
     lambda r: r["panel"]["checks"]["j_star_matches_truth_labels"]
     and all(v["checks"]["truth_and_j_star_match_s3"]
             for v in r["independent_recomputation"].values())),
    ("Rendered prompts byte-identical to the 14B protocol",
     lambda r: all(v["checks"]["rendered_prompts_byte_identical_to_14b"]
                   for v in r["row_integrity"].values())),
    ("A/B semantic mapping unchanged (single-token A = met, B = not met)",
     lambda r: all(v["label_met"] == "A" and v["label_not_met"] == "B"
                   and v["single_token_equivalence"]
                   for v in r["ab_semantics_unchanged"].values())),
    ("No result rows duplicated",
     lambda r: all(v["checks"]["no_duplicate_rows"] for v in r["row_integrity"].values())),
    ("No missing rows without explicit terminal status",
     lambda r: all(c["completed"] == qc.N_ROWS and c["missing"] == 0
                   for c in r["completeness"].values())),
    ("Metrics recomputed independently from saved margins agree",
     lambda r: all(v["status"] == "PASS"
                   for v in r["independent_recomputation"].values())),
    ("Existing 14B metrics reproduce exactly",
     lambda r: r["reference_14b"]["status"] == "PASS"),
    ("No existing frozen artifact modified",
     lambda r: r["frozen_record"]["status"] == "PASS"),
)


def integrity_document(root: Path) -> str:
    report = load(root, qc.INTEGRITY_REPORT_PATH)
    preflight = load(root, qc.PREFLIGHT_PATH)
    out: list[str] = []
    add = out.append
    add(f"# {qc.STUDY_NAME} — Integrity")
    add("")
    add(f"Overall status: **{report['status']}** (checked `{report['checked_at']}`).")
    add("")
    add("## 1. Audit note: what was reused")
    add("")
    add("Reused read-only, never regenerated or modified:")
    add("")
    for path in qc.READ_ONLY_INPUTS:
        add(f"- `{path}`")
    add("")
    add("Reused code, unchanged: `formalcrrc.scoring`, `formalcrrc.prompts`, "
        "`formalcrrc.config`, `formalcrrc.metrics`, `formalcrrc.s3`, "
        "`formalcrrc.s2_bootstrap.crossing_index`, "
        "`formalcrrc.rts_ablation_bootstrap` (paired plan), and the s-3 model "
        "loader in `scripts/run_s3_inference.py`.")
    add("")
    add("## 2. The ten required checks")
    add("")
    add("| # | Check | Result |")
    add("|---:|---|---|")
    for index, (label, test) in enumerate(CHECK_LABELS, start=1):
        add(f"| {index} | {label} | {'PASS' if test(report) else 'FAIL'} |")
    add("")
    add("## 3. Preregistration")
    add("")
    p = report["preregistration"]
    add("| | |")
    add("|---|---|")
    add(f"| Frozen at | `{p['frozen_at']}` |")
    add(f"| Amendments | {p['amendments']} |")
    add(f"| Checksum verifies | {p['checksum_verifies']} |")
    add(f"| Pinned analysis code unchanged | {p['analysis_code_unchanged']} |")
    add(f"| Frozen copy matches document | {p['frozen_copy_matches_document']} |")
    add(f"| Pinned code manifest in force | {p['pinned_code_source']} |")
    add(f"| model_revisions.json checksum verifies | {p['model_revisions_checksum_verifies']} |")
    add(f"| Runs used the pinned revisions | {p['runs_used_pinned_revisions']} |")
    add("")
    add("## 4. Pre-flight (no S3 prompt scored)")
    add("")
    add(f"Status **{preflight['status']}**, SLURM job "
        f"{preflight['slurm']['job_id']} on `{preflight['slurm']['node']}`.")
    add("")
    for name, ok in preflight["stack_checks"].items():
        add(f"- {'PASS' if ok else 'FAIL'} `{name}`")
    ref = preflight["reference_14b"]
    add(f"- {'PASS' if ref['status'] == 'PASS' else 'FAIL'} frozen 14B reproduces its "
        f"recorded smoke-prompt scores ({ref['smoke_raw_score_met']}, "
        f"{ref['smoke_raw_score_not_met']})")
    add("")
    for tag, record in preflight["models"].items():
        add(f"### {qc.MODEL_SHORT_NAMES[tag]} — {record['status']}")
        add("")
        add(f"Revision `{record['revision']}`; parameters "
            f"{record.get('parameter_count', 0):,}; peak GPU "
            f"{record.get('peak_gpu_memory_gib')} GiB at load + development prompts.")
        add("")
        for name, ok in record["checks"].items():
            add(f"- {'PASS' if ok else 'FAIL'} `{name}`")
        add("")
    add("## 5. Row integrity of the new runs")
    add("")
    for tag, block in report["row_integrity"].items():
        add(f"### {qc.MODEL_SHORT_NAMES[tag]} — {block['status']}")
        add("")
        for name, ok in block["checks"].items():
            add(f"- {'PASS' if ok else 'FAIL'} `{name}`")
        add("")
    add("## 6. Independent recomputation")
    add("")
    add("Recomputed from saved margins without the primary metric modules: "
        "reachability by brute-force intercept enumeration, oracle errors over a "
        "redundant superset of cuts, decisions from the margin sign.")
    add("")
    for tag, block in report["independent_recomputation"].items():
        add(f"### {qc.MODEL_SHORT_NAMES[tag]} — {block['status']} (`{block['source']}`)")
        add("")
        for name, ok in block["checks"].items():
            add(f"- {'PASS' if ok else 'FAIL'} `{name}`")
        add("")
    add("## 7. 14B reference reproduction")
    add("")
    add("| Metric | Recomputed | Expected |")
    add("|---|---:|---:|")
    for name, expected in report["reference_14b"]["expected_counts"].items():
        got = report["reference_14b"]["counts"][name]
        add(f"| {name} | {got[0]}/{got[1]} | {expected[0]}/{expected[1]} |")
    add("")
    for name, ok in report["reference_14b"]["checks"].items():
        add(f"- {'PASS' if ok else 'FAIL'} `{name}`")
    add("")
    add("## 8. Frozen record")
    add("")
    f = report["frozen_record"]
    add("| | |")
    add("|---|---|")
    add(f"| Protected files | {f['n_recorded']} |")
    add(f"| Combined digest before | `{f['combined_sha256_before']}` |")
    add(f"| Combined digest after | `{f['combined_sha256_after']}` |")
    add(f"| Changed | {f['changed'] or 'NONE'} |")
    add(f"| Missing | {f['missing'] or 'NONE'} |")
    add(f"| Added | {f['added'] or 'NONE'} |")
    add("")
    add("## 9. Completeness and receipts")
    add("")
    add("| Model | Planned | Completed | Failed | Missing |")
    add("|---|---:|---:|---:|---:|")
    for tag, c in report["completeness"].items():
        add(f"| {qc.MODEL_SHORT_NAMES[tag]} | {c['planned']} | {c['completed']} | "
            f"{c['failed']} | {c['missing']} |")
    add("")
    add("Per-invocation receipts: "
        + ", ".join(f"`{qc.receipts_path(t)}`" for t in qc.NEW_TAGS) + ".")
    add("")
    add("## 10. Infrastructure-only failures")
    add("")
    note = root / qc.INFRASTRUCTURE_LOG_DIR / "README.json"
    if note.is_file():
        infra = json.loads(note.read_text(encoding="utf-8"))
        add(infra["statement"])
        add("")
        add("| SLURM job | Stage | Node | State | What happened |")
        add("|---|---|---|---|---|")
        for job in infra["jobs"]:
            add(f"| {job['job_id']} | {job['stage']} | {job['node']} | {job['state']} | "
                f"{job['what_happened']} |")
        add("")
        add(f"Logs preserved under `{qc.INFRASTRUCTURE_LOG_DIR}/`.")
    else:
        add("None recorded.")
    add("")
    manifest = load(root, qc.RUN_MANIFEST_PATH)
    add("## 11. Provenance-only code")
    add("")
    add(manifest["provenance_only_code"]["note"])
    add("")
    for name, sha in manifest["provenance_only_code"]["files"].items():
        add(f"- `{name}` `{sha}`")
    add("")
    add("## 12. Checksums")
    add("")
    add(f"`{qc.CHECKSUMS_PATH}` lists the SHA-256 of every final machine-readable "
        "result file:")
    add("")
    add("```text")
    add((root / qc.CHECKSUMS_PATH).read_text(encoding="utf-8").rstrip())
    add("```")
    add("")
    add("---")
    add("")
    add(f"*Generated by `scripts/write_results_qwen_scale.py` from "
        f"`{qc.INTEGRITY_REPORT_PATH}` and `{qc.PREFLIGHT_PATH}`.*")
    add("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = Path(args.root)
    (root / qc.RESULTS_MD_PATH).write_text(
        results_document(root), encoding="utf-8", newline="\n"
    )
    (root / qc.INTEGRITY_MD_PATH).write_text(
        integrity_document(root), encoding="utf-8", newline="\n"
    )
    print(f"wrote {qc.RESULTS_MD_PATH}")
    print(f"wrote {qc.INTEGRITY_MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
