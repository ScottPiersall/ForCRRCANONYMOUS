#!/usr/bin/env python
"""Freeze the Qwen2.5 same-family scale-ablation preregistration.

Fails closed. It refuses to run when any 7B or 32B result, partial result or
receipt exists, so the study cannot be represented as prospectively frozen after
the fact, and it refuses to overwrite an existing freeze: a change after
freezing goes through a numbered amendment that leaves the original intact.

Before writing anything it re-verifies the frozen S3 panel from primitive fields
and recomputes the Qwen2.5-14B reference rung from its saved margins, requiring
every reference value -- including FSRR = 136/288 and the identity
``FSR_x = 1 iff E*_x = 0`` on all 288 nontrivial artifacts -- to hold exactly.

The machine form and the human-readable document are rendered from one object.
The document cites the machine form's SHA-256; the machine form does not cite
the document's, because stamping a digest into the document changes it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import dataset as ds  # noqa: E402
from formalcrrc import day3_config as d3c  # noqa: E402
from formalcrrc import prompts, provenance  # noqa: E402
from formalcrrc import qwen_scale as qs  # noqa: E402
from formalcrrc import qwen_scale_config as qc  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    INFERENCE_BATCH_SIZE,
    LABEL_MET,
    LABEL_NOT_MET,
    PROMPT_TEMPLATE,
)

ROOT = provenance.repo_root()


def refuse_if_not_prospective() -> None:
    blockers = [ROOT / qc.PREREG_JSON_PATH, ROOT / qc.PREREG_SHA_PATH]
    for tag in qc.NEW_TAGS:
        blockers += [
            ROOT / qc.results_path(tag),
            ROOT / qc.partial_results_path(tag),
            ROOT / qc.run_record_path(tag),
            ROOT / qc.receipts_path(tag),
        ]
    blockers += [ROOT / qc.SUMMARY_PATH, ROOT / qc.MODEL_REVISIONS_PATH]
    existing = [p for p in blockers if p.exists()]
    if existing:
        print("REFUSING TO FREEZE: these files already exist:")
        for path in existing:
            print(f"  {path.relative_to(ROOT).as_posix()}")
        raise SystemExit(2)


def verify_panel() -> dict:
    frame = pd.read_parquet(ROOT / d3c.DATASET_PATH)
    manifest = json.loads((ROOT / d3c.DATASET_MANIFEST_PATH).read_text(encoding="utf-8"))
    verdict = qs.validate_panel(frame)
    verdict["checks"].update(
        {
            "seed_is_45": int(manifest["seed"]) == qc.SOURCE_SEED,
            "dataset_file_matches_manifest": provenance.sha256_file(
                ROOT / d3c.DATASET_PATH
            )
            == manifest["hashes"]["dataset_file_sha256"],
            "dataset_content_matches_manifest": ds.dataset_content_hash(frame)
            == manifest["hashes"]["dataset_content_sha256"],
            "prompt_manifest_matches_manifest": provenance.sha256_file(
                ROOT / d3c.PROMPT_MANIFEST_PATH
            )
            == manifest["hashes"]["prompt_manifest_file_sha256"],
            "prompt_template_matches_manifest": prompts.prompt_template_sha256()
            == manifest["hashes"]["prompt_template_sha256"],
        }
    )
    verdict["status"] = "PASS" if all(verdict["checks"].values()) else "FAIL"
    verdict["hashes"] = manifest["hashes"]
    verdict["dataset_path"] = d3c.DATASET_PATH
    verdict["prompt_manifest_path"] = d3c.PROMPT_MANIFEST_PATH
    return verdict


def verify_reference() -> dict:
    frame = pd.read_parquet(ROOT / d3c.DATASET_PATH)
    raw = pd.read_parquet(ROOT / qc.REFERENCE_RAW_SCORES_PATH)
    run = json.loads((ROOT / qc.REFERENCE_RUN_RECORD_PATH).read_text(encoding="utf-8"))
    table = qs.artifact_table(qs.attach_truth(raw, frame), qc.REFERENCE_TAG)
    frozen = pd.read_parquet(ROOT / d3c.METRICS_PATH)
    frozen = frozen[frozen["model_id"] == qc.MODEL_IDS[qc.REFERENCE_TAG]]
    verdict = qs.verify_reference(table, frozen)
    verdict["checks"].update(
        {
            "raw_scores_sha256_matches": provenance.sha256_file(
                ROOT / qc.REFERENCE_RAW_SCORES_PATH
            )
            == qc.REFERENCE_RAW_SCORES_SHA256
            == run["raw_scores_sha256"],
            "revision_matches": run["revision"] == qc.REFERENCE_REVISION,
            "chat_template_sha256_matches": run["chat_template_sha256"]
            == qc.REFERENCE_CHAT_TEMPLATE_SHA256,
            "scoring_method_is_single_token_next_logit": run["scoring_method"]
            == qc.REQUIRED_SCORING_METHOD,
            "stack_matches": all(
                run["environment"]["packages"].get(k) == v
                for k, v in qc.REFERENCE_STACK.items()
            ),
            "all_2916_scored": run["prompts_scored"] == qc.N_ROWS and not run["failures"],
        }
    )
    verdict["status"] = "PASS" if all(verdict["checks"].values()) else "FAIL"
    verdict["metrics"] = qs.summarise_model(table)["metrics"]
    verdict["by_family"] = qs.family_summary(table)
    verdict["run"] = {
        "model_id": run["model_id"],
        "revision": run["revision"],
        "dtype": run["dtype"],
        "scoring_method": run["scoring_method"],
        "label_tokenization": run["label_tokenization"],
        "chat_template_sha256": run["chat_template_sha256"],
        "slurm": run["slurm"],
        "total_seconds": run["total_seconds"],
        "gpu": run["environment"]["gpu"]["devices"][0]["name"],
        "packages": run["environment"]["packages"],
        "python_version": run["environment"]["python_version"],
        "torch_cuda_version": run["environment"]["gpu"]["torch_cuda_version"],
    }
    verdict["provenance_note"] = qc.REFERENCE_PROVENANCE_NOTE
    return verdict


def build(panel: dict, reference: dict, fingerprint: dict) -> dict:
    return {
        "study": qc.STUDY_NAME,
        "study_type": qc.STUDY_TYPE,
        "amendments": 0,
        "frozen_at": provenance.utc_now(),
        "freeze_statement": qc.FREEZE_STATEMENT,
        "not_preregistered_for_14b": qc.NOT_PREREGISTERED_FOR_14B,
        "not_a_rerun": qc.NOT_A_RERUN,
        "document": {"path": qc.PREREG_MD_PATH},
        "question": qc.PRIMARY_QUESTION,
        "ladder": {
            "order": list(qc.LADDER),
            "models": qc.MODEL_IDS,
            "short_names": qc.MODEL_SHORT_NAMES,
            "new": list(qc.NEW_TAGS),
            "reference": qc.REFERENCE_TAG,
            "run_order": list(qc.RUN_ORDER),
        },
        "protocol": {
            "name": qc.PROTOCOL,
            "held_fixed": list(qc.HELD_FIXED),
            "prompt_template": PROMPT_TEMPLATE,
            "prompt_template_sha256": prompts.prompt_template_sha256(),
            "label_met": LABEL_MET,
            "label_not_met": LABEL_NOT_MET,
            "label_resolution": qc.LABEL_RESOLUTION,
            "required_scoring_method": qc.REQUIRED_SCORING_METHOD,
            "canonical_margin": qc.CANONICAL_MARGIN,
            "probability": qc.PROBABILITY,
            "dtype": qc.MODEL_DTYPE,
            "quantization": "none",
            "batch_size": INFERENCE_BATCH_SIZE,
            "continuity_requirements": list(qc.CONTINUITY_REQUIREMENTS),
            "stop_rule": qc.STOP_RULE,
            "revision_rule": qc.REVISION_RULE,
            "preflight_scope": qc.PREFLIGHT_SCOPE,
            "preflight_code_status": qc.PREFLIGHT_CODE_STATUS,
            "development_prompts": list(qc.DEVELOPMENT_PROMPTS),
            "reference_stack": qc.REFERENCE_STACK,
            "hardware": "one NVIDIA H100 PCIe 80 GB per model",
        },
        "dataset": {
            "partition": qc.SOURCE_PARTITION,
            "seed": qc.SOURCE_SEED,
            "n_artifacts": qc.N_ARTIFACTS,
            "n_rows": qc.N_ROWS,
            "n_nontrivial": qc.N_NONTRIVIAL,
            "nontrivial_rule": qc.NONTRIVIAL_RULE,
            "validation": panel,
        },
        "reference_14b": reference,
        "completeness": {
            "rule": qc.COMPLETENESS_RULE,
            "retry_rule": qc.RETRY_RULE,
            "no_peeking_rule": qc.NO_PEEKING_RULE,
        },
        "metrics": {
            "primary": list(qc.PRIMARY_METRICS),
            "secondary": list(qc.SECONDARY_METRICS),
            "definitions": qc.METRIC_DEFINITIONS,
            "secondary_definitions": qc.SECONDARY_DEFINITIONS,
            "population": qc.METRIC_POPULATION,
            "higher_is_better": list(qc.HIGHER_IS_BETTER),
            "lower_is_better": list(qc.LOWER_IS_BETTER),
            "crossing_convention": qc.CROSSING_CONVENTION,
            "decision_convention": qc.DECISION_CONVENTION,
            "tie_policy": qc.TIE_POLICY,
            "mvr_tolerance": qc.MVR_TOLERANCE,
            "mvr_tolerance_strict": qc.MVR_TOLERANCE_STRICT,
        },
        "statistics": {
            "bootstrap_replicates": qc.BOOTSTRAP_REPLICATES,
            "bootstrap_seed": qc.BOOTSTRAP_SEED,
            "bootstrap_scheme": qc.BOOTSTRAP_SCHEME,
            "bootstrap_ci": qc.BOOTSTRAP_CI_DESCRIPTION,
            "primary_contrasts": [list(c) for c in qc.PRIMARY_CONTRASTS],
            "secondary_contrasts": [list(c) for c in qc.SECONDARY_CONTRASTS],
            "paired_primary_metrics": list(qc.PAIRED_PRIMARY_METRICS),
            "paired_secondary_metrics": list(qc.PAIRED_SECONDARY_METRICS),
            "zero_count_rule": qc.ZERO_COUNT_RULE,
            "mcnemar_family": qc.MCNEMAR_FAMILY,
            "mcnemar_outcomes": list(qc.MCNEMAR_OUTCOMES),
            "mcnemar_alpha": qc.MCNEMAR_ALPHA,
        },
        "trend": {
            "rule": qc.TREND_RULE,
            "labels": list(qc.TREND_LABELS),
            "prohibitions": list(qc.TREND_PROHIBITIONS),
        },
        "interpretation": {
            "substantial_unreachable_min": qc.SUBSTANTIAL_UNREACHABLE_MIN,
            "near_perfect_trr_min": qc.NEAR_PERFECT_TRR_MIN,
            "cases": qc.INTERPRETATION_CASES,
            "fallback": qc.INTERPRETATION_FALLBACK,
            "claims_not_made": list(qc.CLAIMS_NOT_MADE),
        },
        "family_analysis": qc.FAMILY_ANALYSIS,
        "amendment_rule": qc.AMENDMENT_RULE,
        "outputs": {
            "preregistration_md": qc.PREREG_MD_PATH,
            "results_md": qc.RESULTS_MD_PATH,
            "integrity_md": qc.INTEGRITY_MD_PATH,
            "model_revisions": qc.MODEL_REVISIONS_PATH,
            "run_manifest": qc.RUN_MANIFEST_PATH,
            "results": {tag: qc.results_path(tag) for tag in qc.NEW_TAGS},
            "combined_scale_results": qc.COMBINED_RESULTS_PATH,
            "artifact_level_metrics": qc.ARTIFACT_METRICS_PATH,
            "paired_effects": qc.PAIRED_EFFECTS_PATH,
            "family_results": qc.FAMILY_RESULTS_PATH,
            "summary": qc.SUMMARY_PATH,
            "integrity_report": qc.INTEGRITY_REPORT_PATH,
            "checksums": qc.CHECKSUMS_PATH,
            "tex_table": qc.TEX_TABLE_PATH,
        },
        "read_only_inputs": provenance.source_manifest(qc.READ_ONLY_INPUTS, ROOT),
        "frozen_record": {
            "n_files": fingerprint["n_files"],
            "combined_sha256": fingerprint["combined_sha256"],
        },
        "analysis_code": provenance.source_manifest(qc.ANALYSIS_CODE_PATHS, ROOT),
        "git": provenance.git_state(ROOT),
    }


def render(prereg: dict, digest: str) -> str:
    ref = prereg["reference_14b"]
    counts = ref["counts"]
    run = ref["run"]
    p = prereg["protocol"]
    out: list[str] = []
    add = out.append

    add(f"# {prereg['study']} — Preregistration")
    add("")
    add(f"**{prereg['freeze_statement']}**")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Study type | {prereg['study_type']} |")
    add(f"| Frozen at (UTC) | `{prereg['frozen_at']}` |")
    add(f"| Machine form | `{qc.PREREG_JSON_PATH}` |")
    add(f"| Machine form SHA-256 | `{digest}` |")
    add(f"| Amendments | {prereg['amendments']} |")
    add("")
    add("## 1. Status of this document")
    add("")
    add(prereg["not_preregistered_for_14b"])
    add("")
    add(prereg["not_a_rerun"])
    add("")
    add(prereg["amendment_rule"])
    add("")
    add("## 2. Scientific question")
    add("")
    add(f"> {prereg['question']}")
    add("")
    add(
        "The study addresses the possible objection that the boundary-localization "
        "and boundary-recoverability failures FormalCRRC measures mainly reflect "
        "limited model capability. It is a same-family model-scale comparison "
        "under a common evaluation protocol. The checkpoints differ in scale and "
        "potentially in training dynamics, so parameter count is not claimed to be "
        "the causal mechanism under any outcome."
    )
    add("")
    add("## 3. The scale ladder")
    add("")
    add("| Rung | Model | Role |")
    add("|---|---|---|")
    for tag in prereg["ladder"]["order"]:
        role = (
            "existing frozen S3 result; **not rerun**"
            if tag == prereg["ladder"]["reference"]
            else "new, run under this preregistration"
        )
        add(f"| {tag.upper()} | `{prereg['ladder']['models'][tag]}` | {role} |")
    add("")
    add(
        "Run order is fixed: "
        + ", then ".join(t.upper() for t in prereg["ladder"]["run_order"])
        + "."
    )
    add("")
    add("## 4. Frozen S3 panel (reused, not regenerated)")
    add("")
    d = prereg["dataset"]
    add(f"Partition `{d['partition']}`, seed {d['seed']}: {d['n_artifacts']} artifacts "
        f"x 9 thresholds = {d['n_rows']} rubric conditions per model; "
        f"{d['n_nontrivial']} nontrivial artifacts. {d['nontrivial_rule']}")
    add("")
    add("| Asset | SHA-256 |")
    add("|---|---|")
    add(f"| `{d['validation']['dataset_path']}` | `{d['validation']['hashes']['dataset_file_sha256']}` |")
    add(f"| dataset content | `{d['validation']['hashes']['dataset_content_sha256']}` |")
    add(f"| `{d['validation']['prompt_manifest_path']}` | `{d['validation']['hashes']['prompt_manifest_file_sha256']}` |")
    add(f"| prompt template | `{d['validation']['hashes']['prompt_template_sha256']}` |")
    add("")
    add("Panel assertions verified at freeze:")
    add("")
    for name, ok in d["validation"]["checks"].items():
        add(f"- {'PASS' if ok else 'FAIL'} `{name}`")
    add("")
    add("No artifact, threshold, rubric text, truth label or prompt is regenerated.")
    add("")
    add("## 5. The existing Qwen2.5-14B reference")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Raw margins | `{qc.REFERENCE_RAW_SCORES_PATH}` |")
    add(f"| Raw margins SHA-256 | `{qc.REFERENCE_RAW_SCORES_SHA256}` |")
    add(f"| Model and tokenizer revision | `{run['revision']}` |")
    add(f"| Chat template SHA-256 | `{run['chat_template_sha256']}` |")
    add(f"| Label tokens | A = {run['label_tokenization']['met_token_ids']}, "
        f"B = {run['label_tokenization']['not_met_token_ids']} (single token each) |")
    add(f"| Scoring | `{run['scoring_method']}` |")
    add(f"| Precision | {run['dtype']}, no quantization, batch size {p['batch_size']} |")
    add(f"| Stack | torch {run['packages']['torch']} (CUDA {run['torch_cuda_version']}), "
        f"transformers {run['packages']['transformers']}, "
        f"tokenizers {run['packages']['tokenizers']}, Python {run['python_version']} |")
    add(f"| Hardware | {run['gpu']}, node `{run['slurm']['node']}`, "
        f"SLURM job {run['slurm']['job_id']}, {run['total_seconds']} s |")
    add("")
    add("Reference values recomputed from the saved margins at freeze. The analysis "
        "recomputes them again and refuses to proceed unless every count matches.")
    add("")
    add("| Metric | Count | Value |")
    add("|---|---:|---:|")
    for name in qc.REFERENCE_COUNTS:
        numerator, denominator = counts[name]
        add(f"| {name} | {numerator}/{denominator} | {numerator / denominator:.6f} |")
    add("")
    add(ref["provenance_note"])
    add("")
    add("Reference assertions verified at freeze:")
    add("")
    for name, ok in ref["checks"].items():
        add(f"- {'PASS' if ok else 'FAIL'} `{name}`")
    add("")
    add("## 6. Inference protocol")
    add("")
    add("The 7B and 32B conditions reproduce the 14B immediate-readout protocol. "
        "Held fixed:")
    add("")
    for item in p["held_fixed"]:
        add(f"- {item}")
    add("")
    add("The prompt template, verbatim:")
    add("")
    add("```text")
    add(p["prompt_template"])
    add("```")
    add("")
    add(f"Canonical margin: `{p['canonical_margin']}`, where A = met and B = not met. "
        f"`{p['probability']}`. Label tokens are resolved by `{p['label_resolution']}`.")
    add("")
    add("The scoring path is the existing implementation, called unchanged: the "
        "Day-3 model loader, `formalcrrc.scoring.render_chat_prompt` and "
        "`formalcrrc.scoring.score_prompt`. No reason-then-score or other "
        "generation protocol is used.")
    add("")
    add("## 7. Model provenance and continuity requirements")
    add("")
    add(p["revision_rule"])
    add("")
    add("Recorded per checkpoint in `" + qc.MODEL_REVISIONS_PATH + "`: full commit "
        "hash, license, `config.json`, measured parameter count, tokenizer "
        "configuration, chat-template hash and label tokenization.")
    add("")
    add("Each new checkpoint must satisfy all of the following before any S3 row "
        "is scored:")
    add("")
    for item in p["continuity_requirements"]:
        add(f"1. {item}")
    add("")
    add(p["stop_rule"])
    add("")
    add("Tokenization is not assumed identical because the checkpoints share a "
        "family; it is checked on every prompt.")
    add("")
    add("## 8. Pre-flight")
    add("")
    add(p["preflight_scope"])
    add("")
    add(p["preflight_code_status"])
    add("")
    add("Permanently designated development prompts:")
    add("")
    for text in p["development_prompts"]:
        add("```text")
        add(text)
        add("```")
    add("")
    add(f"Hardware: {p['hardware']}. Required stack: "
        + ", ".join(f"{k} {v}" for k, v in p["reference_stack"].items()) + ".")
    add("")
    add("## 9. Completeness")
    add("")
    c = prereg["completeness"]
    add(c["rule"])
    add("")
    add(c["retry_rule"])
    add("")
    add(c["no_peeking_rule"])
    add("")
    add("Reported per model: planned, completed, failed, missing, retried.")
    add("")
    add("## 10. Metrics")
    add("")
    m = prereg["metrics"]
    add(f"Conventions: {m['crossing_convention']} {m['decision_convention']}. "
        f"{m['tie_policy']}")
    add("")
    add("### Primary")
    add("")
    for name in m["primary"]:
        add(f"- **{name}** — {m['definitions'][name]}")
    add("")
    add("### Secondary (descriptive)")
    add("")
    for name in m["secondary"]:
        add(f"- **{name}** — {m['secondary_definitions'][name]}")
    add("")
    add("No metric is added after outcomes are inspected.")
    add("")
    add("## 11. Primary comparison table")
    add("")
    add("One table, columns " + ", ".join(prereg["ladder"]["short_names"][t] for t in qc.LADDER)
        + ", rows: " + ", ".join(m["primary"]) + ". The 14B column is recomputed "
        "from the frozen S3 margins, never typed in.")
    add("")
    add("## 12. Paired uncertainty")
    add("")
    s = prereg["statistics"]
    add(f"{s['bootstrap_scheme']}")
    add("")
    add(f"Replicates {s['bootstrap_replicates']}, seed {s['bootstrap_seed']}, "
        f"interval {s['bootstrap_ci']}.")
    add("")
    add("Primary paired contrasts (first minus second): "
        + "; ".join(f"{a.upper()} − {b.upper()}" for a, b in s["primary_contrasts"])
        + ". Secondary, descriptive: "
        + "; ".join(f"{a.upper()} − {b.upper()}" for a, b in s["secondary_contrasts"]) + ".")
    add("")
    add("Paired differences with intervals are reported for: "
        + ", ".join(s["paired_primary_metrics"]) + " (primary); "
        + ", ".join(s["paired_secondary_metrics"]) + " (secondary).")
    add("")
    add("The artifact is the sampling unit. No test treats the 2,916 rubric "
        "conditions as independent.")
    add("")
    add(s["zero_count_rule"])
    add("")
    add("### Null-hypothesis tests")
    add("")
    add(s["mcnemar_family"])
    add("")
    add("## 13. Scale trend (descriptive)")
    add("")
    add(prereg["trend"]["rule"])
    add("")
    for item in prereg["trend"]["prohibitions"]:
        add(f"- {item}")
    add("")
    add("## 14. Prespecified interpretation")
    add("")
    add("The qualitative cases are given operational conditions here, before "
        "inference, so the wording cannot follow the result. They key on TRR. "
        "Cases are not mutually exclusive.")
    add("")
    add("| Case | Condition | Interpretation |")
    add("|---|---|---|")
    for key, case in prereg["interpretation"]["cases"].items():
        add(f"| {key} | {case['condition']} | {case['interpretation']} |")
    add("")
    add(prereg["interpretation"]["fallback"])
    add("")
    add("Under no outcome will the report claim:")
    add("")
    for claim in prereg["interpretation"]["claims_not_made"]:
        add(f"- {claim}")
    add("")
    add("## 15. Family-level analysis")
    add("")
    add(prereg["family_analysis"])
    add("")
    add("## 16. Integrity checks before results are accepted")
    add("")
    for item in (
        "dataset IDs exactly match the S3 panel",
        "nine thresholds exist for every artifact",
        "truth labels and j* reproduce the S3 truth exactly",
        "rendered prompts are byte-identical to the 14B run's for every prompt_id",
        "A/B semantic mapping is unchanged",
        "no result rows are duplicated",
        "no row is missing without an explicit terminal status",
        "every metric recomputed independently from saved margins agrees with the "
        "primary analysis",
        "the 14B metrics reproduce exactly",
        f"none of the {prereg['frozen_record']['n_files']} fingerprinted frozen files "
        f"has changed (combined SHA-256 `{prereg['frozen_record']['combined_sha256']}`)",
    ):
        add(f"1. {item}")
    add("")
    add("## 17. Outputs")
    add("")
    for label, path in prereg["outputs"].items():
        if isinstance(path, dict):
            for tag, sub in path.items():
                add(f"- `{sub}` ({label} {tag})")
        else:
            add(f"- `{path}`")
    add("")
    add("The main manuscript is not edited by this study. Manuscript-ready text and "
        "the LaTeX table are provided for author approval.")
    add("")
    add("## 18. Analysis code pinned at freeze")
    add("")
    add(f"Tree SHA-256 `{prereg['analysis_code']['tree_sha256']}`. The inference "
        "runner and the analysis refuse to run if any pinned file has changed.")
    add("")
    add("| File | SHA-256 |")
    add("|---|---|")
    for name, sha in prereg["analysis_code"]["files"].items():
        add(f"| `{name}` | `{sha}` |")
    add("")
    add("---")
    add("")
    add("*Rendered by `scripts/freeze_qwen_scale_preregistration.py` from the same "
        "object as the machine form.*")
    add("")
    return "\n".join(out)


def main() -> int:
    print("=" * 74)
    print("QWEN2.5 SCALE ABLATION PREREGISTRATION FREEZE")
    print("=" * 74)

    refuse_if_not_prospective()
    print("hard stop        PASS  no 7B/32B inference and no prior freeze exist")

    panel = verify_panel()
    print(f"S3 panel         {panel['status']}")
    for name, ok in panel["checks"].items():
        print(f"    {'PASS' if ok else 'FAIL'}  {name}")
    if panel["status"] != "PASS":
        return 2

    reference = verify_reference()
    print(f"14B reference    {reference['status']}")
    for name, ok in reference["checks"].items():
        print(f"    {'PASS' if ok else 'FAIL'}  {name}")
    if reference["status"] != "PASS":
        print("REFUSING TO FREEZE: the 14B reference does not reproduce.")
        print(f"    recomputed counts: {reference['counts']}")
        return 2

    missing = [p for p in qc.ANALYSIS_CODE_PATHS if not (ROOT / p).is_file()]
    if missing:
        print(f"REFUSING TO FREEZE: analysis code missing: {missing}")
        return 2

    fingerprint = qs.build_frozen_fingerprint(ROOT)
    print(f"frozen record    {fingerprint['n_files']} files  "
          f"{fingerprint['combined_sha256'][:16]}...")

    prereg = build(panel, reference, fingerprint)
    digest = provenance.write_json(ROOT / qc.PREREG_JSON_PATH, prereg)
    provenance.write_checksum_file(ROOT / qc.PREREG_JSON_PATH, ROOT / qc.PREREG_SHA_PATH)

    document = render(prereg, digest)
    (ROOT / qc.PREREG_MD_PATH).write_text(document, encoding="utf-8", newline="\n")
    (ROOT / qc.PREREG_FROZEN_COPY_PATH).write_text(document, encoding="utf-8", newline="\n")
    provenance.write_json(ROOT / qc.FROZEN_FINGERPRINT_PATH, fingerprint)

    print()
    print(f"frozen_at              {prereg['frozen_at']}")
    print(f"preregistration.json   {digest}")
    print(f"document sha256        {provenance.sha256_file(ROOT / qc.PREREG_MD_PATH)}")
    print("amendments             0")
    print("FREEZE COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
