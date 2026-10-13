#!/usr/bin/env python
"""Record a numbered amendment to the Qwen2.5 scale-ablation preregistration.

The original freeze is never rewritten. It must still verify against its
checksum when this runs, and it is left byte-identical. The amendment is a
separate checksummed record stating the defect, why the change is
outcome-independent, exactly what changed, and the pinned code manifest that is
in force from this point on.

An amendment that claims to precede inference is only written if that is true:
the script refuses when any 7B or 32B result, partial result, receipt or run
record exists.

Usage: amend_qwen_scale_preregistration.py --number 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import provenance  # noqa: E402
from formalcrrc import qwen_scale as qs  # noqa: E402
from formalcrrc import qwen_scale_config as qc  # noqa: E402

ROOT = provenance.repo_root()


def inference_outputs() -> list[str]:
    paths = []
    for tag in qc.NEW_TAGS:
        paths += [
            qc.results_path(tag),
            qc.partial_results_path(tag),
            qc.run_record_path(tag),
            qc.receipts_path(tag),
        ]
    paths.append(qc.SUMMARY_PATH)
    return [p for p in paths if (ROOT / p).exists()]


def render(amendment: dict, digest: str) -> str:
    out: list[str] = []
    add = out.append
    add(f"# {qc.STUDY_NAME} — Amendment {amendment['number']:03d}")
    add("")
    add(f"**{amendment['title']}**")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Created at (UTC) | `{amendment['created_at']}` |")
    add(f"| Amends freeze | `{amendment['preregistration_sha256']}` "
        f"(frozen `{amendment['preregistration_frozen_at']}`) |")
    add(f"| Machine form | `{qc.amendment_json_path(amendment['number'])}` |")
    add(f"| Machine form SHA-256 | `{digest}` |")
    add(f"| 7B/32B inference existed | {amendment['inference_existed']} |")
    add(f"| Revisions already pinned | {amendment['model_revisions_existed']} |")
    add("")
    add("The original preregistration is preserved byte-identical and still "
        "verifies against its checksum.")
    add("")
    add("## 1. The defect")
    add("")
    add(amendment["defect"])
    add("")
    add("## 2. Why the change is outcome-independent")
    add("")
    add(amendment["outcome_independence"])
    add("")
    add("## 3. Exactly what changed")
    add("")
    add(amendment["what_changed"])
    add("")
    add("### Clarifications now in force")
    add("")
    add(f"**Primary endpoint.** {amendment['clarifications']['primary_endpoint']}")
    add("")
    add(f"**Interpretation cases.** {amendment['clarifications']['case_status']}")
    add("")
    for name, definition in amendment["clarifications"]["scale_steps"].items():
        add(f"- `{name} = {definition}`")
    add("")
    add(f"**Revision pinning.** {amendment['clarifications']['revision_pinning']}")
    add("")
    add("**Unchanged.** The four prespecified exact McNemar tests with Holm "
        "correction; the 32B − 7B contrast, descriptive only; every metric "
        "definition, population, bootstrap setting, case condition and protocol "
        "element. No figure is produced.")
    add("")
    add("### Pinned code")
    add("")
    add("| File | Status | SHA-256 now in force |")
    add("|---|---|---|")
    for name, entry in amendment["code_changes"].items():
        add(f"| `{name}` | {entry['status']} | `{entry['sha256']}` |")
    add("")
    add(f"Pinned tree SHA-256 now in force: `{amendment['analysis_code']['tree_sha256']}`.")
    add("")
    add("### Provenance-only code (hashed, not execution-blocking)")
    add("")
    for name, sha in amendment["provenance_only_code"]["files"].items():
        add(f"- `{name}` `{sha}`")
    add("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--number", type=int, required=True)
    args = parser.parse_args()
    number = args.number

    if number not in qc.AMENDMENTS:
        print(f"REFUSING: no amendment {number} is defined in qwen_scale_config")
        return 2
    if (ROOT / qc.amendment_json_path(number)).exists():
        print(f"REFUSING: amendment {number:03d} already exists")
        return 2
    existing = qs.load_amendments(ROOT)
    if len(existing) != number - 1:
        print(f"REFUSING: {len(existing)} amendments exist; cannot write {number:03d}")
        return 2
    if not provenance.verify_checksum_file(
        ROOT / qc.PREREG_JSON_PATH, ROOT / qc.PREREG_SHA_PATH
    ):
        print("REFUSING: the original preregistration does not verify")
        return 2
    outputs = inference_outputs()
    if outputs:
        print("REFUSING: this amendment claims to precede inference, but these exist:")
        for path in outputs:
            print(f"  {path}")
        return 2
    missing = [p for p in qc.ANALYSIS_CODE_PATHS if not (ROOT / p).is_file()]
    if missing:
        print(f"REFUSING: pinned code missing: {missing}")
        return 2

    prereg = json.loads((ROOT / qc.PREREG_JSON_PATH).read_text(encoding="utf-8"))
    previous = qs.effective_code_manifest(ROOT)["files"]
    current = provenance.source_manifest(qc.ANALYSIS_CODE_PATHS, ROOT)
    changes = {}
    for name, sha in current["files"].items():
        if name not in previous:
            status = "added"
        elif previous[name] != sha:
            status = "changed"
        else:
            status = "unchanged"
        changes[name] = {"status": status, "sha256": sha, "previous_sha256": previous.get(name)}

    text = qc.AMENDMENTS[number]
    amendment = {
        "study": qc.STUDY_NAME,
        "number": number,
        "title": text["title"],
        "created_at": provenance.utc_now(),
        "preregistration_sha256": (ROOT / qc.PREREG_SHA_PATH)
        .read_text(encoding="utf-8")
        .split()[0],
        "preregistration_frozen_at": prereg["frozen_at"],
        "inference_existed": False,
        "model_revisions_existed": (ROOT / qc.MODEL_REVISIONS_PATH).exists(),
        "defect": text["defect"],
        "outcome_independence": text["outcome_independence"],
        "what_changed": text["what_changed"],
        "clarifications": {
            "primary_endpoint": qc.PRIMARY_ENDPOINT_STATEMENT,
            "primary_endpoint_metric": qc.PRIMARY_ENDPOINT,
            "supporting_endpoints": list(qc.SUPPORTING_ENDPOINTS),
            "case_status": qc.CASE_STATUS,
            "scale_steps": qc.SCALE_STEP_DEFINITIONS,
            "revision_pinning": qc.REVISION_PINNING_RULE,
        },
        "code_changes": changes,
        "analysis_code": current,
        "provenance_only_code": provenance.source_manifest(
            qc.PROVENANCE_ONLY_CODE_PATHS, ROOT
        ),
        "document": {"path": qc.amendment_md_path(number)},
    }
    digest = provenance.write_json(ROOT / qc.amendment_json_path(number), amendment)
    provenance.write_checksum_file(
        ROOT / qc.amendment_json_path(number), ROOT / qc.amendment_sha_path(number)
    )
    (ROOT / qc.amendment_md_path(number)).write_text(
        render(amendment, digest), encoding="utf-8", newline="\n"
    )

    print(f"AMENDMENT {number:03d} RECORDED")
    print(f"created_at   {amendment['created_at']}")
    print(f"sha256       {digest}")
    for name, entry in changes.items():
        if entry["status"] != "unchanged":
            print(f"  {entry['status']:8s} {name}")
    print("original preregistration verifies: "
          f"{provenance.verify_checksum_file(ROOT / qc.PREREG_JSON_PATH, ROOT / qc.PREREG_SHA_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
