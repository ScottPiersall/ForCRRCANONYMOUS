#!/usr/bin/env python
"""Phase A: fingerprint the completed s-1 and s-2 experiments.

Read-only with respect to every prior file. Writes only
``artifacts/s3/prior_experiments_manifest.json``, which is the reference the
final s-3 integrity check re-verifies against.

Also confirms, as a separate check, that no source file either prior experiment
pinned by hash has been modified — s 3 extends shared modules only by adding
new ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import dataset as ds  # noqa: E402
from formalcrrc import s3, s3_config as d3c, prompts, provenance  # noqa: E402


def check_prior_source_files(root: Path) -> dict:
    """Every source hash a prior integrity manifest pinned must still hold.

    One documented exception exists and predates s 3. s 2 extended
    ``src/formalcrrc/dataset.py`` backward-compatibly to accept a seed and a
    partition tag, recording the change in its own preregistration under
    ``s1_SOURCE_EXTENSIONS``. That file therefore no longer matches the hash
    s 1 recorded, while matching the one s 2 recorded, and s 1's dataset
    still reproduces its frozen content hash.

    A change is classified EXPLAINED only if it is on that documented list *and*
    agrees with the later manifest. Anything else is UNEXPLAINED, and s 3 must
    add nothing to either category.
    """
    from formalcrrc import s2_config

    documented = set(s2_config.s1_SOURCE_EXTENSIONS)
    s2_manifest = json.loads(
        (root / "artifacts/s2/integrity_manifest.json").read_text(encoding="utf-8")
    )
    s2_recorded = s2_manifest["source_manifest"]["files"]

    result: dict = {}
    for s, relative in (
        ("s1", "artifacts/s1/integrity_manifest.json"),
        ("s2", "artifacts/s2/integrity_manifest.json"),
    ):
        manifest = json.loads((root / relative).read_text(encoding="utf-8"))
        recorded = manifest["source_manifest"]["files"]
        explained, unexplained = [], []
        for path, digest in recorded.items():
            if digest is None:
                continue
            candidate = root / path
            if not candidate.is_file():
                unexplained.append(path)
                continue
            current = provenance.sha256_file(candidate)
            if current == digest:
                continue
            if (
                s == "s1"
                and path in documented
                and s2_recorded.get(path) == current
            ):
                explained.append(path)
            else:
                unexplained.append(path)
        result[s] = {
            "n_files": len(recorded),
            "explained_by_documented_s2_extension": explained,
            "unexplained_changes": unexplained,
            "status": "PASS" if not unexplained else "FAIL",
            "recorded_tree_sha256": manifest["source_manifest"]["tree_sha256"],
        }
    result["documented_s2_extensions"] = sorted(documented)
    result["s3_added_no_extension"] = (
        set(result["s1"]["explained_by_documented_s2_extension"]) <= documented
        and not result["s1"]["unexplained_changes"]
        and not result["s2"]["unexplained_changes"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = Path(args.root)

    manifest_path = root / d3c.PRIOR_MANIFEST_PATH
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"prior manifest already exists ({manifest['n_files']} files)", flush=True)
    else:
        manifest = s3.build_prior_manifest(root)
        provenance.write_json(manifest_path, manifest)
        print(
            f"wrote {d3c.PRIOR_MANIFEST_PATH}  ({manifest['n_files']} files: "
            f"{manifest['by_experiment']['s1']['n_files']} s-1, "
            f"{manifest['by_experiment']['s2']['n_files']} s-2)",
            flush=True,
        )

    verification = s3.verify_prior_manifest(manifest, root)
    print("", flush=True)
    print(f"s-1 files: {verification['s1']['status']} "
          f"(changed={verification['s1']['changed'] or 'none'})", flush=True)
    print(f"s-2 files: {verification['s2']['status']} "
          f"(changed={verification['s2']['changed'] or 'none'})", flush=True)
    print(f"added: {verification['added'] or 'none'}", flush=True)

    sources = check_prior_source_files(root)
    print("", flush=True)
    for s in ("s1", "s2"):
        block = sources[s]
        print(
            f"{s} pinned source files: {block['status']} "
            f"({block['n_files']} files)",
            flush=True,
        )
        if block["explained_by_documented_s2_extension"]:
            print(
                "    explained by the documented s-2 extension: "
                f"{block['explained_by_documented_s2_extension']}",
                flush=True,
            )
        if block["unexplained_changes"]:
            print(f"    UNEXPLAINED: {block['unexplained_changes']}", flush=True)
    print(f"s 3 added no source extension: {sources['s3_added_no_extension']}",
          flush=True)

    s1_hash = ds.dataset_content_hash(ds.build_dataset())
    reproduces = s1_hash == d3c.s1_DATASET_CONTENT_SHA256
    print("", flush=True)
    print(f"s-1 dataset reproduces: {reproduces} ({s1_hash[:16]})", flush=True)
    print(
        f"prompt template: {prompts.prompt_template_sha256()[:16]}",
        flush=True,
    )

    ok = (
        verification["status"] == "PASS"
        and sources["s1"]["status"] == "PASS"
        and sources["s2"]["status"] == "PASS"
        and sources["s3_added_no_extension"]
        and reproduces
    )
    print("", flush=True)
    print(f"PRIOR_EXPERIMENT_IMMUTABILITY = {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
