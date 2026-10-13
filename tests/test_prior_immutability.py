"""ss 1 and 2 must survive s 3 untouched.

Two guarantees: their recorded outputs still hash to what they hashed to, and no
s-3 code path is capable of writing into a prior experiment's location.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from formalcrrc import dataset as ds
from formalcrrc import s3, s3_config, prompts

MANIFEST = "artifacts/s3/prior_experiments_manifest.json"

#: s-3 code that must never write into a s-1 or s-2 location.
s3_SOURCES = (
    "scripts/generate_s3_dataset.py",
    "scripts/freeze_s3_preregistration.py",
    "scripts/verify_s3_ready.py",
    "scripts/run_s3_inference.py",
    "scripts/compute_s3_oracle.py",
    "scripts/analyze_s3.py",
    "scripts/figures_s3.py",
    "scripts/verify_s3_integrity.py",
    "scripts/audit_s3_priors.py",
    "src/formalcrrc/s3.py",
    "src/formalcrrc/s3_bootstrap.py",
    "src/formalcrrc/s3_config.py",
)

PRIOR_WRITE_TARGETS = ("artifacts/s1", "artifacts/s2", "figures/s1", "figures/s2")


class TestPriorManifest:
    def test_manifest_exists(self, repo_root: Path):
        assert (repo_root / MANIFEST).is_file(), "run scripts/audit_s3_priors.py"

    def test_manifest_covers_every_prior_file(self, repo_root: Path):
        manifest = json.loads((repo_root / MANIFEST).read_text(encoding="utf-8"))
        recorded = set(manifest["files"])
        for tree in s3_config.PRIOR_TREES:
            for path in (repo_root / tree).rglob("*"):
                if path.is_file():
                    assert path.relative_to(repo_root).as_posix() in recorded

    def test_manifest_covers_the_prior_documents(self, repo_root: Path):
        manifest = json.loads((repo_root / MANIFEST).read_text(encoding="utf-8"))
        for relative in s3_config.PRIOR_DOCS:
            if (repo_root / relative).is_file():
                assert relative in manifest["files"], relative

    def test_both_experiments_are_separately_recorded(self, repo_root: Path):
        manifest = json.loads((repo_root / MANIFEST).read_text(encoding="utf-8"))
        assert manifest["by_experiment"]["s1"]["n_files"] > 0
        assert manifest["by_experiment"]["s2"]["n_files"] > 0

    def test_nothing_changed(self, repo_root: Path):
        manifest = json.loads((repo_root / MANIFEST).read_text(encoding="utf-8"))
        verification = s3.verify_prior_manifest(manifest, repo_root)
        assert verification["s1"]["changed"] == []
        assert verification["s1"]["missing"] == []
        assert verification["s2"]["changed"] == []
        assert verification["s2"]["missing"] == []
        assert verification["status"] == "PASS"

    def test_combined_hash_refuses_an_empty_list(self):
        with pytest.raises(ValueError):
            s3._combined({})


class TestPriorBehaviourPreserved:
    def test_s1_dataset_still_reproduces(self, formal_dataset):
        assert (
            ds.dataset_content_hash(formal_dataset)
            == s3_config.s1_DATASET_CONTENT_SHA256
        )

    def test_s2_partitions_still_reproduce(
        self, repo_root: Path, s2_calibration, s2_test
    ):
        for path, frame in (
            ("artifacts/s2/calibration_manifest.json", s2_calibration),
            ("artifacts/s2/test_manifest.json", s2_test),
        ):
            manifest = json.loads((repo_root / path).read_text(encoding="utf-8"))
            assert manifest["hashes"]["dataset_content_sha256"] == (
                ds.dataset_content_hash(frame)
            )

    def test_prompt_template_unchanged_across_all_three_ss(self, repo_root: Path):
        for path in (
            "artifacts/s1/preregistration.json",
            "artifacts/s2/preregistration.json",
        ):
            document = json.loads((repo_root / path).read_text(encoding="utf-8"))
            assert prompts.prompt_template_sha256() == (
                document["prompt"]["template_sha256"]
            )

    def test_s2_pinned_source_files_unchanged(self, repo_root: Path):
        """s 3 must not have edited any file s 2 pinned by hash."""
        from formalcrrc import provenance

        manifest = json.loads(
            (repo_root / "artifacts/s2/integrity_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        for relative, recorded in manifest["source_manifest"]["files"].items():
            if recorded is None:
                continue
            candidate = repo_root / relative
            assert candidate.is_file(), relative
            assert provenance.sha256_file(candidate) == recorded, relative

    def test_s1_pinned_sources_changed_only_by_the_documented_extension(
        self, repo_root: Path
    ):
        """One s-1 source hash is stale, and it was s 2 that made it so.

        s 2 extended ``dataset.py`` backward-compatibly and recorded the
        change in its own preregistration. That file must therefore match s
        2's recorded hash rather than s 1's, and nothing else may differ.
        """
        from formalcrrc import s2_config, provenance

        s1 = json.loads(
            (repo_root / "artifacts/s1/integrity_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        s2 = json.loads(
            (repo_root / "artifacts/s2/integrity_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        documented = set(s2_config.s1_SOURCE_EXTENSIONS)

        differing = []
        for relative, recorded in s1["source_manifest"]["files"].items():
            if recorded is None:
                continue
            candidate = repo_root / relative
            assert candidate.is_file(), relative
            if provenance.sha256_file(candidate) != recorded:
                differing.append(relative)

        assert set(differing) == documented, (
            f"unexpected divergence from the s-1 source manifest: "
            f"{sorted(set(differing) - documented)}"
        )
        for relative in differing:
            assert provenance.sha256_file(repo_root / relative) == (
                s2["source_manifest"]["files"][relative]
            ), relative

    def test_s3_introduced_no_new_source_extension(self):
        """The documented extension list is s 2's and s 3 adds nothing."""
        from formalcrrc import s2_config

        assert set(s2_config.s1_SOURCE_EXTENSIONS) == {
            "src/formalcrrc/dataset.py"
        }


class Tests3CannotWritePriors:
    @pytest.mark.parametrize("relative", s3_SOURCES)
    def test_no_prior_write_target_in_s3_code(self, repo_root: Path, relative: str):
        path = repo_root / relative
        if not path.is_file():
            pytest.skip(f"{relative} not present yet")
        for line in path.read_text(encoding="utf-8").splitlines():
            if "write_json" not in line and "to_parquet" not in line:
                continue
            for target in PRIOR_WRITE_TARGETS:
                assert target not in line, f"{relative}: {line.strip()}"

    def test_s3_config_paths_are_all_under_s3(self):
        for name in dir(s3_config):
            if not name.endswith("_PATH"):
                continue
            value = getattr(s3_config, name)
            if isinstance(value, str):
                assert value.startswith(("artifacts/s3", "figures/s3")), (
                    name,
                    value,
                )

    def test_s3_artifact_and_figure_dirs_are_namespaced(self):
        assert s3_config.ARTIFACT_DIR == "artifacts/s3"
        assert s3_config.FIGURE_DIR == "figures/s3"
