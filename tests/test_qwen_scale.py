"""Tests for the Qwen2.5 same-family scale ablation.

Covers the two quantities this study defines (full-sequence recovery and oracle
minimum errors), the frozen 14B reference assertions the author required before
the freeze, the paired bootstrap wiring, and the descriptive trend labels.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from formalcrrc import s3
from formalcrrc import s3_config as d3c
from formalcrrc import qwen_scale as qs
from formalcrrc import qwen_scale_config as qc


def step(j_star: int) -> list[int]:
    return [1 if s < j_star else 0 for s in range(9)]


# --------------------------------------------------------------------------
# E* and FSR on hand-built curves
# --------------------------------------------------------------------------


def test_perfect_step_has_zero_oracle_errors():
    margins = [8, 7, 6, 5, -1, -2, -3, -4, -5]
    assert qs.oracle_min_errors(margins, step(4)) == 0
    assert qs.full_sequence_recovered(margins, step(4))


def test_wrong_location_but_right_order_is_fully_recovered():
    # Every verdict says "met", yet the ordering is perfect: one cut fixes it.
    margins = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert qs.oracle_min_errors(margins, step(3)) == 0
    assert qs.full_sequence_recovered(margins, step(3))


def test_single_inversion_costs_one_error():
    margins = [9, 8, 1, 6, 5, 4, 3, 2, 0]  # level 2 sits below levels 3..7
    truth = step(5)
    assert not qs.full_sequence_recovered(margins, truth)
    assert qs.oracle_min_errors(margins, truth) == 1


def test_tie_across_the_boundary_blocks_full_recovery():
    margins = [5, 4, 3, 3, 2, 1, 0, -1, -2]  # levels 2 and 3 tie, boundary at 3
    truth = step(3)
    assert not qs.full_sequence_recovered(margins, truth)
    assert qs.oracle_min_errors(margins, truth) == 1


def test_tie_within_one_side_is_harmless():
    margins = [5, 5, 5, 1, 1, 1, 0, 0, 0]
    assert qs.full_sequence_recovered(margins, step(3))
    assert qs.oracle_min_errors(margins, step(3)) == 0


def test_all_pass_and_all_fail_states_are_enumerated():
    reversed_curve = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # truth: first 2 met. Best cut is all-fail (2 errors); all-pass costs 7.
    assert qs.oracle_min_errors(reversed_curve, step(2)) == 2
    # truth: first 7 met. Best cut is all-pass (2 errors); all-fail costs 7.
    assert qs.oracle_min_errors(reversed_curve, step(7)) == 2
    errors = qs.scalar_cut_errors(reversed_curve, step(2))
    assert errors[0] == 7 and errors[-1] == 2


def test_constant_curve_has_two_states():
    errors = qs.scalar_cut_errors([0.0] * 9, step(4))
    assert errors.tolist() == [5, 4]


def test_constant_truth_is_vacuously_recovered_with_zero_errors():
    margins = [3, -1, 4, -1, 5, -9, 2, -6, 5]
    assert qs.full_sequence_recovered(margins, step(9))
    assert qs.oracle_min_errors(margins, step(9)) == 0


def test_oracle_errors_match_brute_force_and_fsr_identity():
    rng = np.random.default_rng(7)
    for _ in range(400):
        margins = rng.integers(-3, 4, size=9).astype(float)  # many ties
        j_star = int(rng.integers(1, 9))
        truth = np.array(step(j_star))
        values = sorted(set(margins))
        cuts = values + [values[-1] + 1.0] + [
            0.5 * (a + b) for a, b in zip(values, values[1:])
        ]
        brute = min(int(np.sum((margins >= c).astype(int) != truth)) for c in cuts)
        assert qs.oracle_min_errors(margins, truth) == brute
        assert qs.full_sequence_recovered(margins, truth) == (brute == 0)
        if brute == 0:
            assert s3.translation_reachable(margins, j_star)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        qs.oracle_min_errors([1, 2, 3], [1, 0, 0])
    with pytest.raises(ValueError):
        qs.oracle_min_errors([float("nan")] * 9, step(3))
    with pytest.raises(ValueError):
        qs.full_sequence_recovered([1] * 9, [2] * 9)


# --------------------------------------------------------------------------
# Frozen S3 panel and the 14B reference
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frozen_dataset(repo_root):
    return pd.read_parquet(repo_root / d3c.DATASET_PATH)


@pytest.fixture(scope="module")
def reference_table(repo_root, frozen_dataset):
    raw = pd.read_parquet(repo_root / qc.REFERENCE_RAW_SCORES_PATH)
    return qs.artifact_table(qs.attach_truth(raw, frozen_dataset), qc.REFERENCE_TAG)


def test_panel_structure(frozen_dataset):
    verdict = qs.validate_panel(frozen_dataset)
    assert verdict["status"] == "PASS", verdict["checks"]
    assert verdict["n_artifacts"] == 324
    assert verdict["n_nontrivial"] == 288
    assert verdict["checks"]["every_artifact_has_exactly_9_thresholds"]


def test_reference_fsrr_is_136_over_288(reference_table):
    nontrivial = reference_table[reference_table["nontrivial"]]
    assert len(reference_table) == 324
    assert len(nontrivial) == 288
    assert int(nontrivial["full_sequence_recovered"].sum()) == 136
    assert qs.metric_value(reference_table, "fsrr") == pytest.approx(136 / 288)


def test_reference_fsr_iff_zero_oracle_errors(reference_table):
    nontrivial = reference_table[reference_table["nontrivial"]]
    assert (
        nontrivial["full_sequence_recovered"].astype(bool)
        == (nontrivial["oracle_min_errors"] == 0)
    ).all()


def test_reference_counts_reproduce_exactly(repo_root, reference_table):
    frozen = pd.read_parquet(repo_root / d3c.METRICS_PATH)
    frozen = frozen[frozen["model_id"] == qc.MODEL_IDS[qc.REFERENCE_TAG]]
    verdict = qs.verify_reference(reference_table, frozen)
    assert verdict["status"] == "PASS", verdict
    assert qs.metric_value(reference_table, "trr") == pytest.approx(138 / 288)
    assert qs.metric_value(reference_table, "oracle_errors") == pytest.approx(201 / 288)
    assert qs.metric_value(reference_table, "oracle_errors_all") == pytest.approx(
        201 / 324
    )
    assert qs.metric_value(reference_table, "tce") == pytest.approx(3.373457, abs=1e-6)
    assert qs.metric_value(reference_table, "accuracy") == pytest.approx(
        0.767833, abs=1e-6
    )


def test_constant_truth_artifacts_have_zero_oracle_error(reference_table):
    constant = reference_table[~reference_table["nontrivial"]]
    assert len(constant) == 36
    assert (constant["oracle_min_errors"] == 0).all()


def test_attach_truth_rejects_duplicates_and_missing(repo_root, frozen_dataset):
    raw = pd.read_parquet(repo_root / qc.REFERENCE_RAW_SCORES_PATH)
    with pytest.raises(ValueError):
        qs.attach_truth(pd.concat([raw, raw.head(1)]), frozen_dataset)
    with pytest.raises(ValueError):
        qs.attach_truth(raw.iloc[1:], frozen_dataset)


# --------------------------------------------------------------------------
# Paired bootstrap, tests, trends
# --------------------------------------------------------------------------


def _ladder(reference_table):
    """Three aligned rungs: 14B itself plus two deterministic perturbations."""
    worse = reference_table.copy()
    worse["tce"] = worse["tce"] + 1
    worse["translation_reachable"] = False
    better = reference_table.copy()
    better["tce"] = 0
    better["translation_reachable"] = True
    return {qc.TAG_7B: worse, qc.TAG_14B: reference_table, qc.TAG_32B: better}


def test_bootstrap_is_paired_and_deterministic(reference_table):
    tables = _ladder(reference_table)
    first = qs.bootstrap_scale(tables, replicates=200, seed=3)
    second = qs.bootstrap_scale(tables, replicates=200, seed=3)
    assert first == second

    delta = first["paired"]["7b_minus_14b"]["metrics"]["tce"]
    # A constant +1 shift on every artifact has no paired resampling variance.
    assert delta["point"] == pytest.approx(1.0)
    assert delta["ci_low"] == pytest.approx(1.0) and delta["ci_high"] == pytest.approx(1.0)

    trr_32 = first["models"][qc.TAG_32B]["trr"]
    assert trr_32["point"] == 1.0
    assert trr_32["exact_interval"]["method"] == "clopper_pearson"
    assert trr_32["exact_interval"]["ci_low"] < 1.0

    identical = first["paired"]["32b_minus_14b"]["metrics"]["accuracy"]
    assert identical["point"] == 0.0 and not identical["excludes_zero"]
    assert set(first["by_family"]) == {"coverage", "max_violation", "numeric_tolerance"}


def test_misaligned_tables_are_refused(reference_table):
    tables = _ladder(reference_table)
    tables[qc.TAG_7B] = tables[qc.TAG_7B].iloc[1:]
    with pytest.raises(ValueError):
        qs.align_tables(tables)


def test_mcnemar_counts_and_holm():
    new = np.array([1, 1, 1, 0, 0, 1], dtype=bool)
    ref = np.array([0, 0, 1, 1, 0, 0], dtype=bool)
    result = qs.mcnemar_exact(new, ref)
    assert (result["gains"], result["losses"]) == (3, 1)
    assert result["p_value"] == pytest.approx(0.625)
    assert qs.mcnemar_exact(ref, ref)["p_value"] == 1.0

    adjusted = qs.holm_adjust([0.01, 0.04, 0.03, 0.5])
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert all(a >= p for a, p in zip(adjusted, [0.01, 0.04, 0.03, 0.5]))


def test_mcnemar_family_has_exactly_four_tests(reference_table):
    tests = qs.mcnemar_family(_ladder(reference_table))
    assert len(tests) == 4
    assert {(t["new"], t["outcome"]) for t in tests} == set(
        itertools.product(qc.NEW_TAGS, qc.MCNEMAR_OUTCOMES)
    )
    assert all(t["n_pairs"] == 288 for t in tests)


@pytest.mark.parametrize(
    "values, higher, expected",
    [
        ([0.2, 0.5, 0.9], True, "monotonic improvement"),
        ([0.2, 0.2, 0.9], True, "monotonic improvement"),
        ([0.9, 0.5, 0.2], True, "monotonic decline"),
        ([3.0, 2.0, 1.0], False, "monotonic improvement"),
        ([1.0, 2.0, 2.0], False, "monotonic decline"),
        ([0.5, 0.5, 0.5], True, "flat"),
        ([0.2, 0.9, 0.5], True, "not monotonic"),
        ([2.0, 1.0, 3.0], False, "not monotonic"),
    ],
)
def test_trend_labels(values, higher, expected):
    assert qs.trend_label(values, higher) == expected
    assert expected in qc.TREND_LABELS


def test_interpretation_cases(reference_table):
    tables = _ladder(reference_table)
    summaries = {tag: qs.summarise_model(t) for tag, t in tables.items()}
    bootstrap = qs.bootstrap_scale(tables, replicates=200, seed=3)
    trends = qs.scale_trends(summaries, bootstrap)
    cases = qs.interpretation_cases(summaries, bootstrap, trends)
    # TRR is 0 -> 138/288 -> 1 here: strict, significant improvement to perfect.
    assert trends["trr"]["label"] == "monotonic improvement"
    assert cases["cases"]["B"]["holds"] and cases["cases"]["C"]["holds"]
    assert not cases["cases"]["A"]["holds"] and not cases["cases"]["D"]["holds"]


# --------------------------------------------------------------------------
# Configuration invariants
# --------------------------------------------------------------------------


def test_config_is_internally_consistent():
    assert qc.LADDER == (qc.TAG_7B, qc.TAG_14B, qc.TAG_32B)
    assert set(qc.NEW_TAGS) | {qc.REFERENCE_TAG} == set(qc.LADDER)
    assert set(qc.PRIMARY_METRICS) == set(qc.HIGHER_IS_BETTER) | set(qc.LOWER_IS_BETTER)
    for metric in qc.PRIMARY_METRICS + qc.SECONDARY_METRICS:
        assert metric in qc.METRIC_POPULATION and metric in qc.METRIC_COLUMN
        assert qc.METRIC_COLUMN[metric] in qs.ARTIFACT_VALUE_COLUMNS
    for metric in ("trr", "fsrr", "oracle_errors"):
        assert qc.METRIC_POPULATION[metric] == "nontrivial"
    assert qc.N_ARTIFACTS * qc.N_STRICTNESS_LEVELS == qc.N_ROWS == 2916


def test_reference_revision_matches_frozen_s3_provenance(repo_root):
    import json

    document = json.loads(
        (repo_root / d3c.MODEL_PROVENANCE_PATH).read_text(encoding="utf-8")
    )
    record = document["models"][qc.MODEL_IDS[qc.REFERENCE_TAG]]
    assert record["revision"] == qc.REFERENCE_REVISION
    assert record["tokenizer_revision"] == qc.REFERENCE_TOKENIZER_REVISION
    assert record["chat_template_sha256"] == qc.REFERENCE_CHAT_TEMPLATE_SHA256
    assert record["smoke_test"]["prompt"] == qc.SMOKE_PROMPT
    assert record["smoke_test"]["raw_score_met"] == qc.REFERENCE_SMOKE_SCORE_MET
    assert record["smoke_test"]["raw_score_not_met"] == qc.REFERENCE_SMOKE_SCORE_NOT_MET


def test_study_paths_do_not_touch_protected_trees():
    for path in (qc.ARTIFACT_DIR, qc.FIGURE_DIR):
        assert path not in qc.PROTECTED_TREES
    assert all(not qc.ARTIFACT_DIR.startswith(tree + "/") for tree in qc.PROTECTED_TREES)


# --------------------------------------------------------------------------
# Amendment 001
# --------------------------------------------------------------------------


def test_primary_endpoint_is_nontrivial_trr():
    assert qc.PRIMARY_ENDPOINT == "trr"
    assert qc.METRIC_POPULATION[qc.PRIMARY_ENDPOINT] == "nontrivial"
    assert set(qc.SUPPORTING_ENDPOINTS) | {qc.PRIMARY_ENDPOINT} == set(qc.PRIMARY_METRICS)
    assert qc.SCALE_STEP_DEFINITIONS == {
        "Delta_7_to_14": "TRR_14B - TRR_7B",
        "Delta_14_to_32": "TRR_32B - TRR_14B",
    }


def test_scale_steps_are_signed_larger_minus_smaller(reference_table):
    tables = _ladder(reference_table)
    summaries = {tag: qs.summarise_model(t) for tag, t in tables.items()}
    bootstrap = qs.bootstrap_scale(tables, replicates=200, seed=3)
    cases = qs.interpretation_cases(
        summaries, bootstrap, qs.scale_trends(summaries, bootstrap)
    )
    trr = cases["trr"]
    lower, upper = cases["scale_steps"]["Delta_7_to_14"], cases["scale_steps"]["Delta_14_to_32"]
    assert lower["point"] == pytest.approx(trr[qc.TAG_14B] - trr[qc.TAG_7B])
    assert upper["point"] == pytest.approx(trr[qc.TAG_32B] - trr[qc.TAG_14B])
    assert lower["point"] > 0 and lower["ci_low"] <= lower["point"] <= lower["ci_high"]
    assert cases["primary_endpoint"] == "trr"


def test_amendment_supersedes_the_code_manifest(tmp_path):
    from formalcrrc import provenance

    pinned = tmp_path / "a.py"
    pinned.parent.mkdir(parents=True, exist_ok=True)
    pinned.write_text("one\n", encoding="utf-8")
    prereg = tmp_path / qc.PREREG_JSON_PATH
    provenance.write_json(
        prereg,
        {"analysis_code": {"files": {"a.py": provenance.sha256_file(pinned)}, "tree_sha256": "t0"}},
    )
    original = provenance.write_checksum_file(prereg, tmp_path / qc.PREREG_SHA_PATH)
    assert qs.effective_code_manifest(tmp_path)["source"] == "freeze"
    assert qs.changed_pinned_files(tmp_path) == []

    pinned.write_text("two\n", encoding="utf-8")
    assert qs.changed_pinned_files(tmp_path) == ["a.py"]

    amendment = tmp_path / qc.amendment_json_path(1)
    provenance.write_json(
        amendment,
        {
            "number": 1,
            "preregistration_sha256": original,
            "analysis_code": {
                "files": {"a.py": provenance.sha256_file(pinned)},
                "tree_sha256": "t1",
            },
        },
    )
    with pytest.raises(ValueError):  # an amendment without a checksum is refused
        qs.effective_code_manifest(tmp_path)
    provenance.write_checksum_file(amendment, tmp_path / qc.amendment_sha_path(1))
    in_force = qs.effective_code_manifest(tmp_path)
    assert in_force["source"] == "amendment 001" and in_force["n_amendments"] == 1
    assert qs.changed_pinned_files(tmp_path) == []
    assert provenance.verify_checksum_file(prereg, tmp_path / qc.PREREG_SHA_PATH)

    amendment.write_text(amendment.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError):  # a tampered amendment is refused
        qs.effective_code_manifest(tmp_path)
