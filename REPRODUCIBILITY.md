# FormalCRRC reproducibility archive

This archive accompanies the paper *FormalCRRC: Locating and Repairing Rubric Decision Boundaries in LLM Judges*. It is the anonymized review release, version 1.1.0-review, built on 2026-10-11.

It contains the source code, frozen study artifacts (datasets, prompts, raw judge scores, preregistrations, integrity records), and the analysis code needed to reproduce the synthetic results in the paper. You can reproduce every analysis from the saved scores without a GPU and without model inference. See `SCOPE.md` for what is and is not included.

## Start here

| File | Purpose |
|---|---|
| `REPRODUCIBILITY.md` | This file: how to reproduce the results |
| `SCOPE.md` | What the archive covers and what it leaves out |
| `VALIDATION.md` | The checks run when this archive was built, and their results |
| `protocols/index.json` | Machine-readable map from each study to its preregistration, scripts, raw scores and integrity records |
| `REDACTIONS.md` | What was anonymized, and the effect on recorded hashes |
| `README.md` | The original project README (study-by-study pipeline) |
| `CITATION.cff`, `.zenodo.json`, `codemeta.json` | Citation and deposit metadata |
| `LICENSE`, `LICENSE-DATA.md`, `THIRD_PARTY_NOTICES.md` | Licensing |

## Layout

```
src/formalcrrc/          library: predicates, datasets, prompts, scoring, metrics, bootstrap, provenance
scripts/                 one script per study phase (generate, freeze, verify, infer, analyze, figures, results)
scripts/posthoc/         stand-alone post-hoc analyses reported in the paper (standard-library Python)
tests/                   test suite
artifacts/<study>/       frozen datasets, preregistrations, raw scores, metrics, integrity records
artifacts/posthoc_manuscript/   outputs of scripts/posthoc (as reproduced when this archive was built)
figures/<study>/         predeclared figures
slurm/, cluster/         job scripts used for GPU inference (site-specific values replaced by placeholders)
tables/                  LaTeX tables generated for the paper
```

## Studies and where they appear in the paper

| ID | Study | Artifacts | Paper |
|---|---|---|---|
| S1 | Threshold sensitivity, fresh partition (seed 42) | `artifacts/s1/` | Appendix "Reference Strategies, S1 Results, and Held-Out Correction" |
| S2 | Held-out additive calibration (CAL seed 43, TEST seed 44) | `artifacts/s2/` | Section 4.2, Q1 table; Q2 |
| S3 | Oracle boundary recoverability (seed 45) | `artifacts/s3/` | Section 4.2, Q2; appendices |
| LS | A/B label-swap counterbalance | `artifacts/labelswap/` | Section 4.2, Robustness; Appendix "Label-Swap Family Detail" |
| Q3 | Qwen3-30B-A3B native reasoning anchor | `artifacts/reasoning_anchor/` | Section 4.2, Q3 |
| RTS | Qwen2.5-14B reason-then-score ablation | `artifacts/reason_then_score_ablation/` | Section 4.2, Q3; Appendix "Synthetic Full-Sequence Recovery" |
| SCALE | Qwen2.5 7B/14B/32B scale ablation | `artifacts/qwen_scale_ablation/` | Section 4.2, Q1; Appendix "Same-Family Qwen2.5 Scale Ablation" |
| — | Artifact-blind references | `artifacts/poststudy_baselines/` | Section 4.1 |
| — | Reachability/theory audit | `artifacts/poststudy_theory/` | Appendices A and "Strict-Order Diagnostics" |
| — | Post-hoc manuscript analyses | `artifacts/posthoc_manuscript/` | See `protocols/index.json` |

All studies use $L=9$ threshold levels; each partition has 324 artifacts, 288 of them with a nontrivial boundary.

## Level 1: check the files (seconds)

```bash
sha256sum -c MANIFEST-SHA256.txt        # Linux; on macOS: shasum -a 256 -c MANIFEST-SHA256.txt
```

## Level 2: reproduce the post-hoc analyses in the paper (about 30 seconds, no dependencies)

Needs only Python 3.8 or later; `scripts/posthoc/minipq.py` reads the Parquet files without third-party packages.

```bash
bash scripts/posthoc/run_all.sh artifacts /tmp/posthoc_check
python3 - <<'PY'
import json
for f in ["grid_results.json", "tolerance_results.json", "mvr_results.json", "s2_tr_results.json",
          "reviewer_results.json", "reviewer_split.json"]:
    a = json.load(open("artifacts/posthoc_manuscript/" + f)); b = json.load(open("/tmp/posthoc_check/" + f))
    print(f, "identical" if a == b else "DIFFERENT")
PY
```

These scripts first recompute published values from the raw margins:
- S3 TRR and FSRR at $L=9$;
- S2 raw, global and family TCE;
- the saved Qwen3 and RTS MVR values.

They then compute the post-hoc quantities: $L=5$ re-scoring, the permutation null and 5% scale-normalized tolerance, MVR at several values of $\delta$, the TR split against held-out S2 correction, the level-0 analysis (share of first failures at $s=0$; TCE, accuracy, TRR and FSRR with and without $s=0$; refitted correction on levels 1–8), held-out S2 calibration at four granularities (global, family, level, level × family) with exact-placement rates split by TR, and family-stratified bootstrap intervals (5,000 replicates) for the main S2 table (`reviewer_results.json`, `reviewer_split.json`).

## Level 3: recompute each study's metrics from the frozen raw scores (minutes, CPU)

```bash
python -m venv .venv && . .venv/bin/activate      # Python 3.11 or later
pip install -e ".[dev]"
python -m pytest                                    # test suite
python scripts/analyze_s1.py                      # likewise analyze_s2/3, analyze_labelswap,
                                                    # analyze_reasoning_anchor, analyze_rts_ablation, analyze_qwen_scale
python scripts/compute_reference_baselines.py
python scripts/audit_step_proposition.py
```

The analysis scripts write into `artifacts/<study>/`. Run them on a copy of the archive if you want to compare their outputs with the frozen files. `protocols/index.json` lists, for each study, its analysis, figure and integrity-verification scripts.

## Level 4: rerun model inference (GPU, optional)

Inference uses Hugging Face checkpoints at the exact revisions recorded in each study's `model_provenance.json` or `model_revisions.json`, in bf16 without quantization. Install with `pip install -e ".[inference]"`.

The SLURM scripts in `slurm/` show the exact invocations. Before using them, set the placeholders `<CLUSTER_HOME>` and `<SLURM_ACCOUNT>` in `slurm/common.sh` for your site. Next-token logits can differ in the last bits across GPU types and library versions, which is why the analyses above start from the saved scores.

## Determinism

- Datasets are generated from fixed seeds (S1 42; S2 43/44; S3 45).
- Bootstrap seeds are recorded in each preregistration and in `scripts/posthoc/*.py`.
- The post-hoc permutation null is seeded per artifact, so reruns are bit-identical.
