# Validation record


## 1. Post-hoc analyses reproduced from the frozen scores

`scripts/posthoc/run_all.sh` was run on this archive. All six outputs are numerically identical to the values used in the paper: `grid_results.json`, `tolerance_results.json`, `mvr_results.json`, `s2_tr_results.json`, `reviewer_results.json` and `reviewer_split.json` (`mvr_results.json` and `reviewer_split.json` are also byte-identical). `reviewer_results.json` also reproduces the frozen S2 global and family calibration intercepts exactly (`check` fields).

Before computing anything new, the scripts recompute published values from the raw margins:

- **S3:** TRR and FSRR for all four immediate-readout judges, Qwen2.5 RTS and Qwen3, at $L=9$.
- **S2:** raw, global and family-corrected TCE for all four judges (3.556/3.216/2.954, 4.920/3.176/2.830, 2.867/3.077/2.284, 3.373/3.250/1.966).
- **MVR:** the saved Qwen3 values (0.418 at $\delta=0$, 0.095 at $\delta=0.05$) and RTS values (0.449, 0.034).

## 2. Recorded integrity hashes

Every per-file SHA-256 recorded in the study JSON records under `artifacts/` (integrity manifests, preregistrations, fingerprints, run records) was recomputed against this archive. That is 1,331 entries:

- **1,290 match.**
- **21 are aggregate hashes** (dataset content hashes and combined or source-tree digests), not single files.
- **20 do not match, and each is accounted for:**
  - **4 redacted provenance files.** `model_provenance.json` and `preflight.json` for RTS and Qwen3 differ only in the cluster cache path. They are referenced by later studies' fingerprints. See `REDACTIONS.md` for their pre-redaction hashes.
  - **1 self-referential entry.** `artifacts/labelswap/integrity_manifest.json` records a hash of itself, which can never match.
  - **Source files extended after an earlier freeze:**
    - `src/formalcrrc/config.py`, `dataset.py` and `prompts.py` as recorded at the S1 freeze;
    - `scripts/figures_reasoning_anchor.py`;
    - the five scale-ablation code files as recorded in the original scale preregistration.

    In every case the current file's hash is itself recorded in a later study record. For the scale ablation that record is Amendment 001 (`artifacts/qwen_scale_ablation/amendment_001.json`), made before inference. The exact earlier versions are in the project's git history, which the anonymized release omits (`SCOPE.md`).

## 3. Not run in this build environment

The test suite (`python -m pytest`) and the per-study analysis scripts need Python 3.11 or later with numpy, pandas, pyarrow and scipy, which were not available offline in the environment that built this archive. Run them as described in `REPRODUCIBILITY.md` (Level 3). The paper's integrity record reports 1,097 repository tests passing at the time of the theory audit.

