#!/usr/bin/env bash
# Reproduce the post-hoc analyses reported in the manuscript from the frozen
# artifacts. Standard-library Python only (minipq.py reads Parquet); no model
# inference. Usage: bash scripts/posthoc/run_all.sh [artifacts_dir] [out_dir]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
A="${1:-$HERE/../../artifacts}"
OUT="${2:-$A/posthoc_manuscript}"
mkdir -p "$OUT"
python3 -I "$HERE/grid_robustness.py"   "$A" > "$OUT/grid_results.json"       # L=5 re-scoring (Appendix: robustness to L)
python3 -I "$HERE/tolerance_analysis.py" "$A" > "$OUT/tolerance_results.json"  # permutation null + 5% scale-normalized tolerance
python3 -I "$HERE/mvr_sensitivity.py"   "$A" > "$OUT/mvr_results.json"        # MVR at delta in {0, .01, .02, .05, .10}
python3 -I "$HERE/s2_tr_split.py"       "$A" > "$OUT/s2_tr_results.json"      # TR split vs held-out S2 correction
python3 -I "$HERE/reviewer_analyses.py" "$A" > "$OUT/reviewer_results.json"   # level 0; per-level calibration; S2 table intervals
python3 -I "$HERE/reviewer_split.py"    "$A" > "$OUT/reviewer_split.json"     # level x family fixes: TR=0 only-at-s0 vs interior
echo "wrote $OUT"
