#!/bin/bash
#SBATCH --job-name=fcrrc_qs_infer
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=160G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc24,evc25,evc26,evc28,evc29,evc31,evc33,evc34,evc35,evc38,evc40,evc42,evc43,evc44,evc45,evc46,evc48,evc49,evc50
#SBATCH --time=04:00:00
#SBATCH --output=<CLUSTER_HOME>/FormalCRRC/logs/qwen_scale_infer_%j.out
#SBATCH --error=<CLUSTER_HOME>/FormalCRRC/logs/qwen_scale_infer_%j.err

# Qwen2.5 scale ablation: score the frozen S3 panel with ONE new rung.
# Usage: sbatch slurm/run_qwen_scale_inference.sbatch <7b|32b>
# Console output is operational only; no metric is computed here.

set -uo pipefail
source <CLUSTER_HOME>/FormalCRRC/slurm/common.sh
cd "$BASE"

TAG="${1:?usage: sbatch run_qwen_scale_inference.sbatch <7b|32b>}"

report_node
echo "tag=$TAG"

timeout 180 "$PY" - <<'PY'
import sys, torch
try:
    torch.zeros(8, dtype=torch.bfloat16, device="cuda").sum().item()
except Exception as exc:
    print(f"CUDA smoke test failed: {type(exc).__name__}: {exc}", flush=True)
    sys.exit(1)
print("CUDA smoke test OK", flush=True)
PY
smoke=$?
if [ "$smoke" -ne 0 ]; then
  echo "FATAL: $(hostname) cannot create a CUDA context (rc=$smoke)"
  exit 1
fi

"$PY" scripts/run_qwen_scale_inference.py \
  --root "$BASE" \
  --tag "$TAG" \
  --hf-token-file "$HF_TOKEN_FILE" \
  --cache-dir "$HF_HOME/hub"
rc=$?

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== DONE rc=$rc ==="
exit $rc
