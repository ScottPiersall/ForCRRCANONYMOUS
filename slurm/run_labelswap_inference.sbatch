#!/bin/bash
#SBATCH --job-name=fcrrc_swap
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc31,evc45
#SBATCH --time=08:00:00
#SBATCH --output=<CLUSTER_HOME>/FormalCRRC/logs/labelswap_infer_%j.out
#SBATCH --error=<CLUSTER_HOME>/FormalCRRC/logs/labelswap_infer_%j.err

# Label-Swap robustness study: score the frozen S3 partition under the
# SWAPPED label mapping for ONE judge.
# Usage: sbatch slurm/run_labelswap_inference.sbatch <model_id>
# Console output is operational only; no metric is computed here.

set -uo pipefail
source <CLUSTER_HOME>/FormalCRRC/slurm/common.sh
cd "$BASE"

MODEL="${1:?usage: sbatch run_labelswap_inference.sbatch <model_id>}"

report_node
echo "model=$MODEL"
echo "condition=swapped"

"$PY" scripts/run_labelswap_inference.py \
  --root "$BASE" \
  --model "$MODEL" \
  --hf-token-file "$HF_TOKEN_FILE" \
  --cache-dir "$HF_HOME/hub"
rc=$?

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== DONE rc=$rc ==="
exit $rc
