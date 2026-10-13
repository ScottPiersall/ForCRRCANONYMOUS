#!/bin/bash
#SBATCH --job-name=fcrrc2_infer
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc31,evc45
#SBATCH --time=08:00:00
#SBATCH --output=<CLUSTER_HOME>/FormalCRRC/logs/s2_infer_%j.out
#SBATCH --error=<CLUSTER_HOME>/FormalCRRC/logs/s2_infer_%j.err

# S2 Phase F: score one judge over BOTH partitions.
# Usage: sbatch slurm/run_s2_inference.sbatch <model_id>
# Console output is operational only; no calibration is fitted here.

set -uo pipefail
source <CLUSTER_HOME>/FormalCRRC/slurm/common.sh
cd "$BASE"

MODEL="${1:?usage: sbatch run_s2_inference.sbatch <model_id>}"

report_node
echo "model=$MODEL"

for SPLIT in calibration test; do
  echo "--- partition=$SPLIT"
  "$PY" scripts/run_s2_inference.py \
    --root "$BASE" \
    --model "$MODEL" \
    --partition "$SPLIT" \
    --hf-token-file "$HF_TOKEN_FILE" \
    --cache-dir "$HF_HOME/hub" || { echo "FATAL: $SPLIT failed"; exit 1; }
done

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== DONE rc=0 ==="
