#!/bin/bash
#SBATCH --job-name=fcrrc_infer
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc31
#SBATCH --time=08:00:00
#SBATCH --output=<CLUSTER_HOME>/FormalCRRC/logs/infer_%j.out
#SBATCH --error=<CLUSTER_HOME>/FormalCRRC/logs/infer_%j.err

# Phase E: score all 2916 rubric-response pairs for ONE judge model.
# Usage: sbatch slurm/run_inference.sbatch <model_id>
# Console output is operational only; no scientific summary is produced here.

set -uo pipefail
source <CLUSTER_HOME>/FormalCRRC/slurm/common.sh
cd "$BASE"

MODEL="${1:?usage: sbatch run_inference.sbatch <model_id>}"

report_node
echo "model=$MODEL"

"$PY" scripts/run_s1_inference.py \
  --root "$BASE" \
  --model "$MODEL" \
  --hf-token-file "$HF_TOKEN_FILE" \
  --cache-dir "$HF_HOME/hub"
rc=$?

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== DONE rc=$rc ==="
exit $rc
