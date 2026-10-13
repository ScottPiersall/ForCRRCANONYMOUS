#!/bin/bash
#SBATCH --job-name=fcrrc3_audit
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc31,evc45
#SBATCH --time=04:00:00
#SBATCH --output=<CLUSTER_HOME>/FormalCRRC/logs/s3_audit_%j.out
#SBATCH --error=<CLUSTER_HOME>/FormalCRRC/logs/s3_audit_%j.err

# S3 Phase D: continuity audit against the immutable S1/S2 provenance.
# No FormalCRRC study prompt is scored here.

set -uo pipefail
source <CLUSTER_HOME>/FormalCRRC/slurm/common.sh
cd "$BASE"

report_node

"$PY" scripts/audit_s3_models.py \
  --root "$BASE" \
  --hf-token-file "$HF_TOKEN_FILE" \
  --cache-dir "$HF_HOME/hub"
rc=$?

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== DONE rc=$rc ==="
exit $rc
