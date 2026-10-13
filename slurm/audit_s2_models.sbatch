#!/bin/bash
#SBATCH --job-name=fcrrc2_audit
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc31
#SBATCH --time=04:00:00
#SBATCH --output=<CLUSTER_HOME>/FormalCRRC/logs/s2_audit_%j.out
#SBATCH --error=<CLUSTER_HOME>/FormalCRRC/logs/s2_audit_%j.err

# S2 Phase D: continuity audit against the immutable S1 provenance.
# Loads each judge at the exact S1 revision and re-checks every
# scoring-relevant property. No FormalCRRC study prompt is scored here.

set -uo pipefail
source <CLUSTER_HOME>/FormalCRRC/slurm/common.sh
cd "$BASE"

report_node

"$PY" scripts/audit_s2_models.py \
  --root "$BASE" \
  --hf-token-file "$HF_TOKEN_FILE" \
  --cache-dir "$HF_HOME/hub"
rc=$?

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== DONE rc=$rc ==="
exit $rc
