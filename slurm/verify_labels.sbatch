#!/bin/bash
#SBATCH --job-name=fcrrc_labels
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH --exclude=evc31
#SBATCH --time=00:30:00
#SBATCH --output=<CLUSTER_HOME>/FormalCRRC/logs/labels_%j.out
#SBATCH --error=<CLUSTER_HOME>/FormalCRRC/logs/labels_%j.err

# Phase C: verify that the scored label token is the token each model must
# actually emit. Tokenizers only -- no GPU, no model weights, no study scoring.

set -uo pipefail
source <CLUSTER_HOME>/FormalCRRC/slurm/common.sh
cd "$BASE"

echo "host=$(hostname) job=${SLURM_JOB_ID}"
"$PY" scripts/verify_label_tokens.py --root "$BASE" --hf-token-file "$HF_TOKEN_FILE"
rc=$?
echo "=== DONE rc=$rc ==="
exit $rc
