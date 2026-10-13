#!/bin/bash
#SBATCH --job-name=fcrrc_rts_pre
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc29,evc31,evc43,evc45
#SBATCH --time=02:00:00
#SBATCH --output=<CLUSTER_HOME>/FormalCRRC/logs/rts_preflight_%j.out
#SBATCH --error=<CLUSTER_HOME>/FormalCRRC/logs/rts_preflight_%j.err

# Reason-then-score ablation PRE-FREEZE preflight.
#
# Synthetic prompts only; touches no FormalCRRC study data, so running it
# cannot contaminate the preregistration.

set -uo pipefail
source <CLUSTER_HOME>/FormalCRRC/slurm/common.sh
cd "$BASE"

export PATH="$SHARED/envs/rcjc311/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0

report_node
echo "stage=preflight (synthetic prompts only, pre-freeze)"

# Bound the smoke test in wall clock. A faulty node fails two different ways:
# evc43 returned cudaErrorDevicesUnavailable immediately, while evc29 blocked
# forever inside CUDA init with the process idle at 2s of CPU. An unbounded
# check catches only the first and lets the second consume the allocation, so
# the timeout is part of the guard, not a convenience.
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
if [ "$smoke" -eq 124 ]; then
  echo "FATAL: $(hostname) hung during CUDA init (timed out after 180s)"
  exit 1
elif [ "$smoke" -ne 0 ]; then
  echo "FATAL: $(hostname) cannot create a CUDA context (rc=$smoke)"
  exit 1
fi

"$PY" scripts/preflight_rts_ablation.py \
  --root "$BASE" \
  --hf-token-file "$HF_TOKEN_FILE" \
  --cache-dir "$HF_HOME/hub"
rc=$?
echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== DONE rc=$rc ==="
exit $rc
