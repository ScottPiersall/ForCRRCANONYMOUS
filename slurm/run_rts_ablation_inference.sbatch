#!/bin/bash
#SBATCH --job-name=fcrrc_rts
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc29,evc31,evc43,evc45
#SBATCH --time=08:00:00
#SBATCH --output=<CLUSTER_HOME>/FormalCRRC/logs/rts_infer_%j.out
#SBATCH --error=<CLUSTER_HOME>/FormalCRRC/logs/rts_infer_%j.err

# Reason-then-score ablation: the frozen elicited protocol over all 2916 rows
# of the S3 TEST partition, on the exact frozen s-3 Qwen2.5 revision.
#
# Checkpointed in shards, so re-submitting resumes rather than restarts.
# Console output is operational only; no metric is computed here.

set -uo pipefail
source <CLUSTER_HOME>/FormalCRRC/slurm/common.sh
cd "$BASE"

export PATH="$SHARED/envs/rcjc311/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0

report_node
echo "stage=study inference (post-freeze)"

# Bounded guard: a faulty node either errors immediately (evc43) or blocks
# forever inside CUDA init (evc29). Both must fail fast, not eat the allocation.
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
  echo "FATAL: $(hostname) hung during CUDA init (timed out after 180s)"; exit 1
elif [ "$smoke" -ne 0 ]; then
  echo "FATAL: $(hostname) cannot create a CUDA context (rc=$smoke)"; exit 1
fi

"$PY" scripts/run_rts_ablation_inference.py \
  --root "$BASE" \
  --hf-token-file "$HF_TOKEN_FILE" \
  --cache-dir "$HF_HOME/hub" \
  --max-model-len 10240 \
  --gpu-memory-utilization 0.90 \
  --chunk-size 128
rc=$?

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== DONE rc=$rc ==="
exit $rc
