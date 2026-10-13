#!/bin/bash
#SBATCH --job-name=fcrrc_ranchor
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=192G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc31,evc43,evc45
#SBATCH --time=08:00:00
#SBATCH --output=<CLUSTER_HOME>/FormalCRRC/logs/ranchor_infer_%j.out
#SBATCH --error=<CLUSTER_HOME>/FormalCRRC/logs/ranchor_infer_%j.err

# Reasoning-anchor study inference: the frozen reason-then-score protocol over
# all 2916 rows of the s-3 TEST partition.
#
# The run is checkpointed in shards, so re-submitting this script after a
# timeout or a node failure resumes where it stopped. A resumed row keeps its
# original seed, prompt and configuration, so the retry is the same experiment.
#
# Console output is operational only; no metric is computed here.

set -uo pipefail
source <CLUSTER_HOME>/FormalCRRC/slurm/common.sh
cd "$BASE"

# FlashInfer JIT-compiles its sampling kernel on first use and shells out to
# `ninja`, which is installed as a Python package but is not on the batch PATH.
# Put the venv's bin directory in front so any JIT build can find it.
export PATH="$SHARED/envs/rcjc311/bin:$PATH"

# Take the JIT out of the sampling path as well. vLLM's native top-k/top-p
# sampler needs no compilation, so a fresh node cannot fail here, and the
# sampling backend has no bearing on the frozen protocol: seeds are per row and
# the generation parameters are unchanged either way.
export VLLM_USE_FLASHINFER_SAMPLER=0

report_node
echo "stage=study inference (post-freeze)"

# Fail fast on a node whose GPU cannot actually take a CUDA context.
# `nvidia-smi -L` is not sufficient: evc43 reports a healthy H100 with 0 MiB
# used and Default compute mode, yet every context creation returns
# cudaErrorDevicesUnavailable. Failing here costs seconds; failing after the
# model load costs the allocation.
# Bounded in wall clock: a faulty node may error immediately (evc43) or block
# forever inside CUDA init (evc29). An unbounded check catches only the first.
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

"$PY" scripts/run_reasoning_anchor_inference.py \
  --root "$BASE" \
  --hf-token-file "$HF_TOKEN_FILE" \
  --cache-dir "$HF_HOME/hub" \
  --max-model-len 10240 \
  --gpu-memory-utilization 0.92 \
  --chunk-size 96
rc=$?

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== DONE rc=$rc ==="
exit $rc
