#!/bin/bash
#SBATCH --job-name=fcrrc_ranchor_pre
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=192G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc31,evc43,evc45
#SBATCH --time=04:00:00
#SBATCH --output=<CLUSTER_HOME>/FormalCRRC/logs/ranchor_preflight_%j.out
#SBATCH --error=<CLUSTER_HOME>/FormalCRRC/logs/ranchor_preflight_%j.err

# Reasoning-anchor PRE-FREEZE preflight.
#
# Validates model mechanics on synthetic prompts only. Touches no FormalCRRC
# study data, so running it cannot contaminate the preregistration. Console
# output is operational; no study metric is produced here.

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

REVISION="${1:?usage: sbatch preflight_reasoning_anchor.sbatch <model_revision>}"

report_node
echo "revision=$REVISION"
echo "stage=preflight (synthetic prompts only, pre-freeze)"

# Fail fast on a node whose GPU cannot actually take a CUDA context.
# `nvidia-smi -L` is not sufficient: evc43 reports a healthy H100 with 0 MiB
# used and Default compute mode, yet every context creation returns
# cudaErrorDevicesUnavailable. Without this gate the failure only surfaces after
# a 61 GB model load, burning the allocation.
"$PY" - <<'PY' || { echo "FATAL: $(hostname) cannot create a CUDA context"; exit 1; }
import sys, torch
try:
    torch.zeros(8, dtype=torch.bfloat16, device="cuda").sum().item()
except Exception as exc:
    print(f"CUDA smoke test failed: {type(exc).__name__}: {exc}", flush=True)
    sys.exit(1)
print("CUDA smoke test OK", flush=True)
PY

"$PY" scripts/preflight_reasoning_anchor.py \
  --root "$BASE" \
  --revision "$REVISION" \
  --hf-token-file "$HF_TOKEN_FILE" \
  --cache-dir "$HF_HOME/hub"
rc=$?

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== DONE rc=$rc ==="
exit $rc
