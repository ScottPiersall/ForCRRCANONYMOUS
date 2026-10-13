#!/bin/bash
# Shared environment for every FormalCRRC job on GPU Cluster.
#
# Sourced by the sbatch scripts in this directory. Encodes the cluster-specific
# fixes documented in docs/CLUSTER_NEWTON.md: no CUDA toolkit at the default
# location, no `module` in non-login shells, and a shared uv-managed CPython
# whose headers Triton needs.
#
# The Hugging Face token path is exported for the study scripts to read from
# disk. Its contents are never echoed, logged, or written into any artifact.

BASE=<CLUSTER_HOME>/FormalCRRC
SHARED=<CLUSTER_HOME>/RCJC          # shared infrastructure: venv + model cache

export HF_HOME=$SHARED/hf_cache
export HF_TOKEN_FILE=$SHARED/.hf_token
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

export CUDA_HOME=/apps/cuda/cuda-13.1.0
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

PY=$SHARED/envs/rcjc311/bin/python

mkdir -p "$BASE/logs"

report_node() {
  echo "host=$(hostname) job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-none}"
  echo "nvcc=$(command -v nvcc || echo MISSING)"
  nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv
  nvidia-smi -L 2>/dev/null | grep -q GPU || {
    echo "FATAL: node $(hostname) exposes no usable GPU"
    exit 1
  }
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
