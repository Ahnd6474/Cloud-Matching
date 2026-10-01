#!/usr/bin/env bash
set -euo pipefail

# This file is intentionally a manual launcher. Preparing the server does not
# execute it or start training.
repo="${HOME}/Cloud-Matching"
python_env="${repo}/.venv"
dataset="${repo}/data/prepared_df2k_adaptive_microstep_v2"
checkpoint="${repo}/outputs/l40_adaptive_gate_fast_ustat_s4_epoch1_30/best.pt"
output="${repo}/outputs/l40_adaptive_gate_fast_ustat_s4_continue_epoch29_59"

for required in \
  "${dataset}/train/manifest.json" \
  "${dataset}/valid/manifest.json" \
  "${checkpoint}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 1
  fi
done

if pgrep -u "$(id -u)" -f "kaggle/train_div2k_ddp.py" >/dev/null; then
  echo "Cloud-Matching training already appears to be running" >&2
  exit 2
fi

mkdir -p "${output}"
cd "${repo}"

# Two ranks, one A6000 per rank.  Keep CPU math single-threaded because each
# rank already owns eight persistent data-loader workers.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${HOME}/.cache/torchinductor-cloud-matching}"

exec "${python_env}/bin/torchrun" \
  --standalone \
  --nproc-per-node=2 \
  kaggle/train_div2k_ddp.py \
  --config configs/cloud_l40_adaptive_gate_fast_ustat.yaml \
  --train-prepared "${dataset}/train" \
  --val-prepared "${dataset}/valid" \
  --output "${output}" \
  --epochs 59 \
  --resume "${checkpoint}" \
  --warmup-to-lr 0.00001 \
  --warmup-epochs 1 \
  --fixed-lr 0.00001 \
  --tensorboard-dir "${output}/tensorboard" \
  --validation-images 4
