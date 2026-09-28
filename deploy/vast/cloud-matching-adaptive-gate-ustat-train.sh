#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

source /venv/main/bin/activate
export TORCH_HOME=/workspace/models/torch

repo=/workspace/Cloud-Matching
output=${repo}/outputs/l40_adaptive_gate_fast_ustat_s4_scratch
dataset=/workspace/data/prepared_df2k_adaptive_microstep_v2
mkdir -p "${output}"
exec > >(tee -a "${output}/service.log") 2>&1

for split in train valid; do
  if [[ ! -f "${dataset}/${split}/manifest.json" ]]; then
    echo "adaptive dataset is incomplete: ${dataset}/${split}" >&2
    exit 1
  fi
done

cd "${repo}"

checkpoint_args=()
if [[ -f "${output}/latest.pt" ]]; then
  checkpoint_args=(--resume "${output}/latest.pt")
fi

exec python -u kaggle/train_div2k_ddp.py \
  --config configs/cloud_l40_adaptive_gate_fast_ustat.yaml \
  --train-prepared "${dataset}/train" \
  --val-prepared "${dataset}/valid" \
  --output "${output}" \
  --epochs 30 \
  --validation-images 4 \
  "${checkpoint_args[@]}"
