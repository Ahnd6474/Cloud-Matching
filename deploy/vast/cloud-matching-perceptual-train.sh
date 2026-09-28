#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

source /venv/main/bin/activate
export TORCH_HOME=/workspace/models/torch

repo=/workspace/Cloud-Matching
output=${repo}/outputs/l40_paired_perceptual
mkdir -p "${output}"
exec > >(tee -a "${output}/service.log") 2>&1

cd "${repo}"

checkpoint_args=()
if [[ -f "${output}/latest.pt" ]]; then
  checkpoint_args=(--resume "${output}/latest.pt")
elif [[ -f "${repo}/init/bounded_energy_best.pt" ]]; then
  checkpoint_args=(--init-checkpoint "${repo}/init/bounded_energy_best.pt")
fi

exec python -u kaggle/train_div2k_ddp.py \
  --config configs/cloud_l40_paired_perceptual.yaml \
  --train-prepared /workspace/data/prepared_df2k_seed_v2/train \
  --val-prepared /workspace/data/prepared_df2k_seed_v2/valid \
  --output "${output}" \
  --epochs 30 \
  --reset-energy-head \
  "${checkpoint_args[@]}"
