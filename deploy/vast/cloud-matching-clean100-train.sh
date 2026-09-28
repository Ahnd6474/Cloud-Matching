#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

source /venv/main/bin/activate
export TORCH_HOME=/workspace/models/torch

repo=/workspace/Cloud-Matching
output=${repo}/outputs/l40_paired_perceptual_clean100
dataset=/workspace/data/prepared_df2k_clean100_seed_v3
mkdir -p "${output}"
exec > >(tee -a "${output}/service.log") 2>&1

if [[ ! -f "${dataset}/.complete" ]]; then
  echo "clean100 dataset is not complete: ${dataset}" >&2
  exit 1
fi

cd "${repo}"

checkpoint_args=()
if [[ -f "${output}/latest.pt" ]]; then
  checkpoint_args=(--resume "${output}/latest.pt")
elif [[ -f "${repo}/init/bounded_energy_best.pt" ]]; then
  checkpoint_args=(--init-checkpoint "${repo}/init/bounded_energy_best.pt")
fi

exec python -u kaggle/train_div2k_ddp.py \
  --config configs/cloud_l40_paired_perceptual_clean100.yaml \
  --train-prepared "${dataset}/train" \
  --val-prepared "${dataset}/valid" \
  --output "${output}" \
  --epochs 30 \
  --reset-energy-head \
  "${checkpoint_args[@]}"
