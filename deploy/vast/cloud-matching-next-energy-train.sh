#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

source /venv/main/bin/activate
export TORCH_HOME=/workspace/models/torch

repo=/workspace/Cloud-Matching
output=${repo}/outputs/l40_random_attention_next_energy
dataset=/workspace/data/prepared_df2k_next_energy_v1
mkdir -p "${output}"
exec > >(tee -a "${output}/service.log") 2>&1

if [[ ! -f "${dataset}/.complete" ]]; then
  echo "next-state dataset is not complete: ${dataset}" >&2
  exit 1
fi

cd "${repo}"

checkpoint_args=()
if [[ -f "${output}/latest.pt" ]]; then
  checkpoint_args=(--resume "${output}/latest.pt")
elif [[ -f "${repo}/outputs/l40_paired_perceptual_clean100/best.pt" ]]; then
  checkpoint_args=(--init-checkpoint "${repo}/outputs/l40_paired_perceptual_clean100/best.pt")
elif [[ -f "${repo}/init/bounded_energy_best.pt" ]]; then
  checkpoint_args=(--init-checkpoint "${repo}/init/bounded_energy_best.pt")
fi

exec python -u kaggle/train_div2k_ddp.py \
  --config configs/cloud_l40_random_attention_next_energy.yaml \
  --train-prepared "${dataset}/train" \
  --val-prepared "${dataset}/valid" \
  --output "${output}" \
  --epochs 30 \
  --validation-images 4 \
  --warmup-to-lr 0.00004 \
  --warmup-epochs 1 \
  --fixed-lr 0.00004 \
  "${checkpoint_args[@]}"
