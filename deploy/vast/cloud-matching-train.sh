#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

source /venv/main/bin/activate

repo=/workspace/Cloud-Matching
output=${repo}/outputs/l40_softplus_long
mkdir -p "${output}"
exec > >(tee -a "${output}/service.log") 2>&1

cd "${repo}"

checkpoint_args=(
  --init-checkpoint "${repo}/init/bounded_energy_best.pt"
)
if [[ -f "${output}/latest.pt" ]]; then
  checkpoint_args=(--resume "${output}/latest.pt")
fi

exec python -u kaggle/train_div2k_ddp.py \
  --config configs/cloud_l40_softplus_long.yaml \
  --train-prepared /workspace/data/prepared_softplus_v8/train \
  --val-prepared /workspace/data/prepared_softplus_v8/valid \
  --output "${output}" \
  --epochs 30 \
  "${checkpoint_args[@]}"
