#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

source /venv/main/bin/activate

repo=/workspace/Cloud-Matching
output=/workspace/data/prepared_df2k_seed_v2
train_output=${output}/train
valid_output=${output}/valid
log=/workspace/data/prepared_df2k_seed_v2.log
mkdir -p /workspace/data
exec > >(tee -a "${log}") 2>&1

if supervisorctl status cloud_matching_train | grep -q RUNNING; then
  echo "refusing dataset preparation while training is running" >&2
  exit 2
fi
if [[ -e "${output}" ]]; then
  echo "output already exists; refusing to overwrite: ${output}" >&2
  exit 3
fi
available=$(df --output=avail -B1 /workspace | tail -1 | tr -d ' ')
if [[ "${available}" -lt 12000000000 ]]; then
  echo "at least 12 GB free space is required; available=${available}" >&2
  exit 4
fi

cd "${repo}"

echo "Preparing DF2K train: 3450 sources x 8 variants = 27600 records"
python -u prepare_dataset.py \
  --config configs/cloud_l40_paired_mean_deviation.yaml \
  --data /workspace/data/div2k/Dataset/DIV2K_train_HR \
  --data /workspace/data/flickr2k/Flickr2K/Flickr2K_HR \
  --output "${train_output}" \
  --variants 8 \
  --device cuda

echo "Preparing DIV2K validation: 100 sources x 2 variants = 200 records"
python -u prepare_dataset.py \
  --config configs/cloud_l40_paired_mean_deviation.yaml \
  --data /workspace/data/div2k/Dataset/DIV2K_valid_HR \
  --output "${valid_output}" \
  --variants 2 \
  --device cuda

python -u - <<'PY'
import json
from pathlib import Path

import torch

from stochastic_bridge.prepared import PreparedBridgeDataset, materialize_target_noise

root = Path("/workspace/data/prepared_df2k_seed_v2")
for split, expected in (("train", 27_600), ("valid", 200)):
    dataset = PreparedBridgeDataset(root / split)
    manifest = dataset.manifest
    if len(dataset) != expected:
        raise RuntimeError(f"{split} length mismatch: {len(dataset)} != {expected}")
    if manifest.get("format_version") != 2:
        raise RuntimeError(f"{split} is not format v2")
    if manifest.get("target_noise_storage") != "seed":
        raise RuntimeError(f"{split} does not use seed noise storage")
    records = [dataset[index] for index in range(min(2, len(dataset)))]
    raw = {
        "target_cloud": torch.stack([record["target_cloud"] for record in records]),
        "target_noise_seed": torch.stack(
            [record["target_noise_seed"] for record in records]
        ),
    }
    noise = materialize_target_noise(raw, "cuda")
    if noise is None or not torch.isfinite(noise).all():
        raise RuntimeError(f"{split} noise reconstruction failed")
    print(
        json.dumps(
            {
                "split": split,
                "records": len(dataset),
                "format_version": manifest["format_version"],
                "target_noise_storage": manifest["target_noise_storage"],
                "shards": len(manifest["shards"]),
            }
        )
    )
(root / ".complete").write_text("validated\n", encoding="utf-8")
PY

echo "DF2K seed-v2 dataset complete"
du -sh "${train_output}" "${valid_output}"
df -h /workspace
