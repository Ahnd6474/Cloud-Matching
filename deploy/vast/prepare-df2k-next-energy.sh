#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
source /venv/main/bin/activate

repo=/workspace/Cloud-Matching
output=/workspace/data/prepared_df2k_next_energy_v1
train_output=${output}/train
valid_output=${output}/valid
log=/workspace/data/prepared_df2k_next_energy_v1.log
mkdir -p /workspace/data
exec > >(tee -a "${log}") 2>&1

for process in cloud_matching_clean100 cloud_matching_next_energy; do
  if supervisorctl status "${process}" 2>/dev/null | grep -q RUNNING; then
    echo "refusing dataset preparation while ${process} is running" >&2
    exit 2
  fi
done
if [[ -e "${output}" ]]; then
  echo "output already exists; refusing to overwrite: ${output}" >&2
  exit 3
fi
available=$(df --output=avail -B1 /workspace | tail -1 | tr -d ' ')
if [[ "${available}" -lt 5500000000 ]]; then
  echo "at least 5.5 GB free space is required; available=${available}" >&2
  exit 4
fi

cd "${repo}"

echo "Preparing next-state Energy-Distance train data: 3450 x 4 = 13800 records"
python -u prepare_dataset.py \
  --config configs/cloud_l40_random_attention_next_energy.yaml \
  --data /workspace/data/div2k/Dataset/DIV2K_train_HR \
  --data /workspace/data/flickr2k/Flickr2K/Flickr2K_HR \
  --output "${train_output}" \
  --variants 4 \
  --device cuda

echo "Preparing validation data: 100 x 2 = 200 records"
python -u prepare_dataset.py \
  --config configs/cloud_l40_random_attention_next_energy.yaml \
  --data /workspace/data/div2k/Dataset/DIV2K_valid_HR \
  --output "${valid_output}" \
  --variants 2 \
  --device cuda

python -u - <<'PY'
import json
from pathlib import Path

import torch

from stochastic_bridge.prepared import PreparedBridgeDataset

root = Path("/workspace/data/prepared_df2k_next_energy_v1")
for split, expected in (("train", 13_800), ("valid", 200)):
    dataset = PreparedBridgeDataset(root / split)
    manifest = dataset.manifest
    if len(dataset) != expected:
        raise RuntimeError(f"{split} length mismatch: {len(dataset)} != {expected}")
    schedule = manifest["config"]["schedule"]
    if not schedule.get("goal_from_clean") or not schedule.get("answer_from_current"):
        raise RuntimeError(f"{split} does not contain next-state transitions")
    records = [dataset[index] for index in range(min(32, len(dataset)))]
    current = torch.stack([record["current_level"] for record in records])
    goal = torch.stack([record["goal_level"] for record in records])
    answer = torch.stack([record["answer_level"] for record in records])
    if torch.count_nonzero(goal):
        raise RuntimeError(f"{split} goal levels are not clean-based")
    if not torch.equal(answer, (current - 10).clamp_min(0)):
        raise RuntimeError(f"{split} answer levels are not current-relative")
    diversity = torch.stack(
        [record["target_cloud"].std(dim=0, unbiased=False).mean() for record in records]
    ).mean()
    if not torch.isfinite(diversity) or diversity <= 0:
        raise RuntimeError(f"{split} target cloud has no stochastic diversity")
    print(json.dumps({
        "split": split,
        "records": len(dataset),
        "shards": len(manifest["shards"]),
        "mean_target_diversity": float(diversity),
    }))
(root / ".complete").write_text("validated\n", encoding="utf-8")
PY

echo "next-state Energy-Distance dataset complete"
du -sh "${train_output}" "${valid_output}"
df -h /workspace

echo "Starting cloud_matching_next_energy"
supervisorctl start cloud_matching_next_energy
