#!/usr/bin/env bash
set -euo pipefail

repo="${HOME}/Cloud-Matching"
python="${repo}/.venv/bin/python"
output="${repo}/data/prepared_df2k_adaptive_microstep_v2/train"

cd "${repo}"

if [[ -f "${output}/manifest.json" ]]; then
  echo "prepared train dataset already exists: ${output}"
  exit 0
fi

mkdir -p "${repo}/data/prepared_df2k_adaptive_microstep_v2"

div2k_root="$(${python} - <<'PY' | tail -n 1
import kagglehub
print(kagglehub.dataset_download("takihasan/div2k-dataset-for-super-resolution"))
PY
)"
flickr_root="$(${python} - <<'PY' | tail -n 1
import kagglehub
print(kagglehub.dataset_download("daehoyang/flickr2k"))
PY
)"

readarray -t roots < <(${python} - "${div2k_root}" "${flickr_root}" <<'PY'
from pathlib import Path
import sys

def choose(root: Path, expected: int, preferred: tuple[str, ...]) -> Path:
    candidates = []
    for directory in [root, *root.rglob("*")]:
        if not directory.is_dir():
            continue
        count = sum(1 for _ in directory.glob("*.png"))
        if count >= expected:
            score = sum(token.lower() in directory.name.lower() for token in preferred)
            candidates.append((score, -abs(count - expected), directory))
    if not candidates:
        raise RuntimeError(f"could not find {expected} PNG images below {root}")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]

print(choose(Path(sys.argv[1]), 800, ("div2k", "train", "hr")))
print(choose(Path(sys.argv[2]), 2650, ("flickr2k", "hr")))
PY
)

div2k_train="${roots[0]}"
flickr_train="${roots[1]}"
echo "DIV2K train: ${div2k_train}"
echo "Flickr2K train: ${flickr_train}"

${python} -u prepare_dataset.py \
  --config configs/cloud_l40_adaptive_gate_fast_ustat.yaml \
  --data "${div2k_train}" \
  --data "${flickr_train}" \
  --output "${output}" \
  --variants 4 \
  --device cuda

${python} - "${output}" <<'PY'
from pathlib import Path
import sys
from stochastic_bridge.prepared import PreparedBridgeDataset

root = Path(sys.argv[1])
dataset = PreparedBridgeDataset(root)
if len(dataset) != 13_800:
    raise RuntimeError(f"prepared length mismatch: {len(dataset)} != 13800")
if dataset.manifest.get("format_version") != 2:
    raise RuntimeError("prepared dataset is not format v2")
print(f"prepared dataset verified: records={len(dataset)}, root={root}")
PY
