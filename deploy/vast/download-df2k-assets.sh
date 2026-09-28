#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

source /venv/main/bin/activate
export TORCH_HOME=/workspace/models/torch

log=/workspace/data/df2k-assets.log
mkdir -p /workspace/data/flickr2k "${TORCH_HOME}"
exec > >(tee -a "${log}") 2>&1

hr_root=/workspace/data/flickr2k/Flickr2K/Flickr2K_HR
mkdir -p "${hr_root}"
existing=$(find "${hr_root}" -maxdepth 1 -type f -iname '*.png' | wc -l)
if [[ "${existing}" -lt 2650 ]]; then
  echo "Streaming the Kaggle ZIP and extracting only Flickr2K_HR."
  # The archive also contains several LR pyramids. stream-unzip avoids keeping
  # the 10.9 GB ZIP or writing those unused copies to local disk.
  python -c 'import stream_unzip' 2>/dev/null \
    || uv pip install 'stream-unzip==0.0.101'
  python -u /workspace/Cloud-Matching/deploy/vast/download_flickr2k_hr.py
fi

images=$(find "${hr_root}" -maxdepth 1 -type f -iname '*.png' | wc -l)
if [[ "${images}" -ne 2650 ]]; then
  echo "expected 2650 Flickr2K HR images, found ${images}" >&2
  exit 1
fi
echo "Flickr2K ready: ${images} HR images below ${hr_root}"

python -u - <<'PY'
import torch
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).eval()
with torch.inference_mode():
    value = torch.zeros(1, 3, 128, 128)
    shapes = []
    for index, block in enumerate(model.features):
        value = block(value)
        if index in {1, 2, 3, 5, 7}:
            shapes.append(tuple(value.shape))
print(f"EfficientNet-B0 cached at {torch.hub.get_dir()}")
print(f"128px feature shapes: {shapes}")
PY

df -h /workspace
du -sh /workspace/data/flickr2k "${TORCH_HOME}"
