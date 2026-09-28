#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
source /venv/main/bin/activate

repo=/workspace/Cloud-Matching
source_root=/workspace/data/prepared_df2k_seed_v2
output_root=/workspace/data/prepared_df2k_clean100_seed_v3
log=/workspace/data/prepared_df2k_clean100_seed_v3.log
exec > >(tee -a "${log}") 2>&1

if supervisorctl status cloud_matching_perceptual 2>/dev/null | grep -q RUNNING; then
  echo "refusing to convert while cloud_matching_perceptual is running" >&2
  exit 1
fi

cd "${repo}"
python -u convert_prepared_clean_endpoint.py \
  --source "${source_root}" \
  --output "${output_root}"

du -sh "${output_root}"
df -h /workspace
