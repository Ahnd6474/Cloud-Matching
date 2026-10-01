#!/usr/bin/env bash
set -euo pipefail

repo="${HOME}/Cloud-Matching"
logdir="${repo}/outputs/l40_adaptive_gate_fast_ustat_s4_continue_epoch29_59/tensorboard"
port="${TENSORBOARD_PORT:-8080}"

mkdir -p "${logdir}"
cd "${repo}"

exec "${repo}/.venv/bin/tensorboard" \
  --logdir "${logdir}" \
  --host 127.0.0.1 \
  --port "${port}" \
  --reload_interval 30 \
  --samples_per_plugin scalars=2000,images=40,text=20
