#!/usr/bin/env bash
set -euo pipefail

# Stage-1：RICO+Screen2Words 预适配训练（Linux）

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_FILE="configs/stage1.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[Error] Missing $CONFIG_FILE" >&2
  exit 1
fi

accelerate launch --config_file configs/accelerate_config.yaml \
  src/train_stage1_align.py \
  --config "$CONFIG_FILE" \
  --use_wandb

