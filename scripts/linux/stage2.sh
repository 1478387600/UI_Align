#!/usr/bin/env bash
set -euo pipefail

# Stage-2：自建数据微调（对齐+分类）（Linux）

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_FILE="configs/stage2.yaml"
STAGE1_DIR="outputs/stage1"

if [[ ! -d "$STAGE1_DIR" ]]; then
  echo "[Error] $STAGE1_DIR not found. Run stage1 first." >&2
  exit 1
fi

accelerate launch --config_file configs/accelerate_config.yaml \
  src/train_stage2_align_cls.py \
  --config "$CONFIG_FILE" \
  --use_wandb

