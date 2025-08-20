#!/usr/bin/env bash
set -euo pipefail

# 推理演示脚本（Linux）

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="outputs/stage2/best/pytorch_model.bin"
LABEL_MAP="data/custom_app/label_map.json"
IMAGE_PATH="${1:-}"

if [[ -z "${IMAGE_PATH}" ]]; then
  echo "Usage: $0 <image_path>" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[Error] Model not found at $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$LABEL_MAP" ]]; then
  echo "[Error] Label map not found at $LABEL_MAP" >&2
  exit 1
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "[Error] Image not found at $IMAGE_PATH" >&2
  exit 1
fi

python src/infer_demo.py \
  --model_path "$MODEL_PATH" \
  --label_map "$LABEL_MAP" \
  --image "$IMAGE_PATH" \
  --save_visualization \
  --output_dir demo_results

