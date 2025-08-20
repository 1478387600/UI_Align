#!/usr/bin/env bash
set -euo pipefail

# 评估脚本（Linux）

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="outputs/stage2/best/pytorch_model.bin"
TEST_IMAGES="data/custom_app/images"
TEST_CAPTIONS="data/custom_app/captions.jsonl"
TEST_LABELS="data/custom_app/labels.jsonl"
LABEL_MAP="data/custom_app/label_map.json"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[Error] Model not found at $MODEL_PATH" >&2
  exit 1
fi

python src/eval_retrieval.py \
  --model_path "$MODEL_PATH" \
  --test_images "$TEST_IMAGES" \
  --test_captions "$TEST_CAPTIONS" \
  --test_labels "$TEST_LABELS" \
  --label_map "$LABEL_MAP" \
  --output_dir eval_results/retrieval \
  --save_examples

python src/eval_cls.py \
  --model_path "$MODEL_PATH" \
  --test_images "$TEST_IMAGES" \
  --test_labels "$TEST_LABELS" \
  --label_map "$LABEL_MAP" \
  --test_captions "$TEST_CAPTIONS" \
  --output_dir eval_results/classification

