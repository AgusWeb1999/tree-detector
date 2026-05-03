#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:?Usage: scripts/predict_example.sh /path/to/image.tiff /path/to/output_dir}"
OUT_DIR="${2:?Usage: scripts/predict_example.sh /path/to/image.tiff /path/to/output_dir}"

python3 ml_tree_detector.py predict \
  --image "$IMAGE" \
  --model "models/young_tree_model_v3.joblib" \
  --out-dir "$OUT_DIR" \
  --vegetation-percentile 62 \
  --min-score 0.08 \
  --prob-threshold 0.55 \
  --nms-spacing-factor 0.65
