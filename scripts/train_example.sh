#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:?Usage: scripts/train_example.sh /path/to/image.tiff /path/to/reference_points.gpkg /path/to/model_dir}"
REFERENCE_POINTS="${2:?Usage: scripts/train_example.sh /path/to/image.tiff /path/to/reference_points.gpkg /path/to/model_dir}"
OUT_DIR="${3:?Usage: scripts/train_example.sh /path/to/image.tiff /path/to/reference_points.gpkg /path/to/model_dir}"

python3 ml_tree_detector.py train \
  --image "$IMAGE" \
  --reference-points "$REFERENCE_POINTS" \
  --out-dir "$OUT_DIR" \
  --rgb-bands 1 2 3 \
  --spacing-m 1.5 \
  --vegetation-percentile 62 \
  --min-score 0.08 \
  --positive-distance-m 0.55 \
  --negative-distance-m 1.00 \
  --anchor-positive-fraction 0.05
