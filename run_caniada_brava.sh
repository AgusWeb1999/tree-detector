#!/usr/bin/env bash
set -euo pipefail

python3 "$(dirname "$0")/young_tree_counter.py" \
  "/Users/agusmazzini/Downloads/CANIADA BRAVA-MOSAICO-240326-PRENDIMIENTO-PARTE1DE2.tiff" \
  --out-dir "/Users/agusmazzini/Downloads/salida_jovenes" \
  --rgb-bands 1 2 3 \
  --expected-spacing-m 1.5 \
  --min-row-support 2 \
  --vegetation-percentile 78 \
  --min-score 0.36 \
  --debug \
  --write-points-tif
