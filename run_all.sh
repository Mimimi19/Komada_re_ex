#!/bin/bash
set -e

echo "Starting ret2p ROI-average training with band_spearman..."

for roi in $(seq 1 14); do
  echo "=== ROI ${roi} ==="
  uv run python main.py \
    data=ret2p-1 \
    data.output_file=data/ret2p/roi_ave/response_ave_roi${roi}.txt \
    hyper_params.objective_type=band_spearman \
    hydra.run.dir=scripts/results/Baccus_ret2pAve/roi_${roi}
done

echo "All ROI-average jobs completed!"
