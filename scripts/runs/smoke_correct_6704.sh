#!/usr/bin/env bash
# Smoke test of correct_photos.py on 4 bear 1 photos with a saved calibration. Writes only to exports/IMG_6704_footprint_test.
# Usage: bash scripts/runs/smoke_correct_6704.sh
source ~/miniforge3/bin/activate gsplat
cd ~/Claude_Test/scripts || exit 1
E=../exports
python correct_photos.py \
  --surfels $E/IMG_6704_2dgs_fromsplat/surfels_params.pt --ply $E/IMG_6704/splat.ply \
  --labels $E/IMG_6704_sunmap3/sun_visibility.npz --sun $E/IMG_6704_sun_refit_obj/sun_refit.npz \
  --object $E/IMG_6704_object/object.json --plane-from $E/IMG_6704_shadowfix_o/stats.json \
  --colmap ../data/IMG_6704/colmap/sparse/0_refined --images ../data/IMG_6704/images \
  --masks ../data/IMG_6704/shadow_masks --eps 0.122 --calib-from $E/IMG_6704_colourtest/calibration.json \
  --frames frame_00501,frame_00131,frame_00351,frame_00051 --out $E/IMG_6704_footprint_test
echo "smoke test exit code $?"
