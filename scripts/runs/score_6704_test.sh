#!/usr/bin/env bash
# Score the 4-photo bear 1 test: the earlier correction (exports/IMG_6704_colourtest) and the one with the ground limited to the
# bear's own shadow (exports/IMG_6704_footprint_test). Output: exports/IMG_6704_score_{old,new}/score.json
# Usage: bash scripts/runs/score_6704_test.sh
source ~/miniforge3/bin/activate gsplat
cd ~/Claude_Test/scripts || exit 1
E=../exports
for v in "colourtest old" "footprint_test new"; do
  set -- $v
  echo; echo "##### corrected photos: $1"
  python score_correction.py \
    --surfels $E/IMG_6704_2dgs_fromsplat/surfels_params.pt --ply $E/IMG_6704/splat.ply \
    --labels $E/IMG_6704_sunmap3/sun_visibility.npz --sun $E/IMG_6704_sun_refit_obj/sun_refit.npz \
    --object $E/IMG_6704_object/object.json --plane-from $E/IMG_6704_shadowfix_o/stats.json \
    --colmap ../data/IMG_6704/colmap/sparse/0_refined --images ../data/IMG_6704/images \
    --masks ../data/IMG_6704/shadow_masks --eps 0.122 \
    --fixed $E/IMG_6704_$1/images --frames frame_00501,frame_00131,frame_00351,frame_00051 --out $E/IMG_6704_score_$2
done
echo "score run done"
