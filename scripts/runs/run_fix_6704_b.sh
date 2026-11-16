#!/bin/bash
# Bear 1 (IMG_6704): correct the shadows in all 650 photos with the lighting physics (calibrated on every 13th photo),
# then train a normal splatfacto-big splat on them with the original run's settings and poses.
# Needs stage A (run_fix_6704_a.sh). 10 mm label margin = 0.122 units on bear 1 (12.18 units/m).
# Run: scripts/runs/run_fix_6704_b.sh 2>&1 | tee logs/IMG_6704_fix_b.log
set -o pipefail
source ~/miniforge3/bin/activate gsplat; cd ~/Claude_Test
S=IMG_6704; E=exports; F=data/${S}_fixed; COL=data/$S/colmap/sparse/0_refined; PL=$E/${S}_shadowfix_o/stats.json
step() { echo; echo "=== [$1] $2 $(date +%T)"; }
fail() { echo "$1 FAILED $(date +%T)"; exit 1; }

step 1 "correct all photos (calibrate the concrete ratio and the k scales, then every photo)"
python scripts/correct_photos.py --surfels $E/${S}_2dgs_fromsplat/surfels_params.pt --ply $E/$S/splat.ply \
  --labels $E/${S}_sunmap3/sun_visibility.npz --sun $E/${S}_sun_refit_obj/sun_refit.npz --object $E/${S}_object/object.json \
  --plane-from $PL --colmap $COL --images data/$S/images --masks data/$S/shadow_masks --eps 0.122 --out $F --compare-every 50 \
  || fail CORRECT

step 2 "dataset folder for training (corrected images, same poses)"
cp data/$S/transforms.json data/$S/transforms_nsprocess.json $F/ || fail DATASET
ln -sfn ../$S/colmap $F/colmap
ln -sf ../$S/sparse_pc.ply $F/sparse_pc.ply
ln -sf ../$S/sparse_pc_nsprocess.ply $F/sparse_pc_nsprocess.ply
[ "$(ls $F/images | wc -l)" = "$(ls data/$S/images | wc -l)" ] || fail "IMAGE COUNT"

step 3 "train splatfacto-big on the corrected photos (30k, stop-split 7000)"
STOP_SPLIT=7000 ./scripts/make_splat_hq.sh /nonexistent.MOV ${S}_fixed || fail TRAIN
echo; echo "=== STAGE B DONE $(date +%T)"
