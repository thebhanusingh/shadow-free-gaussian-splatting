#!/bin/bash
# Finish the second full pass (run_fix_v2.sh) that the laptop reset cut off on 2026-11-10 at bear 1 training step 7640.
# Bear 1: its 650 corrected photos are already in data/IMG_6704_fixed (summary.json present), so only train.
# Bear 2: correct all photos (concrete ratio measured from its own photos, saturation cap), then train.
# Each stage is skipped when its result is already there, so the script can be started again after a stop.
# About 45 min (bear 1 training) + 1 h 45 min (bear 2).
# Run: scripts/runs/run_fix_v2_resume.sh 2>&1 | tee logs/fix_v2_resume.log
set -o pipefail
source ~/miniforge3/bin/activate gsplat; cd ~/Claude_Test
step() { echo; echo "=== [$1] $2 $(date +%T)"; }
fail() { echo "$1 FAILED $(date +%T)"; exit 1; }
dataset() {  # $1 = scene
  S=$1; F=data/${S}_fixed
  cp data/$S/transforms.json data/$S/transforms_nsprocess.json $F/ || fail DATASET
  ln -sfn ../$S/colmap $F/colmap; ln -sf ../$S/sparse_pc.ply $F/sparse_pc.ply; ln -sf ../$S/sparse_pc_nsprocess.ply $F/sparse_pc_nsprocess.ply
  [ "$(ls $F/images | wc -l)" = "$(ls data/$S/images | wc -l)" ] || fail "IMAGE COUNT $S"
}
corrected() { [ -f data/${1}_fixed/summary.json ] && [ "$(ls data/${1}_fixed/images | wc -l)" = "$(ls data/$1/images | wc -l)" ]; }
E=exports

step 1 "bear 1: corrected photos"
corrected IMG_6704 || fail "BEAR 1 PHOTOS NOT COMPLETE (run run_fix_v2.sh instead)"
echo "already there: $(ls data/IMG_6704_fixed/images | wc -l) photos"
dataset IMG_6704
step 2 "bear 1: train on the corrected photos"
if [ -f $E/IMG_6704_fixed/splat.ply ]; then echo "already there, skipped"
else STOP_SPLIT=7000 ./scripts/make_splat_hq.sh /nonexistent.MOV IMG_6704_fixed || fail TRAIN_6704; fi

step 3 "bear 2: correct all photos"
if corrected IMG_6705; then echo "already there, skipped"
else
  [ -d data/IMG_6705_fixed ] && fail "data/IMG_6705_fixed exists but is incomplete: look at it before rerunning"
  python scripts/correct_photos.py --surfels $E/IMG_6705_2dgs_fromsplat/surfels_params.pt --ply $E/IMG_6705/splat.ply \
    --labels $E/IMG_6705_sunmap3/sun_visibility.npz --sun $E/IMG_6705_sun_refit_obj/sun_refit.npz --object $E/IMG_6705_object/object.json \
    --plane-from $E/IMG_6705_shadowfix_o/stats.json --colmap data/IMG_6705/colmap/sparse/0_refined --images data/IMG_6705/images \
    --masks data/IMG_6705/shadow_masks --eps 0.093 --out data/IMG_6705_fixed --compare-every 50 || fail CORRECT_6705
fi
dataset IMG_6705
step 4 "bear 2: train on the corrected photos"
if [ -f $E/IMG_6705_fixed/splat.ply ]; then echo "already there, skipped"
else STOP_SPLIT=7000 ./scripts/make_splat_hq.sh /nonexistent.MOV IMG_6705_fixed || fail TRAIN_6705; fi
echo; echo "=== ALL DONE $(date +%T)"
