#!/bin/bash
# Second full pass (colour fix: saturation of the lifted shadow capped per hue at the sunlit paint's) for bear 1 then bear 2:
# correct all 650 photos, train splatfacto-big on them (same settings and poses as the originals). The first pass is kept
# as *_v1 (data, exports). About 1 h 45 min per bear.
# Run: scripts/runs/run_fix_v2.sh 2>&1 | tee logs/fix_v2.log
set -o pipefail
source ~/miniforge3/bin/activate gsplat; cd ~/Claude_Test
step() { echo; echo "=== [$1] $2 $(date +%T)"; }
fail() { echo "$1 FAILED $(date +%T)"; exit 1; }
keep_v1() {  # $1 = scene
  for d in data/${1}_fixed exports/${1}_fixed; do [ -d $d ] && [ ! -d ${d}_v1 ] && mv $d ${d}_v1; done
  return 0
}
dataset() {  # $1 = scene
  S=$1; F=data/${S}_fixed
  cp data/$S/transforms.json data/$S/transforms_nsprocess.json $F/ || fail DATASET
  ln -sfn ../$S/colmap $F/colmap; ln -sf ../$S/sparse_pc.ply $F/sparse_pc.ply; ln -sf ../$S/sparse_pc_nsprocess.ply $F/sparse_pc_nsprocess.ply
  [ "$(ls $F/images | wc -l)" = "$(ls data/$S/images | wc -l)" ] || fail "IMAGE COUNT $S"
}
E=exports

step 0 "keep the first pass as _v1"
keep_v1 IMG_6704; keep_v1 IMG_6705

step 1 "bear 1: correct all photos"
python scripts/correct_photos.py --surfels $E/IMG_6704_2dgs_fromsplat/surfels_params.pt --ply $E/IMG_6704/splat.ply \
  --labels $E/IMG_6704_sunmap3/sun_visibility.npz --sun $E/IMG_6704_sun_refit_obj/sun_refit.npz --object $E/IMG_6704_object/object.json \
  --plane-from $E/IMG_6704_shadowfix_o/stats.json --colmap data/IMG_6704/colmap/sparse/0_refined --images data/IMG_6704/images \
  --masks data/IMG_6704/shadow_masks --eps 0.122 --out data/IMG_6704_fixed --compare-every 50 || fail CORRECT_6704
dataset IMG_6704
step 2 "bear 1: train on the corrected photos"
STOP_SPLIT=7000 ./scripts/make_splat_hq.sh /nonexistent.MOV IMG_6704_fixed || fail TRAIN_6704

step 3 "bear 2: correct all photos"
python scripts/correct_photos.py --surfels $E/IMG_6705_2dgs_fromsplat/surfels_params.pt --ply $E/IMG_6705/splat.ply \
  --labels $E/IMG_6705_sunmap3/sun_visibility.npz --sun $E/IMG_6705_sun_refit_obj/sun_refit.npz --object $E/IMG_6705_object/object.json \
  --plane-from $E/IMG_6705_shadowfix_o/stats.json --colmap data/IMG_6705/colmap/sparse/0_refined --images data/IMG_6705/images \
  --masks data/IMG_6705/shadow_masks --out data/IMG_6705_fixed --compare-every 50 || fail CORRECT_6705
dataset IMG_6705
step 4 "bear 2: train on the corrected photos"
STOP_SPLIT=7000 ./scripts/make_splat_hq.sh /nonexistent.MOV IMG_6705_fixed || fail TRAIN_6705
echo; echo "=== ALL DONE $(date +%T)"
