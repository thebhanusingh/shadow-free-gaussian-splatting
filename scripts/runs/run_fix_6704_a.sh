#!/bin/bash
# Bear 1 (IMG_6704): the geometry and shadow-label steps that bear 2 got by hand, in one script (about 1 hour):
# object, 2DGS surfels from the existing splat, borrowed surface directions, refitted sun (only bear surfels cast),
# sun-shadow labels of the bear. Lengths scaled from bear 2: bear 1 is 12.18 units/m (2.13 units, 0.175 m), so
# 3 mm = 0.036 units and the 10 mm label margin = 0.122 units.
# Run: scripts/runs/run_fix_6704_a.sh 2>&1 | tee logs/IMG_6704_fix_a.log
set -o pipefail
source ~/miniforge3/bin/activate gsplat; cd ~/Claude_Test
S=IMG_6704; E=exports; COL=data/$S/colmap/sparse/0_refined; PL=$E/${S}_shadowfix_o/stats.json
CFG=$(ls -t outputs/$S/splatfacto/*/config.yml | head -1)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
step() { echo; echo "=== [$1] $2 $(date +%T)"; }
fail() { echo "$1 FAILED $(date +%T)"; exit 1; }

step 1 "find the bear and the scale (bear 1: 0.175 m, reference scene, K = 1)"
python scripts/find_object.py --ply $E/$S/splat.ply --colmap $COL --plane-from $PL --height-m 0.175 --ref-units-per-m 12.18 \
  --out $E/${S}_object || fail FIND_OBJECT

step 2 "2DGS surfels from the existing splat (10k steps, no splitting)"
python scripts/train_2dgs_ref.py --config $CFG --out $E/${S}_2dgs_fromsplat --init-ply $E/$S/splat.ply --steps 10000 \
  --dist-start 0 --normal-start 1000 --refine-stop 0 || fail 2DGS

step 3 "borrow surface directions for the splat blobs"
python scripts/borrow_normals.py --ply $E/$S/splat.ply --surfels $E/${S}_2dgs_fromsplat/surfels.npz --object $E/${S}_object/object.json \
  --plane-from $PL --out $E/${S}_2dgs_borrow || fail BORROW

step 4 "sun refit, only the bear's surfels cast"
python scripts/sun_refit.py --surfels $E/${S}_2dgs_fromsplat/surfels_params.pt --ply $E/$S/splat.ply \
  --from $E/${S}_shadow4/shadow_flags.npz --votes $E/${S}_sun/votes.npz --object $E/${S}_object/object.json --plane-from $PL \
  --colmap $COL --images data/$S/images --out $E/${S}_sun_refit_obj --object-casters --eps 0.036 || fail SUN_REFIT

step 5 "sun-shadow labels of the bear"
python scripts/sun_shadow_map.py --surfels $E/${S}_2dgs_fromsplat/surfels_params.pt --ply $E/$S/splat.ply \
  --borrow $E/${S}_2dgs_borrow/borrowed.npz --sun-from $E/${S}_shadow4/shadow_flags.npz --object $E/${S}_object/object.json \
  --plane-from $PL --out $E/${S}_sunmap3 --sun-npz $E/${S}_sun_refit_obj/sun_refit.npz --object-casters --test point --eps 0.122 || fail SUN_MAP
echo; echo "=== STAGE A DONE $(date +%T)"
