#!/usr/bin/env bash
# Check that every input the photo-correction pass needs exists for the given scenes. Prints one line per file.
# Usage: bash scripts/runs/check_inputs.sh IMG_6704 IMG_6705
cd ~/Claude_Test || exit 1
bad=0
for S in "$@"; do
  for f in exports/${S}_2dgs_fromsplat/surfels_params.pt exports/${S}/splat.ply exports/${S}_sunmap3/sun_visibility.npz \
           exports/${S}_sun_refit_obj/sun_refit.npz exports/${S}_object/object.json exports/${S}_shadowfix_o/stats.json \
           data/${S}/colmap/sparse/0_refined/images.bin data/${S}/colmap/sparse/0_refined/cameras.bin \
           data/${S}/transforms.json data/${S}/transforms_nsprocess.json; do
    if [ -e "$f" ]; then echo "ok       $f"; else echo "MISSING  $f"; bad=$((bad+1)); fi
  done
  echo "photos $(ls data/$S/images 2>/dev/null | wc -l), shadow masks $(ls data/$S/shadow_masks 2>/dev/null | wc -l)  ($S)"
done
echo "missing files: $bad"
