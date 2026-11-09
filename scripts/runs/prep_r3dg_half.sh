#!/usr/bin/env bash
# R3DG data at half size: R3DG loads every image at full resolution in float64 before resizing (about 32 GB for
# 650 frames at 1903 x 1066), so the frames are undistorted straight to max 960 px (cameras scaled to match) and
# R3DG's image loader is switched to float32 (original kept as scene/utils.py.orig).
set -eo pipefail  # no -u: conda activation scripts use unset variables
cd ~/Claude_Test
source ~/miniforge3/bin/activate gsplat
D=data/IMG_6705_r3dg_half
colmap image_undistorter --image_path data/IMG_6705/images --input_path data/IMG_6705/colmap/sparse/0_refined \
    --output_path $D --output_type COLMAP --max_image_size 960
mkdir -p $D/sparse/0 && mv $D/sparse/*.bin $D/sparse/0/
python scripts/make_r3dg_masks.py --data $D --ply exports/IMG_6705/splat.ply --object exports/IMG_6705_object/object.json --grow 3
U=~/third_party/Relightable3DGaussian/scene/utils.py
[ -f $U.orig ] || cp $U $U.orig
sed -i "s|        img = img / 255$|        img = (img / 255).astype(np.float32)  # local fix: float32 (was float64)|" $U
grep -n "img = (img / 255)" $U
grep -n "^import numpy\|^import numpy as np" $U
ls $D/images | wc -l; ls $D/masks | wc -l
echo "R3DG HALF DATA DONE"
