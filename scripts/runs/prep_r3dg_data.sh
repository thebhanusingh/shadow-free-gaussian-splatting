#!/usr/bin/env bash
# Step 4 of the R3DG test: bear 2 in the form R3DG reads (COLMAP sparse/0 with PINHOLE cameras, images/, masks/).
# - undistort the 650 frames with COLMAP (R3DG only accepts undistorted pinhole cameras); same coordinates as our
#   splat, so our sun direction applies unchanged;
# - masks keep the bear and the ground within 2 x its height (make_r3dg_masks.py);
# - fix R3DG's COLMAP mask line, which otherwise has no effect (mask = 1 - (0/1)/255 is about 1 everywhere);
#   the original file is kept as dataset_readers.py.orig.
set -eo pipefail  # no -u: conda compiler activation scripts use unset variables
cd ~/Claude_Test
source ~/miniforge3/bin/activate gsplat
D=data/IMG_6705_r3dg
colmap image_undistorter --image_path data/IMG_6705/images --input_path data/IMG_6705/colmap/sparse/0_refined \
    --output_path $D --output_type COLMAP
mkdir -p $D/sparse/0 && mv $D/sparse/*.bin $D/sparse/0/
python scripts/make_r3dg_masks.py --data $D --ply exports/IMG_6705/splat.ply --object exports/IMG_6705_object/object.json
R=~/third_party/Relightable3DGaussian/scene/dataset_readers.py
[ -f $R.orig ] || cp $R $R.orig
sed -i "s|        mask = 1.0 - load_mask_bool(mask_path) / 255|        mask = load_mask_bool(mask_path)  # local fix: 1 = keep|" $R
grep -n "mask = load_mask_bool(mask_path)" $R
ls $D; ls $D/images | wc -l; ls $D/masks | wc -l
echo "R3DG DATA DONE"
