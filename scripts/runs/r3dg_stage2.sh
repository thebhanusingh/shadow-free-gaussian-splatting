#!/usr/bin/env bash
# Step 6 of the R3DG test: stage 2 (NeILF: base colour, roughness, environment light, ray-traced visibility) on the
# stage 1 checkpoint, geometry frozen, settings from R3DG's run_tnt.sh, plus OUR sun as a fixed-direction light
# (patch_r3dg_sun.py; direction = the refit sun from sun_refit.py --object-casters, COLMAP / splat coordinates).
set -eo pipefail  # no -u: conda activation scripts use unset variables
ITERS=${1:-50000}
SAMPLES=${2:-32}
source ~/miniforge3/bin/activate r3dg
python ~/Claude_Test/scripts/patch_r3dg_sun.py
export R3DG_SUN=$(python -c "import numpy as np; s=np.load('$HOME/Claude_Test/exports/IMG_6705_sun_refit_obj/sun_refit.npz')['sun']; print(','.join(f'{v:.6f}' for v in s))")
export R3DG_SUN_INIT=3.0
echo "sun direction $R3DG_SUN"
cd ~/third_party/Relightable3DGaussian
O=~/Claude_Test/outputs_r3dg/IMG_6705
python train.py --eval -s ~/Claude_Test/data/IMG_6705_r3dg_half -m $O/neilf_sun -c $O/3dgs/chkpnt30000.pth -r 1 \
    --data_device cpu --save_training_vis --position_lr_init 0 --position_lr_final 0 --normal_lr 0 --sh_lr 0 \
    --opacity_lr 0 --scaling_lr 0 --rotation_lr 0 --iterations $ITERS --lambda_base_color_smooth 1 \
    --lambda_roughness_smooth 0.2 --lambda_light_smooth 1 --lambda_light 0.01 -t neilf --sample_num $SAMPLES \
    --save_training_vis_iteration 200 --lambda_env_smooth 0.01
echo "R3DG STAGE 2 DONE"
