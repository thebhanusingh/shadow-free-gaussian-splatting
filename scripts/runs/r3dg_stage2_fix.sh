#!/usr/bin/env bash
# R3DG stage 2 retry: the traced sun visibility is right (96.6% of photo-shadow ground blocked, 0.4% of lit ground),
# but the first run put the shadow in the paint. So: sun strength FIXED from the concrete (a neutral concrete of
# albedo 0.6 lit at the measured linear brightness), no per-Gaussian local light, paint smoothness guided by the photo
# divided by the modelled light (a shadow edge the visibility explains is no excuse for a paint edge), low sky start.
set -eo pipefail  # no -u: conda activation scripts use unset variables
ITERS=${1:-40000}
SAMPLES=${2:-8}
source ~/miniforge3/bin/activate r3dg
python ~/Claude_Test/scripts/patch_r3dg_sun.py
export R3DG_SUN=$(python -c "import numpy as np; s=np.load('$HOME/Claude_Test/exports/IMG_6705_sun_refit_obj/sun_refit.npz')['sun']; print(','.join(f'{v:.6f}' for v in s))")
export R3DG_SUN_FIX=3.8,3.5,3.0
export R3DG_NO_LOCAL=1
export R3DG_SHADE_GUIDE=1
echo "sun direction $R3DG_SUN, fixed strength $R3DG_SUN_FIX"
cd ~/third_party/Relightable3DGaussian
O=~/Claude_Test/outputs_r3dg/IMG_6705
python train.py --eval -s ~/Claude_Test/data/IMG_6705_r3dg_half -m $O/neilf_sun_fix -c $O/3dgs/chkpnt30000.pth -r 1 \
    --data_device cpu --save_training_vis --position_lr_init 0 --position_lr_final 0 --normal_lr 0 --sh_lr 0 \
    --opacity_lr 0 --scaling_lr 0 --rotation_lr 0 --iterations $ITERS --lambda_base_color_smooth 1 \
    --lambda_roughness_smooth 0.2 --lambda_light_smooth 1 --lambda_light 0.01 -t neilf --sample_num $SAMPLES \
    --save_training_vis_iteration 200 --lambda_env_smooth 0.01 --light_init 0.05
python eval_sun.py -m $O/neilf_sun_fix -c $O/neilf_sun_fix/chkpnt$ITERS.pth --sample_num $SAMPLES
echo "R3DG STAGE 2 FIX DONE"
