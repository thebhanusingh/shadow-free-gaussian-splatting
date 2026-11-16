#!/bin/bash
# Train a normal splatfacto-big splat on the shadow-corrected photos of bear 2 (data/IMG_6705_fixed, made by
# scripts/correct_photos.py). Same settings as the original IMG_6705 run (run_full_6705.sh step 1): 30k steps, full
# resolution, stop-split 7000 on the 8 GB laptop card, same COLMAP poses (copied, not re-run), so the two splats can be
# compared view for view. Run: scripts/runs/run_train_fixed.sh 2>&1 | tee logs/IMG_6705_fixed_train.log
set -o pipefail
cd ~/Claude_Test
STOP_SPLIT=7000 ./scripts/make_splat_hq.sh /nonexistent.MOV IMG_6705_fixed
