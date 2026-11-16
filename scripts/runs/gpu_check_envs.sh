#!/usr/bin/env bash
# Quick check after a laptop reset: does each conda env still see the GPU?
# Usage: bash scripts/runs/gpu_check_envs.sh  (log goes to logs/gpu_check_envs.log)
source ~/miniforge3/etc/profile.d/conda.sh
cd ~/Claude_Test
mkdir -p logs
LOG=logs/gpu_check_envs.log
: > "$LOG"
for env in gsplat r3dg gsir gaussian_shader sddnet hloc colmap4; do
  echo "=== $env ===" | tee -a "$LOG"
  conda run --no-capture-output -n "$env" python scripts/runs/gpu_check_env.py 2>&1 | tee -a "$LOG"
done
echo "=== colmap binary (colmap4) ===" | tee -a "$LOG"
conda run --no-capture-output -n colmap4 colmap -h 2>&1 | head -n 2 | tee -a "$LOG"
