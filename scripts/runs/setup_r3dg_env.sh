#!/usr/bin/env bash
# Step 2 of the R3DG test: a separate conda env "r3dg" (the working env "gsplat" is not touched).
# The repo pins Python 3.7 / PyTorch 1.12 + CUDA 11.6 and recommends compiling with CUDA 11.8, so: Python 3.9,
# PyTorch 2.0.1 + cu118, conda CUDA toolkit 11.8 and gcc/g++ 11 (the system gcc 13/15 is too new for CUDA 11.8).
set -euo pipefail
source ~/miniforge3/bin/activate
mamba create -y -n r3dg python=3.9
conda activate r3dg
mamba install -y -c nvidia/label/cuda-11.8.0 cuda-toolkit
mamba install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch_scatter==2.1.1 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install kornia==0.6.12 numpy==1.23.5 matplotlib==3.5.2 tensorboard tqdm dearpygui==1.10.1 imageio==2.31.2 \
    opencv-python==4.8.1.78 pillow==9.5.0 plyfile==0.9 scipy==1.10.1 ninja
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvcc --version | tail -1
echo "R3DG ENV OK"
