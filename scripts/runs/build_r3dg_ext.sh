#!/usr/bin/env bash
# Step 3 of the R3DG test: build R3DG's CUDA extensions in env r3dg with the env's own CUDA 11.8 and gcc 11,
# for the RTX 4070 only (sm_89). nvdiffrast 0.3.1 is installed from NVlabs' GitHub (it compiles its kernels on first use).
set -eo pipefail  # no -u: conda compiler activation scripts use unset variables
source ~/miniforge3/bin/activate r3dg
export CUDA_HOME=$CONDA_PREFIX
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export TORCH_CUDA_ARCH_LIST=8.9
cd ~/third_party/Relightable3DGaussian
pip install --no-build-isolation ./submodules/simple-knn
pip install --no-build-isolation ./bvh
pip install --no-build-isolation ./r3dg-rasterization
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git@v0.3.1
python -c "import simple_knn._C, bvh_tracing._C, r3dg_rasterization, nvdiffrast.torch; print('imports ok')" 2>&1 | tail -1
echo "R3DG EXT DONE"
