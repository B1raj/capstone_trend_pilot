#!/usr/bin/env bash
set -euo pipefail

# Build PyTorch in WSL with support for sm_120 (RTX 5070)
# Usage: run inside WSL from project root or call via Windows: wsl bash -ic "/mnt/c/Users/rakib/Desktop/Work/Capstone/visual_generation/image/build_pytorch_wsl.sh"

ROOT_DIR="/mnt/c/Users/rakib/Desktop/Work/Capstone"
cd "$ROOT_DIR"

echo "[1/8] Update apt packages"
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-dev python3-venv ninja-build libopenblas-dev libblas-dev liblapack-dev libffi-dev libssl-dev pkg-config libjpeg-dev libpng-dev

echo "[2/8] Setup Python venv"
python3 -m venv ~/pytorch-build-venv
source ~/pytorch-build-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pyyaml setuptools wheel ninja

echo "[3/8] Install CUDA toolkit in WSL if not present"
# On WSL, NVIDIA drivers should be available; if CUDA toolkit isn't installed, user should follow NVIDIA WSL instructions.
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Please install CUDA toolkit in WSL following NVIDIA instructions: https://docs.nvidia.com/cuda/wsl-user-guide/index.html"
  echo "Continuing; build may fail if CUDA toolchain is missing."
fi

echo "[4/8] Clone PyTorch (if needed)"
if [ ! -d "$ROOT_DIR/pytorch" ]; then
  git clone --recursive https://github.com/pytorch/pytorch.git "$ROOT_DIR/pytorch"
else
  echo "PyTorch repo already exists; pulling latest"
  cd "$ROOT_DIR/pytorch"
  git fetch origin
  git checkout main
  git pull --rebase
  git submodule sync
  git submodule update --init --recursive
  cd "$ROOT_DIR"
fi

cd "$ROOT_DIR/pytorch"

echo "[5/8] Set build env for sm_120"
export TORCH_CUDA_ARCH_LIST="sm_120"
export USE_CUDA=1
export MAX_JOBS=$(nproc)

echo "[6/8] Install Python requirements"
python -m pip install -r requirements.txt || true

echo "[7/8] Build PyTorch (this will take a long time)"
python setup.py build -j ${MAX_JOBS}

echo "[8/8] Install PyTorch into venv"
python -m pip install -e .

echo "PyTorch build complete. Verify with: python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))'"
