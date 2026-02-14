GPU setup and PyTorch GPU fix (Windows + WSL)

Goal
- Ensure `torch` is installed with CUDA support and contains CUDA kernels compatible with NVIDIA GeForce RTX 5070 (compute capability sm_120).
- Allow `visual_generation/image/generate_image.py` to use GPU for inference.

Summary
- Your system currently detects the RTX 5070, but the installed PyTorch binary does not contain compiled kernels for sm_120 and fails with:
  `RuntimeError: CUDA error: no kernel image is available for execution on the device`.
- Fix options:
  1. Install an official/ nightly PyTorch wheel that includes sm_120 support (if available). This is the fastest fix.
  2. Build PyTorch from source including `sm_120` (reliable but long).
  3. Use WSL + prebuilt Linux wheel that supports sm_120 if available.

Quick checks
1. Activate your venv and run the check script:

```powershell
.\.venv\Scripts\activate
python visual_generation/image/check_cuda.py
```

2. If `Torch version` shows `+cpu`, you must reinstall a CUDA-enabled wheel.

Fast attempt: install nightly wheel (may or may not include sm_120)

```powershell
.\.venv\Scripts\activate
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
python visual_generation/image/check_cuda.py
```

If the `check_cuda.py` output still warns `not compatible with the current PyTorch installation` for `sm_120`, proceed to build from source.

Recommended (reliable) — Build PyTorch from source (WSL/Linux recommended)

Why use WSL: building PyTorch on Linux/WSL is better-supported and typically simpler than on native Windows. If you must build on Windows, ensure you have Visual Studio and required build tools.

Minimal WSL build steps (Linux environment):

```bash
# in WSL (Ubuntu recommended)
# Install system deps (example for Ubuntu 22.04)
sudo apt update && sudo apt install -y build-essential git cmake python3-dev python3-venv ninja-build libopenblas-dev libblas-dev liblapack-dev

# Install CUDA toolkit in WSL (follow NVIDIA WSL guide)
# After driver + CUDA setup, create venv
python3 -m venv ~/pytorch-venv
source ~/pytorch-venv/bin/activate
python -m pip install --upgrade pip
pip install pyyaml setuptools wheel ninja

# Clone PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# Set the arch list to include sm_120
export TORCH_CUDA_ARCH_LIST="sm_120"
export USE_CUDA=1
# Optional: enable distributed build flags etc.
python -m pip install -r requirements.txt
python setup.py build
python -m pip install -e .
```

Notes
- Building can take a long time (1–4+ hours) depending on machine.
- Ensure you have sufficient disk space (tens of GB) and correct CUDA toolkit matching drivers.
- After build, re-run `visual_generation/image/check_cuda.py`.

Windows native build (advanced)
- Install Visual Studio 2022 with C++ desktop development and Windows SDK.
- Install CUDA toolkit for Windows matching driver.
- Use `Developer Command Prompt` and follow https://github.com/pytorch/pytorch#from-source
- Set `TORCH_CUDA_ARCH_LIST=sm_120` before building.

Alternative: use CPU for development and small/fast images until a GPU-capable PyTorch wheel is available.

Follow-up actions
- Attempt nightly wheel installation and re-check (fast).
- If nightly fails, build from source (WSL recommended).

I can proceed to attempt the nightly install again, or start the WSL build process (will take long). Which should I do next?"}]}]