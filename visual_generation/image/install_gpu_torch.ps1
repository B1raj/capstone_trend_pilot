# PowerShell helper: attempt installing GPU-enabled PyTorch and verify
# Usage: open PowerShell, activate venv, then run this script:
# .\.venv\Scripts\Activate
# .\visual_generation\image\install_gpu_torch.ps1

Write-Host "Checking current torch / CUDA status..."
python visual_generation/image/check_cuda.py

Write-Host "Trying to install latest stable cu121 wheel (may already be installed)..."
pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121

Write-Host "Try nightly build if stable wheel doesn't support your GPU (optional)..."
$choice = Read-Host "Install nightly (may be large download ~2-3GB)? (y/n)"
if ($choice -eq 'y') {
    pip install --pre --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
}

Write-Host "Re-checking torch / CUDA status..."
python visual_generation/image/check_cuda.py

Write-Host "If the output warns about incompatible compute capability (sm_120), follow visual_generation/image/GPU_SETUP.md to build from source or use WSL."
