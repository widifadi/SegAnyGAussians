# SegAnyGAussians - Quick Start Guide (Windows)

**Last Updated**: 2025-12-03
**Status**: ✅ TESTED AND WORKING on Windows 11 + RTX 4060 Ti

This guide helps you set up SegAnyGAussians on another PC using the provided environment files.

---

## Prerequisites

- Windows 10/11 64-bit
- NVIDIA GPU (Compute Capability 6.0+)
- NVIDIA Driver 520.61+ (for CUDA 11.8)
- Visual Studio 2019/2022 with C++ build tools
- Conda (Anaconda or Miniconda)
- Git for Windows

---

## Method 1: Using environment.yml (RECOMMENDED)

### Step 1: Clone Repository

```cmd
cd D:\Work\Softwares
git clone https://github.com/YOUR_USERNAME/SegAnyGAussians.git
cd SegAnyGAussians\SegAnyGAussians
git submodule update --init --recursive
```

### Step 2: Create Environment from YAML

**Open x64 Native Tools Command Prompt for VS 2022**

```cmd
conda env create -f environment.yml
conda activate SAGA
```

This installs:
- ✅ Python 3.10
- ✅ CUDA Toolkit 11.8
- ✅ PyTorch 2.0.1 + CUDA 11.8
- ✅ NumPy 1.26.x (pinned <2)
- ✅ opencv-python 4.10.x (pinned <4.12)
- ✅ PyTorch3D 0.7.8
- ✅ All dependencies

### Step 3: Set Environment Variables

```cmd
set CUDA_PATH=%CONDA_PREFIX%
set CUDA_HOME=%CONDA_PREFIX%
set PATH=%CONDA_PREFIX%\bin;%PATH%
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=6.0 6.1 7.0 7.5 8.0 8.6 8.9+PTX
set MAX_JOBS=2
set FORCE_CUDA=1
```

### Step 4: Fix simple-knn (CRITICAL for Windows)

**Edit** `submodules\simple-knn\setup.py`:

Add this line after `name="simple_knn",`:
```python
packages=['simple_knn'],
```

**Create** `submodules\simple-knn\simple_knn\__init__.py`:
```python
import os
import sys

# Add torch DLL directory to search path (Windows only)
try:
    import torch
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if sys.platform == 'win32' and os.path.exists(torch_lib_dir):
        os.add_dll_directory(torch_lib_dir)
except Exception:
    pass

from simple_knn._C import *
```

### Step 5: Compile CUDA Extensions

```cmd
cd submodules\diff-gaussian-rasterization
pip install --no-build-isolation .
cd ..\..

cd submodules\diff-gaussian-rasterization-depth
pip install --no-build-isolation .
cd ..\..

cd submodules\diff-gaussian-rasterization_contrastive_f
pip install --no-build-isolation .
cd ..\..

cd submodules\simple-knn
pip install --no-build-isolation .
cd ..\..
```

Each takes 5-15 minutes to compile.

### Step 6: Install Third-Party Packages

```cmd
cd third_party\segment-anything
pip install -e .
cd ..\..

cd third_party\kmeans_pytorch
pip install -e .
cd ..\..
```

### Step 7: Download SAM Checkpoint

```cmd
mkdir third_party\segment-anything\sam_ckpt
powershell -Command "Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth' -OutFile 'third_party\segment-anything\sam_ckpt\sam_vit_h_4b8939.pth'"
```

Or download manually (2.4GB):
- URL: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
- Save to: `third_party\segment-anything\sam_ckpt\sam_vit_h_4b8939.pth`

### Step 8: Verify Installation

```cmd
python -c "import torch, numpy, cv2, pytorch3d; print('PyTorch:', torch.__version__); print('NumPy:', numpy.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch: 2.0.1+cu118
NumPy: 1.26.4
CUDA: True
```

### Step 9: Test SAGA GUI

```cmd
python saga_gui.py -f 10000 -m "path\to\your\trained\model"
```

---

## Method 2: Manual Installation (Alternative)

If `environment.yml` doesn't work, follow the complete step-by-step guide in **`WINDOWS_INSTALLATION.txt`**.

---

## Key Differences from Original Repository

This fork includes Windows compatibility fixes:

1. **PyTorch 2.0 API compatibility** (6 lines of code):
   - `torch.eig` → `torch.linalg.eig` (saga_gui.py)
   - Added explicit `dtype=torch.float32` (cameras.py, network_gui.py)

2. **Windows-specific fixes**:
   - simple-knn setup.py modification
   - simple-knn __init__.py for DLL loading

3. **Critical version constraints**:
   - NumPy <2 (PyTorch 2.0.1 requires NumPy 1.x)
   - opencv-python <4.12 (for NumPy 1.x compatibility)

**No logic changes** - only compatibility fixes!

---

## Troubleshooting

### "RuntimeError: Numpy is not available"
→ Ensure NumPy <2 is installed:
```cmd
pip install "numpy<2"
```

### "ModuleNotFoundError: No module named 'simple_knn._C'"
→ Missing simple-knn setup.py fix. See Step 4.

### "DLL load failed" for simple-knn
→ Missing __init__.py file. See Step 4.

### CUDA extension compilation fails
→ Ensure you're in x64 Native Tools Command Prompt and environment variables are set.

---

## Files Summary

| File | Purpose |
|------|---------|
| `environment.yml` | Conda environment with all dependencies |
| `requirements.txt` | Pip package list (reference only) |
| `WINDOWS_INSTALLATION.txt` | Complete step-by-step guide (17 steps) |
| `QUICK_START.md` | This file - streamlined installation |
| `CHANGES_SUMMARY.md` | Detailed list of all code changes |
| `NUMPY2_COMPATIBILITY_REPORT.md` | Technical analysis |

---

## Estimated Installation Time

- Environment creation: 10-15 minutes
- CUDA extension compilation: 30-45 minutes
- Third-party packages: 5 minutes
- **Total**: ~1 hour (mostly automated)

---

## Next Steps

After installation:
1. Prepare your data (COLMAP format)
2. Train a scene: `python train_scene.py -s <path> -m <output>`
3. Extract masks: `python extract_segment_everything_masks.py`
4. Launch GUI: `python saga_gui.py -f 10000 -m <model_path>`

For detailed usage, see the [original repository](https://github.com/Jumpat/SegAnyGAussians).

---

**Working Stack**:
- Python 3.10
- PyTorch 2.0.1 + CUDA 11.8
- NumPy 1.26.x
- PyTorch3D 0.7.8 (MiroPsota wheels)

✅ **TESTED AND WORKING** on Windows 11 + RTX 4060 Ti

---

For questions or issues, refer to:
- `WINDOWS_INSTALLATION.txt` - Complete guide
- `CHANGES_SUMMARY.md` - What changed from original
- Original repository: https://github.com/Jumpat/SegAnyGAussians
