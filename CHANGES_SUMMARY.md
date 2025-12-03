# Changes Summary - Windows-Compatible Fork

**Last Updated**: 2025-12-03
**Status**: ✅ Tested and working on Windows 11 + RTX 4060 Ti

This fork adds Windows compatibility and updates to modern PyTorch/NumPy versions.

---

## Key Changes

### 1. Updated Software Stack

| Component | Original | This Fork |
|-----------|----------|-----------|
| Python | 3.7 | **3.10** |
| PyTorch | 1.12.1 + CUDA 11.6 | **2.0.1 + CUDA 11.8** |
| NumPy | Any | **<2.0 (1.26.x)** |
| OpenCV | Any | **<4.12** |
| PyTorch3D | Build from source | **0.7.8 (prebuilt wheels)** |

### 2. Code Compatibility Fixes (6 lines total)

**saga_gui.py:552** - Replace deprecated `torch.eig()` with `torch.linalg.eig()`

**scene/cameras.py:62** - Add explicit `dtype=torch.float32` to `torch.tensor()`

**gaussian_renderer/network_gui.py:74,77** - Add explicit `dtype=torch.float32` to `torch.tensor()`

### 3. Windows-Specific Fixes

**simple-knn Windows DLL Fix:**
- Modified `submodules/simple-knn/setup.py` to include `packages=['simple_knn']`
- Created `submodules/simple-knn/simple_knn/__init__.py` to load torch DLLs on Windows

### 4. Critical Version Constraints

⚠️ **IMPORTANT**: PyTorch 2.0.1 requires NumPy <2.0

```bash
pip install "numpy<2" "opencv-python<4.12"
```

**Why?** PyTorch 2.0.1 was compiled with NumPy 1.x C API and cannot use NumPy 2.x (released June 2024).

---

## Installation

📖 **Quick Start**: See [QUICK_START.md](QUICK_START.md) for streamlined Windows installation (~1 hour)

📚 **Detailed Guide**: See [WINDOWS_INSTALLATION.txt](WINDOWS_INSTALLATION.txt) for complete 17-step guide with troubleshooting

🔬 **Technical Analysis**: See [docs/NUMPY2_COMPATIBILITY_ANALYSIS.md](docs/NUMPY2_COMPATIBILITY_ANALYSIS.md) for detailed compatibility analysis

---

## What Hasn't Changed

✅ **No logic modifications** - Only API compatibility updates
✅ **All original features work** - Training, segmentation, GUI, rendering
✅ **Same model format** - Compatible with original SAGA checkpoints
✅ **Same data format** - Use original SAGA datasets

---

## Credits

- **Original SAGA**: [Jumpat/SegAnyGAussians](https://github.com/Jumpat/SegAnyGAussians)
- **PyTorch3D Windows wheels**: [MiroPsota/torch_packages_builder](https://github.com/MiroPsota/torch_packages_builder)
- **simple-knn Windows fix**: Adapted from [3dgs-mcmc](https://github.com/ubc-vision/3dgs-mcmc)
