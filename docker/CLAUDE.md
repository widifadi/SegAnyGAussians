# SegAnyGAussians (SAGA) - Development Workflow Guide

**Last Updated:** 2025-10-31
**Purpose:** Docker-based development environment setup for SAGA with custom code integration

---

## 🎯 Current Progress Tracker

### ✅ Completed Setup
- [x] Docker container with volume mounts configured
- [x] VS Code attached to running container
- [x] Python environment auto-activation (.bashrc + settings.json)
- [x] WSL2 memory increased to 12GB (via .wslconfig)
- [x] evaluation_toolkit installed and imports working
- [x] Path fixes applied (parent directory for SAGA modules)
- [x] `01_create_ground_truth.py` - Export working (full_scene.ply generated)

### 🔄 In Progress
- [ ] Manual segmentation in CloudCompare (fence, roof, etc.)
- [ ] Custom code development for manual segmentation assistance

### 📋 Next Steps
1. Complete manual segmentation in CloudCompare
2. Test `01_create_ground_truth.py --create-mask` to generate GT masks
3. Test `02_evaluate_segmentation.py` with real predictions
4. Test remaining evaluation scripts (03, 04, 05)
5. Document complete workflow for PC 2 deployment

### 📁 Data Locations
- **Training Data:** `D:\Work\data\sitinggil_3D` → `/workspace/data`
- **SAGA Output:** `D:\Work\Softwares\SegAnyGAussians\output\sitinggil_3D_r8` → `/workspace/output`
- **Segmentation Results:** `D:\Work\Softwares\SegAnyGAussians\segmentation_res` → `/workspace/segmentation_res`
- **Existing Predictions:** `sitinggil_3D.pt`, `sitinggil_3D_fence.pt`, `sitinggil_3D_roof2.pt`

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [VS Code Dev Container Setup (RECOMMENDED)](#vs-code-dev-container-setup-recommended)
3. [Docker Development Options](#docker-development-options)
4. [Project Structure & Key Components](#project-structure--key-components)
5. [Adding Custom Code](#adding-custom-code)
6. [SAGA Workflow Understanding](#saga-workflow-understanding)
7. [Reference Commands](#reference-commands)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites
- Docker Desktop installed and running
- NVIDIA Docker runtime (nvidia-docker2) installed
- VS Code with "Dev Containers" extension (for Option 2)
- Docker image `seganyGaussians` already built

### Verify Docker Setup
```bash
# Check Docker is running
docker ps

# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi

# Check image exists
docker images | grep seganyGaussians
```

---

## VS Code Dev Container Setup (RECOMMENDED)

This setup provides the best development experience with full IDE features inside the container.

### Step 1: Create Dev Container Configuration

Create `.devcontainer/devcontainer.json` in your project root:

```json
{
  "name": "SegAnyGAussians Dev",
  "image": "seganyGaussians",
  "runArgs": [
    "--gpus", "all"
  ],
  "mounts": [
    "source=${localWorkspaceFolder},target=/workspace/SegAnyGAussians,type=bind"
  ],
  "workspaceFolder": "/workspace/SegAnyGAussians",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/workspace/SegAnyGAussians/gaussian_env/bin/python",
        "python.terminal.activateEnvironment": true
      }
    }
  },
  "postStartCommand": "source /workspace/SegAnyGAussians/gaussian_env/bin/activate"
}
```

### Step 2: Open in Container

1. Open VS Code in project directory: `D:\Work\Softwares\SegAnyGAussians`
2. Press `F1` → type "Dev Containers: Reopen in Container"
3. Wait for container to start (first time may take a moment)
4. VS Code will reload inside the container

### Step 3: Verify Setup

Open terminal in VS Code (it will be inside the container):

```bash
# Check Python environment
which python
# Should show: /workspace/SegAnyGAussians/gaussian_env/bin/python

# Verify CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Check working directory
pwd
# Should show: /workspace/SegAnyGAussians
```

### Benefits of Dev Container
- ✅ Full IntelliSense and code completion
- ✅ Integrated debugging (F5)
- ✅ Git integration
- ✅ Terminal runs inside container automatically
- ✅ Edit files seamlessly with all IDE features
- ✅ No manual volume mounting needed

---

## Docker Development Options

### Option 1: Simple Volume Mount (For Quick Testing)

```bash
# Windows PowerShell
docker run -it --gpus all `
  -v D:\Work\Softwares\SegAnyGAussians:/workspace/SegAnyGAussians `
  seganyGaussians

# Inside container
source /workspace/SegAnyGAussians/gaussian_env/bin/activate
cd /workspace/SegAnyGAussians
```

**Use case:** Quick testing, running scripts, checking outputs

### Option 2: VS Code Dev Containers (RECOMMENDED)

See detailed setup above.

**Use case:** Full development, debugging, adding custom code

### Option 3: Docker Compose

Create `docker-compose.yml` in project root:

```yaml
version: '3.8'
services:
  saga-dev:
    image: seganyGaussians
    runtime: nvidia
    volumes:
      - .:/workspace/SegAnyGAussians
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    working_dir: /workspace/SegAnyGAussians
    stdin_open: true
    tty: true
    command: /bin/bash
```

Run with:
```bash
docker-compose run --rm saga-dev
```

**Use case:** Team collaboration, reproducible environment

---

## Project Structure & Key Components

```
SegAnyGAussians/
├── .devcontainer/
│   └── devcontainer.json          # VS Code container config
├── submodules/
│   └── diff-gaussian-rasterization/
├── gaussian_renderer/             # Core rendering code
├── scene/                         # Scene and Gaussian models
├── utils/                         # Utility functions
├── train.py                       # Training script
├── train_contrastive_feature.py   # Feature training
├── render.py                      # Rendering script
├── prompt_segmenting.ipynb        # Interactive segmentation notebook
├── evaluation_toolkit/            # Segmentation evaluation tools
│   ├── 01_create_ground_truth.py
│   ├── 02_evaluate_segmentation.py
│   ├── 03_create_consensus_gt.py
│   ├── 04_transfer_dense_labels.py
│   ├── 05_batch_evaluation.py
│   └── utils.py
└── CLAUDE.md                      # This file
```

### Key Python Modules

**Core Components:**
- `scene/gaussian_model.py` - 3D Gaussian splatting model
- `scene/contrastive_feature_gaussian_model.py` - Feature-based Gaussians
- `gaussian_renderer/__init__.py` - Rendering pipeline
- `utils/graphics_utils.py` - Graphics utilities
- `utils/loss_utils.py` - Loss functions

**Your Custom Code Should Go In:**
- `evaluation_toolkit/` - For evaluation-related code
- Create new modules in root directory for custom experiments
- Extend classes in `scene/` for model modifications

---

## Adding Custom Code

### Workflow for Custom Development

1. **Edit Files in VS Code (Windows)**
   - Changes are immediately synced to container
   - Full IDE features available

2. **Run/Test in Container Terminal**
   - Terminal inside VS Code runs in container
   - Access to all dependencies and GPU

3. **No Rebuild Required When:**
   - Adding new Python files
   - Modifying existing code
   - Installing packages via pip (though they won't persist after container stops)

4. **Rebuild Required When:**
   - Adding system-level dependencies
   - Modifying Dockerfile
   - Installing new apt packages
   - Permanently adding Python packages to environment

### Example: Adding Custom Segmentation Module

**Step 1:** Create your module (Windows side, in VS Code)

```python
# D:\Work\Softwares\SegAnyGAussians\custom_segmentation\my_module.py
import torch
import numpy as np

class CustomSegmenter:
    def __init__(self, feature_dim=32):
        self.feature_dim = feature_dim

    def segment(self, features, query_point):
        """Your custom segmentation logic"""
        # This code runs with access to all container dependencies
        pass
```

**Step 2:** Use it in the container (terminal in VS Code)

```bash
# Terminal is already inside container with environment activated
cd /workspace/SegAnyGAussians
python -c "from custom_segmentation.my_module import CustomSegmenter; print('Import successful')"
```

**Step 3:** Integrate with existing SAGA code

Modify `prompt_segmenting.ipynb` or create a new notebook:

```python
from custom_segmentation.my_module import CustomSegmenter

# Use with existing SAGA components
segmenter = CustomSegmenter()
result = segmenter.segment(point_features, query_feature)
```

### Installing Additional Python Packages

**Temporary (current session only):**
```bash
pip install package_name
```

**Permanent (add to Dockerfile):**
1. Modify Dockerfile to include package
2. Rebuild image:
```bash
docker build -t seganyGaussians .
```

---

## SAGA Workflow Understanding

### Core Workflow: prompt_segmenting.ipynb

The notebook provides **alternative segmentation tools**, not sequential steps:

```
Initialization
    ↓
Point Prompt (select query point & scale)
    ↓
    ├─→ Cluster in 2D    (2D visualization)
    ├─→ Segmentation 3D  (point-based 3D segmentation)
    └─→ Cluster in 3D    (automatic multi-object)
```

### Key Shared Variables

These variables are shared across notebook sections via Jupyter's global scope:

1. **`scale`** - Object scale (0, 0.5, or 1.0)
2. **`gates`** - Scale conditioning gates from `scale_gate(scale)`
3. **`query_feature`** - Feature vector of selected point
4. **`scale_conditioned_feature`** - 2D rendered features (for 2D clustering)
5. **`normed_features`** - Normalized features
6. **`point_features`** - Raw 3D Gaussian features

### 2D vs 3D Feature Usage

**2D Clustering:**
```python
3D point_features
→ render_contrastive_feature()
→ rendered_feature (HxWxC image)
→ multiply by gates
→ scale_conditioned_feature
→ cluster pixels in 2D image
```

**3D Segmentation:**
```python
3D point_features
→ multiply by gates directly
→ scale_conditioned_point_features (NxC)
→ find 3D Gaussians similar to query_feature
```

**Why different?**
- 2D works with rendered features from a specific viewpoint (image space)
- 3D works with raw point features (3D space)
- Both derive from same source: `feature_gaussians.get_point_features`

### Training Pipeline

1. **Train base 3D Gaussians:**
   ```bash
   python train.py -s <path_to_scene> -m <output_model>
   ```

2. **Train contrastive features:**
   ```bash
   python train_contrastive_feature.py -m <model_path>
   ```

3. **Interactive segmentation:**
   - Open `prompt_segmenting.ipynb`
   - Run initialization cells
   - Use point prompts and segmentation tools

---

## Reference Commands

### Docker Management

```bash
# List running containers
docker ps

# List all containers
docker ps -a

# Stop container
docker stop <container_id>

# Remove container
docker rm <container_id>

# View logs
docker logs <container_id>

# Execute command in running container
docker exec -it <container_id> bash
```

### VS Code Dev Container

```
F1 → "Dev Containers: Reopen in Container"
F1 → "Dev Containers: Rebuild Container"
F1 → "Dev Containers: Attach to Running Container"
F1 → "Dev Containers: Reopen Folder Locally"
```

### Python Environment (Inside Container)

```bash
# Activate environment
source /workspace/SegAnyGAussians/gaussian_env/bin/activate

# Check installed packages
pip list

# Install new package (temporary)
pip install package_name

# Run tests
pytest tests/

# Run evaluation
cd evaluation_toolkit
python 02_evaluate_segmentation.py --help
```

### Git Operations

```bash
# Check status (works inside container)
git status

# Create branch for custom work
git checkout -b feature/custom-segmentation

# Commit changes
git add .
git commit -m "Add custom segmentation module"
```

---

## Troubleshooting

### Issues Solved During Setup

#### ✅ Import Error: "No module named 'scene'"
**Problem:** Scripts couldn't import SAGA modules when run from evaluation_toolkit/

**Solution:** Add parent directory to Python path in each script:
```python
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
```

#### ✅ Import Error: "No module named 'utils.system_utils'"
**Problem:** Name collision - evaluation_toolkit/utils.py conflicted with SAGA's utils/ package

**Solution:** Renamed evaluation_toolkit/utils.py → eval_utils.py

#### ✅ Import Error: "name 'Namespace' is not defined"
**Problem:** cfg_args file uses argparse.Namespace but it wasn't imported

**Solution:** Add to imports:
```python
from argparse import Namespace
```

#### ✅ Memory Error: Script Crashes Silently
**Problem:** Container had only 7.7GB RAM, not enough for 1.6M Gaussians

**Solution:** Create C:\Users\<Username>\.wslconfig:
```ini
[wsl2]
memory=12GB
swap=4GB
processors=4
```
Then: `wsl --shutdown` and restart Docker Desktop

#### ✅ VS Code Environment Not Auto-Activating
**Problem:** New terminals didn't activate gaussian_env automatically

**Solution:**
1. Add to .bashrc: `echo 'source /workspace/SegAnyGAussians/gaussian_env/bin/activate' >> ~/.bashrc`
2. Update .vscode/settings.json with proper paths and bash login shell

---

### Common Issues

### Docker Desktop Not Starting
- Check if WSL 2 is enabled
- Verify Docker Desktop settings → Use WSL 2 backend
- Restart Docker Desktop

### GPU Not Visible in Container
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi

# If fails, check nvidia-docker2 installation
# Windows: Make sure NVIDIA Container Toolkit is installed
```

### Volume Mount Not Working
- Docker Desktop → Settings → Resources → File Sharing
- Ensure `D:\` drive is shared
- Try absolute path: `D:\Work\Softwares\SegAnyGAussians`

### Python Environment Not Activated
```bash
# Manually activate
source /workspace/SegAnyGAussians/gaussian_env/bin/activate

# Check it's activated
which python
# Should show: /workspace/SegAnyGAussians/gaussian_env/bin/python
```

### Changes Not Reflected in Container
- Verify volume mount in `docker run` command or `devcontainer.json`
- Check file permissions
- Restart container if needed

### Import Errors in Custom Code
```bash
# Make sure you're in the right directory
cd /workspace/SegAnyGAussians

# Add to Python path if needed
export PYTHONPATH="${PYTHONPATH}:/workspace/SegAnyGAussians"

# Or modify sys.path in your code
import sys
sys.path.insert(0, '/workspace/SegAnyGAussians')
```

### Dev Container Won't Start
- Check Docker Desktop is running
- Verify image exists: `docker images | grep seganyGaussians`
- Check `.devcontainer/devcontainer.json` syntax
- View logs: VS Code → Output → Dev Containers

---

## Best Practices

### Development Workflow

1. **Use Dev Containers for development** - best experience
2. **Use volume mount for quick tests** - faster startup
3. **Keep Dockerfile updated** - document all system dependencies
4. **Use git branches** - separate custom work from base SAGA code
5. **Document your changes** - update this file with custom additions

### Code Organization

```
custom_segmentation/           # Your custom modules
├── __init__.py
├── models.py                  # Custom models
├── utils.py                   # Custom utilities
└── experiments/               # Experimental code
    └── experiment_01.py

notebooks/                     # Custom notebooks
└── custom_experiment.ipynb
```

### Testing Custom Code

```bash
# Unit tests
pytest custom_segmentation/tests/

# Integration with SAGA
python -c "from custom_segmentation import CustomSegmenter; print('OK')"

# Run full pipeline
python custom_segmentation/experiments/experiment_01.py
```

---

## Useful Resources

**Project Links:**
- Repository: https://github.com/Jumpat/SegAnyGAussians
- Paper: Segment Any 3D Gaussians (AAAI-25)

**Docker Documentation:**
- Dev Containers: https://code.visualstudio.com/docs/devcontainers/containers
- NVIDIA Docker: https://github.com/NVIDIA/nvidia-docker

**SAGA Components:**
- 3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
- Segment Anything: https://github.com/facebookresearch/segment-anything

---

## Quick Commands Reference

```bash
# Start container with volume mount
docker run -it --gpus all -v D:\Work\Softwares\SegAnyGAussians:/workspace/SegAnyGAussians seganyGaussians

# Inside container setup
source /workspace/SegAnyGAussians/gaussian_env/bin/activate
cd /workspace/SegAnyGAussians

# Test CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Run evaluation
cd evaluation_toolkit
python 02_evaluate_segmentation.py --pred <pred.pt> --gt <gt.pt> --output <results.json>

# Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

---

**Remember:**
- Edit code in VS Code on Windows
- Run/test in container (terminal in VS Code)
- Changes sync automatically via volume mount
- No rebuild needed for code changes
- Rebuild only for system dependencies

**End of Guide**
