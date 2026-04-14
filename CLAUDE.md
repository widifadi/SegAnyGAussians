# CLAUDE.md — SegAnyGAussians Development Guide

## Project Context

This is a 3D Gaussian Splatting (3DGS) scene segmentation system. Training and inference pipelines are **complete and working**. The goal of this development track is to add a lightweight GUI on top of the existing inference pipeline — do NOT change the core training or rendering code.

Hardware context: RTX 5080 (sm_120), CUDA 12.8, Python 3.10, PyTorch 2.8.0+cu128.

---

## What NOT to Touch

These files implement the core pipeline and must not be modified:

- `train_scene.py`, `train_contrastive_feature.py` — training pipelines
- `render.py`, `gaussian_renderer/` — rendering
- `scene/gaussian_model.py`, `scene/gaussian_model_ff.py` — Gaussian data structures
- `scene/dataset_readers.py`, `scene/cameras.py`, `scene/colmap_loader.py`
- `submodules/` — CUDA extensions
- `third_party/` — SAM
- `extract_segment_everything_masks.py`, `get_scale.py`, `get_clip_features.py`
- `clip_utils/` — CLIP/SAM mask generation utilities
- `arguments/` — training config
- `utils/` — loss, graphics, image, camera utilities
- `lpipsPyTorch/`

Do not refactor, rename, or reorganize any existing code unless a bug in it directly blocks the new GUI.

---

## New Feature: Simple Segmentation GUI

**Target file:** `segmentation_gui.py` (new file in repo root)

This is a standalone GUI that replicates the `prompt_segmenting.ipynb` workflow with a proper interface. It is NOT a replacement or rewrite of `saga_gui.py` — it is a simpler, separate tool.

### Framework

Use **PyQt5** with an embedded matplotlib figure for the image canvas. Rationale:
- Handles click events on images natively
- No browser required (unlike Gradio)
- No real-time rasterization loop needed (unlike DearPyGUI)
- Familiar widget toolkit

Dependencies that are already available in the SAGA conda env: `PyQt5`, `matplotlib`, `numpy`, `torch`. Do NOT add new conda/pip dependencies unless absolutely necessary.

### Workflow the GUI Must Replicate

This mirrors `prompt_segmenting.ipynb` exactly:

```
Load model paths
  → load scene GaussianModel from scene_point_cloud.ply
  → load feature GaussianModel from feature_point_cloud.ply
  → load scale_gate.pt network
  → load training camera views (for reference image list)
Select reference view (from dropdown of training cameras)
  → render feature map at that view
  → render RGB image at that view (for display background)
User clicks point(s) on the image
  → sample feature at clicked pixel
  → compute cosine similarity across all pixels
  → threshold → 2D preview mask
  → threshold → 3D segmentation (apply to Gaussians)
Export → save_ply() of segmented Gaussians
```

### Feature 1: Multi-Object Selection Within Same Class

**What "same class" means here:** multiple spatially separate instances of similar objects (e.g., two chairs). The contrastive feature space encodes appearance/scale similarity, not semantic class labels, so "same class" = objects with visually similar feature vectors.

**Implementation approach** (already proven in `saga_gui.py` lines 631–660):
- Maintain a list of query feature vectors
- Each click appends one 32-dim feature vector to the list
- Similarity is `max` over all query features (or mean — test both, use max)
- A "Clear points" button resets the list

Key code pattern from `saga_gui.py` to adapt:
```python
# Accumulate: chosen_feature shape (32, N_clicks)
chosen_feature = torch.cat([chosen_feature, new_feature.unsqueeze(-1)], dim=-1)
# Similarity: score per pixel = max similarity to any query
score_map = (feature_map @ chosen_feature).max(dim=-1).values
```

**Note:** SAM is not involved in the interactive segmentation at inference time. The interactive segmentation is purely feature-similarity-based. Multi-object selection is entirely feasible and already proven.

### Feature 2: Export Segmented Object as Single PLY

After 3D segmentation (`GaussianModel.segment(mask)` has already been called), the segmented Gaussians are already isolated in memory. Export is:

```python
# The segmented scene model already has only the target Gaussians
# after segment() has been called
scene_gaussian_model.save_ply(output_path)
```

`save_ply()` is defined in `scene/gaussian_model.py` (~line 215). It exports the full Gaussian attributes: xyz, SH coefficients, opacity, scale, rotation. This is a valid `.ply` readable by standard 3DGS viewers.

GUI requirement:
- "Export PLY" button, enabled only after a 3D segmentation has been run
- File dialog to choose output path
- After export, offer "Roll back" to restore full scene for another segmentation pass

**Important:** After `segment()`, the model is mutated. Use `roll_back()` before running a new segmentation, or reload models. Maintain a `segmented = False` state flag.

### Feature 3: Skip Open-Vocab / CLIP

Do not include any CLIP-based text prompt functionality. No `clip_utils/` imports in the new GUI.

### Feature 4: High-Resolution Training Support

**Decision: use existing downsampling only, do not change core code.**

The training pipeline already supports `--resolution` flags (`-r 1`, `-r 2`, `-r 4`, `-r 8`) in both `train_scene.py` and `train_contrastive_feature.py`. The `dataset_readers.py` handles this transparently.

For large-area scenes with high-res images, the correct approach is:
1. Use `--resolution 2` or `--resolution 4` when calling `train_scene.py` and `train_contrastive_feature.py`
2. Use `--resolution 2` or `--resolution 4` in `extract_segment_everything_masks.py`

Add a simple note in the GUI's "Load Model" section reminding the user to train with `--resolution` flags for large scenes. Do NOT implement patching or any changes to training code — that would change core functionality.

---

## GUI Layout

```
Window: "SAGA Segmentation"
┌─────────────────────────────────────────────────────────────┐
│ [Left Panel - Controls]    │ [Right Panel - Image Canvas]   │
│                            │                                 │
│ Model Paths:               │  [Matplotlib figure, click-    │
│  Scene PLY: [path] [...]   │   able, shows RGB + mask       │
│  Feature PLY: [path] [..]  │   overlay]                    │
│  Scale Gate: [path] [...]  │                                 │
│  Data Path: [path] [...]   │                                 │
│  [Load Models]             │                                 │
│                            │                                 │
│ Reference View:            │                                 │
│  [Dropdown camera list]    │                                 │
│  [Render View]             │                                 │
│                            │                                 │
│ Segmentation:              │                                 │
│  Scale: [slider 0-1]       │                                 │
│  Threshold: [slider 0-1]   │                                 │
│  [Add Point] [Clear Points]│                                 │
│  Points: 0 selected        │                                 │
│  [Run 3D Segment]          │                                 │
│  [Roll Back]               │                                 │
│                            │                                 │
│ Export:                    │                                 │
│  [Export PLY...]           │                                 │
│  [Export Mask (.pt)...]    │                                 │
│                            │                                 │
│ Status: [status label]     │                                 │
└─────────────────────────────────────────────────────────────┘
```

- Canvas click while "Add Point" mode is active → samples feature at pixel → shows point marker → updates similarity preview
- Similarity preview: semi-transparent red overlay on top of RGB image
- Scale slider controls scale gate input (same as `saga_gui.py`)
- Threshold slider controls segmentation cutoff

---

## Key Functions to Reuse (Read These Before Implementing)

| Source | Function/Class | What It Does |
|--------|---------------|--------------|
| `scene/gaussian_model.py` | `GaussianModel` | Scene Gaussian storage, `load_ply`, `save_ply`, `segment`, `roll_back` |
| `scene/gaussian_model_ff.py` | `FeatureGaussianModel` | Feature vectors, same segment/rollback interface |
| `scene/__init__.py` | `Scene` | Loads scene + cameras from data path |
| `gaussian_renderer/__init__.py` | `render()` | RGB render from a camera |
| `gaussian_renderer/feature_renderer.py` | `render_contrastive_feature()` | Feature map render |
| `arguments/__init__.py` | `ModelParams`, `PipelineParams`, `OptimizationParams` | Config dataclasses for model loading |
| `utils/sh_utils.py` | `SH2RGB()` | Convert SH DC component to RGB (for feature viz) |

How `prompt_segmenting.ipynb` loads everything — replicate this exactly:
```python
# Arguments
parser = ArgumentParser()
model = ModelParams(parser)
pipeline = PipelineParams(parser)
args = parser.parse_args(["--model_path", MODEL_PATH, "--source_path", SOURCE_PATH])

# Scene (cameras + Gaussians)
scene = Scene(model.extract(args), GaussianModel(0), load_iteration=SCENE_ITER, shuffle=False)

# Feature model
feature_gaussians = FeatureGaussianModel(FEATURE_DIM)
feature_gaussians.load_ply(FEATURE_PCD_PATH)

# Scale gate
scale_gate = torch.nn.Sequential(torch.nn.Linear(1, FEATURE_DIM), torch.nn.Sigmoid())
scale_gate.load_state_dict(torch.load(SCALE_GATE_PATH))
scale_gate = scale_gate.cuda()
```

---

## Implementation Order (fits in one day)

1. **Scaffold** (`segmentation_gui.py`): PyQt5 window, left/right panels, all widgets wired to stubs. ~1 hour
2. **Model loading**: Reuse the notebook's load pattern, triggered by [Load Models] button. ~30 min
3. **Render view**: Render RGB + feature map for selected camera, display in canvas. ~45 min
4. **Click → feature sample → 2D preview**: Click handler, feature lookup, similarity heatmap overlay. ~1 hour
5. **Multi-point accumulation**: Maintain query list, max-similarity over queries, Clear Points. ~30 min
6. **3D segmentation**: Call `feature_gaussians.segment()` and `scene_gaussians.segment()` with threshold mask. ~30 min
7. **PLY export**: File dialog → `scene_gaussians.save_ply(path)`. ~20 min
8. **Roll back + state management**: `roll_back()` on both models, reset query list. ~20 min
9. **Polish**: Status messages, error handling for missing paths, disable buttons in wrong state. ~30 min

Total: ~5.5 hours of focused implementation.

---

## Common Pitfalls

- Feature render returns a `(H, W, 32)` tensor. Normalize before similarity: `feats = feats / feats.norm(dim=-1, keepdim=True)`
- Scale gate input must be a `(1, 1)` CUDA float tensor matching the slider value
- After `segment()`, the model's `_xyz`, `_features_dc` etc. are replaced with the subset. `roll_back()` restores originals — keep the original PLY paths available for a full reload if needed
- Camera views from `scene.getTrainCameras()` — use these for the dropdown; each has `.image_name` attribute
- `render()` returns a dict; RGB image is at key `"render"` with shape `(3, H, W)`, range [0, 1]
- `render_contrastive_feature()` returns dict; feature map is at key `"render"` with shape `(32, H, W)`
- For the matplotlib canvas, use `FigureCanvasQTAgg` from `matplotlib.backends.backend_qtagg`
- Run all model inference in a `QThread` worker to avoid freezing the UI