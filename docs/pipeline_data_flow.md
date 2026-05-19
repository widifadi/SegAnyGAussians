# SAGA Pipeline: Data Flow from SAM Masks to 3D Segmentation

This document explains in plain terms what each intermediate file contains, how it is produced, and how everything connects to the 32-dimensional contrastive feature that drives segmentation.

> **Paper reference**: Cen et al., *"Segment Any 3D Gaussians"*, AAAI 2025.
> ArXiv: [2312.00860](https://arxiv.org/abs/2312.00860)

---

## The Big Picture

```
Training images
      │
      ▼
extract_segment_tiled_masks.py
      │  SAM segments every image into N binary masks
      ▼
sam_masks/*.pt          ← (N_masks, H, W) bool — which pixels belong to which mask
      │
      ▼
get_scale.py
      │  Renders depth per camera, back-projects mask pixels to 3D, measures spread
      ▼
mask_scales/*.pt        ← (N_masks,) float — physical 3D size of each mask in metres
      │
      ▼
train_contrastive_feature.py
      │  Learns 32-dim fingerprints so same-mask Gaussians cluster together
      ▼
contrastive_feature_point_cloud.ply   ← 32 floats per Gaussian
scale_gate.pt                         ← tiny network: scale value → 32-dim gate
      │
      ▼
segmentation_gui / prompt_segmenting.ipynb
      │  Click a pixel → sample its fingerprint → match against all Gaussians
      ▼
3D segmentation mask
```

---

## Step 1 — `sam_masks/*.pt`

### What is stored

One `.pt` file per training image. Loading it gives a **3-D boolean tensor**:

```
shape: (N_masks, H, W)   dtype: torch.bool
```

Real example from sitinggil dataset:

```python
torch.load('sam_masks/DJI_001.pt').shape
# torch.Size([81, 735, 981])
# → 81 masks, each covering a 735×981 pixel image
```

Each `mask[i]` is a **binary image** — `True` where pixels belong to segment `i`, `False` everywhere else. The masks can overlap (a pixel can belong to a small mask and a larger mask that contains it — SAM intentionally generates hierarchical, multi-scale masks).

### Where it comes from

SAM (*Segment Anything Model*, Meta AI) runs in "segment everything" mode — no prompts, just automatic detection of every plausible region at every scale. The tiled variant (`extract_segment_tiled_masks.py`) splits large images into overlapping tiles first so SAM can handle aerial/high-resolution imagery.

> **From the SAGA paper**: *"We use SAM to generate a set of masks for each training view. These masks naturally form a hierarchy — from fine-grained parts to whole objects — and serve as the spatial supervision signal for contrastive feature learning."*

---

## Step 2 — `get_scale.py` and `mask_scales/*.pt`

### The problem it solves

Each SAM mask is a 2D region in one image. To train scale-aware features, SAGA needs to know **how large that region is in 3D space** — a mask covering a teacup should have a very different scale from a mask covering a building facade.

### How it works

For each training camera:

1. **Render a depth map** — the 3DGS model renders a `(H, W)` depth image, giving the distance from the camera to each Gaussian at every pixel.

2. **Back-project to 3D** — using the camera's focal length (`fx`, `fy`) and principal point (`cx`, `cy`), each pixel `(u, v)` with depth `d` is converted to a 3D point:

   ```
   X = (u - cx) × d / fx
   Y = (v - cy) × d / fy
   Z = d
   ```

   This gives a `(H, W, 3)` point cloud where every pixel has an (X, Y, Z) world position.

3. **Measure each mask's 3D spread** — for every mask `i`, select only the 3D points that fall inside that mask region, then compute:

   ```python
   scale[i] = (points_in_mask.std(dim=0) * 2).norm()
   #            └─ std along X, Y, Z separately ─┘   └─ combine into one number
   ```

   `std × 2` approximates the diameter (2σ covers ~95% of a normal distribution). `.norm()` collapses the three spatial dimensions into a single scalar.

### What is stored

One `.pt` file per training image, shape `(N_masks,)` — one float per mask:

```python
torch.load('mask_scales/DJI_001.pt')
# tensor([0.363, 1.204, 4.353, ..., 12.841])   shape: (81,)
# units: roughly metres in the reconstructed scene
```

Real value range from sitinggil: **0.36 – 12.84**, mean **4.35** (metres).

A small scale (≈0.36) means the mask covered a compact region in 3D (a small stone, a tile). A large scale (≈12.84) means the mask covered a large region (a wall section, a tree canopy).

> **From the SAGA paper**: *"We estimate the 3D scale of each mask by projecting the rendered depth map and computing the standard deviation of the 3D point cloud within the mask region. This scale serves as a continuous supervision signal that allows the model to learn features that are consistent across viewpoints at the appropriate granularity."*

---

## Step 3 — `train_contrastive_feature.py` and the 32-dim fingerprint

### What the 32-dim vector is

Every Gaussian ends up with a learned **32-dimensional float vector** — its fingerprint. These 32 numbers have no individual human-readable meaning; they are purely geometric coordinates in a learned embedding space where:

- **Same-object Gaussians** → nearby in 32-dim space → high dot product
- **Different-object Gaussians** → far apart in 32-dim space → low dot product

```
Gaussian A (part of car):    [0.12, -0.34, 0.87, ...]   ← 32 floats
Gaussian B (same car):       [0.11, -0.31, 0.89, ...]   ← similar → high dot product
Gaussian C (road surface):   [0.90,  0.22, -0.10, ...]  ← different → low dot product
```

The dimensionality 32 is a design choice (`FEATURE_DIM = 32`). It is large enough to distinguish many objects in a scene, small enough to render efficiently.

### How training uses `sam_masks` and `mask_scales`

The training loop for one iteration:

1. **Pick a random camera view.**

2. **Render a `(32, H, W)` feature map** — the 3DGS feature rasterizer splats each Gaussian's 32-dim vector onto the image, the same way RGB rendering splats colour.

3. **Sample random pixel pairs.** For each pair `(pixel_A, pixel_B)`:
   - Look up which SAM masks contain pixel A, which contain pixel B
   - **Positive pair**: both pixels are inside the same SAM mask → their features *should* be similar
   - **Negative pair**: pixels are in different SAM masks → their features *should* be different

4. **Apply the scale gate** — before comparing features, weight the 32 dimensions by a gate derived from the mask's 3D scale (`mask_scales`). This ensures features are compared at the correct granularity (small-scale details vs large-scale structure).

5. **Compute contrastive loss** and backpropagate — this nudges the 32-dim vectors of positive pairs closer together and negative pairs further apart.

> **From the SAGA paper**: *"We adopt a contrastive learning objective where pixel pairs within the same SAM mask are treated as positives and pairs across different masks as negatives. The loss encourages the rendered feature map to be consistent with the 2D mask supervision, while the 3D Gaussian representation ensures multi-view consistency without additional cross-view supervision."*

### The scale hierarchy — why masks at multiple scales matter

SAM produces masks at many scales simultaneously. Without scale awareness, the model would be confused: should a Gaussian on a car door be similar to other car-door Gaussians (fine scale) or to all-car Gaussians (coarse scale)? Both are valid depending on what you want to segment.

SAGA solves this by training the gate to answer: *"at scale s, which of the 32 dimensions matter?"*

```
Small scale (0.0) → gate emphasises dimensions that encode fine parts
Large scale (1.0) → gate emphasises dimensions that encode whole objects
```

This is why the scale slider in the GUI changes what gets segmented — the same 32-dim fingerprints are reweighted by the gate before the dot product comparison.

> **From the SAGA paper**: *"To enable scale-aware segmentation, we introduce a scale gate module — a lightweight MLP that takes the normalised 3D scale as input and outputs a per-dimension attention weight. This allows users to interactively control the granularity of segmentation at inference time without retraining."*

---

## Step 4 — Output files

### `contrastive_feature_point_cloud.ply`

The main output of training. Stores the same Gaussian positions as `scene_point_cloud.ply` but replaces the SH/colour attributes with the 32 learned floats per Gaussian.

### `scale_gate.pt`

A tiny two-layer network:

```python
scale_gate = torch.nn.Sequential(
    torch.nn.Linear(1, 32),   # maps one scalar → 32 values
    torch.nn.Sigmoid()         # squash to (0, 1) — acts as per-dimension on/off weights
)
```

Input: one scalar (the slider value, 0–1, representing normalised 3D scale).
Output: 32-dim gate vector used to reweight the fingerprint before computing similarity.

---

## Summary: what each file does at a glance

| File | Shape | Contains | Used by |
|---|---|---|---|
| `sam_masks/*.pt` | `(N_masks, H, W)` bool | Binary pixel masks from SAM | `get_scale.py`, `train_contrastive_feature.py` |
| `mask_scales/*.pt` | `(N_masks,)` float | 3D physical size of each mask | `train_contrastive_feature.py` |
| `contrastive_feature_point_cloud.ply` | `(N_gaussians, 32)` float | Learned fingerprint per Gaussian | Segmentation GUI / notebook |
| `scale_gate.pt` | Linear(1→32) weights | Scale-to-gate mapping network | Segmentation GUI / notebook |

---

## Why this works without per-scene class labels

SAGA never needs you to define what classes exist. The SAM masks are **class-agnostic** — they just say "these pixels go together" without naming what they are. The contrastive training learns to honour those groupings in 3D. At inference you supply the class implicitly by clicking — the clicked pixel's fingerprint becomes the query, and similarity thresholding finds all Gaussians that the training decided belong to the same group.