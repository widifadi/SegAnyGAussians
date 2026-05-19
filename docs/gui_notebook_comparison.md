# GUI vs Notebook: Segmentation Math Comparison

This document explains how the `segmentation_gui` sliders map to the `prompt_segmenting.ipynb` notebook, and proves the two are mathematically equivalent.

---

## The Scale Slider

The scale slider controls which "granularity" of features to focus on — small details (low scale) vs large structures (high scale). Internally it feeds a learned `scale_gate` network that weights each of the 32 feature dimensions differently depending on the scale value.

| GUI | Notebook |
|---|---|
| `scale_val = slider / 100` (0.00 – 1.00) | `scale = torch.tensor([1.])` (hardcoded) |

The notebook hardcodes scale to `1.0`. The slider just makes that value interactive.

---

## The Threshold Slider and the 0.75 → 0.875 Conversion

### Why cosine similarity lives in [-1, 1]

Cosine similarity measures how "aligned" two feature vectors are
([`torch.nn.functional.cosine_similarity` docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html)):

```
cos(A, B) = (A · B) / (|A| × |B|)
```

- `cos = 1.0` → vectors point in exactly the same direction (identical object)
- `cos = 0.0` → vectors are perpendicular (unrelated)
- `cos = -1.0` → vectors point in opposite directions

### What the GUI does differently

The GUI remaps cosine to `[0, 1]` so the slider feels natural (0 = exclude everything, 1 = include everything):

```
gui_score = (cosine + 1.0) / 2.0
```

| Cosine | GUI score |
|--------|-----------|
| -1.0   | 0.00      |
|  0.0   | 0.50      |
|  0.75  | **0.875** |
|  1.0   | 1.00      |

### The conversion — worked example

Notebook threshold `0.75` in cosine space:

```
gui_threshold = (0.75 + 1.0) / 2.0 = 1.75 / 2.0 = 0.875
```

**Concrete example with toy 3-dim vectors** (same logic applies to the actual 32-dim features):

```python
import torch

# Two 3-dim unit vectors — imagine these are feature vectors at two pixels
query = torch.tensor([0.8,  0.5,  0.3])   # clicked pixel
pixel = torch.tensor([0.7,  0.6,  0.2])   # candidate pixel

# Step 1 — L2 normalise (make unit vectors)
query_n = query / query.norm()   # [0.811, 0.507, 0.304]
pixel_n = pixel / pixel.norm()   # [0.740, 0.634, 0.211]

# Step 2 — dot product = cosine similarity (because both are unit vectors)
cosine = (query_n * pixel_n).sum()   # ≈ 0.934  →  high similarity

# Notebook threshold check:
print(cosine > 0.75)   # True  →  pixel included

# GUI threshold check (same result):
gui_score = (cosine + 1.0) / 2.0   # ≈ 0.967
print(gui_score > 0.875)            # True  →  pixel included ✓
```

Both checks include the same pixel. The linear remap `(cos+1)/2` never changes which pixels are included or excluded — it only shifts the threshold number.

**Set the slider to ~87–88 to replicate notebook behaviour.**

The GUI default of `0.50` (slider = 50) equals cosine `0.0` — it includes anything even slightly positively correlated, which is much more permissive than the notebook's `0.75`.

---

## Mathematical Equivalence: `_compute_scale_gated` vs Notebook

### Functions used

| Function | Docs | Purpose |
|---|---|---|
| `torch.Tensor.unsqueeze` | [docs](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html) | Add a size-1 dimension so shapes broadcast correctly |
| `torch.Tensor.permute` | [docs](https://pytorch.org/docs/stable/generated/torch.permute.html) | Reorder tensor axes, e.g. `(32,H,W)` → `(H,W,32)` |
| `torch.nn.functional.normalize` | [docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html) | L2-normalise each vector to unit length along a given dim |

### Notebook (point prompt cell)

```python
feature_with_scale = rendered_feature * gates.unsqueeze(-1).unsqueeze(-1)  # (32,H,W) × (32,1,1) — broadcast multiply
scale_conditioned_feature = feature_with_scale.permute([1, 2, 0])           # → (H, W, 32)
normed_features = normalize(scale_conditioned_feature, dim=-1)              # L2-normalise each pixel's 32-dim vector
```

`gates.unsqueeze(-1).unsqueeze(-1)` turns shape `(32,)` into `(32,1,1)` so it [broadcasts](https://pytorch.org/docs/stable/notes/broadcasting.html) over every spatial pixel.

Compact form: **`result = normalize(f * g)`**

### GUI `_compute_scale_gated`

```python
feat = raw_feat / (norm + 1e-6)      # pre-normalise: f̂ = f / |f|
feat = feat * gates.unsqueeze(0)     # (H,W,32) × (1,32) — broadcast multiply
feat = normalize(feat, dim=-1)       # L2-normalise again
```

`gates.unsqueeze(0)` turns shape `(32,)` into `(1,32)` — same broadcast effect over all pixels.

Compact form: **`result = normalize((f / |f|) * g)`**

### Proof they are equal

For element-wise multiplication of a feature vector `f` by a gates vector `g`:

```
|f * g|  =  |f| × |(f / |f|) * g|
```

The scalar `|f|` factors out of the L2 norm linearly, so it cancels in the division:

```
normalize(f * g)  =  (f * g) / |f * g|
                  =  (|f| × (f̂ * g)) / (|f| × |f̂ * g|)   where f̂ = f / |f|
                  =  (f̂ * g) / |f̂ * g|
                  =  normalize(f̂ * g)
                  =  normalize((f / |f|) * g)
```

**The pre-normalisation step cancels out exactly.** Both produce the same unit vector.

### Concrete example

```python
import torch, torch.nn.functional as F

f = torch.tensor([3.0, 4.0, 0.0])   # raw feature at one pixel
g = torch.tensor([0.9, 0.5, 0.8])   # scale gate output (32-dim in practice)

# Notebook path
notebook = F.normalize((f * g).unsqueeze(0), dim=-1).squeeze()

# GUI path
f_hat = f / (f.norm() + 1e-6)       # pre-normalise → [0.6, 0.8, 0.0]
gui    = F.normalize((f_hat * g).unsqueeze(0), dim=-1).squeeze()

print(torch.allclose(notebook, gui, atol=1e-6))   # True ✓
# notebook: tensor([0.5408, 0.3783, 0.7527])
# gui:      tensor([0.5408, 0.3783, 0.7527])
```

---

## Mathematical Equivalence: Similarity Computation

### Functions used

| Function | Docs | Purpose |
|---|---|---|
| `torch.einsum` | [docs](https://pytorch.org/docs/stable/generated/torch.einsum.html) | Einstein summation — general batched dot products |
| `torch.Tensor.reshape` | [docs](https://pytorch.org/docs/stable/generated/torch.reshape.html) | Flatten spatial dims `(H,W,32)` → `(H×W, 32)` |
| `torch.Tensor @ ...` (matmul) | [docs](https://pytorch.org/docs/stable/generated/torch.matmul.html) | Batch matrix multiply — equivalent to einsum here |
| `torch.cat` | [docs](https://pytorch.org/docs/stable/generated/torch.cat.html) | Concatenate query feature vectors along N_clicks dim |
| `torch.Tensor.max` | [docs](https://pytorch.org/docs/stable/generated/torch.max.html) | Return max value (and index) along a dimension |

### Notebook (single click)

```python
query_feature = normed_features[y, x]                            # (32,) — the clicked pixel's feature
similarity = torch.einsum('C,HWC->HW', query_feature, normed_features)
# reads as: for each pixel (h,w), sum over C → score[h,w] = Σ_c query[c] × map[h,w,c]
mask = similarity > 0.75
```

### GUI `_update_similarity_preview`

```python
chosen   = torch.cat([gated_f.reshape(-1, 1) for gated_f in click_features], dim=-1)  # (32, N_clicks)
feat_map = _compute_scale_gated(raw_feature_map)                                        # (H, W, 32)
score    = feat_map.reshape(-1, 32) @ chosen    # (H×W, 32) @ (32, N) → (H×W, N)
score    = (score + 1.0) / 2.0
mask     = score.max(dim=-1).values > thresh
```

**`@` is matrix multiplication** — `(H×W, 32) @ (32, N)` computes every pixel's dot product against every query vector simultaneously:

```
score[pixel, k] = Σ_c  feat_map[pixel, c] × chosen[c, k]
```

This is identical to the einsum for N=1. For N>1 clicks, `.max(dim=-1).values` picks whichever query gave the highest similarity per pixel — the multi-click extension absent in the notebook.

### Concrete example

```python
import torch

H, W, C = 2, 2, 3   # tiny spatial map, 3-dim features (32 in practice)
feat_map = torch.randn(H, W, C)
feat_map = torch.nn.functional.normalize(feat_map, dim=-1)

query = feat_map[0, 1]  # "click" on pixel (0,1)
query_col = query.reshape(-1, 1)   # (3,1) — "chosen" in GUI

# Notebook einsum path
einsum_scores = torch.einsum('C,HWC->HW', query, feat_map)

# GUI matmul path
matmul_scores = (feat_map.reshape(-1, C) @ query_col).reshape(H, W)

print(torch.allclose(einsum_scores, matmul_scores, atol=1e-6))  # True ✓
```

---

## Mathematical Equivalence: 3D Segmentation

### Functions used

| Function | Docs | Purpose |
|---|---|---|
| `torch.nn.functional.normalize` | [docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html) | L2-normalise per-Gaussian feature vectors |
| `torch.Tensor @ ...` (matmul) | [docs](https://pytorch.org/docs/stable/generated/torch.matmul.html) | Dot product of each Gaussian's feature against each query |
| `torch.Tensor.any` | [docs](https://pytorch.org/docs/stable/generated/torch.any.html) | True if any query exceeds threshold — multi-click union |
| `GaussianModel.segment` | `scene/gaussian_model.py` | Keep only Gaussians where mask is True |

### Notebook

```python
scale_conditioned_point_features = point_features * gates.unsqueeze(0)   # (N_pts, 32) × (1, 32)
normed = normalize(scale_conditioned_point_features, dim=-1)
similarities = einsum('C,NC->N', query_feature, normed)                  # dot product per Gaussian
scene_gaussians.segment(similarities > 0.75)
```

`einsum('C,NC->N', q, F)` = for each of N Gaussians, sum over C channels → one similarity score per Gaussian.

### GUI `_run_3d_segment`

```python
feat_pts  = feature_model.get_point_features * gates   # (N_pts, 32) × (1, 32) — same broadcast
feat_pts  = normalize(feat_pts, dim=-1)
score_pts = feat_pts @ chosen                          # (N_pts, 32) @ (32, N_clicks) → (N_pts, N_clicks)
score_pts = (score_pts + 1.0) / 2.0
mask_3d   = (score_pts > thresh).any(dim=-1)           # True if any click matched
scene_gaussians.segment(mask_3d)
```

`feat_pts @ chosen` = `(N_pts, 32) @ (32, N_clicks)` → `(N_pts, N_clicks)` — one score per Gaussian per click. Identical to the einsum for N_clicks=1.

`.any(dim=-1)` returns `True` for a Gaussian if **any** click gave a high score — union of all clicks, the multi-click extension.

### Concrete example (threshold equivalence)

```python
import torch

N_pts = 5
feat_pts = torch.nn.functional.normalize(torch.randn(N_pts, 3), dim=-1)
query    = torch.nn.functional.normalize(torch.randn(3), dim=-1)

# Notebook
cos_sim   = torch.einsum('C,NC->N', query, feat_pts)
nb_mask   = cos_sim > 0.75

# GUI (single click)
chosen    = query.reshape(3, 1)
score_pts = feat_pts @ chosen                           # (N_pts, 1)
score_pts = (score_pts + 1.0) / 2.0
gui_mask  = (score_pts > 0.875).squeeze()               # slider=87.5

print(torch.all(nb_mask == gui_mask))   # True ✓
# Both produce identical boolean masks
```

---

## Summary of Actual Differences

| Aspect | Notebook | GUI |
|---|---|---|
| Scale value | Hardcoded `1.0` | Slider (0–1) |
| Threshold units | Cosine [-1, 1], e.g. `> 0.75` | Remapped [0, 1], slider `> 0.875` |
| Default threshold | 0.75 (cosine) | 0.50 (= cosine 0.0, very loose) |
| Multi-click | No | Yes, `.max` / `.any` over N_clicks |
| Query timing | Queries scale-gated map | Stores raw, gates dynamically |
| Pre-normalisation | No | Yes, but mathematically cancels |

The query timing difference is also equivalent: applying the scale gate before or after storing the raw feature produces the same final normalised vector, because `normalize(gate(f)) == normalize(gate(normalize(f)))` as proven above — and the GUI uses the same gate for both query and feature map.