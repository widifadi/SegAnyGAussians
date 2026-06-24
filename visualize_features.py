"""
visualize_features.py

Convert contrastive-feature .pt files (C, H, W) produced by
  render.py --target contrastive_feature
into viewable RGB images via PCA (32 dims → 3 RGB channels).

PCA is fit on a random subsample drawn from ALL frames in the directory
so that colours are globally consistent across views within one scene.
Each directory gets its own PCA fit (independent colour mapping).

Outputs (written next to each renders/ folder):
  renders_pca/          — plain PCA colour images
  renders_pca_blend/    — PCA overlaid on the GT image (with --blend)

Usage:
  python visualize_features.py --scale 8 ^
    --renders_dir "output/ori_sitinggil_.../train/ours_-1/renders"

  # With GT overlay (requires gt/ folder next to renders/)
  python visualize_features.py --scale 8 --blend 0.5 ^
    --renders_dir "output/ori_sitinggil_.../train/ours_-1/renders" ^
                  "output/ori_stras_.../train/ours_-1/renders"
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import PCA


def process_dir(renders_dir: str,
                subsample: int = 100_000,
                scale: int = 1,
                blend_alpha: float = 0.0) -> None:

    pt_files = sorted(f for f in os.listdir(renders_dir) if f.endswith(".pt"))
    if not pt_files:
        print(f"  [skip] No .pt files in {renders_dir}")
        return

    parent   = os.path.dirname(renders_dir)
    gt_dir   = os.path.join(parent, "gt")
    out_dir  = os.path.join(parent, "renders_pca")
    os.makedirs(out_dir, exist_ok=True)

    do_blend = blend_alpha > 0.0 and os.path.isdir(gt_dir)
    if blend_alpha > 0.0 and not os.path.isdir(gt_dir):
        print(f"  [warn] --blend requested but gt/ not found at {gt_dir} — skipping blend")
    if do_blend:
        blend_dir = os.path.join(parent, "renders_pca_blend")
        os.makedirs(blend_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Pass 1 — load tensors, collect subsample for PCA fitting
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    per_file = []       # list of (fname, vecs (H*W,C), H, W)
    sample_parts = []

    print(f"  Reading {len(pt_files)} feature maps…")
    for fname in tqdm(pt_files, desc="  load"):
        t = torch.load(os.path.join(renders_dir, fname), map_location="cpu")  # (C,H,W)
        C, H, W = t.shape
        vecs = t.permute(1, 2, 0).reshape(-1, C).numpy().astype(np.float32)
        per_file.append((fname, vecs, H, W))

        n   = max(1, subsample // len(pt_files))
        idx = rng.choice(len(vecs), min(n, len(vecs)), replace=False)
        sample_parts.append(vecs[idx])

    # ------------------------------------------------------------------
    # Fit PCA
    # ------------------------------------------------------------------
    sample = np.concatenate(sample_parts, axis=0)
    print(f"  Fitting PCA on {len(sample):,} sample vectors…")
    pca = PCA(n_components=3)
    pca.fit(sample)

    # Project all frames and find global range for consistent normalisation
    projections = [pca.transform(vecs) for _, vecs, _, _ in per_file]
    g_min = min(p.min() for p in projections)
    g_max = max(p.max() for p in projections)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print(f"  Saving to {out_dir}" + (f" + {blend_dir}" if do_blend else "") + "…")
    for (fname, _, H, W), proj in tqdm(zip(per_file, projections),
                                       total=len(pt_files), desc="  save"):
        # Normalise PCA projection → uint8 RGB
        rgb = (proj - g_min) / (g_max - g_min + 1e-8)
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8).reshape(H, W, 3)

        pca_img = Image.fromarray(rgb)
        if scale != 1:
            pca_img = pca_img.resize((W * scale, H * scale), Image.LANCZOS)

        stem = os.path.splitext(fname)[0]
        pca_img.save(os.path.join(out_dir, stem + ".png"))

        if do_blend:
            # Load matching GT image (same stem, any common extension)
            gt_img = None
            for ext in (".png", ".jpg", ".JPG", ".jpeg"):
                gt_path = os.path.join(gt_dir, stem + ext)
                if os.path.exists(gt_path):
                    gt_img = Image.open(gt_path).convert("RGB")
                    break

            if gt_img is None:
                continue

            # GT is already at render resolution (H×W); upscale both to same size
            if scale != 1:
                gt_img = gt_img.resize((W * scale, H * scale), Image.LANCZOS)

            blended = Image.blend(gt_img, pca_img, alpha=blend_alpha)
            blended.save(os.path.join(blend_dir, stem + ".png"))

    print(f"  Done — {len(pt_files)} images written.")


def main():
    parser = argparse.ArgumentParser(
        description="PCA-visualise contrastive-feature .pt renders as RGB images")
    parser.add_argument(
        "--renders_dir", nargs="+", required=True,
        help="Path(s) to renders/ folders containing .pt feature files")
    parser.add_argument(
        "--subsample", type=int, default=100_000,
        help="Pixel vectors sampled for PCA fitting (default 100k)")
    parser.add_argument(
        "--scale", type=int, default=1,
        help="Upscale factor for output images (e.g. 8 to undo -r 8 training)")
    parser.add_argument(
        "--blend", type=float, default=0.0, metavar="ALPHA",
        help="Blend PCA over GT image. 0=off, 0.5=half-half, 1.0=PCA only. "
             "Requires a gt/ folder next to each renders/ folder.")
    args = parser.parse_args()

    for d in args.renders_dir:
        print(f"\n{'='*60}\nProcessing: {d}\n{'='*60}")
        process_dir(os.path.normpath(d),
                    subsample=args.subsample,
                    scale=args.scale,
                    blend_alpha=args.blend)


if __name__ == "__main__":
    main()