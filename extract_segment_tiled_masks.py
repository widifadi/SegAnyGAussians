"""
extract_segment_tiled_masks.py

Drop-in replacement for extract_segment_everything_masks.py that adds tiling
support for large images (e.g. aerial photogrammetry with 100k×100k pixels).

Output format is identical: per-image .pt files containing a bool tensor of
shape [N_masks, H_out, W_out] saved under <image_root>/sam_masks/.

When the image fits within --tile_size (default 3000), processing is identical
to the original script (single tile = whole image).  When tiling kicks in:
  1. The image is sliced into overlapping tiles (overlap = --overlap pixels).
  2. SamAutomaticMaskGenerator runs independently on each tile.
  3. Tile masks are OR-merged back into full-resolution boolean arrays.
  4. Unique merged masks are stacked and saved exactly like the original.

Usage:
    python extract_segment_tiled_masks.py \
        --image_root /path/to/scene \
        --sam_checkpoint_path ./third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth \
        --tile_size 3000 \
        --overlap 300

Optional flags (same as original):
    --downsample 4          Resize each tile by 1/N before running SAM
    --downsample_type mask  Downsample after segmenting (original behaviour)
    --sam_arch vit_h
"""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------

def _make_tiles(h: int, w: int, tile_size: int, overlap: int):
    """Return list of (y0, y1, x0, x1) tile boxes covering the full image."""
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    ys = list(range(0, h, stride))
    xs = list(range(0, w, stride))

    tiles = []
    for y in ys:
        y0 = y
        y1 = min(y + tile_size, h)
        for x in xs:
            x0 = x
            x1 = min(x + tile_size, w)
            tiles.append((y0, y1, x0, x1))
            if x1 == w:
                break
        if y1 == h:
            break
    return tiles


def _generate_masks_tiled(mask_generator: SamAutomaticMaskGenerator,
                           img: np.ndarray,
                           tile_size: int,
                           overlap: int) -> list:
    """
    Run SamAutomaticMaskGenerator on tiles and OR-merge results.

    Returns a list of H×W bool numpy arrays, one per unique merged mask.
    Each mask covers the full image resolution.
    """
    h, w = img.shape[:2]

    if h <= tile_size and w <= tile_size:
        # Fast path: whole image fits in one tile
        raw = mask_generator.generate(img)
        return [m["segmentation"].astype(bool) for m in raw]

    tiles = _make_tiles(h, w, tile_size, overlap)
    # Accumulate: list of full-res bool arrays
    merged: list[np.ndarray] = []

    for y0, y1, x0, x1 in tiles:
        tile_img = img[y0:y1, x0:x1]
        tile_masks = mask_generator.generate(tile_img)

        for m in tile_masks:
            seg = m["segmentation"].astype(bool)  # shape: (y1-y0, x1-x0)

            # Place tile mask into a full-res canvas
            canvas = np.zeros((h, w), dtype=bool)
            canvas[y0:y1, x0:x1] = seg

            # OR-merge with any existing mask that substantially overlaps
            merged_flag = False
            for i, existing in enumerate(merged):
                intersection = np.logical_and(existing, canvas).sum()
                union = np.logical_or(existing, canvas).sum()
                if union > 0 and intersection / union > 0.3:
                    merged[i] = np.logical_or(existing, canvas)
                    merged_flag = True
                    break

            if not merged_flag:
                merged.append(canvas)

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = ArgumentParser(
        description="SAM segment-everything mask extraction with tiling support"
    )
    parser.add_argument("--image_root", default="./", type=str,
                        help="Scene root directory (must contain images/ subdir)")
    parser.add_argument("--sam_checkpoint_path",
                        default="./third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth",
                        type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--downsample", default=1, type=int,
                        help="Downsample factor (applied per-tile when tiling)")
    parser.add_argument("--downsample_type", default="image", type=str,
                        choices=["image", "mask"],
                        help="Downsample then segment (image) or segment then downsample (mask)")
    parser.add_argument("--tile_size", default=3000, type=int,
                        help="Max tile dimension in pixels. Images smaller than this "
                             "are processed whole (same as original script).")
    parser.add_argument("--overlap", default=300, type=int,
                        help="Overlap between adjacent tiles in pixels.")

    args = parser.parse_args()

    print("Initializing SAM...")
    sam = sam_model_registry[args.sam_arch](
        checkpoint=args.sam_checkpoint_path
    ).to("cuda")

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        box_nms_thresh=0.7,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    # Resolve image directory (mirrors original downsample logic)
    downsample_manually = False
    if args.downsample == 1 or args.downsample_type == "mask":
        IMAGE_DIR = os.path.join(args.image_root, "images")
    else:
        IMAGE_DIR = os.path.join(args.image_root, "images_" + str(args.downsample))
        if not os.path.exists(IMAGE_DIR):
            IMAGE_DIR = os.path.join(args.image_root, "images")
            downsample_manually = True
            print("No pre-downsampled image folder found; downsampling manually.")

    assert os.path.exists(IMAGE_DIR), \
        f"Image directory not found: {IMAGE_DIR}. Set --image_root correctly."

    OUTPUT_DIR = os.path.join(args.image_root, "sam_masks")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"tile_size={args.tile_size}  overlap={args.overlap}  "
          f"downsample={args.downsample} ({args.downsample_type})")
    print("Extracting SAM masks...")

    for filename in tqdm(sorted(os.listdir(IMAGE_DIR))):
        name = filename.rsplit(".", 1)[0]
        img = cv2.imread(os.path.join(IMAGE_DIR, filename))
        if img is None:
            continue

        # --- image-mode downsampling (resize before SAM) ---
        if downsample_manually or (args.downsample > 1 and args.downsample_type == "image"):
            new_w = img.shape[1] // args.downsample
            new_h = img.shape[0] // args.downsample
            img = cv2.resize(img, dsize=(new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

        full_h, full_w = img.shape[:2]

        # --- run SAM (with tiling when needed) ---
        bool_masks = _generate_masks_tiled(
            mask_generator, img, args.tile_size, args.overlap
        )

        # --- mask-mode downsampling (resize after SAM) ---
        mask_list = []
        for seg in bool_masks:
            m_score = torch.from_numpy(seg).float().to("cuda")

            if args.downsample > 1 and args.downsample_type == "mask":
                out_h = full_h // args.downsample
                out_w = full_w // args.downsample
                m_score = torch.nn.functional.interpolate(
                    m_score.unsqueeze(0).unsqueeze(0),
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                m_score = (m_score >= 0.5).bool()
            else:
                m_score = m_score.bool()

            # Drop degenerate masks (all-zero or all-one)
            if len(m_score.unique()) < 2:
                continue
            mask_list.append(m_score)

        if len(mask_list) == 0:
            print(f"  Warning: no valid masks for {filename}, skipping.")
            continue

        masks = torch.stack(mask_list, dim=0)  # [N, H, W] bool
        torch.save(masks, os.path.join(OUTPUT_DIR, name + ".pt"))