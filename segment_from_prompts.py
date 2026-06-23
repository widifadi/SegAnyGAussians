"""
segment_from_prompts.py

Run 3D segmentation from prompt features saved via the GUI's "Save Prompts"
button, without launching the full GUI.

Outputs (written to --output_dir):
  <name>.pt   — boolean mask over all Gaussians (compatible with render.py --precomputed_mask)
  <name>.ply  — PLY of the selected Gaussians only

Usage:
  python segment_from_prompts.py \
    --model_path "output/ori_sitinggil_3D_2025_subset_local" \
    --prompts    "output/ori_sitinggil_3D_2025_subset_local/segmented_pt/prompts.pt" \
    --output_dir "output/ori_sitinggil_3D_2025_subset_local/segmented_pt" \
    --output_name roofs \
    --scale 0.5 \
    --threshold 0.5 \
    --scene_iter 15000 \
    --feature_iter 10000
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scene import GaussianModel, FeatureGaussianModel

FEATURE_DIM = 32


def main():
    parser = argparse.ArgumentParser(
        description="3D segmentation from saved GUI prompt features")
    parser.add_argument("--model_path",   required=True,
                        help="SAGA output directory (contains point_cloud/)")
    parser.add_argument("--prompts",      required=True,
                        help="(N, 32) .pt tensor saved by GUI Save Prompts button")
    parser.add_argument("--output_dir",   default=None,
                        help="Where to write outputs. Default: <model_path>/segmented_pt")
    parser.add_argument("--output_name",  default="segmented",
                        help="Stem for output files (<name>.pt and <name>.ply)")
    parser.add_argument("--scale",        type=float, default=0.5,
                        help="Scale gate value (0–1), match what you used in the GUI")
    parser.add_argument("--threshold",    type=float, default=0.5,
                        help="Cosine similarity threshold (0–1)")
    parser.add_argument("--scene_iter",   type=int, default=15000,
                        help="Iteration of the scene PLY to load")
    parser.add_argument("--feature_iter", type=int, default=10000,
                        help="Iteration of the feature PLY / scale_gate.pt to load")
    parser.add_argument("--chunk",        type=int, default=500_000,
                        help="Gaussians per similarity chunk; reduce if still OOM")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.model_path, "segmented_pt")
    os.makedirs(output_dir, exist_ok=True)

    scene_ply   = os.path.join(args.model_path,
                               f"point_cloud/iteration_{args.scene_iter}/scene_point_cloud.ply")
    feature_ply = os.path.join(args.model_path,
                               f"point_cloud/iteration_{args.feature_iter}"
                               "/contrastive_feature_point_cloud.ply")
    gate_path   = os.path.join(args.model_path,
                               f"point_cloud/iteration_{args.feature_iter}/scale_gate.pt")

    for p, label in [(scene_ply, "scene PLY"), (feature_ply, "feature PLY"),
                     (gate_path, "scale gate"), (args.prompts, "prompts")]:
        if not os.path.exists(p):
            print(f"[ERROR] {label} not found:\n  {p}")
            sys.exit(1)

    print("Loading scene Gaussians…")
    scene_model = GaussianModel(3)
    scene_model.load_ply(scene_ply)
    N_pts = scene_model.get_xyz.shape[0]
    print(f"  {N_pts:,} Gaussians loaded.")

    print("Loading feature Gaussians…")
    feature_model = FeatureGaussianModel(FEATURE_DIM)
    feature_model.load_ply(feature_ply)

    print("Loading scale gate…")
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, FEATURE_DIM, bias=True),
        torch.nn.Sigmoid(),
    ).cuda()
    scale_gate.load_state_dict(torch.load(gate_path, map_location="cuda"))
    scale_gate.eval()

    print(f"Loading prompts from {args.prompts}…")
    raw_prompts = torch.load(args.prompts, map_location="cpu")  # (N, 32)
    if raw_prompts.dim() != 2 or raw_prompts.shape[1] != FEATURE_DIM:
        print(f"[ERROR] Expected (N, {FEATURE_DIM}) tensor, got {tuple(raw_prompts.shape)}")
        sys.exit(1)
    print(f"  {raw_prompts.shape[0]} prompt(s) loaded.")

    print(f"Segmenting — scale={args.scale}, threshold={args.threshold}…")
    with torch.no_grad():
        gates = scale_gate(torch.tensor([[args.scale]], device="cuda"))  # (1, 32)

        # Gate + normalise each prompt → (32, N_prompts)
        normed = F.normalize(raw_prompts.cuda(), dim=-1, p=2)
        gated  = normed * gates
        chosen = F.normalize(gated, dim=-1, p=2).T  # (32, N_prompts)

        feat_pts = feature_model.get_point_features  # (N_pts, 32)
        feat_pts = feat_pts * gates
        feat_pts = F.normalize(feat_pts, dim=-1, p=2)

        try:
            # Fast path: single (N_pts × N_prompts) matrix multiply.
            score = feat_pts @ chosen
            score = (score + 1.0) / 2.0
            mask  = (score > args.threshold).any(dim=-1)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM on full matrix — chunked fallback (chunk={args.chunk:,})…")
            mask = torch.zeros(N_pts, dtype=torch.bool, device="cuda")
            for start in range(0, N_pts, args.chunk):
                end = min(start + args.chunk, N_pts)
                s   = feat_pts[start:end] @ chosen
                s   = (s + 1.0) / 2.0
                mask[start:end] = (s > args.threshold).any(dim=-1)

    n_sel = int(mask.sum().item())
    print(f"Selected {n_sel:,} / {N_pts:,} Gaussians ({n_sel / N_pts * 100:.1f}%)")

    if n_sel == 0:
        print("[WARNING] Empty mask — try lowering --threshold or loading more prompts.")
        sys.exit(0)

    mask_path = os.path.join(output_dir, f"{args.output_name}.pt")
    torch.save(mask, mask_path)
    print(f"Mask  → {mask_path}")

    scene_model.segment(mask)
    ply_path = os.path.join(output_dir, f"{args.output_name}.ply")
    scene_model.save_ply(ply_path)
    print(f"PLY   → {ply_path}")


if __name__ == "__main__":
    main()