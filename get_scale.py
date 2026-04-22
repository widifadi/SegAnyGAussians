import torch


import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from argparse import ArgumentParser, Namespace
import cv2

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel, FeatureGaussianModel

import gaussian_renderer
import importlib

importlib.reload(gaussian_renderer)

import os

FEATURE_DIM = 32

DATA_ROOT = "./data/nerf_llff_data_for_3dgs/"
# MODEL_PATH = './output/figurines_lerf_poses/'
# MODEL_PATH = './output/figurines/'

ALLOW_PRINCIPLE_POINT_SHIFT = False


def get_combined_args(parser: ArgumentParser):
    # cmdlne_string = ['--model_path', model_path]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args()

    target_cfg_file = "cfg_args"

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file found: {}".format(cfgfilepath))
        pass
    args_cfgfile = eval(cfgfile_string)

    # for k in args_cfgfile.__dict__.keys():
    # print(k, args_cfgfile.__dict__[k], "?")

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v

    # for k in merged_dict.keys():
    # print(k, merged_dict[k])
    return Namespace(**merged_dict)


def generate_grid_index(depth):
    h, w = depth.shape
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid = torch.stack(grid, dim=-1)
    return grid


if __name__ == "__main__":

    parser = ArgumentParser(description="Get scales for SAM masks")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument("--idx", default=0, type=int)
    parser.add_argument("--precomputed_mask", default=None, type=str)

    parser.add_argument(
        "--image_root", default="/datasets/nerf_data/360_v2/garden/", type=str
    )

    args = get_combined_args(parser)

    dataset = model.extract(args)
    dataset.need_features = False
    dataset.need_masks = False

    # ALLOW_PRINCIPLE_POINT_SHIFT = 'lerf' in args.model_path
    dataset.allow_principle_point_shift = ALLOW_PRINCIPLE_POINT_SHIFT

    feature_gaussians = None
    scene_gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(
        dataset,
        scene_gaussians,
        feature_gaussians,
        load_iteration=-1,
        feature_load_iteration=-1,
        shuffle=False,
        mode="eval",
        target="scene",
    )

    assert (
        os.path.exists(os.path.join(dataset.source_path, "images"))
        and "Please specify a valid image root."
    )
    assert (
        os.path.join(dataset.source_path, "sam_masks")
        and "Please run extract_segment_everything_masks first."
    )

    from tqdm import tqdm

    IMAGE_DIR = os.path.join(dataset.source_path, "images")
    SAM_MASK_DIR = os.path.join(dataset.source_path, "sam_masks")

    # ------------------------------------------------------------------
    # Memory estimation: predict RAM for pre-loading all masks at once,
    # and for the batched upsample+erosion step inside the render loop.
    # Sample one .pt file to get (N_masks, H, W) and one image for
    # render resolution, then extrapolate.
    # ------------------------------------------------------------------
    use_lazy = False
    use_per_mask = False
    sample_pts = sorted(f for f in os.listdir(SAM_MASK_DIR) if f.endswith(".pt"))
    if sample_pts:
        sample_mask = torch.load(os.path.join(SAM_MASK_DIR, sample_pts[0]))
        n_masks_sample, h_mask, w_mask = sample_mask.shape
        del sample_mask

        # Estimate render resolution from the first image in IMAGE_DIR.
        image_files = sorted(f for f in os.listdir(IMAGE_DIR) if not f.startswith("."))
        n_images = len(image_files)
        h_render, w_render = h_mask, w_mask  # fallback: assume same as mask
        if image_files:
            try:
                _img = cv2.imread(os.path.join(IMAGE_DIR, image_files[0]))
                if _img is not None:
                    h_render, w_render = _img.shape[:2]
                del _img
            except Exception:
                pass

        bytes_est = n_images * n_masks_sample * h_mask * w_mask * 4  # float32
        gb_est = bytes_est / (1024**3)

        # Batched upsample + conv2d: 2 tensors of (N_masks, 1, H_render, W_render) float32.
        # 1.5× safety factor accounts for torch.conv2d workspace buffers (observed ~1.3× overhead).
        gb_batch_mask = n_masks_sample * h_render * w_render * 4 * 2 * 1.5 / (1024**3)

        print(f"\n{'='*60}")
        print(f"  get_scale memory estimate (pre-load mode)")
        print(f"    Images        : {n_images}")
        print(f"    Masks / image : ~{n_masks_sample}  (from '{sample_pts[0]}')")
        print(f"    Mask size     : {h_mask} x {w_mask}")
        print(f"    Render size   : {h_render} x {w_render}")
        print(f"    Estimated RAM : {gb_est:.1f} GB  (pre-load all masks)")
        print(
            f"    Batch mask RAM: {gb_batch_mask:.1f} GB  (per-view upsample+erosion, incl. overhead)"
        )

        try:
            import psutil

            avail_gb = psutil.virtual_memory().available / (1024**3)
            total_gb = psutil.virtual_memory().total / (1024**3)
            print(f"    Available RAM : {avail_gb:.1f} GB  /  {total_gb:.1f} GB total")
            use_lazy = gb_est > avail_gb * 0.7
            use_per_mask = gb_batch_mask > avail_gb * 0.15
        except ImportError:
            # psutil not available — fall back to conservative hard thresholds
            use_lazy = gb_est > 8.0
            use_per_mask = gb_batch_mask > 4.0
            print(f"    (psutil not found; using 8 GB / 4 GB thresholds)")

        print(f"    Mask load     : {'LAZY'     if use_lazy     else 'PRE-LOAD'}")
        print(f"    Mask ops      : {'PER-MASK' if use_per_mask else 'BATCHED'}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Pre-load path (original behaviour): load every image's masks into
    # a dict before the render loop.  Fast but memory-intensive.
    # ------------------------------------------------------------------
    images_masks = {}
    if not use_lazy:
        for i, image_path in tqdm(
            enumerate(sorted(os.listdir(IMAGE_DIR))), desc="Pre-loading masks"
        ):
            image = cv2.imread(os.path.join(IMAGE_DIR, image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = torch.load(
                os.path.join(
                    SAM_MASK_DIR,
                    image_path.replace("jpg", "pt")
                    .replace("JPG", "pt")
                    .replace("png", "pt"),
                )
            )
            # N_mask, C
            images_masks[image_path.split(".")[0]] = masks.cpu().float()

    OUTPUT_DIR = os.path.join(args.image_root, "mask_scales")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cameras = scene.getTrainCameras()

    background = torch.zeros(scene_gaussians.get_mask.shape[0], 3, device="cuda")

    for it, view in tqdm(enumerate(cameras)):

        with torch.no_grad():
            rendered_pkg = gaussian_renderer.render_with_depth(
                view, scene_gaussians, pipeline.extract(args), background
            )
        torch.cuda.synchronize()  # surface async CUDA errors as Python exceptions

        depth = rendered_pkg["depth"]

        # Lazy-load path: load this view's masks on demand, then discard.
        if use_lazy:
            corresponding_masks = (
                torch.load(os.path.join(SAM_MASK_DIR, view.image_name + ".pt"))
                .cpu()
                .float()
            )
        else:
            corresponding_masks = images_masks[view.image_name]

        # generate_grid_index(depth.squeeze())[50, 1]

        depth = depth.cpu().squeeze()
        del rendered_pkg  # free GPU tensor immediately
        torch.cuda.empty_cache()

        grid_index = generate_grid_index(depth)

        points_in_3D = torch.zeros(depth.shape[0], depth.shape[1], 3).cpu()
        points_in_3D[:, :, -1] = depth

        # caluculate cx cy fx fy with FoVx FoVy
        cx = depth.shape[1] / 2
        cy = depth.shape[0] / 2
        fx = cx / np.tan(cameras[0].FoVx / 2)
        fy = cy / np.tan(cameras[0].FoVy / 2)

        points_in_3D[:, :, 0] = (grid_index[:, :, 0] - cx) * depth / fx
        points_in_3D[:, :, 1] = (grid_index[:, :, 1] - cy) * depth / fy

        if not use_per_mask:
            # ----------------------------------------------------------
            # Original batched path: upsample + erode all masks at once.
            # Fast but requires (N_masks × H_render × W_render × 4 × 2) RAM.
            # ----------------------------------------------------------
            upsampled_mask = torch.nn.functional.interpolate(
                corresponding_masks.unsqueeze(1),
                mode="bilinear",
                size=(depth.shape[0], depth.shape[1]),
                align_corners=False,
            )
            del corresponding_masks

            eroded_masks = torch.conv2d(
                upsampled_mask.float(),
                torch.full((3, 3), 1.0).view(1, 1, 3, 3),
                padding=1,
            )
            del upsampled_mask
            # Threshold of 5 means at least 5 pixels in the 3x3 neighborhood belong to the mask.
            # This is a more aggressive erosion than the previous threshold of 3,
            # which required at least 3 pixels in the neighborhood.
            # Change to 3 if train_contrastive_feature.py is failing with dim errors due to too
            # few points in the eroded masks.

            # eroded_masks = (eroded_masks >= 5).squeeze()  # (num_masks, H, W)
            eroded_masks = (eroded_masks >= 3).squeeze()  # (num_masks, H, W)

            scale = torch.zeros(len(eroded_masks) if eroded_masks.dim() == 3 else 1)
            for mask_id in range(scale.shape[0]):
                mask_slice = (
                    eroded_masks[mask_id] if eroded_masks.dim() == 3 else eroded_masks
                )
                point_in_3D_in_mask = points_in_3D[mask_slice == 1]

                scale[mask_id] = (point_in_3D_in_mask.std(dim=0) * 2).norm()

            del eroded_masks, points_in_3D

        else:
            # ----------------------------------------------------------
            # Per-mask path: upsample + erode one mask at a time.
            # Uses only ~(H_render × W_render × 4 × 2) RAM per step.
            # ----------------------------------------------------------
            n_masks = corresponding_masks.shape[0]
            erode_kernel = torch.full((3, 3), 1.0).view(1, 1, 3, 3)
            scale = torch.zeros(n_masks)

            for mask_id in range(n_masks):
                m = corresponding_masks[mask_id : mask_id + 1].unsqueeze(
                    1
                )  # (1, 1, h, w)
                m = torch.nn.functional.interpolate(
                    m,
                    mode="bilinear",
                    size=(depth.shape[0], depth.shape[1]),
                    align_corners=False,
                )
                m = torch.conv2d(m.float(), erode_kernel, padding=1)
                # Similar reasoning for the threshold as in the batched path, 
                # but applied to each mask individually.
                # m = (m >= 5).squeeze()  # (H, W)
                m = (m >= 3).squeeze()  # (H, W)
                point_in_3D_in_mask = points_in_3D[m == 1]
                scale[mask_id] = (point_in_3D_in_mask.std(dim=0) * 2).norm()

            del corresponding_masks, points_in_3D

        torch.save(scale, os.path.join(OUTPUT_DIR, view.image_name + ".pt"))
