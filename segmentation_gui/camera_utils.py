"""Camera loading utilities for non-COLMAP 3DGS outputs."""

import json
import numpy as np
import torch

from scene.cameras import Camera
from utils.graphics_utils import focal2fov


def load_cameras_from_json(json_path: str) -> list:
    """
    Parse cameras.json written by any standard 3DGS training run.

    Format (written by Scene.save() via camera_to_JSON):
      [ { "id": int, "img_name": str, "width": int, "height": int,
          "position": [x, y, z],            <- camera world position (C2W)
          "rotation": [[...], [...], [...]],  <- C2W rotation matrix (R_cw^T)
          "fx": float, "fy": float } ... ]

    Math:
      - R (3×3) = C2W rotation = R_cw^T where R_cw is W2C rotation
      - position = camera centre in world coords
      - Camera() expects R = R_cw^T, T = t_cw = -(R^T @ position)
    """
    with open(json_path) as f:
        data = json.load(f)

    cameras = []
    for entry in data:
        R = np.array(entry["rotation"], dtype=np.float64)    # C2W = R_cw^T
        pos = np.array(entry["position"], dtype=np.float64)  # world position
        T = -(R.T @ pos)                                       # W2C translation

        w, h = int(entry["width"]), int(entry["height"])
        FoVx = focal2fov(entry["fx"], w)
        FoVy = focal2fov(entry["fy"], h)

        dummy = torch.zeros(3, h, w)  # placeholder image

        cam = Camera(
            colmap_id=int(entry["id"]),
            R=R, T=T,
            FoVx=FoVx, FoVy=FoVy,
            image=dummy,
            gt_alpha_mask=None,
            image_name=entry["img_name"],
            uid=int(entry["id"]),
        )
        cam.feature_height = h
        cam.feature_width = w
        cameras.append(cam)

    return cameras