"""
SAGA Segmentation GUI — launcher
---------------------------------
A lightweight PyQt5 interface over the prompt_segmenting.ipynb workflow.

Usage:
    python segmentation_gui.py

Key capabilities:
  - Multi-view feature accumulation: add prompts from different camera angles.
  - Works with any 3DGS pipeline output (LichtFeld, nerfstudio, original 3DGS).
    Source path is optional — if blank, cameras.json from the model dir is used.
  - Prepare menu: extract SAM masks, compute scale, train contrastive features
    directly from the GUI without leaving the application.

Workflow:
    1. Set Model Path and (optionally) Data/Source Path
    2. Use Prepare menu to run pipeline steps if not yet done
    3. Set iterations and click Auto-fill Paths, then Load Models
    4. Pick a reference view and click Render View
    5. Toggle Add Point and click on the image (switch views for multi-view prompts)
    6. Adjust Scale / Threshold sliders
    7. Click Run 3D Segment
    8. Export PLY / mask, or Roll Back to retry
"""

from segmentation_gui import main

if __name__ == "__main__":
    main()
