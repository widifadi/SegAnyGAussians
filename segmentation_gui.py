"""
SAGA Segmentation GUI
---------------------
A lightweight PyQt5 interface over the prompt_segmenting.ipynb workflow.

Usage:
    python segmentation_gui.py

Key capabilities:
  - Multi-view feature accumulation: add prompts from different camera angles.
    The feature space is 3D and view-independent, so features from different
    views combine naturally without any special handling.
  - Works with any 3DGS pipeline output: source_path (COLMAP) is optional.
    If left blank, cameras are loaded from cameras.json in the model directory
    (written by every standard 3DGS training run including lichtfeld, nerfstudio,
    the original 3DGS, etc.)
  - PLY paths are fully configurable; auto-fill from model_path + iterations,
    but any path can be overridden for non-standard directory layouts.
  - Iterations are just path-building helpers — the core segment() and render()
    functions have no concept of iteration numbers.

Workflow:
    1. Set Model Path (output dir) — required
    2. Set Source Path (COLMAP data dir) — optional, leave blank to use cameras.json
    3. Set iterations and click "Auto-fill Paths", then "Load Models"
       OR manually browse to each PLY / .pt file and load
    4. Pick a reference view from the dropdown and click "Render View"
    5. Toggle "Add Point" and left-click on the image
       — switch views and keep clicking to build multi-view prompts
    6. Adjust Scale / Threshold sliders to tune the live similarity preview
    7. Click "Run 3D Segment"
    8. Export PLY or mask, or Roll Back to try again
"""

import sys
import os
import json

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import numpy as np
from argparse import Namespace

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSlider, QFileDialog,
    QGroupBox, QMessageBox, QSpinBox, QSizePolicy, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scene import GaussianModel, FeatureGaussianModel
from scene.cameras import Camera
from gaussian_renderer import render, render_contrastive_feature
from utils.graphics_utils import focal2fov


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class Worker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.finished.emit(result, "")
        except Exception:
            import traceback
            self.finished.emit(None, traceback.format_exc())


# ---------------------------------------------------------------------------
# Main GUI window
# ---------------------------------------------------------------------------

class SegmentationGUI(QMainWindow):

    FEATURE_DIM = 32

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAGA Segmentation GUI")

        # model state
        self.scene_model: GaussianModel | None = None
        self.feature_model: FeatureGaussianModel | None = None
        self.scale_gate: torch.nn.Sequential | None = None
        self.cameras: list = []
        self.current_camera = None

        # rendering state
        self.rgb_image: np.ndarray | None = None
        self.raw_feature_map: torch.Tensor | None = None

        # segmentation state
        # click_points: pixel coords on the CURRENT view only (for display)
        # click_raw_features: accumulated across ALL views — never auto-cleared
        self.click_points: list = []
        self.click_raw_features: list = []
        self.is_segmented: bool = False
        self.last_segment_mask: torch.Tensor | None = None

        self.pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                              debug=False)
        self.bg_color: torch.Tensor | None = None
        self.bg_feature: torch.Tensor | None = None

        self._worker: Worker | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)
        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_canvas_panel(), stretch=1)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(360)
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)

        layout.addWidget(self._build_paths_group())
        layout.addWidget(self._build_view_group())
        layout.addWidget(self._build_seg_group())
        layout.addWidget(self._build_export_group())

        self.status_label = QLabel("Status: Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #555;")
        layout.addWidget(self.status_label)
        layout.addStretch()
        return panel

    # ---- paths group -------------------------------------------------

    def _build_paths_group(self) -> QGroupBox:
        g = QGroupBox("Model & File Paths")
        v = QVBoxLayout(g)

        # --- model path ---
        v.addWidget(QLabel("Model Path (output dir):"))
        row = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("e.g. ./output/my_scene")
        self.model_path_edit.textChanged.connect(self._on_model_path_changed)
        row.addWidget(self.model_path_edit)
        b = QPushButton("…"); b.setFixedWidth(28)
        b.clicked.connect(lambda: self._browse_dir(self.model_path_edit))
        row.addWidget(b)
        v.addLayout(row)

        # --- source path (optional) ---
        src_lbl = QLabel("Data/Source Path (optional — COLMAP dir):")
        src_lbl.setStyleSheet("color: #666; font-size: 10px;")
        v.addWidget(src_lbl)
        src_hint = QLabel("Leave blank to use cameras.json from model dir.")
        src_hint.setStyleSheet("color: #999; font-size: 9px;")
        v.addWidget(src_hint)
        row2 = QHBoxLayout()
        self.source_path_edit = QLineEdit()
        self.source_path_edit.setPlaceholderText("optional — see hint above")
        row2.addWidget(self.source_path_edit)
        b2 = QPushButton("…"); b2.setFixedWidth(28)
        b2.clicked.connect(lambda: self._browse_dir(self.source_path_edit))
        row2.addWidget(b2)
        v.addLayout(row2)

        # --- iterations + auto-fill ---
        iters = QHBoxLayout()
        iters.addWidget(QLabel("Scene iter:"))
        self.scene_iter_spin = QSpinBox()
        self.scene_iter_spin.setRange(0, 1_000_000)
        self.scene_iter_spin.setValue(30_000)
        self.scene_iter_spin.setSingleStep(1000)
        iters.addWidget(self.scene_iter_spin)
        iters.addWidget(QLabel("Feature iter:"))
        self.feat_iter_spin = QSpinBox()
        self.feat_iter_spin.setRange(0, 1_000_000)
        self.feat_iter_spin.setValue(10_000)
        self.feat_iter_spin.setSingleStep(1000)
        iters.addWidget(self.feat_iter_spin)
        v.addLayout(iters)

        autofill_btn = QPushButton("Auto-fill Paths from Model Path + Iterations")
        autofill_btn.clicked.connect(self._autofill_paths)
        v.addWidget(autofill_btn)

        # --- separator ---
        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken)
        v.addWidget(line)

        # --- explicit PLY / .pt paths ---
        ply_lbl = QLabel("PLY / model file paths (editable / override):")
        ply_lbl.setStyleSheet("font-size: 10px;")
        v.addWidget(ply_lbl)

        self.scene_ply_edit = self._path_row(v, "Scene PLY:", "scene_point_cloud.ply")
        self.feature_ply_edit = self._path_row(v, "Feature PLY:", "contrastive_feature_point_cloud.ply")
        self.scale_gate_edit = self._path_row(v, "Scale Gate:", "scale_gate.pt")

        self.load_btn = QPushButton("Load Models")
        self.load_btn.clicked.connect(self._load_models)
        v.addWidget(self.load_btn)
        return g

    def _path_row(self, parent_layout, label: str, placeholder: str) -> QLineEdit:
        """Helper: add a labelled path row with browse button, return the QLineEdit."""
        parent_layout.addWidget(QLabel(label))
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        row.addWidget(edit)
        b = QPushButton("…"); b.setFixedWidth(28)
        # Lambda capture needs default arg to avoid late-binding
        b.clicked.connect(lambda _=False, e=edit: self._browse_file(e))
        row.addWidget(b)
        parent_layout.addLayout(row)
        return edit

    # ---- view group --------------------------------------------------

    def _build_view_group(self) -> QGroupBox:
        g = QGroupBox("Reference View")
        v = QVBoxLayout(g)
        self.view_combo = QComboBox()
        self.view_combo.setEnabled(False)
        v.addWidget(self.view_combo)
        self.render_view_btn = QPushButton("Render View")
        self.render_view_btn.setEnabled(False)
        self.render_view_btn.clicked.connect(self._render_view)
        v.addWidget(self.render_view_btn)
        hint = QLabel("Tip: render multiple views to add prompts from different angles.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 9px;")
        v.addWidget(hint)
        return g

    # ---- seg group ---------------------------------------------------

    def _build_seg_group(self) -> QGroupBox:
        g = QGroupBox("Segmentation")
        v = QVBoxLayout(g)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scale:"))
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(0, 100); self.scale_slider.setValue(50)
        self.scale_slider.valueChanged.connect(self._on_slider_changed)
        scale_row.addWidget(self.scale_slider)
        self.scale_val_lbl = QLabel("0.50"); self.scale_val_lbl.setFixedWidth(36)
        scale_row.addWidget(self.scale_val_lbl)
        v.addLayout(scale_row)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Threshold:"))
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 100); self.thresh_slider.setValue(50)
        self.thresh_slider.valueChanged.connect(self._on_slider_changed)
        thresh_row.addWidget(self.thresh_slider)
        self.thresh_val_lbl = QLabel("0.50"); self.thresh_val_lbl.setFixedWidth(36)
        thresh_row.addWidget(self.thresh_val_lbl)
        v.addLayout(thresh_row)

        self.click_mode_btn = QPushButton("Add Point  [off]")
        self.click_mode_btn.setCheckable(True)
        self.click_mode_btn.setEnabled(False)
        self.click_mode_btn.toggled.connect(self._on_click_mode_toggled)
        v.addWidget(self.click_mode_btn)

        self.points_lbl = QLabel("Points: 0 total (0 on this view)")
        v.addWidget(self.points_lbl)

        btn_row = QHBoxLayout()
        clear_view_btn = QPushButton("Clear This View")
        clear_view_btn.setToolTip("Remove click markers for current view only")
        clear_view_btn.clicked.connect(self._clear_view_points)
        btn_row.addWidget(clear_view_btn)
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.setToolTip("Remove ALL accumulated prompts from all views")
        clear_all_btn.clicked.connect(self._clear_all_points)
        btn_row.addWidget(clear_all_btn)
        v.addLayout(btn_row)

        self.segment_3d_btn = QPushButton("Run 3D Segment")
        self.segment_3d_btn.setEnabled(False)
        self.segment_3d_btn.clicked.connect(self._run_3d_segment)
        v.addWidget(self.segment_3d_btn)

        self.rollback_btn = QPushButton("Roll Back")
        self.rollback_btn.setEnabled(False)
        self.rollback_btn.clicked.connect(self._roll_back)
        v.addWidget(self.rollback_btn)
        return g

    # ---- export group ------------------------------------------------

    def _build_export_group(self) -> QGroupBox:
        g = QGroupBox("Export")
        v = QVBoxLayout(g)
        self.export_ply_btn = QPushButton("Export Segmented PLY…")
        self.export_ply_btn.setEnabled(False)
        self.export_ply_btn.clicked.connect(self._export_ply)
        v.addWidget(self.export_ply_btn)
        self.export_mask_btn = QPushButton("Export Mask (.pt)…")
        self.export_mask_btn.setEnabled(False)
        self.export_mask_btn.clicked.connect(self._export_mask)
        v.addWidget(self.export_mask_btn)
        return g

    # ---- canvas panel ------------------------------------------------

    def _build_canvas_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Load a model and render a view to begin")
        self.ax.axis("off")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        layout.addWidget(self.canvas)
        return widget

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _browse_dir(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            edit.setText(path)

    def _browse_file(self, edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "All files (*.*)")
        if path:
            edit.setText(path)

    def _set_status(self, msg: str):
        self.status_label.setText(f"Status: {msg}")
        QApplication.processEvents()

    def _on_click_mode_toggled(self, checked: bool):
        self.click_mode_btn.setText(
            "Add Point  [ON]" if checked else "Add Point  [off]")

    def _on_slider_changed(self, _):
        self.scale_val_lbl.setText(f"{self.scale_slider.value()/100:.2f}")
        self.thresh_val_lbl.setText(f"{self.thresh_slider.value()/100:.2f}")
        if self.click_raw_features:
            self._update_similarity_preview()

    def _on_model_path_changed(self, text: str):
        """Auto-fill paths whenever model path field is edited."""
        if text.strip():
            self._autofill_paths()

    def _autofill_paths(self):
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            return
        s_iter = self.scene_iter_spin.value()
        f_iter = self.feat_iter_spin.value()
        self.scene_ply_edit.setText(
            os.path.join(model_path, f"point_cloud/iteration_{s_iter}/scene_point_cloud.ply"))
        self.feature_ply_edit.setText(
            os.path.join(model_path, f"point_cloud/iteration_{f_iter}/contrastive_feature_point_cloud.ply"))
        self.scale_gate_edit.setText(
            os.path.join(model_path, f"point_cloud/iteration_{f_iter}/scale_gate.pt"))

    def _update_points_label(self):
        total = len(self.click_raw_features)
        current = len(self.click_points)
        self.points_lbl.setText(
            f"Points: {total} total ({current} on this view)")

    def _set_controls_enabled(self, has_model: bool, has_render: bool, has_segment: bool):
        self.view_combo.setEnabled(has_model)
        self.render_view_btn.setEnabled(has_model)
        self.click_mode_btn.setEnabled(has_render)
        self.segment_3d_btn.setEnabled(has_render and len(self.click_raw_features) > 0)
        self.rollback_btn.setEnabled(has_segment)
        self.export_ply_btn.setEnabled(has_segment)
        self.export_mask_btn.setEnabled(has_segment)

    # ------------------------------------------------------------------
    # Camera loading helpers
    # ------------------------------------------------------------------

    def _load_cameras_from_json(self, json_path: str) -> list:
        """
        Parse cameras.json written by any standard 3DGS training run.

        cameras.json format (written by Scene via camera_to_JSON):
          [ { "id": int, "img_name": str, "width": int, "height": int,
              "position": [x,y,z],        <- camera world position (C2W)
              "rotation": [[...],[...],[...]],  <- C2W rotation matrix
              "fy": float, "fx": float } ... ]

        Reconstruction:
          - rotation (3×3) = C2W rotation = R_cw^T  where R_cw is W2C rotation
          - position = camera centre in world = -R_cw^T @ t_cw
          - Camera() expects R = R_cw^T = rotation, T = t_cw = -rotation^T @ position
        """
        with open(json_path) as f:
            data = json.load(f)

        cameras = []
        for entry in data:
            R = np.array(entry["rotation"], dtype=np.float64)    # C2W rot = R_cw^T
            pos = np.array(entry["position"], dtype=np.float64)  # world position
            T = -(R.T @ pos)                                      # t_cw (W2C translation)

            w, h = int(entry["width"]), int(entry["height"])
            FoVx = focal2fov(entry["fx"], w)
            FoVy = focal2fov(entry["fy"], h)

            # Dummy black image — only dimensions matter for rendering
            dummy = torch.zeros(3, h, w)

            cam = Camera(
                colmap_id=int(entry["id"]),
                R=R, T=T,
                FoVx=FoVx, FoVy=FoVy,
                image=dummy,
                gt_alpha_mask=None,
                image_name=entry["img_name"],
                uid=int(entry["id"]),
            )
            # Use full image resolution for feature map
            cam.feature_height = h
            cam.feature_width = w
            cameras.append(cam)

        return cameras

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        model_path = self.model_path_edit.text().strip()
        source_path = self.source_path_edit.text().strip()
        scene_ply = self.scene_ply_edit.text().strip()
        feature_ply = self.feature_ply_edit.text().strip()
        scale_gate_path = self.scale_gate_edit.text().strip()

        if not model_path:
            QMessageBox.warning(self, "Missing Path", "Please provide a Model Path.")
            return

        # Auto-fill if fields are blank
        if not scene_ply or not feature_ply or not scale_gate_path:
            self._autofill_paths()
            scene_ply = self.scene_ply_edit.text().strip()
            feature_ply = self.feature_ply_edit.text().strip()
            scale_gate_path = self.scale_gate_edit.text().strip()

        missing = [(p, n) for p, n in [
            (scene_ply, "Scene PLY"),
            (feature_ply, "Feature PLY"),
            (scale_gate_path, "Scale Gate .pt"),
        ] if not os.path.exists(p)]
        if missing:
            QMessageBox.critical(self, "Files Not Found",
                                 "\n".join(f"{n}:\n  {p}" for p, n in missing))
            return

        # Determine camera source
        cameras_json = os.path.join(os.path.abspath(model_path), "cameras.json")
        if not source_path:
            if os.path.exists(cameras_json):
                cam_source = ("json", cameras_json)
            else:
                QMessageBox.warning(
                    self, "No Camera Source",
                    "Source Path is blank and cameras.json was not found in the model "
                    "directory.\n\nEither:\n"
                    "  • Fill in the Data/Source Path (COLMAP directory), or\n"
                    f"  • Ensure cameras.json exists at:\n    {cameras_json}\n\n"
                    "cameras.json is generated automatically during training.")
                return
        else:
            cam_source = ("colmap", source_path)

        self.load_btn.setEnabled(False)
        self._set_status("Loading models…")

        scene_ply_ = scene_ply
        feature_ply_ = feature_ply
        scale_gate_path_ = scale_gate_path
        model_path_ = model_path

        def _do_load():
            # --- load cameras ---
            if cam_source[0] == "json":
                cameras = self.loaded_cameras_from_json = self._load_cameras_from_json(cam_source[1])
            else:
                # COLMAP / Scene path
                from scene import Scene
                args = Namespace(
                    model_path=os.path.abspath(model_path_),
                    source_path=os.path.abspath(cam_source[1]),
                    sh_degree=3, feature_dim=self.FEATURE_DIM,
                    init_from_3dgs_pcd=False, images="images",
                    resolution=-1, white_background=False, data_device="cuda",
                    eval=False, need_features=False, need_masks=False,
                    allow_principle_point_shift=False, feature_model_path="",
                )
                scene_iter = self.scene_iter_spin.value()
                scene_obj = Scene(args, gaussians=None, feature_gaussians=None,
                                  load_iteration=scene_iter, shuffle=False)
                cameras = scene_obj.getTrainCameras()
                # Set full-res feature dimensions
                for cam in cameras:
                    cam.feature_height = cam.image_height
                    cam.feature_width = cam.image_width

            # --- load Gaussians ---
            s_model = GaussianModel(3)
            s_model.load_ply(scene_ply_)

            f_model = FeatureGaussianModel(self.FEATURE_DIM)
            f_model.load_ply(feature_ply_)

            # --- load scale gate ---
            gate = torch.nn.Sequential(
                torch.nn.Linear(1, self.FEATURE_DIM, bias=True),
                torch.nn.Sigmoid(),
            ).cuda()
            gate.load_state_dict(torch.load(scale_gate_path_, map_location="cuda"))
            gate.eval()

            return cameras, s_model, f_model, gate

        self._worker = Worker(_do_load)
        self._worker.finished.connect(self._on_load_finished)
        self._worker.start()

    def _on_load_finished(self, result, error: str):
        self.load_btn.setEnabled(True)
        if error:
            QMessageBox.critical(self, "Load Error", error)
            self._set_status("Error loading models.")
            return

        cameras, s_model, f_model, gate = result
        self.cameras = cameras
        self.scene_model = s_model
        self.feature_model = f_model
        self.scale_gate = gate
        self.bg_color = torch.zeros(3, device="cuda")
        self.bg_feature = torch.zeros(self.FEATURE_DIM, device="cuda")

        self.view_combo.clear()
        for cam in self.cameras:
            self.view_combo.addItem(cam.image_name)

        self._set_controls_enabled(has_model=True, has_render=False,
                                   has_segment=False)
        self._set_status(
            f"Loaded. {len(cameras)} cameras "
            f"({'cameras.json' if not self.source_path_edit.text().strip() else 'COLMAP'}).")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_view(self):
        if self.scene_model is None:
            return
        idx = self.view_combo.currentIndex()
        if idx < 0 or idx >= len(self.cameras):
            return

        cam = self.cameras[idx]
        cam.feature_height = cam.image_height
        cam.feature_width = cam.image_width
        self.current_camera = cam

        self._set_status(f"Rendering {cam.image_name}…")
        self.render_view_btn.setEnabled(False)

        def _do_render():
            with torch.no_grad():
                scene_out = render(cam, self.scene_model, self.pipe, self.bg_color)
                rgb = scene_out["render"].permute(1, 2, 0).cpu().numpy()
                feat_out = render_contrastive_feature(
                    cam, self.feature_model, self.pipe, self.bg_feature)
                feat = feat_out["render"].permute(1, 2, 0)  # (H, W, 32)
            return rgb, feat

        self._worker = Worker(_do_render)
        self._worker.finished.connect(self._on_render_finished)
        self._worker.start()

    def _on_render_finished(self, result, error: str):
        self.render_view_btn.setEnabled(True)
        if error:
            QMessageBox.critical(self, "Render Error", error)
            self._set_status(f"Render error: {error[:120]}")
            return

        rgb, feat = result
        self.rgb_image = np.clip(rgb, 0.0, 1.0)
        self.raw_feature_map = feat  # (H, W, 32) GPU, raw

        # Clear per-view click markers ONLY — keep cross-view feature accumulation
        self.click_points = []
        self._update_points_label()

        # Show similarity preview immediately if we have features from other views
        if self.click_raw_features:
            self._update_similarity_preview()
        else:
            self._display_image()

        self._set_controls_enabled(has_model=True, has_render=True,
                                   has_segment=self.is_segmented)
        cam = self.current_camera
        total = len(self.click_raw_features)
        self._set_status(
            f"Rendered: {cam.image_name}  ({cam.image_width}×{cam.image_height})"
            + (f"  [{total} prompt(s) carried over]" if total else "")
            + ("  [segmented]" if self.is_segmented else "")
        )

    # ------------------------------------------------------------------
    # Canvas display
    # ------------------------------------------------------------------

    def _display_image(self, overlay: np.ndarray | None = None):
        self.ax.clear()
        self.ax.axis("off")
        if self.rgb_image is not None:
            self.ax.imshow(self.rgb_image, origin="upper")
            if overlay is not None:
                rgba = np.zeros((*overlay.shape, 4), dtype=np.float32)
                rgba[overlay > 0.5] = [1.0, 0.2, 0.1, 0.45]
                self.ax.imshow(rgba, origin="upper")
            if self.click_points:
                xs = [p[0] for p in self.click_points]
                ys = [p[1] for p in self.click_points]
                self.ax.scatter(xs, ys, c="lime", s=80, marker="+",
                                linewidths=2, zorder=5)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Click handling → feature sampling
    # ------------------------------------------------------------------

    def _on_canvas_click(self, event):
        if not self.click_mode_btn.isChecked():
            return
        if event.inaxes is not self.ax:
            return
        if event.button != 1:
            return
        if self.raw_feature_map is None or event.xdata is None:
            return

        H, W = self.raw_feature_map.shape[:2]
        px = max(0, min(int(round(event.xdata)), W - 1))
        py = max(0, min(int(round(event.ydata)), H - 1))

        self.click_points.append((px, py))

        with torch.no_grad():
            raw_f = self.raw_feature_map[py, px, :].clone()  # (32,)
        self.click_raw_features.append(raw_f)

        self._update_points_label()
        self.segment_3d_btn.setEnabled(True)
        self._update_similarity_preview()

    # ------------------------------------------------------------------
    # Similarity preview
    # ------------------------------------------------------------------

    def _compute_scale_gated(self, raw_feat: torch.Tensor) -> torch.Tensor:
        """Apply scale gate + L2-normalise. Works for (H,W,32) or (32,) tensors."""
        scale_val = self.scale_slider.value() / 100.0
        with torch.no_grad():
            gates = self.scale_gate(
                torch.tensor([[scale_val]], device="cuda"))  # (1, 32)
            if raw_feat.dim() == 3:
                feat = raw_feat / (raw_feat.norm(dim=-1, keepdim=True) + 1e-6)
                feat = feat * gates.unsqueeze(0)
                feat = torch.nn.functional.normalize(feat, dim=-1, p=2)
            else:
                feat = raw_feat / (raw_feat.norm() + 1e-6)
                feat = feat * gates.squeeze()
                feat = torch.nn.functional.normalize(
                    feat.unsqueeze(0), dim=-1, p=2).squeeze()
        return feat

    def _update_similarity_preview(self):
        if not self.click_raw_features or self.raw_feature_map is None:
            self._display_image()
            return

        thresh = self.thresh_slider.value() / 100.0
        with torch.no_grad():
            chosen = torch.cat(
                [self._compute_scale_gated(f).reshape(-1, 1)
                 for f in self.click_raw_features],
                dim=-1)  # (32, N_total)

            feat_map = self._compute_scale_gated(self.raw_feature_map)
            H, W = feat_map.shape[:2]

            score = feat_map.reshape(-1, self.FEATURE_DIM) @ chosen  # (H*W, N)
            score = (score + 1.0) / 2.0
            mask = (score.max(dim=-1).values > thresh).reshape(H, W).cpu().numpy()

        self._display_image(overlay=mask)

    # ------------------------------------------------------------------
    # Clear points
    # ------------------------------------------------------------------

    def _clear_view_points(self):
        """Remove click markers for the current view only; keep other-view features."""
        n_this_view = len(self.click_points)
        # Remove only the most-recently-added N features (those from this view)
        if n_this_view > 0:
            self.click_raw_features = self.click_raw_features[:-n_this_view]
        self.click_points = []
        self._update_points_label()
        if not self.click_raw_features:
            self.segment_3d_btn.setEnabled(False)
            self._display_image()
        else:
            self._update_similarity_preview()

    def _clear_all_points(self):
        """Remove ALL accumulated prompts from all views."""
        self.click_points = []
        self.click_raw_features = []
        self._update_points_label()
        self.segment_3d_btn.setEnabled(False)
        self._display_image()

    # ------------------------------------------------------------------
    # 3-D segmentation
    # ------------------------------------------------------------------

    def _run_3d_segment(self):
        if not self.click_raw_features or self.scene_model is None:
            return

        thresh = self.thresh_slider.value() / 100.0
        self._set_status(
            f"Running 3D segmentation with {len(self.click_raw_features)} prompt(s)…")
        self.segment_3d_btn.setEnabled(False)

        def _do_segment():
            with torch.no_grad():
                gates = self.scale_gate(
                    torch.tensor([[self.scale_slider.value() / 100.0]],
                                 device="cuda"))
                chosen = torch.cat(
                    [self._compute_scale_gated(f).reshape(-1, 1)
                     for f in self.click_raw_features],
                    dim=-1)  # (32, N)

                feat_pts = self.feature_model.get_point_features  # (N_pts, 32)
                feat_pts = feat_pts * gates
                feat_pts = torch.nn.functional.normalize(feat_pts, dim=-1, p=2)

                score_pts = feat_pts @ chosen   # (N_pts, N_clicks)
                score_pts = (score_pts + 1.0) / 2.0
                mask_3d = (score_pts > thresh).any(dim=-1)  # (N_pts,) bool
            return mask_3d

        self._worker = Worker(_do_segment)
        self._worker.finished.connect(self._on_segment_finished)
        self._worker.start()

    def _on_segment_finished(self, result, error: str):
        self.segment_3d_btn.setEnabled(bool(self.click_raw_features))
        if error:
            QMessageBox.critical(self, "Segment Error", error)
            self._set_status("Segment error.")
            return

        mask_3d: torch.Tensor = result
        n_sel = int(mask_3d.sum().item())
        if n_sel == 0:
            QMessageBox.warning(self, "Empty Mask",
                                "No Gaussians matched.  Lower the Threshold or add more points.")
            self._set_status("3D segment: empty result.")
            return

        self.last_segment_mask = mask_3d.clone()
        self.scene_model.segment(mask_3d)
        self.feature_model.segment(mask_3d)
        self.is_segmented = True

        self._set_controls_enabled(has_model=True, has_render=True, has_segment=True)
        self._set_status(
            f"Segmented — {n_sel:,} Gaussians selected from "
            f"{len(self.click_raw_features)} prompt(s).  Re-rendering…")
        self._render_view()

    # ------------------------------------------------------------------
    # Roll back
    # ------------------------------------------------------------------

    def _roll_back(self):
        if self.scene_model is None:
            return
        self.scene_model.roll_back()
        self.feature_model.roll_back()
        # Clear everything including cross-view features after a rollback
        self.click_points = []
        self.click_raw_features = []
        self._update_points_label()
        self.is_segmented = (self.scene_model.segment_times > 0)
        self.last_segment_mask = None
        self.rgb_image = None
        self.raw_feature_map = None
        self._display_image()
        self._set_controls_enabled(has_model=True, has_render=False,
                                   has_segment=self.is_segmented)
        self._set_status("Rolled back. Press Render View to continue.")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_ply(self):
        if not self.is_segmented or self.scene_model is None:
            QMessageBox.warning(self, "Not Segmented", "Run 3D segmentation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Segmented Gaussians", "segmented_object.ply",
            "PLY files (*.ply)")
        if not path:
            return
        try:
            self.scene_model.save_ply(path)
            self._set_status(f"PLY saved → {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_mask(self):
        if self.last_segment_mask is None:
            QMessageBox.warning(self, "Not Segmented", "Run 3D segmentation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Segmentation Mask", "segmented_mask.pt",
            "PyTorch files (*.pt)")
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            torch.save(self.last_segment_mask, path)
            self._set_status(f"Mask saved → {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    win = SegmentationGUI()
    win.resize(1450, 850)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()