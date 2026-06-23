"""
SAGA Segmentation GUI — main window.

See the top-level segmentation_gui.py docstring for full usage.
"""

import sys
import os

# Ensure repo root is on sys.path so scene/, gaussian_renderer/, etc. are importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import numpy as np
from argparse import Namespace

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSlider, QFileDialog,
    QGroupBox, QMessageBox, QSpinBox, QSizePolicy, QFrame,
    QFormLayout,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scene import GaussianModel, FeatureGaussianModel
from gaussian_renderer import render, render_contrastive_feature

from .workers import Worker
from .prepare import PrepareDialog
from .camera_utils import load_cameras_from_json


# ---------------------------------------------------------------------------
# Main window
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
        # click_points   — pixel coords on the CURRENT view only (display only)
        # click_raw_features — accumulated across ALL views, never auto-cleared
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
        self._build_prepare_menu()

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

        # Model path
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

        # Source / data path (optional)
        src_lbl = QLabel("Data/Source Path (optional — COLMAP dir):")
        src_lbl.setStyleSheet("color: #666; font-size: 10px;")
        v.addWidget(src_lbl)
        src_hint = QLabel("Leave blank to use cameras.json from model dir.")
        src_hint.setStyleSheet("color: #999; font-size: 9px;")
        v.addWidget(src_hint)
        row2 = QHBoxLayout()
        self.source_path_edit = QLineEdit()
        self.source_path_edit.setPlaceholderText("optional — see hint above")
        self.source_path_edit.textChanged.connect(self._on_source_path_changed)
        row2.addWidget(self.source_path_edit)
        b2 = QPushButton("…"); b2.setFixedWidth(28)
        b2.clicked.connect(lambda: self._browse_dir(self.source_path_edit))
        row2.addWidget(b2)
        v.addLayout(row2)

        # Iterations + auto-fill
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

        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken)
        v.addWidget(line)

        ply_lbl = QLabel("PLY / model file paths (editable / override):")
        ply_lbl.setStyleSheet("font-size: 10px;")
        v.addWidget(ply_lbl)

        self.scene_ply_edit    = self._path_row(v, "Scene PLY:",   "scene_point_cloud.ply")
        self.feature_ply_edit  = self._path_row(
            v, "Feature PLY:", "contrastive_feature_point_cloud.ply")
        self.scale_gate_edit   = self._path_row(v, "Scale Gate:",  "scale_gate.pt")

        self.load_btn = QPushButton("Load Models")
        self.load_btn.clicked.connect(self._load_models)
        v.addWidget(self.load_btn)
        return g

    def _path_row(self, parent_layout, label: str, placeholder: str) -> QLineEdit:
        parent_layout.addWidget(QLabel(label))
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        row.addWidget(edit)
        b = QPushButton("…"); b.setFixedWidth(28)
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

        prompt_io_row = QHBoxLayout()
        self.save_prompts_btn = QPushButton("Save Prompts…")
        self.save_prompts_btn.setEnabled(False)
        self.save_prompts_btn.setToolTip(
            "Save accumulated click feature vectors to disk (use before Run 3D Segment "
            "so they survive an OOM, or reuse in a later session / via CLI).")
        self.save_prompts_btn.clicked.connect(self._save_prompts)
        prompt_io_row.addWidget(self.save_prompts_btn)
        self.load_prompts_btn = QPushButton("Load Prompts…")
        self.load_prompts_btn.setEnabled(False)
        self.load_prompts_btn.setToolTip(
            "Load previously saved prompt feature vectors and append them to the current set.")
        self.load_prompts_btn.clicked.connect(self._load_prompts)
        prompt_io_row.addWidget(self.load_prompts_btn)
        v.addLayout(prompt_io_row)

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
    # Prepare menu
    # ------------------------------------------------------------------

    def _build_prepare_menu(self):
        prepare = self.menuBar().addMenu("Prepare")

        self.action_masks = prepare.addAction("Extract SAM Masks…")
        self.action_masks.setEnabled(False)
        self.action_masks.triggered.connect(self._show_masks_dialog)

        prepare.addSeparator()

        self.action_scale = prepare.addAction("Get Scale…")
        self.action_scale.setEnabled(False)
        self.action_scale.triggered.connect(self._show_scale_dialog)

        prepare.addSeparator()

        self.action_contrastive = prepare.addAction("Train Contrastive Features…")
        self.action_contrastive.setEnabled(False)
        self.action_contrastive.triggered.connect(self._show_contrastive_dialog)

    @staticmethod
    def _count_pt_files(path: str) -> int:
        if not os.path.isdir(path):
            return 0
        return sum(1 for f in os.listdir(path) if f.endswith(".pt"))

    def _update_prepare_menu_state(self, _=None):
        """Enable/disable Prepare menu items based on what exists on disk."""
        if not hasattr(self, "action_masks"):
            return

        source = self.source_path_edit.text().strip()
        model  = self.model_path_edit.text().strip()

        has_images = bool(source) and os.path.isdir(os.path.join(source, "images"))

        n_masks  = self._count_pt_files(os.path.join(source, "sam_masks"))  if source else 0
        n_scales = self._count_pt_files(os.path.join(source, "mask_scales")) if source else 0

        # scales are only complete when every mask file has a corresponding scale file
        has_masks  = n_masks > 0
        has_scales = n_scales > 0 and n_scales >= n_masks

        self.action_masks.setEnabled(has_images)
        self.action_scale.setEnabled(has_masks and bool(model))
        self.action_contrastive.setEnabled(has_scales and bool(model))

    # ---- Prepare step dialogs ----------------------------------------

    def _show_masks_dialog(self):
        source = self.source_path_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "Missing Path",
                                "Set the Data/Source Path before extracting masks.")
            return

        dlg = PrepareDialog("Prepare — Extract SAM Masks", self)

        # SAM checkpoint with browse button
        sam_edit = QLineEdit(
            os.path.join(_REPO_ROOT,
                         "third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth"))
        sam_browse = QPushButton("…"); sam_browse.setFixedWidth(28)
        from PyQt5.QtWidgets import QWidget as _W
        sam_row_w = _W(); sam_row_l = QHBoxLayout(sam_row_w)
        sam_row_l.setContentsMargins(0, 0, 0, 0)
        sam_row_l.addWidget(sam_edit); sam_row_l.addWidget(sam_browse)
        def _browse_sam():
            p, _ = QFileDialog.getOpenFileName(dlg, "SAM Checkpoint", "", "*.pth")
            if p:
                sam_edit.setText(p)
        sam_browse.clicked.connect(_browse_sam)

        tile_spin    = QSpinBox(); tile_spin.setRange(512, 20000); tile_spin.setValue(3000)
        overlap_spin = QSpinBox(); overlap_spin.setRange(0, 2000);  overlap_spin.setValue(300)
        ds_combo     = QComboBox(); ds_combo.addItems(["1", "2", "4", "8"]); ds_combo.setCurrentText("4")

        dlg.add_param("Source path:", QLabel(source))
        dlg.add_param("SAM checkpoint:", sam_row_w)
        dlg.add_param("Tile size (px):", tile_spin)
        dlg.add_param("Overlap (px):",   overlap_spin)
        dlg.add_param("Downsample:",     ds_combo)

        def build_cmd():
            return [
                sys.executable,
                os.path.join(_REPO_ROOT, "extract_segment_tiled_masks.py"),
                "--image_root",          source,
                "--sam_checkpoint_path", sam_edit.text().strip(),
                "--tile_size",           str(tile_spin.value()),
                "--overlap",             str(overlap_spin.value()),
                "--downsample",          ds_combo.currentText(),
            ]

        dlg.set_run(build_cmd, _REPO_ROOT)
        dlg.finished.connect(self._update_prepare_menu_state)
        dlg.exec_()

    def _show_scale_dialog(self):
        source = self.source_path_edit.text().strip()
        model  = self.model_path_edit.text().strip()

        dlg = PrepareDialog("Prepare — Get Scale", self)

        iter_spin = QSpinBox()
        iter_spin.setRange(0, 1_000_000)
        iter_spin.setValue(self.scene_iter_spin.value())
        iter_spin.setSingleStep(1000)

        dlg.add_param("Source path:",         QLabel(source or "(not set)"))
        dlg.add_param("Model path:",          QLabel(model  or "(not set)"))
        dlg.add_param("Scene PLY iteration:", iter_spin)

        def build_cmd():
            if not source or not model:
                raise ValueError("Source path and model path must both be set.")
            return [
                sys.executable,
                os.path.join(_REPO_ROOT, "get_scale.py"),
                "--model_path", model,
                "--image_root", source,
                "--iteration",  str(iter_spin.value()),
            ]

        dlg.set_run(build_cmd, _REPO_ROOT)
        dlg.finished.connect(self._update_prepare_menu_state)
        dlg.exec_()

    def _show_contrastive_dialog(self):
        model = self.model_path_edit.text().strip()

        dlg = PrepareDialog("Prepare — Train Contrastive Features", self)

        scene_iter_spin = QSpinBox()
        scene_iter_spin.setRange(0, 1_000_000)
        scene_iter_spin.setValue(self.scene_iter_spin.value())
        scene_iter_spin.setSingleStep(1000)

        train_iter_spin = QSpinBox()
        train_iter_spin.setRange(1000, 200_000)
        train_iter_spin.setValue(10_000)
        train_iter_spin.setSingleStep(1000)

        rays_spin = QSpinBox()
        rays_spin.setRange(10, 10_000)
        rays_spin.setValue(50)
        rays_spin.setSingleStep(50)

        scale_aware_spin = QSpinBox()
        scale_aware_spin.setRange(-1, 32)
        scale_aware_spin.setValue(16)
        scale_aware_spin.setToolTip("-1 = adaptive (full scale gate); 1–31 = fixed partial gate")

        smooth_k_spin = QSpinBox()
        smooth_k_spin.setRange(1, 32)
        smooth_k_spin.setValue(8)
        smooth_k_spin.setToolTip("KNN neighbours for feature smoothing at save time; lower = less VRAM")

        res_combo = QComboBox()
        res_combo.addItems(["1", "2", "4", "8", "16", "32"])
        res_combo.setCurrentText("8")
        res_combo.setToolTip("Render downsample factor — higher = less VRAM. Use 16/32 for very high-res images (>8 MP)")

        dlg.add_param("Model path:",                   QLabel(model or "(not set)"))
        dlg.add_param("Load 3DGS from iteration:",     scene_iter_spin)
        dlg.add_param("Contrastive train iterations:", train_iter_spin)
        dlg.add_param("Sampled rays per iter:",        rays_spin)
        dlg.add_param("Scale aware dim:",              scale_aware_spin)
        dlg.add_param("Smooth K (save-time KNN):",     smooth_k_spin)
        dlg.add_param("Render resolution divisor:",    res_combo)

        def build_cmd():
            if not model:
                raise ValueError("Model path must be set.")
            return [
                sys.executable,
                os.path.join(_REPO_ROOT, "train_contrastive_feature.py"),
                "-m",                  model,
                "--iterations",        str(train_iter_spin.value()),
                "--num_sampled_rays",  str(rays_spin.value()),
                "--iteration",         str(scene_iter_spin.value()),
                "--scale_aware_dim",   str(scale_aware_spin.value()),
                "--smooth_K",          str(smooth_k_spin.value()),
                "--resolution",        res_combo.currentText(),
            ]

        def _on_done(_):
            self._update_prepare_menu_state()
            self._autofill_paths()

        dlg.set_run(build_cmd, _REPO_ROOT)
        dlg.finished.connect(_on_done)
        dlg.exec_()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _browse_dir(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            edit.setText(path)

    def _browse_file(self, edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All files (*.*)")
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
        if text.strip():
            self._autofill_paths()
        self._update_prepare_menu_state()

    def _on_source_path_changed(self, _=None):
        self._update_prepare_menu_state()

    def _autofill_paths(self):
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            return
        s_iter = self.scene_iter_spin.value()
        f_iter = self.feat_iter_spin.value()
        self.scene_ply_edit.setText(
            os.path.join(model_path, f"point_cloud/iteration_{s_iter}/scene_point_cloud.ply"))
        self.feature_ply_edit.setText(
            os.path.join(model_path,
                         f"point_cloud/iteration_{f_iter}/contrastive_feature_point_cloud.ply"))
        self.scale_gate_edit.setText(
            os.path.join(model_path, f"point_cloud/iteration_{f_iter}/scale_gate.pt"))

    def _update_points_label(self):
        total   = len(self.click_raw_features)
        current = len(self.click_points)
        self.points_lbl.setText(f"Points: {total} total ({current} on this view)")
        self.save_prompts_btn.setEnabled(total > 0)

    def _set_controls_enabled(self, has_model: bool, has_render: bool, has_segment: bool):
        self.view_combo.setEnabled(has_model)
        self.render_view_btn.setEnabled(has_model)
        self.click_mode_btn.setEnabled(has_render)
        self.segment_3d_btn.setEnabled(has_render and len(self.click_raw_features) > 0)
        self.rollback_btn.setEnabled(has_segment)
        self.load_prompts_btn.setEnabled(has_model)
        self.export_ply_btn.setEnabled(has_segment)
        self.export_mask_btn.setEnabled(has_segment)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        model_path      = self.model_path_edit.text().strip()
        source_path     = self.source_path_edit.text().strip()
        scene_ply       = self.scene_ply_edit.text().strip()
        feature_ply     = self.feature_ply_edit.text().strip()
        scale_gate_path = self.scale_gate_edit.text().strip()

        if not model_path:
            QMessageBox.warning(self, "Missing Path", "Please provide a Model Path.")
            return

        if not scene_ply or not feature_ply or not scale_gate_path:
            self._autofill_paths()
            scene_ply       = self.scene_ply_edit.text().strip()
            feature_ply     = self.feature_ply_edit.text().strip()
            scale_gate_path = self.scale_gate_edit.text().strip()

        missing = [(p, n) for p, n in [
            (scene_ply,       "Scene PLY"),
            (feature_ply,     "Feature PLY"),
            (scale_gate_path, "Scale Gate .pt"),
        ] if not os.path.exists(p)]
        if missing:
            QMessageBox.critical(self, "Files Not Found",
                                 "\n".join(f"{n}:\n  {p}" for p, n in missing))
            return

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

        scene_ply_       = scene_ply
        feature_ply_     = feature_ply
        scale_gate_path_ = scale_gate_path
        model_path_      = model_path

        def _do_load():
            if cam_source[0] == "json":
                cameras = load_cameras_from_json(cam_source[1])
            else:
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
                scene_obj  = Scene(args, gaussians=None, feature_gaussians=None,
                                   load_iteration=scene_iter, shuffle=False)
                cameras = scene_obj.getTrainCameras()
                for cam in cameras:
                    cam.feature_height = cam.image_height
                    cam.feature_width  = cam.image_width

            s_model = GaussianModel(3)
            s_model.load_ply(scene_ply_)

            f_model = FeatureGaussianModel(self.FEATURE_DIM)
            f_model.load_ply(feature_ply_)

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
        self.cameras       = cameras
        self.scene_model   = s_model
        self.feature_model = f_model
        self.scale_gate    = gate
        self.bg_color      = torch.zeros(3, device="cuda")
        self.bg_feature    = torch.zeros(self.FEATURE_DIM, device="cuda")

        self.view_combo.clear()
        for cam in self.cameras:
            self.view_combo.addItem(cam.image_name)

        self._set_controls_enabled(has_model=True, has_render=False, has_segment=False)
        source_label = ("cameras.json"
                        if not self.source_path_edit.text().strip() else "COLMAP")
        self._set_status(f"Loaded. {len(cameras)} cameras ({source_label}).")

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
        cam.feature_width  = cam.image_width
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
        self.rgb_image      = np.clip(rgb, 0.0, 1.0)
        self.raw_feature_map = feat  # (H, W, 32) GPU, ungated

        self.click_points = []
        self._update_points_label()

        if self.click_raw_features:
            self._update_similarity_preview()
        else:
            self._display_image()

        self._set_controls_enabled(has_model=True, has_render=True,
                                   has_segment=self.is_segmented)
        cam   = self.current_camera
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
    # Click → feature sampling
    # ------------------------------------------------------------------

    def _on_canvas_click(self, event):
        if not self.click_mode_btn.isChecked():
            return
        if event.inaxes is not self.ax or event.button != 1:
            return
        if self.raw_feature_map is None or event.xdata is None:
            return

        H, W = self.raw_feature_map.shape[:2]
        px = max(0, min(int(round(event.xdata)), W - 1))
        py = max(0, min(int(round(event.ydata)), H - 1))

        self.click_points.append((px, py))
        with torch.no_grad():
            raw_f = self.raw_feature_map[py, px, :].clone()
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
            gates = self.scale_gate(torch.tensor([[scale_val]], device="cuda"))  # (1,32)
            if raw_feat.dim() == 3:
                feat = raw_feat / (raw_feat.norm(dim=-1, keepdim=True) + 1e-6)
                feat = feat * gates.unsqueeze(0)
                feat = torch.nn.functional.normalize(feat, dim=-1, p=2)
            else:
                feat = raw_feat / (raw_feat.norm() + 1e-6)
                feat = feat * gates.squeeze()
                feat = torch.nn.functional.normalize(feat.unsqueeze(0), dim=-1, p=2).squeeze()
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
            mask  = (score.max(dim=-1).values > thresh).reshape(H, W).cpu().numpy()

        self._display_image(overlay=mask)

    # ------------------------------------------------------------------
    # Save / load prompts
    # ------------------------------------------------------------------

    def _save_prompts(self):
        if not self.click_raw_features:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Prompts", "prompts.pt", "PyTorch files (*.pt)")
        if not path:
            return
        try:
            prompts = torch.stack([f.cpu() for f in self.click_raw_features])  # (N, 32)
            torch.save(prompts, path)
            self._set_status(
                f"Saved {len(self.click_raw_features)} prompt(s) → {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _load_prompts(self):
        if self.scene_model is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Prompts", "", "PyTorch files (*.pt)")
        if not path:
            return
        try:
            prompts = torch.load(path, map_location="cpu")  # (N, 32)
            if prompts.dim() != 2 or prompts.shape[1] != self.FEATURE_DIM:
                QMessageBox.critical(
                    self, "Shape Mismatch",
                    f"Expected (N, {self.FEATURE_DIM}) tensor, got {tuple(prompts.shape)}.\n"
                    "Only files saved with 'Save Prompts' can be loaded here.")
                return
            self.click_points = []
            self.click_raw_features = [prompts[i] for i in range(prompts.shape[0])]
            self._update_points_label()
            self.segment_3d_btn.setEnabled(self.raw_feature_map is not None)
            self._set_status(
                f"Loaded {prompts.shape[0]} prompt(s) from {os.path.basename(path)}.")
            if self.raw_feature_map is not None:
                self._update_similarity_preview()
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))

    # ------------------------------------------------------------------
    # Clear points
    # ------------------------------------------------------------------

    def _clear_view_points(self):
        n = len(self.click_points)
        if n > 0:
            self.click_raw_features = self.click_raw_features[:-n]
        self.click_points = []
        self._update_points_label()
        if not self.click_raw_features:
            self.segment_3d_btn.setEnabled(False)
            self._display_image()
        else:
            self._update_similarity_preview()

    def _clear_all_points(self):
        self.click_points       = []
        self.click_raw_features = []
        self._update_points_label()
        self.segment_3d_btn.setEnabled(False)
        self._display_image()

    # ------------------------------------------------------------------
    # 3D segmentation
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
                    torch.tensor([[self.scale_slider.value() / 100.0]], device="cuda"))
                chosen = torch.cat(
                    [self._compute_scale_gated(f).reshape(-1, 1)
                     for f in self.click_raw_features],
                    dim=-1)  # (32, N)

                feat_pts = self.feature_model.get_point_features  # (N_pts, 32)
                feat_pts = feat_pts * gates
                feat_pts = torch.nn.functional.normalize(feat_pts, dim=-1, p=2)

                try:
                    # Original fast path: single (N_pts × N_clicks) matrix multiply.
                    score_pts = feat_pts @ chosen
                    score_pts = (score_pts + 1.0) / 2.0
                    mask_3d   = (score_pts > thresh).any(dim=-1)
                except torch.cuda.OutOfMemoryError:
                    # Chunked fallback for large scenes / many prompts.
                    torch.cuda.empty_cache()
                    CHUNK   = 500_000
                    mask_3d = torch.zeros(feat_pts.shape[0], dtype=torch.bool, device="cuda")
                    for start in range(0, feat_pts.shape[0], CHUNK):
                        end   = min(start + CHUNK, feat_pts.shape[0])
                        chunk = feat_pts[start:end] @ chosen
                        chunk = (chunk + 1.0) / 2.0
                        mask_3d[start:end] = (chunk > thresh).any(dim=-1)
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
                                "No Gaussians matched. Lower the Threshold or add more points.")
            self._set_status("3D segment: empty result.")
            return

        self.last_segment_mask = mask_3d.clone()
        self.scene_model.segment(mask_3d)
        self.feature_model.segment(mask_3d)
        self.is_segmented = True

        self._set_controls_enabled(has_model=True, has_render=True, has_segment=True)
        self._set_status(
            f"Segmented — {n_sel:,} Gaussians from {len(self.click_raw_features)} prompt(s)."
            "  Re-rendering…")
        self._render_view()

    # ------------------------------------------------------------------
    # Roll back
    # ------------------------------------------------------------------

    def _roll_back(self):
        if self.scene_model is None:
            return
        self.scene_model.roll_back()
        self.feature_model.roll_back()
        self.click_points       = []
        self.click_raw_features = []
        self._update_points_label()
        self.is_segmented     = self.scene_model.segment_times > 0
        self.last_segment_mask = None
        self.rgb_image         = None
        self.raw_feature_map   = None
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
