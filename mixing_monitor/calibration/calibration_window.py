"""Calibration panel — embeddable QWidget used inside MainWindow."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QButtonGroup, QRadioButton,
    QFileDialog, QMessageBox, QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

from ..common.camera import CameraSource
from ..common.constants import (
    DEFAULT_CONFIG_PATH, VESSEL_COLORS, VESSEL_LABELS,
    MAX_VESSELS, MIN_VESSELS,
)
from ..common.roi_config import (
    load_config, save_config, scale_rois, ConfigValidationError,
)
from .roi_canvas import RoiCanvas

logger = logging.getLogger(__name__)

_PLACEHOLDER_FRAME = np.full((720, 1280, 3), 40, dtype=np.uint8)


class CalibrationWindow(QWidget):
    """Interactive calibration panel: camera selection + ROI drawing + save."""

    # Emitted after a successful save; carries the config path
    calibration_saved = pyqtSignal(Path)
    # Emitted when the user clicks Next →
    next_clicked = pyqtSignal()

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        super().__init__()
        self._config_path = config_path
        self._camera = CameraSource()
        self._devices: list[dict] = []
        self._current_device_index: int = -1
        self._has_unsaved_changes: bool = False
        # Next button is enabled immediately if a config already exists on disk
        self._saved_at_least_once: bool = config_path.exists()

        # Timer must exist before _refresh_devices() calls _timer.stop()
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._grab_frame)

        self._build_ui()
        self._refresh_devices()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # --- Camera row ---
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera:"))
        self._cam_combo = QComboBox()
        self._cam_combo.setMinimumWidth(250)
        self._cam_combo.currentIndexChanged.connect(self._on_camera_selected)
        cam_row.addWidget(self._cam_combo)
        self._refresh_btn = QPushButton("↺ Refresh")
        self._refresh_btn.setFixedWidth(90)
        self._refresh_btn.clicked.connect(self._refresh_devices)
        cam_row.addWidget(self._refresh_btn)
        cam_row.addStretch()
        self._status_label = QLabel("● Disconnected")
        self._status_label.setStyleSheet("color: #ef4444;")
        cam_row.addWidget(self._status_label)
        root.addLayout(cam_row)

        # --- Canvas ---
        self._canvas = RoiCanvas()
        self._canvas.rois_changed.connect(self._on_rois_changed)
        self._canvas.set_frame(_PLACEHOLDER_FRAME)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._canvas, stretch=1)

        # --- Vessel count row ---
        vc_row = QHBoxLayout()
        vc_row.addWidget(QLabel("Number of vessels:"))
        self._vc_group = QButtonGroup(self)
        for n in range(MIN_VESSELS, MAX_VESSELS + 1):
            rb = QRadioButton(str(n))
            if n == 3:
                rb.setChecked(True)
            self._vc_group.addButton(rb, n)
            vc_row.addWidget(rb)
        vc_row.addStretch()
        self._vc_group.idClicked.connect(self._on_vessel_count_changed)
        root.addLayout(vc_row)

        # --- Instructions ---
        instr = QLabel(
            "Instructions: Click and drag on the feed to draw a bounding box for each vessel. "
            "Drag inside to move, drag corners/edges to resize. Right-click to delete."
        )
        instr.setWordWrap(True)
        instr.setStyleSheet("color: #6b7280; font-size: 11px;")
        root.addWidget(instr)

        # --- Separator ---
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #d1d5db;")
        root.addWidget(line)

        # --- Bottom row ---
        btn_row = QHBoxLayout()
        self._load_btn = QPushButton("Load Existing Config")
        self._load_btn.clicked.connect(self._load_config_dialog)
        btn_row.addWidget(self._load_btn)

        btn_row.addStretch()

        self._bottom_status = QLabel("")
        self._bottom_status.setStyleSheet("color: #16a34a;")
        btn_row.addWidget(self._bottom_status)

        self._save_next_btn = QPushButton("Save & Continue →")
        self._save_next_btn.setEnabled(False)
        self._save_next_btn.setToolTip("Draw all bounding boxes to continue.")
        self._save_next_btn.setStyleSheet(
            "QPushButton:enabled { background: #2563eb; color: white; font-weight: bold; "
            "padding: 4px 16px; border-radius: 4px; }"
        )
        self._save_next_btn.clicked.connect(self._save_and_next)
        btn_row.addWidget(self._save_next_btn)

        root.addLayout(btn_row)

        # Call after _save_next_btn exists — set_max_rois emits rois_changed immediately
        self._canvas.set_max_rois(3)

    # ------------------------------------------------------------------
    # Public API (called by MainWindow)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to a blank state after returning from the monitor screen."""
        self._saved_at_least_once = False
        self._has_unsaved_changes = False
        self._save_next_btn.setEnabled(False)
        self._save_next_btn.setToolTip("Draw all bounding boxes to continue.")
        self._bottom_status.setText("")
        vessel_count = self._vc_group.checkedId()
        self._canvas.set_max_rois(vessel_count)  # clears ROIs

    def start_camera(self) -> None:
        """Re-enumerate devices and start the feed. Call after returning from monitor."""
        self._refresh_devices()

    def cleanup(self) -> None:
        """Stop the camera timer and release the device."""
        self._timer.stop()
        self._camera.release()

    def has_unsaved_rois(self) -> bool:
        return self._has_unsaved_changes and self._canvas.roi_count() > 0

    def save_config_now(self) -> None:
        """Trigger save programmatically (used by MainWindow close handler)."""
        self._save_config()

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def _refresh_devices(self) -> None:
        self._timer.stop()
        self._camera.release()
        self._devices = CameraSource.list_devices()

        self._cam_combo.blockSignals(True)
        self._cam_combo.clear()
        if self._devices:
            for d in self._devices:
                self._cam_combo.addItem(d["name"], d["index"])
            self._cam_combo.setCurrentIndex(0)
        else:
            self._cam_combo.addItem("(No cameras found)", -1)
            self._canvas.set_frame(_PLACEHOLDER_FRAME)
            self._set_status_disconnected(
                "No camera detected. Connect a camera and click ↺ Refresh."
            )
        self._cam_combo.blockSignals(False)

        if self._devices:
            self._on_camera_selected(0)

    def _on_camera_selected(self, combo_idx: int) -> None:
        self._timer.stop()
        self._camera.release()
        if combo_idx < 0 or combo_idx >= len(self._devices):
            return
        dev = self._devices[combo_idx]
        ok = self._camera.open(dev["index"])
        if ok:
            w, h = self._camera.actual_resolution
            fps = self._camera.actual_fps
            self._current_device_index = dev["index"]
            self._set_status_connected(w, h, fps)
            self._timer.start()
        else:
            self._canvas.set_frame(_PLACEHOLDER_FRAME)
            self._set_status_disconnected(
                f"Cannot open device '{dev['name']}'. Check connection and try ↺ Refresh."
            )

    def _grab_frame(self) -> None:
        ok, frame = self._camera.read()
        if ok and frame is not None:
            self._canvas.set_frame(frame)
        else:
            self._timer.stop()
            self._set_status_disconnected(
                "Camera disconnected. Reconnect and click ↺ Refresh."
            )

    def _set_status_connected(self, w: int, h: int, fps: float) -> None:
        self._status_label.setText(f"● Connected — {w}×{h} @ {fps:.0f}fps")
        self._status_label.setStyleSheet("color: #16a34a;")

    def _set_status_disconnected(self, msg: str) -> None:
        self._status_label.setText(f"● {msg}")
        self._status_label.setStyleSheet("color: #ef4444;")

    # ------------------------------------------------------------------
    # ROI / vessel count
    # ------------------------------------------------------------------

    def _on_vessel_count_changed(self, n: int) -> None:
        self._canvas.set_max_rois(n)
        self._has_unsaved_changes = True
        self._update_save_button()

    def _on_rois_changed(self) -> None:
        self._has_unsaved_changes = True
        self._update_save_button()

    def _update_save_button(self) -> None:
        vessel_count = self._vc_group.checkedId()
        roi_count = self._canvas.roi_count()
        ready = roi_count == vessel_count
        self._save_next_btn.setEnabled(ready)
        if not ready:
            remaining = vessel_count - roi_count
            self._save_next_btn.setToolTip(f"Draw {remaining} more bounding box(es) to continue.")
        else:
            self._save_next_btn.setToolTip("")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _save_config(self) -> None:
        vessel_count = self._vc_group.checkedId()
        rois = self._canvas.get_rois()
        w, h = self._camera.actual_resolution if self._camera.is_opened else (1280, 720)
        dev_name = ""
        if self._devices:
            combo_idx = self._cam_combo.currentIndex()
            if 0 <= combo_idx < len(self._devices):
                dev_name = self._devices[combo_idx]["name"]

        vessels = []
        for roi_dict in rois:
            vid = roi_dict["vessel_id"]
            vessels.append({
                "id": vid,
                "label": VESSEL_LABELS.get(vid, f"V{vid}"),
                "color": VESSEL_COLORS.get(vid, "#ffffff"),
                "roi": roi_dict["roi"],
            })

        data = {
            "version": 1,
            "camera": {
                "device_index": self._current_device_index,
                "device_name": dev_name,
                "resolution": [w, h],
            },
            "vessel_count": vessel_count,
            "vessels": vessels,
        }

        try:
            save_config(data, self._config_path)
            self._has_unsaved_changes = False
            self._saved_at_least_once = True
            self._bottom_status.setText(f"✓ Saved to {self._config_path.name}")
            self._bottom_status.setStyleSheet("color: #16a34a;")
            self.calibration_saved.emit(self._config_path)
        except Exception as exc:
            self._bottom_status.setText(f"✗ Save failed: {exc}")
            self._bottom_status.setStyleSheet("color: #ef4444;")

    def _save_and_next(self) -> None:
        self._save_config()
        if self._saved_at_least_once:
            self.next_clicked.emit()

    def _load_config_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config", str(self._config_path.parent),
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        self._load_config_from(Path(path))

    def _load_config_from(self, path: Path) -> None:
        try:
            data = load_config(path)
        except (FileNotFoundError, ConfigValidationError) as exc:
            QMessageBox.critical(
                self, "Cannot Load Config",
                f"Cannot load config: {exc}\n\nStart fresh or select a different file."
            )
            return

        vessel_count = data["vessel_count"]
        btn = self._vc_group.button(vessel_count)
        if btn:
            btn.setChecked(True)
        self._canvas.set_max_rois(vessel_count)

        cfg_res = tuple(data["camera"]["resolution"])
        cam_res = self._camera.actual_resolution if self._camera.is_opened else cfg_res
        if cfg_res != cam_res:
            data = scale_rois(data, cam_res)
            QMessageBox.warning(
                self, "Resolution Mismatch",
                f"Config was saved at {cfg_res[0]}×{cfg_res[1]}, "
                f"current camera is {cam_res[0]}×{cam_res[1]}. "
                "ROIs have been scaled. Verify positions."
            )

        rois = [{"vessel_id": v["id"], "roi": v["roi"]} for v in data["vessels"]]
        self._canvas.set_rois(rois)

        cfg_dev_idx = data["camera"]["device_index"]
        for i, dev in enumerate(self._devices):
            if dev["index"] == cfg_dev_idx:
                self._cam_combo.setCurrentIndex(i)
                break
        else:
            if self._devices:
                QMessageBox.warning(
                    self, "Camera Not Found",
                    f"Camera '{data['camera']['device_name']}' (index {cfg_dev_idx}) "
                    "not found. Using the first available device."
                )

        self._config_path = path
        self._has_unsaved_changes = False
        self._update_save_button()
        self._bottom_status.setText(f"✓ Loaded {path.name}")
        self._bottom_status.setStyleSheet("color: #16a34a;")
