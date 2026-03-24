"""Controls panel: buttons, sliders, dropdowns for analysis configuration."""
from __future__ import annotations

from enum import Enum, auto
from typing import Any, Dict

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton,
    QSlider, QVBoxLayout, QWidget,
)


class AppState(Enum):
    IDLE = auto()
    READY = auto()
    CONFIGURED = auto()
    RUNNING = auto()
    PAUSED = auto()


class ControlsPanel(QWidget):
    """Control buttons, config sliders/dropdowns, and state management."""

    video_selected = pyqtSignal(str)
    camera_requested = pyqtSignal(int)
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    export_requested = pyqtSignal()
    roi_mode_requested = pyqtSignal()
    mask_mode_requested = pyqtSignal()
    config_changed = pyqtSignal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        main_layout = QVBoxLayout(self)

        # Row 1: Action buttons
        row1 = QHBoxLayout()
        self._btn_upload = QPushButton("Upload Video")
        self._btn_camera = QPushButton("Start Camera")
        self._btn_roi = QPushButton("Select ROI")
        self._btn_mask = QPushButton("Draw Mask")
        self._btn_start = QPushButton("Start Analysis")
        self._btn_stop = QPushButton("Stop")
        self._btn_export = QPushButton("Export Data")

        self._btn_roi.setCheckable(True)
        self._btn_mask.setCheckable(True)

        for btn in [
            self._btn_upload, self._btn_camera, self._btn_roi,
            self._btn_mask, self._btn_start, self._btn_stop, self._btn_export,
        ]:
            row1.addWidget(btn)
        main_layout.addLayout(row1)

        # Row 2: Config controls
        row2 = QHBoxLayout()

        row2.addWidget(QLabel("Grid:"))
        self._combo_grid = QComboBox()
        self._combo_grid.addItems(["3x3", "4x4", "5x5", "6x6", "8x8", "10x10"])
        self._combo_grid.setCurrentText("5x5")
        row2.addWidget(self._combo_grid)

        row2.addWidget(QLabel("Frame Skip:"))
        self._combo_skip = QComboBox()
        self._combo_skip.addItems(["1", "2", "3", "5", "10"])
        row2.addWidget(self._combo_skip)

        row2.addWidget(QLabel("GLCM Skip:"))
        self._combo_glcm_skip = QComboBox()
        self._combo_glcm_skip.addItems(["1", "2", "3", "5", "10"])
        row2.addWidget(self._combo_glcm_skip)

        row2.addWidget(QLabel("GLCM Levels:"))
        self._combo_levels = QComboBox()
        self._combo_levels.addItems(["8", "16", "32", "64"])
        self._combo_levels.setCurrentText("16")
        row2.addWidget(self._combo_levels)

        row2.addWidget(QLabel("Threshold:"))
        self._slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self._slider_threshold.setRange(0, 255)
        self._slider_threshold.setValue(128)
        self._slider_threshold.setFixedWidth(120)
        row2.addWidget(self._slider_threshold)
        self._lbl_threshold = QLabel("128")
        row2.addWidget(self._lbl_threshold)

        row2.addWidget(QLabel("Export:"))
        self._combo_format = QComboBox()
        self._combo_format.addItems(["csv", "xlsx"])
        row2.addWidget(self._combo_format)

        row2.addStretch()
        main_layout.addLayout(row2)

        # Connections
        self._btn_upload.clicked.connect(self._on_upload)
        self._btn_camera.clicked.connect(self._on_camera)
        self._btn_roi.clicked.connect(lambda: self.roi_mode_requested.emit())
        self._btn_mask.clicked.connect(lambda: self.mask_mode_requested.emit())
        self._btn_start.clicked.connect(lambda: self.start_requested.emit())
        self._btn_stop.clicked.connect(lambda: self.stop_requested.emit())
        self._btn_export.clicked.connect(lambda: self.export_requested.emit())
        self._slider_threshold.valueChanged.connect(
            lambda v: self._lbl_threshold.setText(str(v))
        )

        self.set_state(AppState.IDLE)

    def set_state(self, state: AppState) -> None:
        """Enable/disable controls based on app state."""
        self._state = state
        idle = state == AppState.IDLE
        ready = state == AppState.READY
        configured = state == AppState.CONFIGURED
        running = state == AppState.RUNNING
        paused = state == AppState.PAUSED

        self._btn_upload.setEnabled(idle or ready or configured)
        self._btn_camera.setEnabled(idle or ready or configured)
        self._btn_roi.setEnabled(ready or configured)
        self._btn_mask.setEnabled(ready or configured)
        self._btn_start.setEnabled(configured)
        self._btn_stop.setEnabled(running or paused)
        self._btn_export.setEnabled(running or paused or configured)

        for w in [
            self._combo_grid, self._combo_skip, self._combo_glcm_skip,
            self._combo_levels, self._slider_threshold, self._combo_format,
        ]:
            w.setEnabled(not running and not paused)

    def get_config(self) -> Dict[str, Any]:
        """Build config dict from current control values."""
        grid_text = self._combo_grid.currentText()
        grid_size = int(grid_text.split("x")[0])
        return {
            "grid_rows": grid_size,
            "grid_cols": grid_size,
            "frame_skip": int(self._combo_skip.currentText()),
            "glcm_frame_skip": int(self._combo_glcm_skip.currentText()),
            "glcm_gray_levels": int(self._combo_levels.currentText()),
            "contact_threshold": self._slider_threshold.value(),
            "export_format": self._combo_format.currentText(),
            "glcm_offset": [1, 1],
            "camera_index": 0,
            "video_fps_override": None,
            "brightness_change_threshold": 0.2,
        }

    def _on_upload(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
        )
        if path:
            self.video_selected.emit(path)

    def _on_camera(self) -> None:
        self.camera_requested.emit(0)
