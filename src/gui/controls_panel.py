"""Controls panel: buttons, sliders, dropdowns for analysis configuration."""
from __future__ import annotations

from enum import Enum, auto
from typing import Any, Dict

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QSlider, QVBoxLayout, QWidget,
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
    clear_roi_requested = pyqtSignal()
    clear_mask_requested = pyqtSignal()
    config_changed = pyqtSignal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 4, 8, 4)

        # Row 1: Action buttons
        row1 = QHBoxLayout()

        self._btn_upload = QPushButton("Open Video")
        self._btn_upload.setToolTip(
            "Open a video file (MP4, AVI, MOV) to analyze"
        )

        self._btn_camera = QPushButton("Live Camera")
        self._btn_camera.setToolTip(
            "Start analyzing a live camera feed"
        )

        # Separator
        row1.addWidget(self._btn_upload)
        row1.addWidget(self._btn_camera)
        row1.addSpacing(16)

        self._btn_roi = QPushButton("Select ROI")
        self._btn_roi.setToolTip(
            "Draw a rectangle around the region to analyze.\n"
            "Click and drag on the video to select."
        )
        self._btn_roi.setCheckable(True)

        self._btn_clear_roi = QPushButton("Clear ROI")
        self._btn_clear_roi.setToolTip("Remove the selected region and analyze the full frame")

        self._btn_mask = QPushButton("Exclude Areas")
        self._btn_mask.setToolTip(
            "Paint over areas to EXCLUDE from analysis\n"
            "(e.g. tubes, clamps, labels blocking the view).\n"
            "Scroll wheel to change brush size."
        )
        self._btn_mask.setCheckable(True)

        self._btn_clear_mask = QPushButton("Clear Exclusions")
        self._btn_clear_mask.setToolTip("Remove all excluded areas")

        row1.addWidget(self._btn_roi)
        row1.addWidget(self._btn_clear_roi)
        row1.addWidget(self._btn_mask)
        row1.addWidget(self._btn_clear_mask)
        row1.addSpacing(16)

        self._btn_start = QPushButton("Start Analysis")
        self._btn_start.setToolTip("Run all 6 mixing metrics on every video frame")
        self._btn_start.setStyleSheet(
            "QPushButton:enabled { background-color: #2d7d46; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
        )

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setToolTip("Stop the running analysis")
        self._btn_stop.setStyleSheet(
            "QPushButton:enabled { background-color: #c0392b; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
        )

        self._btn_export = QPushButton("Export Data")
        self._btn_export.setToolTip(
            "Save metric results to CSV or Excel file.\n"
            "Works during analysis (exports data collected so far)."
        )

        row1.addWidget(self._btn_start)
        row1.addWidget(self._btn_stop)
        row1.addWidget(self._btn_export)

        main_layout.addLayout(row1)

        # Row 2: Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFixedHeight(16)
        main_layout.addWidget(self._progress_bar)

        # Row 3: Config controls
        row3 = QHBoxLayout()

        row3.addWidget(QLabel("Grid:"))
        self._combo_grid = QComboBox()
        self._combo_grid.setToolTip("Grid size for spatial analysis (rows x columns)")
        self._combo_grid.addItems(["3x3", "4x4", "5x5", "6x6", "8x8", "10x10"])
        self._combo_grid.setCurrentText("5x5")
        row3.addWidget(self._combo_grid)

        row3.addWidget(QLabel("Frame Skip:"))
        self._combo_skip = QComboBox()
        self._combo_skip.setToolTip("Analyze every Nth frame (higher = faster, less detail)")
        self._combo_skip.addItems(["1", "2", "3", "5", "10"])
        row3.addWidget(self._combo_skip)

        row3.addWidget(QLabel("GLCM Skip:"))
        self._combo_glcm_skip = QComboBox()
        self._combo_glcm_skip.setToolTip(
            "Recompute texture metrics every Nth frame\n"
            "(higher = faster, texture metrics hold their last value)"
        )
        self._combo_glcm_skip.addItems(["1", "2", "3", "5", "10"])
        row3.addWidget(self._combo_glcm_skip)

        row3.addWidget(QLabel("GLCM Levels:"))
        self._combo_levels = QComboBox()
        self._combo_levels.setToolTip("Gray level quantization for texture analysis (fewer = faster)")
        self._combo_levels.addItems(["8", "16", "32", "64"])
        self._combo_levels.setCurrentText("16")
        row3.addWidget(self._combo_levels)

        row3.addWidget(QLabel("Threshold:"))
        self._slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self._slider_threshold.setToolTip(
            "Grayscale threshold for Contact metric (0-255).\n"
            "Pixels above = white, below = black."
        )
        self._slider_threshold.setRange(0, 255)
        self._slider_threshold.setValue(128)
        self._slider_threshold.setFixedWidth(120)
        row3.addWidget(self._slider_threshold)
        self._lbl_threshold = QLabel("128")
        row3.addWidget(self._lbl_threshold)

        row3.addWidget(QLabel("Export:"))
        self._combo_format = QComboBox()
        self._combo_format.setToolTip("File format for data export")
        self._combo_format.addItems(["csv", "xlsx"])
        row3.addWidget(self._combo_format)

        row3.addStretch()
        main_layout.addLayout(row3)

        # Connections
        self._btn_upload.clicked.connect(self._on_upload)
        self._btn_camera.clicked.connect(self._on_camera)
        self._btn_roi.clicked.connect(self._on_roi_toggle)
        self._btn_clear_roi.clicked.connect(lambda: self.clear_roi_requested.emit())
        self._btn_mask.clicked.connect(self._on_mask_toggle)
        self._btn_clear_mask.clicked.connect(lambda: self.clear_mask_requested.emit())
        self._btn_start.clicked.connect(lambda: self.start_requested.emit())
        self._btn_stop.clicked.connect(lambda: self.stop_requested.emit())
        self._btn_export.clicked.connect(lambda: self.export_requested.emit())
        self._slider_threshold.valueChanged.connect(
            lambda v: self._lbl_threshold.setText(str(v))
        )

        self.set_state(AppState.IDLE)

    def _on_roi_toggle(self) -> None:
        if self._btn_roi.isChecked():
            self._btn_mask.setChecked(False)
            self._btn_roi.setText("Drawing ROI...")
            self.roi_mode_requested.emit()
        else:
            self._btn_roi.setText("Select ROI")

    def _on_mask_toggle(self) -> None:
        if self._btn_mask.isChecked():
            self._btn_roi.setChecked(False)
            self._btn_mask.setText("Painting... (scroll=size)")
            self.mask_mode_requested.emit()
        else:
            self._btn_mask.setText("Exclude Areas")

    def deactivate_tools(self) -> None:
        """Reset tool buttons to inactive state."""
        self._btn_roi.setChecked(False)
        self._btn_roi.setText("Select ROI")
        self._btn_mask.setChecked(False)
        self._btn_mask.setText("Exclude Areas")

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
        self._btn_clear_roi.setEnabled(ready or configured)
        self._btn_mask.setEnabled(ready or configured)
        self._btn_clear_mask.setEnabled(ready or configured)
        self._btn_start.setEnabled(configured)
        self._btn_stop.setEnabled(running or paused)
        self._btn_export.setEnabled(running or paused or configured)

        # Progress bar visibility
        self._progress_bar.setVisible(running)
        if not running:
            self._progress_bar.setValue(0)

        # Deactivate drawing tools when running
        if running:
            self.deactivate_tools()

        for w in [
            self._combo_grid, self._combo_skip, self._combo_glcm_skip,
            self._combo_levels, self._slider_threshold, self._combo_format,
        ]:
            w.setEnabled(not running and not paused)

    def update_progress(self, current: int, total: int) -> None:
        """Update the progress bar."""
        if total > 0:
            self._progress_bar.setMaximum(total)
            self._progress_bar.setValue(current)
            pct = current / total * 100
            self._progress_bar.setFormat(f"{current}/{total} frames ({pct:.0f}%)")
        else:
            self._progress_bar.setMaximum(0)  # indeterminate
            self._progress_bar.setFormat(f"{current} frames (live)")

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
