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
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    export_requested = pyqtSignal()
    roi_mode_requested = pyqtSignal()
    mask_mode_requested = pyqtSignal()
    erase_mode_requested = pyqtSignal()
    clear_roi_requested = pyqtSignal()
    clear_mask_requested = pyqtSignal()
    view_mode_requested = pyqtSignal()
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
        row1.addWidget(self._btn_upload)
        row1.addSpacing(16)

        self._btn_roi = QPushButton("Select ROI")
        self._btn_roi.setToolTip(
            "Draw a rectangle around the region to analyze.\n"
            "Click and drag on the video to select.\n"
            "Click again to clear and redraw."
        )
        self._btn_roi.setCheckable(True)

        self._btn_mask = QPushButton("Draw Mask")
        self._btn_mask.setToolTip(
            "Click and drag to paint areas to EXCLUDE from analysis\n"
            "(e.g. tubes, clamps, labels blocking the view).\n"
            "Scroll wheel to change brush size.\n"
            "Click button again to clear all masks."
        )
        self._btn_mask.setCheckable(True)

        self._btn_erase = QPushButton("Erase Mask")
        self._btn_erase.setToolTip(
            "Click and drag to RESTORE masked areas.\n"
            "Scroll wheel to change brush size."
        )
        self._btn_erase.setCheckable(True)

        row1.addWidget(self._btn_roi)
        row1.addWidget(self._btn_mask)
        row1.addWidget(self._btn_erase)
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
        self._combo_grid.setToolTip(
            "Grid size for spatial analysis.\n"
            "More cells = finer resolution, fewer = faster."
        )
        self._combo_grid.addItems(["3x3", "4x4", "5x5", "6x6", "8x8", "10x10"])
        self._combo_grid.setCurrentText("5x5")
        row3.addWidget(self._combo_grid)

        row3.addWidget(QLabel("Frame Skip:"))
        self._combo_skip = QComboBox()
        self._combo_skip.setToolTip(
            "Analyze every Nth frame.\n"
            "1 = every frame, 5 = every 5th frame.\n"
            "Higher = faster but less temporal detail."
        )
        self._combo_skip.addItems(["1", "2", "3", "5", "10"])
        row3.addWidget(self._combo_skip)

        row3.addWidget(QLabel("GLCM Skip:"))
        self._combo_glcm_skip = QComboBox()
        self._combo_glcm_skip.setToolTip(
            "Recompute texture metrics (Contrast, Homogeneity, Energy)\n"
            "every Nth analyzed frame. These are the slowest metrics.\n"
            "Higher = faster. Values hold steady between updates."
        )
        self._combo_glcm_skip.addItems(["1", "2", "3", "5", "10"])
        row3.addWidget(self._combo_glcm_skip)

        row3.addWidget(QLabel("GLCM Levels:"))
        self._combo_levels = QComboBox()
        self._combo_levels.setToolTip(
            "Gray level quantization for texture analysis.\n"
            "16 = recommended (fast, robust).\n"
            "64 = more detail but slower."
        )
        self._combo_levels.addItems(["8", "16", "32", "64"])
        self._combo_levels.setCurrentText("16")
        row3.addWidget(self._combo_levels)

        row3.addWidget(QLabel("Threshold:"))
        self._slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self._slider_threshold.setToolTip(
            "Grayscale threshold for Contact metric (0-255).\n"
            "Pixels above = white, below = black.\n"
            "Contact counts the boundary between them.\n"
            "Adjust based on your liquid color."
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
        self._btn_roi.clicked.connect(self._on_roi_toggle)
        self._btn_mask.clicked.connect(self._on_mask_toggle)
        self._btn_erase.clicked.connect(self._on_erase_toggle)
        self._btn_start.clicked.connect(lambda: self.start_requested.emit())
        self._btn_stop.clicked.connect(lambda: self.stop_requested.emit())
        self._btn_export.clicked.connect(lambda: self.export_requested.emit())
        self._slider_threshold.valueChanged.connect(
            lambda v: self._lbl_threshold.setText(str(v))
        )

        self._has_run = False
        self.set_state(AppState.IDLE)

    def mark_analysis_started(self) -> None:
        """Mark that analysis has been run at least once."""
        self._has_run = True

    def reset_for_new_video(self) -> None:
        """Reset state when a new video is loaded."""
        self._has_run = False

    def _on_roi_toggle(self) -> None:
        if self._btn_roi.isChecked():
            self._btn_mask.setChecked(False)
            self._btn_mask.setText("Draw Mask")
            self._btn_roi.setText("Drawing ROI... (click to clear)")
            self.roi_mode_requested.emit()
        else:
            # Uncheck = clear ROI
            self._btn_roi.setText("Select ROI")
            self.clear_roi_requested.emit()
            self.view_mode_requested.emit()

    def _on_mask_toggle(self) -> None:
        if self._btn_mask.isChecked():
            self._btn_roi.setChecked(False)
            self._btn_roi.setText("Select ROI")
            self._btn_erase.setChecked(False)
            self._btn_mask.setText("Painting... (scroll=size, click to clear)")
            self.mask_mode_requested.emit()
        else:
            self._btn_mask.setText("Draw Mask")
            self.clear_mask_requested.emit()
            self.view_mode_requested.emit()

    def _on_erase_toggle(self) -> None:
        if self._btn_erase.isChecked():
            self._btn_roi.setChecked(False)
            self._btn_roi.setText("Select ROI")
            self._btn_mask.setChecked(False)
            self._btn_mask.setText("Draw Mask")
            self._btn_erase.setText("Erasing... (scroll=size)")
            self.erase_mode_requested.emit()
        else:
            self._btn_erase.setText("Erase Mask")
            self.view_mode_requested.emit()

    def deactivate_tools(self) -> None:
        """Reset tool buttons to inactive state."""
        self._btn_roi.setChecked(False)
        self._btn_roi.setText("Select ROI")
        self._btn_mask.setChecked(False)
        self._btn_mask.setText("Draw Mask")
        self._btn_erase.setChecked(False)
        self._btn_erase.setText("Erase Mask")

    def set_state(self, state: AppState) -> None:
        """Enable/disable controls based on app state."""
        self._state = state
        idle = state == AppState.IDLE
        ready = state == AppState.READY
        configured = state == AppState.CONFIGURED
        running = state == AppState.RUNNING
        paused = state == AppState.PAUSED

        self._btn_upload.setEnabled(idle or ready or configured)
        self._btn_roi.setEnabled(ready or configured)
        self._btn_mask.setEnabled(ready or configured)
        self._btn_erase.setEnabled(ready or configured)
        self._btn_start.setEnabled(ready or configured)
        self._btn_stop.setEnabled(running or paused)
        self._btn_export.setEnabled(running or paused or configured)

        # Update button labels based on context
        if not idle:
            self._btn_upload.setText("Change Video")
        else:
            self._btn_upload.setText("Open Video")

        if self._has_run and not running:
            self._btn_start.setText("Restart Analysis")
        elif running:
            self._btn_start.setText("Running...")
        else:
            self._btn_start.setText("Start Analysis")

        self._progress_bar.setVisible(running)
        if not running:
            self._progress_bar.setValue(0)

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
            self._progress_bar.setMaximum(0)
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
