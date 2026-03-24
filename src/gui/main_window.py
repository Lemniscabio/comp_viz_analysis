"""Main application window: three-panel layout with menu bar."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFileDialog, QMainWindow, QMessageBox, QSplitter,
    QStatusBar, QVBoxLayout, QWidget,
)

from src.core.export import DataExporter
from src.gui.analysis_worker import AnalysisWorker
from src.gui.controls_panel import AppState, ControlsPanel
from src.gui.plots_panel import PlotsPanel
from src.gui.roi_selector import InteractionMode
from src.gui.video_panel import VideoPanel
from src.utils.config_loader import load_config


class MainWindow(QMainWindow):
    """Kineticolor main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kineticolor - Mixing Analysis")
        self.setMinimumSize(1200, 700)

        self._worker: Optional[AnalysisWorker] = None
        self._video_path: Optional[str] = None
        self._camera_index: Optional[int] = None
        self._reference_frame_num = 0
        self._current_frame_num = 0
        self._state = AppState.IDLE

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._video_panel = VideoPanel()
        self._plots_panel = PlotsPanel()
        splitter.addWidget(self._video_panel)
        splitter.addWidget(self._plots_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

        self._controls = ControlsPanel()
        layout.addWidget(self._controls)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

    def _setup_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        open_action = QAction("Open Video", self)
        open_action.triggered.connect(self._on_open_video_menu)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        export_menu = menubar.addMenu("Export")
        export_action = QAction("Export Data", self)
        export_action.triggered.connect(self._on_export)
        export_menu.addAction(export_action)

        settings_menu = menubar.addMenu("Settings")
        grid_toggle = QAction("Toggle Grid Overlay", self)
        grid_toggle.setCheckable(True)
        grid_toggle.triggered.connect(self._video_panel.set_grid_visible)
        settings_menu.addAction(grid_toggle)

        heatmap_toggle = QAction("Toggle Heatmap Overlay", self)
        heatmap_toggle.setCheckable(True)
        heatmap_toggle.triggered.connect(self._video_panel.set_heatmap_visible)
        settings_menu.addAction(heatmap_toggle)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(
            lambda: QMessageBox.about(
                self,
                "About Kineticolor",
                "Kineticolor - Computer Vision Mixing Analysis\n\n"
                "Real-time kinetic analysis of mixing phenomena\n"
                "in reaction vessels using 6 complementary metrics.",
            )
        )
        help_menu.addAction(about_action)

    def _connect_signals(self) -> None:
        ctrl = self._controls
        ctrl.video_selected.connect(self._on_video_selected)
        ctrl.camera_requested.connect(self._on_camera_requested)
        ctrl.start_requested.connect(self._on_start)
        ctrl.stop_requested.connect(self._on_stop)
        ctrl.export_requested.connect(self._on_export)
        ctrl.roi_mode_requested.connect(
            lambda: self._video_panel.set_mode(InteractionMode.ROI)
        )
        ctrl.mask_mode_requested.connect(
            lambda: self._video_panel.set_mode(InteractionMode.MASK)
        )
        ctrl.clear_roi_requested.connect(self._on_clear_roi)
        ctrl.clear_mask_requested.connect(self._on_clear_mask)
        self._video_panel.roi_selected.connect(self._on_roi_selected)
        self._video_panel.reference_frame_requested.connect(self._on_set_reference)

    def _set_state(self, state: AppState) -> None:
        self._state = state
        self._controls.set_state(state)

    def _on_video_selected(self, path: str) -> None:
        self._video_path = path
        self._camera_index = None
        self._status.showMessage(f"Video loaded: {Path(path).name}")

        import cv2

        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        if ret:
            self._video_panel.update_frame(frame)
        cap.release()

        self._set_state(AppState.READY)

    def _on_camera_requested(self, index: int) -> None:
        self._camera_index = index
        self._video_path = None
        self._status.showMessage(f"Camera {index} selected")
        self._set_state(AppState.READY)

    def _on_roi_selected(self, roi: tuple) -> None:
        x, y, w, h = roi
        self._status.showMessage(
            f"ROI selected: {w}x{h} pixels at ({x}, {y}) -- Click 'Start Analysis' to begin"
        )
        self._video_panel.set_mode(InteractionMode.VIEW)
        self._controls.deactivate_tools()
        self._set_state(AppState.CONFIGURED)

    def _on_clear_roi(self) -> None:
        self._video_panel.selector.clear_roi()
        self._video_panel._refresh_display()
        self._status.showMessage("ROI cleared -- full frame will be analyzed")
        self._set_state(AppState.READY)

    def _on_clear_mask(self) -> None:
        self._video_panel.selector.clear_mask()
        self._video_panel._refresh_display()
        self._status.showMessage("Exclusion areas cleared")

    def _on_set_reference(self) -> None:
        self._reference_frame_num = self._current_frame_num
        self._status.showMessage(
            f"Reference frame set to {self._reference_frame_num}"
        )

    def _on_open_video_menu(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
        )
        if path:
            self._on_video_selected(path)

    def _on_start(self) -> None:
        config = self._controls.get_config()
        grid_size = config["grid_rows"]
        self._video_panel.set_grid_size(grid_size, grid_size)
        self._plots_panel.clear_data()

        self._worker = AnalysisWorker(
            config=config,
            video_path=self._video_path,
            camera_index=self._camera_index,
            roi=self._video_panel.selector.roi,
            mask=self._video_panel.selector.mask,
            reference_frame_num=self._reference_frame_num,
        )

        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.progress.connect(self._on_progress)
        self._worker.analysis_finished.connect(self._on_analysis_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.brightness_warning.connect(self._on_brightness_warning)

        self._worker.start()
        self._set_state(AppState.RUNNING)
        self._status.showMessage("Analysis running...")

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker.wait(5000)
        self._set_state(AppState.CONFIGURED)
        self._status.showMessage("Analysis stopped")

    def _on_frame_ready(
        self,
        frame_num: int,
        frame: np.ndarray,
        pixel_delta_e: np.ndarray,
        metrics: dict,
    ) -> None:
        self._current_frame_num = frame_num
        self._video_panel.update_frame(
            frame, pixel_delta_e if pixel_delta_e.size > 1 else None
        )
        row_avg = metrics.get("row_avg")
        col_avg = metrics.get("col_avg")
        self._plots_panel.append_data(
            metrics, metrics.get("timestamp", 0.0),
            row_avg=row_avg, col_avg=col_avg,
        )

    def _on_progress(self, current: int, total: int) -> None:
        self._controls.update_progress(current, total)
        if total > 0:
            pct = current / total * 100
            self._status.showMessage(
                f"Analyzing: {current}/{total} frames ({pct:.0f}%)"
            )
        else:
            self._status.showMessage(f"Analyzing: {current} frames (live)")

    def _on_analysis_finished(self) -> None:
        self._set_state(AppState.CONFIGURED)
        count = 0
        if self._worker and self._worker.engine:
            count = len(self._worker.engine.results)
        self._status.showMessage(
            f"Analysis complete. {count} frames processed."
        )

    def _on_error(self, msg: str) -> None:
        self._status.showMessage(f"Warning: {msg}")

    def _on_brightness_warning(self, msg: str) -> None:
        self._status.showMessage(f"WARNING: {msg}", 5000)

    def _on_export(self) -> None:
        if not self._worker or not self._worker.engine or not self._worker.engine.results:
            QMessageBox.warning(self, "Export", "No data to export.")
            return

        config = self._controls.get_config()
        fmt = config["export_format"]
        default_name = f"results.{fmt}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", default_name,
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)",
        )
        if path:
            results_snapshot = list(self._worker.engine.results)
            exporter = DataExporter()
            exporter.export(results_snapshot, path, fmt=fmt)
            self._status.showMessage(
                f"Exported {len(results_snapshot)} rows to {path}"
            )

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        event.accept()
