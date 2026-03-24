"""Main application window: three-panel layout with menu bar."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget, QFileDialog, QMainWindow, QMessageBox, QPushButton,
    QSplitter, QStatusBar, QVBoxLayout, QWidget,
)

from src.core.export import DataExporter
from src.gui.analysis_worker import AnalysisWorker
from src.gui.controls_panel import AppState, ControlsPanel
from src.gui.plots_panel import PlotsPanel
from src.gui.roi_selector import InteractionMode
from src.gui.video_panel import VideoPanel


class MainWindow(QMainWindow):
    """Kineticolor main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kineticolor - Mixing Analysis")
        self.setMinimumSize(1200, 700)

        self._worker: Optional[AnalysisWorker] = None
        self._video_path: Optional[str] = None
        self._reference_frame_num = 0
        self._current_frame_num = 0
        self._state = AppState.IDLE

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

    def _setup_ui(self) -> None:
        # Plots as central widget
        self._plots_panel = PlotsPanel()
        self.setCentralWidget(self._plots_panel)

        # Video panel as a dockable sidebar (left)
        self._video_panel = VideoPanel()
        self._video_dock = QDockWidget("Video", self)
        self._video_dock.setWidget(self._video_panel)
        self._video_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self._video_dock.setTitleBarWidget(QWidget())  # hide title bar
        self._video_dock.setMinimumWidth(300)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._video_dock)

        # Controls as a dockable panel (bottom)
        self._controls = ControlsPanel()
        self._controls_dock = QDockWidget("Controls", self)
        self._controls_dock.setWidget(self._controls)
        self._controls_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._controls_dock)

        # Toolbar with sidebar toggle
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        self._btn_sidebar = QPushButton("Toggle Video")
        self._btn_sidebar.setCheckable(True)
        self._btn_sidebar.setChecked(True)
        self._btn_sidebar.setToolTip("Show/hide the video panel")
        self._btn_sidebar.setFixedSize(100, 28)
        self._btn_sidebar.clicked.connect(self._toggle_video_panel)
        # Sync state when dock is closed via its own X button
        self._video_dock.visibilityChanged.connect(
            lambda visible: self._btn_sidebar.setChecked(visible)
        )
        toolbar.addWidget(self._btn_sidebar)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Open a video file to begin")

    def _toggle_video_panel(self) -> None:
        visible = self._btn_sidebar.isChecked()
        self._video_dock.setVisible(visible)

    def _setup_menu(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open Video", self)
        open_action.triggered.connect(self._on_open_video_menu)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Export menu
        export_menu = menubar.addMenu("Export")
        export_action = QAction("Export Data", self)
        export_action.triggered.connect(self._on_export)
        export_menu.addAction(export_action)

        # View menu
        view_menu = menubar.addMenu("View")

        video_toggle = self._video_dock.toggleViewAction()
        video_toggle.setText("Show Video Panel")
        view_menu.addAction(video_toggle)

        view_menu.addSeparator()

        grid_toggle = QAction("Toggle Grid Overlay", self)
        grid_toggle.setCheckable(True)
        grid_toggle.triggered.connect(self._video_panel.set_grid_visible)
        view_menu.addAction(grid_toggle)

        heatmap_toggle = QAction("Toggle Heatmap Overlay", self)
        heatmap_toggle.setCheckable(True)
        heatmap_toggle.triggered.connect(self._video_panel.set_heatmap_visible)
        view_menu.addAction(heatmap_toggle)

        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(
            lambda: QMessageBox.about(
                self,
                "About Kineticolor",
                "Kineticolor - Computer Vision Mixing Analysis\n\n"
                "Real-time kinetic analysis of mixing phenomena\n"
                "in reaction vessels using 6 complementary metrics.\n\n"
                "Workflow:\n"
                "1. Open a video file\n"
                "2. Select ROI (region of interest)\n"
                "3. Optionally draw mask to exclude areas\n"
                "4. Click Start Analysis\n"
                "5. Export results when done",
            )
        )
        help_menu.addAction(about_action)

    def _connect_signals(self) -> None:
        ctrl = self._controls
        ctrl.video_selected.connect(self._on_video_selected)
        ctrl.start_requested.connect(self._on_start)
        ctrl.stop_requested.connect(self._on_stop)
        ctrl.export_requested.connect(self._on_export)
        ctrl.roi_mode_requested.connect(
            lambda: self._video_panel.set_mode(InteractionMode.ROI)
        )
        ctrl.mask_mode_requested.connect(
            lambda: self._video_panel.set_mode(InteractionMode.MASK)
        )
        ctrl.view_mode_requested.connect(
            lambda: self._video_panel.set_mode(InteractionMode.VIEW)
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
        self._controls.reset_for_new_video()
        self._plots_panel.clear_data()
        self._status.showMessage(
            f"Video loaded: {Path(path).name} -- Select ROI or click Start Analysis"
        )

        import cv2

        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        if ret:
            self._video_panel.update_frame(frame)
            self._video_dock.setVisible(True)
        cap.release()

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
        if self._video_path:
            self._set_state(AppState.READY)

    def _on_clear_mask(self) -> None:
        self._video_panel.selector.clear_mask()
        self._video_panel._refresh_display()
        self._status.showMessage("Mask cleared")

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

        # Allow start without explicit ROI (uses full frame)
        roi = self._video_panel.selector.roi
        mask = self._video_panel.selector.mask

        self._worker = AnalysisWorker(
            config=config,
            video_path=self._video_path,
            roi=roi,
            mask=mask,
            reference_frame_num=self._reference_frame_num,
        )

        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.progress.connect(self._on_progress)
        self._worker.analysis_finished.connect(self._on_analysis_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.brightness_warning.connect(self._on_brightness_warning)

        self._worker.start()
        self._video_panel.set_interaction_locked(True)
        self._controls.mark_analysis_started()
        self._set_state(AppState.RUNNING)
        self._status.showMessage("Analysis running...")

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker.wait(5000)
        self._video_panel.set_interaction_locked(False)
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
            self._status.showMessage(f"Analyzing: {current} frames")

    def _on_analysis_finished(self) -> None:
        self._video_panel.set_interaction_locked(False)
        self._set_state(AppState.CONFIGURED)
        count = 0
        if self._worker and self._worker.engine:
            count = len(self._worker.engine.results)
        self._status.showMessage(
            f"Analysis complete. {count} frames processed. Click 'Export Data' to save results."
        )

    def _on_error(self, msg: str) -> None:
        self._status.showMessage(f"Warning: {msg}")

    def _on_brightness_warning(self, msg: str) -> None:
        self._status.showMessage(f"WARNING: {msg}", 5000)

    def _on_export(self) -> None:
        if not self._worker or not self._worker.engine or not self._worker.engine.results:
            QMessageBox.warning(self, "Export", "No data to export. Run analysis first.")
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
