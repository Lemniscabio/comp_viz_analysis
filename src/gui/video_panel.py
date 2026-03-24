"""Video display panel with overlay support and mouse interaction."""
from __future__ import annotations

from typing import Optional, List

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QEvent
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.gui.heatmap_overlay import create_heatmap_overlay
from src.gui.roi_selector import InteractionMode, RoiSelector


class VideoPanel(QWidget):
    """Displays video frames with ROI, mask, grid, and heatmap overlays."""

    roi_selected = pyqtSignal(tuple)
    reference_frame_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(320, 240)
        self._label.setStyleSheet("background-color: #1a1a2e;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self._selector = RoiSelector()
        self._current_frame: Optional[np.ndarray] = None
        self._pixel_delta_e: Optional[np.ndarray] = None
        self._show_grid = False
        self._show_heatmap = False
        self._grid_rows = 5
        self._grid_cols = 5
        self._valid_cells: Optional[List[bool]] = None

        self._label.setMouseTracking(True)
        self._label.installEventFilter(self)

    @property
    def selector(self) -> RoiSelector:
        return self._selector

    def set_mode(self, mode: InteractionMode) -> None:
        self._selector.mode = mode

    def set_grid_visible(self, visible: bool) -> None:
        self._show_grid = visible
        self._refresh_display()

    def set_heatmap_visible(self, visible: bool) -> None:
        self._show_heatmap = visible
        self._refresh_display()

    def set_grid_size(self, rows: int, cols: int) -> None:
        self._grid_rows = rows
        self._grid_cols = cols

    def set_valid_cells(self, valid: List[bool]) -> None:
        self._valid_cells = valid

    def update_frame(
        self, frame_bgr: np.ndarray, pixel_delta_e: Optional[np.ndarray] = None
    ) -> None:
        """Update the displayed frame and optional delta-E data."""
        self._current_frame = frame_bgr.copy()
        self._pixel_delta_e = pixel_delta_e
        h, w = frame_bgr.shape[:2]
        self._selector.set_frame_size(w, h)
        self._refresh_display()

    def _refresh_display(self) -> None:
        if self._current_frame is None:
            return

        display = self._current_frame.copy()

        if self._show_heatmap and self._pixel_delta_e is not None:
            roi = self._selector.roi
            if roi:
                x, y, w, h = roi
                roi_heatmap = create_heatmap_overlay(
                    display[y : y + h, x : x + w], self._pixel_delta_e
                )
                display[y : y + h, x : x + w] = roi_heatmap
            else:
                display = create_heatmap_overlay(display, self._pixel_delta_e)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        label_size = self._label.size()
        pixmap = QPixmap.fromImage(qt_image).scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self._selector.set_display_size(pixmap.width(), pixmap.height())

        painter = QPainter(pixmap)
        self._selector.draw_roi_overlay(painter)
        self._selector.draw_mask_overlay(painter, pixmap.width(), pixmap.height())

        if self._show_grid:
            self._draw_grid(painter, pixmap.width(), pixmap.height())

        painter.end()
        self._label.setPixmap(pixmap)

    def _draw_grid(self, painter: QPainter, pw: int, ph: int) -> None:
        """Draw N x N grid lines on the display. Invalid cells shown in red."""
        cell_w = pw / self._grid_cols
        cell_h = ph / self._grid_rows

        if self._valid_cells:
            for i, valid in enumerate(self._valid_cells):
                if not valid:
                    r = i // self._grid_cols
                    c = i % self._grid_cols
                    painter.fillRect(
                        int(c * cell_w),
                        int(r * cell_h),
                        int(cell_w),
                        int(cell_h),
                        QColor(255, 0, 0, 40),
                    )

        pen = QPen(QColor(255, 255, 255, 120), 1, Qt.PenStyle.DotLine)
        painter.setPen(pen)
        for r in range(1, self._grid_rows):
            y = int(r * cell_h)
            painter.drawLine(0, y, pw, y)
        for c in range(1, self._grid_cols):
            x = int(c * cell_w)
            painter.drawLine(x, 0, x, ph)

    _HANDLE_CURSORS = {
        "tl": Qt.CursorShape.SizeFDiagCursor,
        "br": Qt.CursorShape.SizeFDiagCursor,
        "tr": Qt.CursorShape.SizeBDiagCursor,
        "bl": Qt.CursorShape.SizeBDiagCursor,
        "t": Qt.CursorShape.SizeVerCursor,
        "b": Qt.CursorShape.SizeVerCursor,
        "l": Qt.CursorShape.SizeHorCursor,
        "r": Qt.CursorShape.SizeHorCursor,
    }

    def _update_cursor(self, pos) -> None:
        """Update cursor based on position and mode."""
        if self._selector._resizing:
            handle = self._selector._resize_handle
            self._label.setCursor(self._HANDLE_CURSORS.get(handle, Qt.CursorShape.ArrowCursor))
            return
        if self._selector._dragging:
            self._label.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # Check resize handles first
        handle = self._selector.get_resize_handle(pos)
        if handle and self._selector.mode in (InteractionMode.ROI, InteractionMode.VIEW):
            self._label.setCursor(self._HANDLE_CURSORS[handle])
            return

        if self._selector.mode == InteractionMode.ROI:
            if self._selector._is_inside_roi_display(pos):
                self._label.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self._label.setCursor(Qt.CursorShape.CrossCursor)
        elif self._selector.mode == InteractionMode.MASK:
            self._label.setCursor(Qt.CursorShape.CrossCursor)
        elif self._selector.mode == InteractionMode.VIEW:
            if self._selector.roi and self._selector._is_inside_roi_display(pos):
                self._label.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self._label.setCursor(Qt.CursorShape.ArrowCursor)

    def eventFilter(self, obj, event) -> bool:
        if obj is self._label:
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.RightButton:
                    self.reference_frame_requested.emit()
                    return True
                self._selector.on_mouse_press(event.pos())
                self._update_cursor(event.pos())
                return True
            elif event.type() == QEvent.Type.MouseMove:
                self._selector.on_mouse_move(event.pos())
                self._update_cursor(event.pos())
                if (self._selector.mode != InteractionMode.VIEW
                        or self._selector._dragging
                        or self._selector._resizing):
                    self._refresh_display()
                return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                self._selector.on_mouse_release(event.pos())
                self._update_cursor(event.pos())
                if self._selector.roi:
                    self.roi_selected.emit(self._selector.roi)
                self._refresh_display()
                return True
            elif event.type() == QEvent.Type.Wheel:
                self._selector.on_wheel(event.angleDelta().y())
                return True
        return super().eventFilter(obj, event)
