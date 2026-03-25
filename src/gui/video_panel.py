"""Video display panel with overlay support and mouse interaction."""
from __future__ import annotations

from typing import Optional, List

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QEvent
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

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
        self._label.setStyleSheet("background-color: #1a1a2e;")

        # Scroll area wrapping the label — enables pan when zoomed
        self._scroll = QScrollArea()
        self._scroll.setWidget(self._label)
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setStyleSheet("QScrollArea { background-color: #1a1a2e; border: none; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll, stretch=1)

        # Info panel below video
        self._info_label = QLabel("")
        self._info_label.setStyleSheet(
            "color: #cccccc; background-color: #2a2a3e; padding: 6px; font-size: 11px;"
        )
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        # Zoom controls
        zoom_row = QHBoxLayout()
        self._btn_zoom_out = QPushButton("-")
        self._btn_zoom_in = QPushButton("+")
        self._btn_zoom_fit = QPushButton("Fit")
        for btn in [self._btn_zoom_out, self._btn_zoom_in, self._btn_zoom_fit]:
            btn.setFixedSize(36, 24)
        self._btn_zoom_out.setToolTip("Zoom out")
        self._btn_zoom_in.setToolTip("Zoom in")
        self._btn_zoom_fit.setToolTip("Fit to panel")
        zoom_row.addWidget(self._btn_zoom_out)
        zoom_row.addWidget(self._btn_zoom_in)
        zoom_row.addWidget(self._btn_zoom_fit)
        self._zoom_label = QLabel("Fit")
        self._zoom_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        zoom_row.addWidget(self._zoom_label)
        zoom_row.addStretch()
        layout.addLayout(zoom_row)

        self._selector = RoiSelector()
        self._interaction_locked = False
        self._pixmap_offset_x = 0
        self._pixmap_offset_y = 0
        self._zoom_factor = 1.0
        self._current_frame: Optional[np.ndarray] = None
        self._pixel_delta_e: Optional[np.ndarray] = None
        self._show_grid = False
        self._show_heatmap = False
        self._grid_rows = 5
        self._grid_cols = 5
        self._valid_cells: Optional[List[bool]] = None

        self._label.setMouseTracking(True)
        self._label.installEventFilter(self)

        self._btn_zoom_in.clicked.connect(lambda: self._set_zoom(self._zoom_factor * 1.25))
        self._btn_zoom_out.clicked.connect(lambda: self._set_zoom(self._zoom_factor / 1.25))
        self._btn_zoom_fit.clicked.connect(lambda: self._set_zoom(1.0))

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

    def _set_zoom(self, factor: float) -> None:
        self._zoom_factor = max(0.25, min(4.0, factor))
        if self._zoom_factor == 1.0:
            self._zoom_label.setText("Fit")
        else:
            self._zoom_label.setText(f"{self._zoom_factor:.0%}")
        self._refresh_display()

    def _update_info(self) -> None:
        """Update the info panel with current state."""
        lines = []
        if self._current_frame is not None:
            h, w = self._current_frame.shape[:2]
            lines.append(f"Video: {w} x {h} px")

        roi = self._selector.roi
        if roi:
            x, y, rw, rh = roi
            lines.append(f"ROI: {rw} x {rh} px at ({x}, {y})")
            roi_pixels = rw * rh
            lines.append(f"ROI pixels: {roi_pixels:,}")
        else:
            if self._current_frame is not None:
                h, w = self._current_frame.shape[:2]
                lines.append(f"ROI: Full frame ({w * h:,} px)")

        raw_mask = self._selector._mask
        if raw_mask is not None and roi:
            x, y, rw, rh = roi
            roi_mask = raw_mask[y:y+rh, x:x+rw]
            masked_count = int(np.sum(roi_mask == 0))
            total = roi_mask.size
            if masked_count > 0:
                pct = masked_count / total * 100
                lines.append(f"Masked: {masked_count:,} / {total:,} px ({pct:.1f}%)")
                valid_px = total - masked_count
                lines.append(f"Analyzing: {valid_px:,} px ({100-pct:.1f}%)")

        self._info_label.setText("\n".join(lines) if lines else "")

    def update_frame(
        self, frame_bgr: np.ndarray, pixel_delta_e: Optional[np.ndarray] = None
    ) -> None:
        """Update the displayed frame and optional delta-E data."""
        self._current_frame = frame_bgr.copy()
        self._pixel_delta_e = pixel_delta_e
        h, w = frame_bgr.shape[:2]
        self._selector.set_frame_size(w, h)
        self._update_info()
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

        from PyQt6.QtCore import QSize
        viewport_size = self._scroll.viewport().size()

        if self._zoom_factor == 1.0:
            # Fit to viewport
            target_size = viewport_size
        else:
            zoomed_w = int(w * self._zoom_factor)
            zoomed_h = int(h * self._zoom_factor)
            target_size = QSize(zoomed_w, zoomed_h)

        pixmap = QPixmap.fromImage(qt_image).scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self._selector.set_display_size(pixmap.width(), pixmap.height())

        # When fitting, center in viewport. When zoomed, label resizes for scrolling.
        if self._zoom_factor == 1.0:
            self._pixmap_offset_x = (viewport_size.width() - pixmap.width()) // 2
            self._pixmap_offset_y = (viewport_size.height() - pixmap.height()) // 2
        else:
            self._pixmap_offset_x = 0
            self._pixmap_offset_y = 0

        painter = QPainter(pixmap)
        self._selector.draw_roi_overlay(painter)
        self._selector.draw_mask_overlay(painter, pixmap.width(), pixmap.height())

        if self._show_grid:
            self._draw_grid(painter, pixmap.width(), pixmap.height())

        painter.end()
        self._label.setPixmap(pixmap)
        self._label.resize(pixmap.size())

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

    def _label_to_pixmap_pos(self, pos) -> QPoint:
        """Convert label coordinates to pixmap coordinates."""
        from PyQt6.QtCore import QPoint as QP
        # When zoomed, label fills the pixmap so offset is 0
        # When fitting, subtract the centering offset
        return QP(pos.x() - self._pixmap_offset_x, pos.y() - self._pixmap_offset_y)

    _HANDLE_CURSORS = {
        "tl": Qt.CursorShape.SizeFDiagCursor,
        "br": Qt.CursorShape.SizeFDiagCursor,
        "tr": Qt.CursorShape.SizeBDiagCursor,
        "bl": Qt.CursorShape.SizeBDiagCursor,
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
            self._label.setCursor(Qt.CursorShape.ArrowCursor)
        elif self._selector.mode == InteractionMode.VIEW:
            if self._selector.roi and self._selector._is_inside_roi_display(pos):
                self._label.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self._label.setCursor(Qt.CursorShape.ArrowCursor)

    def set_interaction_locked(self, locked: bool) -> None:
        """Lock/unlock mouse interaction (disable during analysis)."""
        self._interaction_locked = locked
        if locked:
            self._label.setCursor(Qt.CursorShape.ArrowCursor)

    def eventFilter(self, obj, event) -> bool:
        if self._interaction_locked:
            return super().eventFilter(obj, event)
        if obj is self._label:
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.RightButton:
                    self.reference_frame_requested.emit()
                    return True
                ppos = self._label_to_pixmap_pos(event.pos())
                self._selector.on_mouse_press(ppos)
                self._update_cursor(ppos)
                return True
            elif event.type() == QEvent.Type.MouseMove:
                ppos = self._label_to_pixmap_pos(event.pos())
                self._selector.on_mouse_move(ppos)
                self._update_cursor(ppos)
                if (self._selector.mode != InteractionMode.VIEW
                        or self._selector._dragging
                        or self._selector._resizing):
                    self._refresh_display()
                return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                ppos = self._label_to_pixmap_pos(event.pos())
                self._selector.on_mouse_release(ppos)
                self._update_cursor(ppos)
                if self._selector.roi:
                    self.roi_selected.emit(self._selector.roi)
                self._update_info()
                self._refresh_display()
                return True
            elif event.type() == QEvent.Type.Wheel:
                self._selector.on_wheel(event.angleDelta().y())
                return True
        return super().eventFilter(obj, event)
