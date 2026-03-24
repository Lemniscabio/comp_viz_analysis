"""ROI rectangle selection and exclusion mask brush for the video panel."""
from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen


class InteractionMode(Enum):
    VIEW = auto()
    ROI = auto()
    MASK = auto()


class RoiSelector:
    """Manages ROI rectangle and exclusion mask state."""

    def __init__(self) -> None:
        self._roi: Optional[Tuple[int, int, int, int]] = None
        self._mask: Optional[np.ndarray] = None
        self._frame_size: Optional[Tuple[int, int]] = None
        self._display_size: Optional[Tuple[int, int]] = None
        self._mode = InteractionMode.VIEW
        self._brush_size = 20
        self._drawing = False
        self._dragging = False
        self._resizing = False
        self._resize_handle: Optional[str] = None  # "tl", "tr", "bl", "br", "t", "b", "l", "r"
        self._drag_offset: Optional[Tuple[int, int]] = None
        self._roi_start: Optional[QPoint] = None
        self._roi_current: Optional[QPoint] = None
        self._handle_size = 12  # pixels in display coords

    @property
    def roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self._roi

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    @property
    def mode(self) -> InteractionMode:
        return self._mode

    @mode.setter
    def mode(self, value: InteractionMode) -> None:
        self._mode = value
        self._drawing = False

    @property
    def brush_size(self) -> int:
        return self._brush_size

    def set_frame_size(self, width: int, height: int) -> None:
        self._frame_size = (width, height)
        if self._mask is None or self._mask.shape != (height, width):
            self._mask = np.ones((height, width), dtype=np.uint8)

    def set_display_size(self, width: int, height: int) -> None:
        self._display_size = (width, height)

    def clear_roi(self) -> None:
        self._roi = None
        self._roi_start = None
        self._roi_current = None

    def clear_mask(self) -> None:
        if self._frame_size:
            w, h = self._frame_size
            self._mask = np.ones((h, w), dtype=np.uint8)

    def _display_to_frame(self, point: QPoint) -> Tuple[int, int]:
        """Convert display coordinates to frame coordinates."""
        if not self._frame_size or not self._display_size:
            return point.x(), point.y()
        fw, fh = self._frame_size
        dw, dh = self._display_size
        x = int(point.x() * fw / dw)
        y = int(point.y() * fh / dh)
        return max(0, min(x, fw - 1)), max(0, min(y, fh - 1))

    def _get_roi_display_rect(self) -> Optional[Tuple[float, float, float, float]]:
        """Get ROI rectangle in display coordinates as (dx, dy, dw, dh)."""
        if not self._roi or not self._frame_size or not self._display_size:
            return None
        fw, fh = self._frame_size
        dw, dh = self._display_size
        x, y, w, h = self._roi
        return (x * dw / fw, y * dh / fh, w * dw / fw, h * dh / fh)

    def get_resize_handle(self, pos: QPoint) -> Optional[str]:
        """Check if pos is near a corner handle. Returns handle id or None."""
        rect = self._get_roi_display_rect()
        if not rect:
            return None
        dx, dy, dw, dh = rect
        px, py = pos.x(), pos.y()
        hs = self._handle_size

        if abs(px - dx) < hs and abs(py - dy) < hs:
            return "tl"
        if abs(px - (dx + dw)) < hs and abs(py - dy) < hs:
            return "tr"
        if abs(px - dx) < hs and abs(py - (dy + dh)) < hs:
            return "bl"
        if abs(px - (dx + dw)) < hs and abs(py - (dy + dh)) < hs:
            return "br"
        return None

    def _is_inside_roi_display(self, pos: QPoint) -> bool:
        """Check if a display-space point is inside the current ROI."""
        if not self._roi or not self._frame_size or not self._display_size:
            return False
        fw, fh = self._frame_size
        dw, dh = self._display_size
        x, y, w, h = self._roi
        dx = x * dw / fw
        dy = y * dh / fh
        dw_roi = w * dw / fw
        dh_roi = h * dh / fh
        return (dx <= pos.x() <= dx + dw_roi and dy <= pos.y() <= dy + dh_roi)

    def on_mouse_press(self, pos: QPoint) -> None:
        if self._mode == InteractionMode.MASK:
            self._drawing = True
            self._paint_mask(pos)
            return

        # Check for resize handle first (ROI or VIEW mode)
        if self._roi and self._mode in (InteractionMode.ROI, InteractionMode.VIEW):
            handle = self.get_resize_handle(pos)
            if handle:
                self._resizing = True
                self._resize_handle = handle
                return

        if self._mode == InteractionMode.ROI:
            if self._roi and self._is_inside_roi_display(pos):
                fx, fy = self._display_to_frame(pos)
                rx, ry, _, _ = self._roi
                self._drag_offset = (fx - rx, fy - ry)
                self._dragging = True
            else:
                self._roi_start = pos
                self._roi_current = pos
                self._drawing = True
        elif self._mode == InteractionMode.VIEW and self._roi:
            if self._is_inside_roi_display(pos):
                fx, fy = self._display_to_frame(pos)
                rx, ry, _, _ = self._roi
                self._drag_offset = (fx - rx, fy - ry)
                self._dragging = True

    def on_mouse_move(self, pos: QPoint) -> None:
        if self._resizing and self._roi and self._resize_handle:
            fx, fy = self._display_to_frame(pos)
            x, y, w, h = self._roi
            right, bottom = x + w, y + h

            handle = self._resize_handle
            if handle == "tl":
                x, y = min(fx, right - 10), min(fy, bottom - 10)
                w, h = right - x, bottom - y
            elif handle == "tr":
                y = min(fy, bottom - 10)
                w, h = max(10, fx - x), bottom - y
            elif handle == "bl":
                x = min(fx, right - 10)
                w, h = right - x, max(10, fy - y)
            elif handle == "br":
                w, h = max(10, fx - x), max(10, fy - y)

            if self._frame_size:
                fw, fh = self._frame_size
                x = max(0, x)
                y = max(0, y)
                w = min(w, fw - x)
                h = min(h, fh - y)

            self._roi = (x, y, w, h)
            return

        if self._dragging and self._roi and self._drag_offset:
            fx, fy = self._display_to_frame(pos)
            ox, oy = self._drag_offset
            x, y, w, h = self._roi
            new_x = fx - ox
            new_y = fy - oy
            if self._frame_size:
                fw, fh = self._frame_size
                new_x = max(0, min(new_x, fw - w))
                new_y = max(0, min(new_y, fh - h))
            self._roi = (new_x, new_y, w, h)
            return

        if not self._drawing:
            return
        if self._mode == InteractionMode.ROI:
            self._roi_current = pos
        elif self._mode == InteractionMode.MASK:
            self._paint_mask(pos)

    def on_mouse_release(self, pos: QPoint) -> None:
        if self._resizing:
            self._resizing = False
            self._resize_handle = None
            return
        if self._dragging:
            self._dragging = False
            self._drag_offset = None
            return
        if self._mode == InteractionMode.ROI and self._roi_start:
            x1, y1 = self._display_to_frame(self._roi_start)
            x2, y2 = self._display_to_frame(pos)
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if w > 5 and h > 5:
                self._roi = (x, y, w, h)
        self._drawing = False

    def on_wheel(self, delta: int) -> None:
        """Adjust brush size via scroll wheel."""
        if self._mode == InteractionMode.MASK:
            self._brush_size = max(5, min(100, self._brush_size + (delta // 120) * 5))

    def _paint_mask(self, pos: QPoint) -> None:
        """Paint exclusion area on the mask using NumPy vectorized circular brush."""
        if self._mask is None or not self._frame_size:
            return
        fx, fy = self._display_to_frame(pos)
        if self._display_size and self._frame_size:
            scale = self._frame_size[0] / self._display_size[0]
            radius = int(self._brush_size * scale / 2)
        else:
            radius = self._brush_size // 2

        h, w = self._mask.shape
        y_min = max(0, fy - radius)
        y_max = min(h, fy + radius)
        x_min = max(0, fx - radius)
        x_max = min(w, fx + radius)

        if y_max > y_min and x_max > x_min:
            yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
            circle = (xx - fx) ** 2 + (yy - fy) ** 2 <= radius ** 2
            self._mask[y_min:y_max, x_min:x_max][circle] = 0

    def draw_roi_overlay(self, painter: QPainter) -> None:
        """Draw ROI rectangle with corner/edge resize handles."""
        if self._roi and self._display_size and self._frame_size:
            fw, fh = self._frame_size
            dw, dh = self._display_size
            x, y, w, h = self._roi
            dx = int(x * dw / fw)
            dy = int(y * dh / fh)
            dw_roi = int(w * dw / fw)
            dh_roi = int(h * dh / fh)

            # Draw rectangle
            pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(dx, dy, dw_roi, dh_roi)

            # Draw resize handles at corners only
            hs = 5  # half handle size
            handle_pen = QPen(QColor(255, 255, 255), 1)
            painter.setPen(handle_pen)
            painter.setBrush(QColor(0, 255, 0))
            corners = [
                (dx, dy),
                (dx + dw_roi, dy),
                (dx, dy + dh_roi),
                (dx + dw_roi, dy + dh_roi),
            ]
            for cx, cy in corners:
                painter.drawRect(cx - hs, cy - hs, hs * 2, hs * 2)
        elif (self._drawing and self._mode == InteractionMode.ROI
              and self._roi_start and self._roi_current):
            pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            rect = QRect(self._roi_start, self._roi_current).normalized()
            painter.drawRect(rect)

    def draw_mask_overlay(
        self, painter: QPainter, display_width: int, display_height: int
    ) -> None:
        """Draw exclusion mask as a semi-transparent red overlay."""
        if self._mask is None:
            return
        excluded = (self._mask == 0)
        if not np.any(excluded):
            return
        h, w = self._mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[excluded] = [255, 0, 0, 80]
        img = QImage(overlay.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        scaled = img.scaled(
            display_width, display_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
        )
        painter.drawImage(0, 0, scaled)
