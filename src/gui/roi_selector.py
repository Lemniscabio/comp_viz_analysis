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
        self._roi_start: Optional[QPoint] = None
        self._roi_current: Optional[QPoint] = None

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

    def on_mouse_press(self, pos: QPoint) -> None:
        if self._mode == InteractionMode.ROI:
            self._roi_start = pos
            self._roi_current = pos
            self._drawing = True
        elif self._mode == InteractionMode.MASK:
            self._drawing = True
            self._paint_mask(pos)

    def on_mouse_move(self, pos: QPoint) -> None:
        if not self._drawing:
            return
        if self._mode == InteractionMode.ROI:
            self._roi_current = pos
        elif self._mode == InteractionMode.MASK:
            self._paint_mask(pos)

    def on_mouse_release(self, pos: QPoint) -> None:
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
        """Draw ROI rectangle on the display."""
        if self._roi and self._display_size and self._frame_size:
            fw, fh = self._frame_size
            dw, dh = self._display_size
            x, y, w, h = self._roi
            dx = int(x * dw / fw)
            dy = int(y * dh / fh)
            dw_roi = int(w * dw / fw)
            dh_roi = int(h * dh / fh)
            pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(dx, dy, dw_roi, dh_roi)
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
