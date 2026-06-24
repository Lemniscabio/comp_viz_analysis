"""Interactive ROI drawing widget for the calibration app.

Displays a live camera feed (or static frame) and lets the user draw,
move, and resize up to 4 color-coded bounding boxes.

All coordinates returned and stored are in *camera pixel space*, not
widget pixel space. The canvas handles the mapping internally via the
letterbox transform computed each time the widget is resized or a
new frame is set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QImage, QPixmap,
)
import numpy as np

from ..common.constants import (
    VESSEL_COLORS, VESSEL_LABELS, MAX_VESSELS,
    MIN_ROI_WIDTH, MIN_ROI_HEIGHT,
)

# Handle size (px in widget space) for corner/edge drag handles
_HANDLE_RADIUS = 5
_HANDLE_HALF = _HANDLE_RADIUS

# Cursor zones on a rectangle
_ZONE_BODY = "body"
_ZONE_TL = "tl"
_ZONE_TR = "tr"
_ZONE_BL = "bl"
_ZONE_BR = "br"
_ZONE_T = "t"
_ZONE_B = "b"
_ZONE_L = "l"
_ZONE_R = "r"


@dataclass
class _Roi:
    vessel_id: int
    # camera-coordinate rect (x, y, w, h) — all ints
    x: int = 0
    y: int = 0
    w: int = 200
    h: int = 200

    @property
    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    def qrect_cam(self) -> QRect:
        return QRect(self.x, self.y, self.w, self.h)


class RoiCanvas(QWidget):
    """Widget that shows a live camera frame with draggable ROI rectangles.

    Signals:
        rois_changed: emitted whenever any ROI is added, moved, resized, or deleted.
    """

    rois_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)

        self._pixmap: QPixmap | None = None
        self._cam_w: int = 1280
        self._cam_h: int = 720

        self._rois: list[_Roi] = []
        self._max_rois: int = MAX_VESSELS

        # Drawing state
        self._drawing: bool = False
        self._draw_start_widget: QPoint | None = None

        # Drag state
        self._drag_roi_idx: int = -1
        self._drag_zone: str = ""
        self._drag_start_widget: QPoint | None = None
        self._drag_start_cam_rect: tuple[int, int, int, int] | None = None  # x,y,w,h

        # Letterbox transform (set in _update_transform)
        self._lx: int = 0  # left offset of image in widget
        self._ly: int = 0  # top offset of image in widget
        self._scale: float = 1.0

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        """Update the displayed frame from a BGR numpy array."""
        h, w = frame_bgr.shape[:2]
        self._cam_w, self._cam_h = w, h
        rgb = frame_bgr[:, :, ::-1].copy()
        qi = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qi)
        self._update_transform()
        self.update()

    def set_max_rois(self, n: int) -> None:
        """Set maximum number of ROIs (vessel count). Clears existing ROIs."""
        self._max_rois = max(1, min(n, MAX_VESSELS))
        self._rois.clear()
        self.update()
        self.rois_changed.emit()

    def get_rois(self) -> list[dict]:
        """Return current ROIs as list of vessel dicts (vessel_id, roi=[x,y,w,h])."""
        return [
            {"vessel_id": r.vessel_id, "roi": r.as_list}
            for r in self._rois
        ]

    def set_rois(self, rois: list[dict]) -> None:
        """Restore ROIs from a list of {vessel_id, roi} dicts."""
        self._rois = [
            _Roi(
                vessel_id=r["vessel_id"],
                x=int(r["roi"][0]),
                y=int(r["roi"][1]),
                w=int(r["roi"][2]),
                h=int(r["roi"][3]),
            )
            for r in rois
        ]
        self.update()
        self.rois_changed.emit()

    def roi_count(self) -> int:
        return len(self._rois)

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def _update_transform(self) -> None:
        if self._pixmap is None:
            return
        ww, wh = self.width(), self.height()
        pw, ph = self._cam_w, self._cam_h
        scale_x = ww / pw
        scale_y = wh / ph
        self._scale = min(scale_x, scale_y)
        img_w = int(pw * self._scale)
        img_h = int(ph * self._scale)
        self._lx = (ww - img_w) // 2
        self._ly = (wh - img_h) // 2

    def _to_cam(self, wp: QPoint) -> tuple[int, int]:
        """Convert widget point to camera pixel coordinates."""
        cx = int((wp.x() - self._lx) / self._scale)
        cy = int((wp.y() - self._ly) / self._scale)
        return cx, cy

    def _to_widget(self, cx: int, cy: int) -> tuple[int, int]:
        """Convert camera pixel coordinates to widget coordinates."""
        wx = int(cx * self._scale + self._lx)
        wy = int(cy * self._scale + self._ly)
        return wx, wy

    def _cam_rect_to_widget(self, roi: _Roi) -> QRect:
        x1w, y1w = self._to_widget(roi.x, roi.y)
        x2w, y2w = self._to_widget(roi.x + roi.w, roi.y + roi.h)
        return QRect(x1w, y1w, x2w - x1w, y2w - y1w)

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def _handle_rects_widget(self, wr: QRect) -> dict[str, QRect]:
        """Return widget-space QRects for each resize handle on a rectangle."""
        cx = wr.left() + wr.width() // 2
        cy = wr.top() + wr.height() // 2

        def hrect(x: int, y: int) -> QRect:
            return QRect(x - _HANDLE_HALF, y - _HANDLE_HALF, _HANDLE_RADIUS * 2, _HANDLE_RADIUS * 2)

        return {
            _ZONE_TL: hrect(wr.left(), wr.top()),
            _ZONE_TR: hrect(wr.right(), wr.top()),
            _ZONE_BL: hrect(wr.left(), wr.bottom()),
            _ZONE_BR: hrect(wr.right(), wr.bottom()),
            _ZONE_T:  hrect(cx, wr.top()),
            _ZONE_B:  hrect(cx, wr.bottom()),
            _ZONE_L:  hrect(wr.left(), cy),
            _ZONE_R:  hrect(wr.right(), cy),
        }

    def _hit_zone(self, pos: QPoint, roi_idx: int) -> str:
        """Return the zone the pointer is in for a given ROI."""
        wr = self._cam_rect_to_widget(self._rois[roi_idx])
        handles = self._handle_rects_widget(wr)
        for zone, rect in handles.items():
            if rect.contains(pos):
                return zone
        if wr.contains(pos):
            return _ZONE_BODY
        return ""

    def _find_hit_roi(self, pos: QPoint) -> tuple[int, str]:
        """Return (roi_index, zone) for the topmost ROI the pointer is over."""
        # Iterate in reverse so last-drawn (topmost) is checked first
        for i in reversed(range(len(self._rois))):
            zone = self._hit_zone(pos, i)
            if zone:
                return i, zone
        return -1, ""

    # ------------------------------------------------------------------
    # Qt event handlers
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        self._update_transform()
        super().resizeEvent(event)

    def mousePressEvent(self, event) -> None:
        pos = event.pos()

        if event.button() == Qt.MouseButton.RightButton:
            idx, _ = self._find_hit_roi(pos)
            if idx >= 0:
                self._rois.pop(idx)
                self.update()
                self.rois_changed.emit()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        idx, zone = self._find_hit_roi(pos)
        if idx >= 0:
            self._drag_roi_idx = idx
            self._drag_zone = zone
            self._drag_start_widget = pos
            roi = self._rois[idx]
            self._drag_start_cam_rect = (roi.x, roi.y, roi.w, roi.h)
        else:
            # Start drawing a new rectangle if quota not reached
            if len(self._rois) < self._max_rois:
                cx, cy = self._to_cam(pos)
                if 0 <= cx < self._cam_w and 0 <= cy < self._cam_h:
                    self._drawing = True
                    self._draw_start_widget = pos

    def mouseMoveEvent(self, event) -> None:
        pos = event.pos()

        if self._drawing and self._draw_start_widget:
            self.update()
            self._update_cursor_for(pos)
            return

        if self._drag_roi_idx >= 0 and self._drag_start_widget and self._drag_start_cam_rect:
            dx_w = pos.x() - self._drag_start_widget.x()
            dy_w = pos.y() - self._drag_start_widget.y()
            dx_c = int(dx_w / self._scale)
            dy_c = int(dy_w / self._scale)
            ox, oy, ow, oh = self._drag_start_cam_rect
            roi = self._rois[self._drag_roi_idx]
            self._apply_drag(roi, self._drag_zone, ox, oy, ow, oh, dx_c, dy_c)
            self.update()
            return

        # Hover: update cursor
        self._update_cursor_for(pos)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self._drawing and self._draw_start_widget:
            self._drawing = False
            start_cam = self._to_cam(self._draw_start_widget)
            end_cam = self._to_cam(event.pos())
            self._finish_draw(start_cam, end_cam)
            self._draw_start_widget = None
            return

        if self._drag_roi_idx >= 0:
            self._drag_roi_idx = -1
            self._drag_zone = ""
            self._drag_start_widget = None
            self._drag_start_cam_rect = None
            self.rois_changed.emit()

    # ------------------------------------------------------------------
    # Internal draw/drag helpers
    # ------------------------------------------------------------------

    def _apply_drag(
        self, roi: _Roi, zone: str,
        ox: int, oy: int, ow: int, oh: int,
        dx: int, dy: int,
    ) -> None:
        """Apply drag delta to the ROI according to the active zone."""
        new_x, new_y, new_w, new_h = ox, oy, ow, oh

        if zone == _ZONE_BODY:
            new_x = ox + dx
            new_y = oy + dy
        elif zone == _ZONE_TL:
            new_x = ox + dx; new_y = oy + dy
            new_w = ow - dx; new_h = oh - dy
        elif zone == _ZONE_TR:
            new_y = oy + dy
            new_w = ow + dx; new_h = oh - dy
        elif zone == _ZONE_BL:
            new_x = ox + dx
            new_w = ow - dx; new_h = oh + dy
        elif zone == _ZONE_BR:
            new_w = ow + dx; new_h = oh + dy
        elif zone == _ZONE_T:
            new_y = oy + dy; new_h = oh - dy
        elif zone == _ZONE_B:
            new_h = oh + dy
        elif zone == _ZONE_L:
            new_x = ox + dx; new_w = ow - dx
        elif zone == _ZONE_R:
            new_w = ow + dx

        # Enforce minimum size and clamp to frame
        new_w = max(MIN_ROI_WIDTH, new_w)
        new_h = max(MIN_ROI_HEIGHT, new_h)
        new_x = max(0, min(new_x, self._cam_w - new_w))
        new_y = max(0, min(new_y, self._cam_h - new_h))
        new_w = min(new_w, self._cam_w - new_x)
        new_h = min(new_h, self._cam_h - new_y)

        roi.x, roi.y, roi.w, roi.h = new_x, new_y, new_w, new_h

    def _finish_draw(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        x1, y1 = start
        x2, y2 = end
        x = max(0, min(x1, x2))
        y = max(0, min(y1, y2))
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        w = min(w, self._cam_w - x)
        h = min(h, self._cam_h - y)

        if w < MIN_ROI_WIDTH or h < MIN_ROI_HEIGHT:
            return  # too small — ignore

        # Determine next vessel ID (smallest unused in 1-4)
        used_ids = {r.vessel_id for r in self._rois}
        next_id = next((i for i in range(1, MAX_VESSELS + 1) if i not in used_ids), None)
        if next_id is None:
            return

        self._rois.append(_Roi(vessel_id=next_id, x=x, y=y, w=w, h=h))
        self._renumber_rois()
        self.update()
        self.rois_changed.emit()

    def _renumber_rois(self) -> None:
        """Sort ROIs left-to-right by center x, reassign IDs 1,2,3,4."""
        self._rois.sort(key=lambda r: r.x + r.w // 2)
        for i, roi in enumerate(self._rois, start=1):
            roi.vessel_id = i

    def _update_cursor_for(self, pos: QPoint) -> None:
        idx, zone = self._find_hit_roi(pos)
        cursor_map = {
            _ZONE_BODY: Qt.CursorShape.SizeAllCursor,
            _ZONE_TL:   Qt.CursorShape.SizeFDiagCursor,
            _ZONE_BR:   Qt.CursorShape.SizeFDiagCursor,
            _ZONE_TR:   Qt.CursorShape.SizeBDiagCursor,
            _ZONE_BL:   Qt.CursorShape.SizeBDiagCursor,
            _ZONE_T:    Qt.CursorShape.SizeVerCursor,
            _ZONE_B:    Qt.CursorShape.SizeVerCursor,
            _ZONE_L:    Qt.CursorShape.SizeHorCursor,
            _ZONE_R:    Qt.CursorShape.SizeHorCursor,
        }
        self.setCursor(cursor_map.get(zone, Qt.CursorShape.CrossCursor))

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor("#1a1a1a"))

        # Camera frame
        if self._pixmap:
            img_w = int(self._cam_w * self._scale)
            img_h = int(self._cam_h * self._scale)
            painter.drawPixmap(self._lx, self._ly, img_w, img_h, self._pixmap)

        # In-progress draw rubber-band
        if self._drawing and self._draw_start_widget:
            cur = self.mapFromGlobal(self.cursor().pos())
            x = min(self._draw_start_widget.x(), cur.x())
            y = min(self._draw_start_widget.y(), cur.y())
            w = abs(cur.x() - self._draw_start_widget.x())
            h = abs(cur.y() - self._draw_start_widget.y())
            # Determine next color
            used_ids = {r.vessel_id for r in self._rois}
            next_id = next((i for i in range(1, MAX_VESSELS + 1) if i not in used_ids), 1)
            color = QColor(VESSEL_COLORS.get(next_id, "#ffffff"))
            pen = QPen(color, 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(x, y, w, h)

        # Draw ROIs
        for roi in self._rois:
            self._paint_roi(painter, roi)

        painter.end()

    def _paint_roi(self, painter: QPainter, roi: _Roi) -> None:
        color = QColor(VESSEL_COLORS.get(roi.vessel_id, "#ffffff"))
        wr = self._cam_rect_to_widget(roi)

        # Border
        pen = QPen(color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(wr)

        # Handles
        handle_color = color.lighter(130)
        painter.setBrush(QBrush(handle_color))
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        for hrect in self._handle_rects_widget(wr).values():
            painter.drawRect(hrect)

        # Label pill
        label = VESSEL_LABELS.get(roi.vessel_id, f"V{roi.vessel_id}")
        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        text_w = fm.horizontalAdvance(label) + 8
        text_h = fm.height() + 4
        pill = QRect(wr.left(), wr.top() - text_h, text_w, text_h)
        # clamp pill inside widget
        if pill.top() < 0:
            pill.moveTop(wr.top())
        pill_color = QColor(VESSEL_COLORS.get(roi.vessel_id, "#ffffff"))
        pill_color.setAlphaF(0.85)
        painter.setBrush(QBrush(pill_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(pill)
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, label)
