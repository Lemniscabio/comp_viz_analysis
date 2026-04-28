"""Mixing-time results panel: T_mix,95 plus per-component breakdown."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from src.core.mixing_time import MixingTimeResult


def _fmt(x) -> str:
    """Format a possibly-NaN/None float as 'X.XX s' or em-dash."""
    if x is None:
        return "—"
    try:
        if x != x:  # NaN
            return "—"
        return f"{float(x):.2f} s"
    except (TypeError, ValueError):
        return "—"


class MixingResultsPanel(QWidget):
    """Docked panel that displays the latest MixingTimeResult."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        self._title = QLabel("Mixing Time")
        self._title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self._title)

        self._main = QLabel("T_mix,95 = —")
        self._main.setStyleSheet("font-size: 22px; color: #2d7d46;")
        layout.addWidget(self._main)

        self._sub = QLabel("T_mix,90 = —    T_mix,99 = —")
        layout.addWidget(self._sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        grid = QGridLayout()
        layout.addLayout(grid)
        self._labels = {}
        rows = [
            ("Bulk Delta-E T95",        "t_deltaE_95"),
            ("Spatial T95",             "t_spatial_95"),
            ("Texture T95",             "t_texture_95"),
            ("Contact T95",             "t_contact_95"),
            ("Contrast T95",            "t_contrast_95"),
            ("Cell slow-region T95",    "t_cell_95"),
            ("Cell variance T95",       "t_variance_95"),
        ]
        for r, (label, key) in enumerate(rows):
            grid.addWidget(QLabel(label), r, 0)
            v = QLabel("—")
            grid.addWidget(v, r, 1)
            self._labels[key] = v

        self._status = QLabel("status: —")
        self._status.setStyleSheet("color: gray;")
        layout.addWidget(self._status)
        self._confidence = QLabel("confidence: —")
        layout.addWidget(self._confidence)
        self._t_start = QLabel("t_start: —")
        layout.addWidget(self._t_start)
        layout.addStretch()

    def clear(self) -> None:
        self._main.setText("T_mix,95 = —")
        self._sub.setText("T_mix,90 = —    T_mix,99 = —")
        for v in self._labels.values():
            v.setText("—")
        self._status.setText("status: —")
        self._confidence.setText("confidence: —")
        self._confidence.setStyleSheet("")
        self._t_start.setText("t_start: —")

    def set_result(self, r: MixingTimeResult) -> None:
        self._main.setText(f"T_mix,95 = {_fmt(r.t_mix.get(0.95))}")
        self._sub.setText(
            f"T_mix,90 = {_fmt(r.t_mix.get(0.90))}    "
            f"T_mix,99 = {_fmt(r.t_mix.get(0.99))}"
        )
        L = 0.95
        m = {
            "t_deltaE_95":   r.t_deltaE.get(L),
            "t_spatial_95":  r.t_spatial.get(L),
            "t_texture_95":  r.t_texture.get(L),
            "t_contact_95":  r.t_contact.get(L),
            "t_contrast_95": r.t_contrast.get(L),
            "t_cell_95":     r.t_cell.get(L),
            "t_variance_95": r.t_variance.get(L),
        }
        for k, v in m.items():
            self._labels[k].setText(_fmt(v))
        self._status.setText(f"status: {r.status}")
        color = {"high": "#2d7d46", "medium": "#cc8400", "low": "#c0392b"}.get(
            r.confidence, "gray"
        )
        self._confidence.setText(f"confidence: {r.confidence}")
        self._confidence.setStyleSheet(f"color: {color};")
        self._t_start.setText(f"t_start: {r.t_start_s:.2f} s")
