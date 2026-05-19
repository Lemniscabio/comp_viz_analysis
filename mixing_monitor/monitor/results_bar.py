"""Horizontal status bar showing all 4 vessel states."""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt6.QtGui import QFont

from ..common.constants import VESSEL_COLORS, VESSEL_LABELS
from .vessel_card import (
    STATE_DISABLED, STATE_IDLE, STATE_ARMED, STATE_MIXING, STATE_COMPLETE,
)

_GRAY = "#6b7280"


class ResultsBar(QWidget):
    """Bottom status bar with one label per vessel slot (always 4 slots)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(36)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(24)

        self._labels: dict[int, QLabel] = {}
        font = QFont("Segoe UI", 10)
        for vid in range(1, 5):
            lbl = QLabel()
            lbl.setFont(font)
            self._labels[vid] = lbl
            layout.addWidget(lbl)
            if vid < 4:
                sep = QLabel("|")
                sep.setStyleSheet(f"color: {_GRAY};")
                layout.addWidget(sep)
        layout.addStretch()

        # Default: all disabled
        for vid in range(1, 5):
            self.update_vessel(vid, STATE_DISABLED)

    def update_vessel(
        self,
        vessel_id: int,
        state: str,
        elapsed: float | None = None,
        mean_a: float | None = None,
        mixing_time: float | None = None,
    ) -> None:
        lbl = self._labels.get(vessel_id)
        if lbl is None:
            return
        label = VESSEL_LABELS.get(vessel_id, f"V{vessel_id}")
        color = VESSEL_COLORS.get(vessel_id, "#ffffff")

        if state == STATE_DISABLED:
            lbl.setText(f"{label}: ○ No vessel")
            lbl.setStyleSheet(f"color: {_GRAY};")
        elif state == STATE_IDLE:
            lbl.setText(f"{label}: ● Idle — Press [{vessel_id}] to arm")
            lbl.setStyleSheet("color: #111827;")
        elif state == STATE_ARMED:
            lbl.setText(f"{label}: ◉ Armed — Waiting for indicator...")
            lbl.setStyleSheet(f"color: {color}; font-style: italic;")
        elif state == STATE_MIXING:
            elapsed_str = f"{elapsed:.1f}s" if elapsed is not None else "—"
            lbl.setText(f"{label}: ◉ Mixing — {elapsed_str} elapsed")
            lbl.setStyleSheet(f"color: {color};")
        elif state == STATE_COMPLETE:
            t_str = f"{mixing_time:.1f}s" if mixing_time is not None else "—"
            lbl.setText(f"{label}: ✓ Complete — Mixed in {t_str}")
            lbl.setStyleSheet("color: #16a34a; font-weight: bold;")
