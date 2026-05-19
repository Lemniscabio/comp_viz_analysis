"""Single vessel display card widget."""

from __future__ import annotations

import math
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPalette, QFont, QImage, QPixmap
import pyqtgraph as pg

from ..common.constants import (
    VESSEL_COLORS, VESSEL_LABELS,
    HEADER_BAR_HEIGHT, SPARKLINE_HEIGHT, SPARKLINE_MAX_SECONDS,
    INACTIVE_CARD_BORDER_COLOR, INACTIVE_CARD_BG_COLOR,
    DELTA_E_STABILITY_THRESHOLD, PINK_FRACTION_COMPLETE,
)

# Vessel states
STATE_DISABLED = "disabled"
STATE_IDLE = "idle"
STATE_ARMED = "armed"
STATE_MIXING = "mixing"
STATE_COMPLETE = "complete"


def _color_with_alpha(hex_color: str, alpha_f: float) -> str:
    c = QColor(hex_color)
    c.setAlphaF(alpha_f)
    return f"rgba({c.red()},{c.green()},{c.blue()},{int(c.alpha())})"


class VesselCard(QWidget):
    """Displays live video crop, metrics, and sparkline for one vessel."""

    def __init__(self, vessel_id: int, active: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._vessel_id = vessel_id
        self._active = active
        self._state = STATE_DISABLED if not active else STATE_IDLE
        self._color = VESSEL_COLORS.get(vessel_id, "#ffffff")
        self._label = VESSEL_LABELS.get(vessel_id, f"V{vessel_id}")

        self._sparkline_data: list[float] = []
        self._sparkline_times: list[float] = []
        self._pulse_on = False

        self._build_ui()

        if active:
            self._pulse_timer = QTimer(self)
            self._pulse_timer.setInterval(500)
            self._pulse_timer.timeout.connect(self._pulse_tick)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        if self._active:
            self.setMinimumSize(280, 280)
        else:
            self.setMinimumSize(120, 60)
            self.setMaximumHeight(80)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        if not self._active:
            self._build_inactive_ui(outer)
        else:
            self._build_active_ui(outer)

        self._apply_card_style()

    def _build_inactive_ui(self, layout: QVBoxLayout) -> None:
        placeholder = QLabel("No Vessel")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet(f"color: {INACTIVE_CARD_BORDER_COLOR}; font-size: 14px;")
        layout.addWidget(placeholder)

    def _build_active_ui(self, layout: QVBoxLayout) -> None:
        # Header bar
        self._header = QWidget()
        self._header.setFixedHeight(HEADER_BAR_HEIGHT)
        h_layout = QHBoxLayout(self._header)
        h_layout.setContentsMargins(8, 0, 8, 0)
        self._header_label = QLabel(self._label)
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        self._header_label.setFont(font)
        self._header_label.setStyleSheet(f"color: {self._color};")
        h_layout.addWidget(self._header_label)
        self._state_indicator = QLabel("● Idle")
        self._state_indicator.setStyleSheet("color: #6b7280; font-size: 10px;")
        h_layout.addStretch()
        h_layout.addWidget(self._state_indicator)
        layout.addWidget(self._header)

        # Video area
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet("background: #1a1a1a;")
        self._video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._video_label.setMinimumHeight(80)
        layout.addWidget(self._video_label, stretch=1)

        # Metrics bar
        self._metrics_bar = QLabel("a*: —      ΔE: —")
        mono = QFont("Consolas", 11)
        self._metrics_bar.setFont(mono)
        self._metrics_bar.setStyleSheet("background: #f0f0f0; padding: 2px 8px;")
        layout.addWidget(self._metrics_bar)

        # Diagnostic strip
        diag_frame = QFrame()
        diag_frame.setStyleSheet(
            "QFrame { background: #f8fafc; border-top: 1px solid #e2e8f0; }"
        )
        diag_layout = QVBoxLayout(diag_frame)
        diag_layout.setContentsMargins(8, 3, 8, 3)
        diag_layout.setSpacing(2)

        diag_font = QFont("Consolas", 9)
        self._lbl_mixing_entered = QLabel("○  Mixing state entered")
        self._lbl_mixing_entered.setFont(diag_font)
        self._lbl_mixing_entered.setStyleSheet(f"color: {INACTIVE_CARD_BORDER_COLOR};")
        diag_layout.addWidget(self._lbl_mixing_entered)

        self._lbl_delta_e = QLabel("   ΔE:  —       peak:   —")
        self._lbl_delta_e.setFont(diag_font)
        self._lbl_delta_e.setStyleSheet(f"color: {INACTIVE_CARD_BORDER_COLOR};")
        diag_layout.addWidget(self._lbl_delta_e)

        self._lbl_stability = QLabel(
            f"   std:   —    / {DELTA_E_STABILITY_THRESHOLD:.2f}  (ΔE stability)"
        )
        self._lbl_stability.setFont(diag_font)
        self._lbl_stability.setStyleSheet(f"color: {INACTIVE_CARD_BORDER_COLOR};")
        diag_layout.addWidget(self._lbl_stability)

        self._lbl_pink = QLabel(
            f"   pink:  —     / {PINK_FRACTION_COMPLETE:.3f}  (complete threshold)"
        )
        self._lbl_pink.setFont(diag_font)
        self._lbl_pink.setStyleSheet(f"color: {INACTIVE_CARD_BORDER_COLOR};")
        diag_layout.addWidget(self._lbl_pink)

        layout.addWidget(diag_frame)

        # Time-series chart (ΔE vs elapsed seconds since arm)
        self._sparkline = pg.PlotWidget()
        self._sparkline.setFixedHeight(SPARKLINE_HEIGHT)
        self._sparkline.setBackground("#f8fafc")
        self._sparkline.showAxis("bottom")
        self._sparkline.showAxis("left")
        self._sparkline.setMouseEnabled(x=False, y=False)
        self._sparkline.getPlotItem().setMenuEnabled(False)

        tick_font = QFont("Consolas", 7)
        bottom = self._sparkline.getAxis("bottom")
        left = self._sparkline.getAxis("left")
        bottom.setTickFont(tick_font)
        left.setTickFont(tick_font)
        bottom.setLabel("s", **{"font-size": "7pt"})
        left.setLabel("ΔE", **{"font-size": "7pt"})
        bottom.setStyle(tickTextOffset=2)
        left.setStyle(tickTextOffset=2)

        self._sparkline.setXRange(0, 10, padding=0)
        self._sparkline.enableAutoRange("y", True)

        pen = pg.mkPen(color=self._color, width=1.5)
        self._spark_curve = self._sparkline.plot([], [], pen=pen)
        layout.addWidget(self._sparkline)

    def _apply_card_style(self) -> None:
        if not self._active:
            self.setStyleSheet(
                f"QWidget {{ border: 2px dashed {INACTIVE_CARD_BORDER_COLOR}; "
                f"background: {INACTIVE_CARD_BG_COLOR}; border-radius: 4px; }}"
            )
        else:
            self.setStyleSheet(
                f"VesselCard {{ border: 2px solid {self._color}; border-radius: 4px; }}"
            )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_state(self, state: str) -> None:
        if not self._active:
            return
        prev = self._state
        self._state = state
        self._update_visuals()

        if state == STATE_ARMED:
            if hasattr(self, "_pulse_timer"):
                self._pulse_timer.start()
        else:
            if hasattr(self, "_pulse_timer"):
                self._pulse_timer.stop()
            self._pulse_on = False
            self._apply_border(solid=True)

    def get_state(self) -> str:
        return self._state

    def _update_visuals(self) -> None:
        state = self._state
        if state == STATE_IDLE:
            self._state_indicator.setText("● Idle")
            self._state_indicator.setStyleSheet("color: #6b7280; font-size: 10px;")
        elif state == STATE_ARMED:
            self._state_indicator.setText("◉ Armed")
            self._state_indicator.setStyleSheet(f"color: {self._color}; font-size: 10px;")
        elif state == STATE_MIXING:
            self._state_indicator.setText("◉ Mixing")
            self._state_indicator.setStyleSheet(f"color: {self._color}; font-size: 10px;")
            self._apply_border(width=3, solid=True)
        elif state == STATE_COMPLETE:
            self._state_indicator.setText("✓ Complete")
            self._state_indicator.setStyleSheet("color: #16a34a; font-size: 10px; font-weight: bold;")
            self.setStyleSheet(
                "VesselCard { border: 2px solid #16a34a; border-radius: 4px; }"
            )

    def _apply_border(self, width: int = 2, solid: bool = True) -> None:
        style = "solid" if solid else "dashed"
        color = self._color if self._state != STATE_COMPLETE else "#16a34a"
        self.setStyleSheet(
            f"VesselCard {{ border: {width}px {style} {color}; border-radius: 4px; }}"
        )

    def _pulse_tick(self) -> None:
        self._pulse_on = not self._pulse_on
        w = 4 if self._pulse_on else 2
        self._apply_border(width=w, solid=True)

    # ------------------------------------------------------------------
    # Data updates
    # ------------------------------------------------------------------

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        if not self._active:
            return
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1].copy()
        qi = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qi)
        # Scale to fit the video label while preserving aspect ratio
        lw = self._video_label.width()
        lh = self._video_label.height()
        if lw > 0 and lh > 0:
            pix = pix.scaled(lw, lh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._video_label.setPixmap(pix)

    def update_metrics(
        self,
        mean_a: float,
        delta_e: float,
        elapsed: float | None = None,
        time_since_arm: float | None = None,
    ) -> None:
        if not self._active:
            return
        elapsed_str = f"  {elapsed:.1f}s" if elapsed is not None else ""
        self._metrics_bar.setText(f"a*: {mean_a:6.1f}   ΔE: {delta_e:6.1f}{elapsed_str}")

        if time_since_arm is not None:
            self._sparkline_times.append(time_since_arm)
            self._sparkline_data.append(delta_e)
            self._spark_curve.setData(x=self._sparkline_times, y=self._sparkline_data)
            # X axis always shows from 0; scale compresses to fit all data up to SPARKLINE_MAX_SECONDS
            x_max = max(time_since_arm * 1.04, 10.0)
            self._sparkline.setXRange(0, x_max, padding=0)

    def reset_sparkline(self) -> None:
        self._sparkline_data.clear()
        self._sparkline_times.clear()
        self._spark_curve.setData(x=[], y=[])
        self._sparkline.setXRange(0, 10, padding=0)
        # Reset metrics bar — clears the green "Mixed in X.Xs" banner
        self._metrics_bar.setText("a*: —      ΔE: —")
        self._metrics_bar.setStyleSheet("background: #f0f0f0; padding: 2px 8px;")
        self.update_diagnostics(False, None, None, None, False, None)

    def update_diagnostics(
        self,
        mixing_entered: bool,
        current_delta_e: float | None,
        peak_delta_e: float | None,
        delta_e_std: float | None,
        stability_met: bool,
        pink_fraction: float | None,
    ) -> None:
        """Update the diagnostic strip below the metrics bar.

        Args:
            mixing_entered:  True once the ΔE trigger was crossed.
            current_delta_e: Current Grand ΔE from reference frame.
            peak_delta_e:    Rolling maximum ΔE seen since arming.
            delta_e_std:     Rolling std of ΔE (None while buffer is filling).
                             Low = color stopped changing = stable.
            stability_met:   True when delta_e_std <= DELTA_E_STABILITY_THRESHOLD.
            pink_fraction:   Fraction of pink pixels (None when not armed).
        """
        if not self._active:
            return

        # --- Mixing entered flag ---
        if mixing_entered:
            self._lbl_mixing_entered.setText("●  Mixing state entered")
            self._lbl_mixing_entered.setStyleSheet(f"color: {self._color}; font-weight: bold;")
        else:
            self._lbl_mixing_entered.setText("○  Mixing state entered")
            self._lbl_mixing_entered.setStyleSheet(f"color: {INACTIVE_CARD_BORDER_COLOR};")

        # --- ΔE current vs peak row ---
        if current_delta_e is not None and peak_delta_e is not None:
            self._lbl_delta_e.setText(
                f"   ΔE: {current_delta_e:6.2f}   peak: {peak_delta_e:6.2f}"
            )
            self._lbl_delta_e.setStyleSheet("color: #374151;")
        elif current_delta_e is not None:
            self._lbl_delta_e.setText(f"   ΔE: {current_delta_e:6.2f}   peak:   —")
            self._lbl_delta_e.setStyleSheet("color: #374151;")
        else:
            self._lbl_delta_e.setText("   ΔE:  —       peak:   —")
            self._lbl_delta_e.setStyleSheet(f"color: {INACTIVE_CARD_BORDER_COLOR};")

        # --- ΔE stability row (rolling std; low = stable = ✓) ---
        if delta_e_std is not None:
            sym = "✓" if stability_met else "…"
            color = "#16a34a" if stability_met else "#f59e0b"
            self._lbl_stability.setText(
                f"   std: {delta_e_std:5.2f}  / {DELTA_E_STABILITY_THRESHOLD:.2f}  {sym}"
            )
            self._lbl_stability.setStyleSheet(
                f"color: {color}; font-weight: {'bold' if stability_met else 'normal'};"
            )
        else:
            self._lbl_stability.setText(
                f"   std:   —    / {DELTA_E_STABILITY_THRESHOLD:.2f}  (filling…)"
            )
            self._lbl_stability.setStyleSheet(f"color: {INACTIVE_CARD_BORDER_COLOR};")

        # --- Pink fraction row ---
        if pink_fraction is not None:
            pink_done = pink_fraction <= PINK_FRACTION_COMPLETE
            sym = "✓" if pink_done else "…"
            color = "#16a34a" if pink_done else "#f59e0b"
            self._lbl_pink.setText(
                f"   pink: {pink_fraction:.3f}  / {PINK_FRACTION_COMPLETE:.3f}  {sym}"
            )
            self._lbl_pink.setStyleSheet(
                f"color: {color}; font-weight: {'bold' if pink_done else 'normal'};"
            )
        else:
            self._lbl_pink.setText(
                f"   pink:  —     / {PINK_FRACTION_COMPLETE:.3f}  (complete threshold)"
            )
            self._lbl_pink.setStyleSheet(f"color: {INACTIVE_CARD_BORDER_COLOR};")

    def show_pipetting_delay(self, delay: float) -> None:
        """Update the mixing-entered label to include the detected pipetting delay."""
        if not self._active:
            return
        self._lbl_mixing_entered.setText(f"●  Mixing entered  (pipette delay: {delay:.1f}s)")
        self._lbl_mixing_entered.setStyleSheet(f"color: {self._color}; font-weight: bold;")

    def show_mixing_time(self, mixing_time: float) -> None:
        if not self._active:
            return
        self._metrics_bar.setText(f"Estimated Mixing time: {mixing_time:.1f}s")
        self._metrics_bar.setStyleSheet(
            "background: #dcfce7; color: #16a34a; font-weight: bold; padding: 2px 8px;"
        )
