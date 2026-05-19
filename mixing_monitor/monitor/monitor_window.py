"""Monitor panel — embeddable QWidget used inside MainWindow."""

from __future__ import annotations

import csv
import logging
import time
from collections import deque
from concurrent.futures import Future, wait as futures_wait
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel,
    QPushButton, QFrame, QHBoxLayout, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, pyqtSignal
from PyQt6.QtGui import QFont, QKeyEvent

from ..common.constants import (
    VESSEL_LABELS, MAX_VESSELS,
    DELTA_E_TRIGGER_THRESHOLD,
    DELTA_E_STABILITY_WINDOW, DELTA_E_STABILITY_THRESHOLD, COMPLETION_CONFIRMATION_FRAMES,
    PINK_FRACTION_TRIGGER, PINK_FRACTION_COMPLETE,
)
from .capture_thread import CaptureThread
from .analysis_pool import AnalysisPool
from .vessel_card import (
    VesselCard,
    STATE_DISABLED, STATE_IDLE, STATE_ARMED, STATE_MIXING, STATE_COMPLETE,
)
from .results_bar import ResultsBar

logger = logging.getLogger(__name__)


class _VesselState:
    def __init__(self, vessel_id: int, active: bool) -> None:
        self.vessel_id = vessel_id
        self.active = active
        self.state: str = STATE_DISABLED if not active else STATE_IDLE
        self.reference_mean_a: float = 0.0
        self.arm_time: float = 0.0
        self.mix_start_time: float = 0.0
        self.mixing_time: float | None = None
        self.mixing_entered: bool = False
        self.peak_delta_e: float = 0.0
        # Pink fraction tracking
        self.last_pink_fraction: float = 0.0
        self.pink_ever_triggered: bool = False  # True if reference or mixing had significant pink
        # ΔE stability (rolling std window)
        self.delta_e_buffer: deque[float] = deque(maxlen=DELTA_E_STABILITY_WINDOW)
        # Combined AND completion counter
        self.both_met_consecutive: int = 0
        self.both_met_first_time: float | None = None
        # Pipetting delay
        self.pipetting_delay: float | None = None  # mix_start_time - arm_time
        # Timeseries (populated during ARMED and MIXING for export)
        self.ts_elapsed: list[float] = []
        self.ts_mean_a: list[float] = []
        self.ts_delta_e: list[float] = []
        self.ts_pink_fraction: list[float] = []
        self.last_mean_a: float = 0.0
        self.last_delta_e: float = 0.0


class MonitorWindow(QWidget):
    """Live monitoring panel: vessel cards, results bar, analysis loop."""

    # Emitted when the user requests to go back to calibration
    recalibrate_requested = pyqtSignal()

    def __init__(self, config: dict) -> None:
        super().__init__()
        self._config = config
        self._vessel_states: dict[int, _VesselState] = {}
        self._pool = AnalysisPool(max_workers=MAX_VESSELS)
        self._frame_count = 0
        self._fps_start = time.monotonic()

        self._build_vessel_states()
        self._build_ui()
        self._start_capture()

    # ------------------------------------------------------------------
    # Public API (called by MainWindow)
    # ------------------------------------------------------------------

    def stop_monitoring(self) -> None:
        """Stop all threads. Safe to call multiple times."""
        if hasattr(self, "_analysis_timer"):
            self._analysis_timer.stop()
        if hasattr(self, "_capture"):
            self._capture.stop()
        if hasattr(self, "_pool"):
            self._pool.shutdown()

    def is_mixing_active(self) -> bool:
        return any(vs.state == STATE_MIXING for vs in self._vessel_states.values())

    def mixing_vessel_labels(self) -> list[str]:
        return [
            VESSEL_LABELS.get(vid, f"V{vid}")
            for vid, vs in self._vessel_states.items()
            if vs.state == STATE_MIXING
        ]

    # ------------------------------------------------------------------
    # Vessel states
    # ------------------------------------------------------------------

    def _build_vessel_states(self) -> None:
        active_ids = {v["id"] for v in self._config.get("vessels", [])}
        for vid in range(1, MAX_VESSELS + 1):
            self._vessel_states[vid] = _VesselState(vid, active=vid in active_ids)

    def _get_roi(self, vessel_id: int) -> list[int] | None:
        for v in self._config.get("vessels", []):
            if v["id"] == vessel_id:
                return v["roi"]
        return None

    def _crop_roi(self, frame: np.ndarray, roi: list[int]) -> np.ndarray:
        x, y, w, h = roi
        fh, fw = frame.shape[:2]
        x1 = max(0, x); y1 = max(0, y)
        x2 = min(fw, x + w); y2 = min(fh, y + h)
        return frame[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 8)
        root.setSpacing(8)

        # 2×2 vessel card grid
        grid = QGridLayout()
        grid.setSpacing(16)
        grid.setContentsMargins(0, 0, 0, 0)
        self._cards: dict[int, VesselCard] = {}
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for vid in range(1, MAX_VESSELS + 1):
            vs = self._vessel_states[vid]
            card = VesselCard(vid, active=vs.active)
            self._cards[vid] = card
            r, c = positions[vid - 1]
            grid.addWidget(card, r, c)
        root.addLayout(grid, stretch=1)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("color: #d1d5db;")
        root.addWidget(sep1)

        # Results bar
        self._results_bar = ResultsBar()
        root.addWidget(self._results_bar)
        for vid in range(1, MAX_VESSELS + 1):
            vs = self._vessel_states[vid]
            self._results_bar.update_vessel(vid, vs.state)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #d1d5db;")
        root.addWidget(sep2)

        # Bottom row: legend + Export + Calibrate Again
        bottom_row = QHBoxLayout()
        legend = QLabel(
            "[1-4] Arm vessel    [R] Reset all    [Q] Quit"
        )
        legend.setStyleSheet("color: #6b7280; font-size: 11px;")
        bottom_row.addWidget(legend)
        bottom_row.addStretch()

        export_btn = QPushButton("Export CSV")
        export_btn.setStyleSheet(
            "QPushButton { background: #2563eb; color: white; font-weight: bold; "
            "padding: 4px 14px; border-radius: 4px; font-size: 11px; }"
        )
        export_btn.clicked.connect(self._export_csv)
        bottom_row.addWidget(export_btn)

        cal_btn = QPushButton("← Calibrate Again")
        cal_btn.setStyleSheet("color: #6b7280; font-size: 11px;")
        cal_btn.setFlat(True)
        cal_btn.clicked.connect(self._on_calibrate_again)
        bottom_row.addWidget(cal_btn)

        root.addLayout(bottom_row)

    # ------------------------------------------------------------------
    # Capture + analysis loop
    # ------------------------------------------------------------------

    def _start_capture(self) -> None:
        cam = self._config["camera"]
        dev_idx = cam["device_index"]
        res = tuple(cam["resolution"])
        self._capture = CaptureThread(dev_idx, res)
        self._capture.start()

        self._analysis_timer = QTimer(self)
        self._analysis_timer.setInterval(33)
        self._analysis_timer.timeout.connect(self._analysis_tick)
        self._analysis_timer.start()

    @pyqtSlot()
    def _analysis_tick(self) -> None:
        frame = self._capture.get_latest_frame()
        if frame is None:
            return

        crops: dict[int, np.ndarray] = {}
        for vid, vs in self._vessel_states.items():
            if not vs.active:
                continue
            roi = self._get_roi(vid)
            if roi is None:
                continue
            crop = self._crop_roi(frame, roi)
            if crop.size == 0:
                continue
            crops[vid] = crop
            self._cards[vid].update_frame(crop)

        analysis_crops = {
            vid: crop for vid, crop in crops.items()
            if self._vessel_states[vid].state in (STATE_ARMED, STATE_MIXING)
        }
        if not analysis_crops:
            return

        futures: dict[int, Future] = self._pool.submit_all(analysis_crops)
        if not futures:
            return

        done, _ = futures_wait(list(futures.values()), timeout=0.05)
        now = time.monotonic()

        for vid, fut in futures.items():
            if fut not in done:
                continue
            try:
                result = fut.result()
            except Exception as exc:
                logger.warning("Analysis error for vessel %d: %s", vid, exc)
                continue
            self._handle_result(vid, result, now)

        # Update FPS in parent window title
        self._frame_count += 1
        elapsed = now - self._fps_start
        if elapsed >= 2.0:
            fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start = now
            win = self.window()
            if win is not self:
                win.setWindowTitle(f"Live Mixing Monitor — Kineticolor  ({fps:.0f} fps)")

    def _handle_result(self, vessel_id: int, result: dict, now: float) -> None:
        vs = self._vessel_states[vessel_id]
        mean_a = result["mean_a_star"]
        delta_e = result["mean_delta_e"]
        pink_fraction = result.get("pink_fraction", 0.0)
        vs.last_mean_a = mean_a
        vs.last_delta_e = delta_e
        vs.last_pink_fraction = pink_fraction

        elapsed = now - vs.mix_start_time if vs.state == STATE_MIXING else None
        ts_elapsed = now - vs.arm_time  # time since arm (valid; only ARMED/MIXING reach here)

        vs.ts_elapsed.append(ts_elapsed)
        vs.ts_mean_a.append(mean_a)
        vs.ts_delta_e.append(delta_e)
        vs.ts_pink_fraction.append(pink_fraction)

        self._cards[vessel_id].update_metrics(mean_a, delta_e, elapsed, time_since_arm=ts_elapsed)

        if vs.state == STATE_ARMED:
            # Track peak ΔE while armed (indicator may be partly added before trigger)
            vs.peak_delta_e = max(vs.peak_delta_e, delta_e)
            self._cards[vessel_id].update_diagnostics(
                mixing_entered=False,
                current_delta_e=delta_e,
                peak_delta_e=vs.peak_delta_e if vs.peak_delta_e > 0 else None,
                delta_e_std=None,
                stability_met=False,
                pink_fraction=pink_fraction,
            )
            # Trigger: ΔE spike indicates HCl addition (disturbance in uniformly colored solution)
            if delta_e >= DELTA_E_TRIGGER_THRESHOLD:
                self._transition_to_mixing(vessel_id, now)

        elif vs.state == STATE_MIXING:
            vs.peak_delta_e = max(vs.peak_delta_e, delta_e)

            if pink_fraction >= PINK_FRACTION_TRIGGER:
                vs.pink_ever_triggered = True

            # --- ΔE stability (rolling std) ---
            vs.delta_e_buffer.append(delta_e)
            if len(vs.delta_e_buffer) >= DELTA_E_STABILITY_WINDOW:
                delta_e_std = float(np.std(list(vs.delta_e_buffer)))
                stability_ok = delta_e_std <= DELTA_E_STABILITY_THRESHOLD
            else:
                delta_e_std = None
                stability_ok = False

            # --- Pink criterion: auto-satisfied if pink was never detected ---
            pink_ok = (not vs.pink_ever_triggered) or (pink_fraction <= PINK_FRACTION_COMPLETE)

            self._results_bar.update_vessel(vessel_id, STATE_MIXING, elapsed=elapsed, mean_a=mean_a)
            self._cards[vessel_id].update_diagnostics(
                mixing_entered=True,
                current_delta_e=delta_e,
                peak_delta_e=vs.peak_delta_e,
                delta_e_std=delta_e_std,
                stability_met=stability_ok,
                pink_fraction=pink_fraction,
            )

            # Complete only when BOTH criteria are simultaneously satisfied
            self._check_completion(vessel_id, now, stability_ok, pink_ok)

    def _transition_to_mixing(self, vessel_id: int, now: float) -> None:
        vs = self._vessel_states[vessel_id]
        vs.state = STATE_MIXING
        vs.mix_start_time = now
        vs.mixing_entered = True
        vs.delta_e_buffer.clear()
        vs.both_met_consecutive = 0
        vs.both_met_first_time = None
        # pink_ever_triggered may already be True from arm time; also check current fraction
        if vs.last_pink_fraction >= PINK_FRACTION_TRIGGER:
            vs.pink_ever_triggered = True
        # Pipetting delay = time between arming and actual HCl addition (ΔE trigger)
        vs.pipetting_delay = now - vs.arm_time
        self._cards[vessel_id].set_state(STATE_MIXING)
        self._cards[vessel_id].show_pipetting_delay(vs.pipetting_delay)
        self._results_bar.update_vessel(vessel_id, STATE_MIXING, elapsed=0.0)
        logger.info(
            "Vessel %d: MIXING started (ΔE=%.2f pink=%.3f pipette_delay=%.1fs)",
            vessel_id, vs.last_delta_e, vs.last_pink_fraction, vs.pipetting_delay,
        )

    def _check_completion(
        self, vessel_id: int, now: float, stability_ok: bool, pink_ok: bool
    ) -> None:
        """Declare mixing complete when BOTH stability AND pink criteria hold simultaneously.

        stability_ok: rolling std of ΔE ≤ DELTA_E_STABILITY_THRESHOLD (color stopped changing).
        pink_ok:      pink_fraction ≤ PINK_FRACTION_COMPLETE, or True if pink was never detected.

        mixing_time measured to the FIRST frame both criteria were simultaneously satisfied.
        """
        vs = self._vessel_states[vessel_id]
        if stability_ok and pink_ok:
            if vs.both_met_first_time is None:
                vs.both_met_first_time = now
            vs.both_met_consecutive += 1
            if vs.both_met_consecutive >= COMPLETION_CONFIRMATION_FRAMES:
                mixing_time = vs.both_met_first_time - vs.mix_start_time
                self._transition_to_complete(vessel_id, mixing_time, vs.last_delta_e)
        else:
            # Either criterion failed — reset the AND counter
            vs.both_met_consecutive = 0
            vs.both_met_first_time = None

    def _transition_to_complete(self, vessel_id: int, mixing_time: float, final_mean_a: float) -> None:
        vs = self._vessel_states[vessel_id]
        vs.state = STATE_COMPLETE
        vs.mixing_time = mixing_time
        self._cards[vessel_id].set_state(STATE_COMPLETE)
        self._cards[vessel_id].show_mixing_time(mixing_time)
        self._results_bar.update_vessel(vessel_id, STATE_COMPLETE, mixing_time=mixing_time)
        logger.info("Vessel %d: COMPLETE — %.2fs (pipette_delay=%.1fs)",
                    vessel_id, mixing_time, vs.pipetting_delay or 0.0)

    # ------------------------------------------------------------------
    # Keyboard controls
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4):
            self._key_vessel(key - Qt.Key.Key_0)
        elif key == Qt.Key.Key_R:
            self._key_reset_all()
        elif key == Qt.Key.Key_E:
            self._export_csv()
        elif key == Qt.Key.Key_Q:
            self.window().close()
        else:
            super().keyPressEvent(event)

    def _key_vessel(self, vessel_id: int) -> None:
        vs = self._vessel_states.get(vessel_id)
        if vs is None or not vs.active:
            return
        if vs.state in (STATE_IDLE, STATE_COMPLETE):
            self._arm_vessel(vessel_id)
        elif vs.state in (STATE_ARMED, STATE_MIXING):
            label = VESSEL_LABELS.get(vessel_id, f"V{vessel_id}")
            state_str = "armed and waiting" if vs.state == STATE_ARMED else "actively mixing"
            reply = QMessageBox.question(
                self, f"Stop monitoring {label}?",
                f"{label} is currently {state_str}.\n\nForce stop and reset this vessel?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._reset_vessel(vessel_id)

    def _arm_vessel(self, vessel_id: int) -> None:
        frame = self._capture.get_latest_frame()
        if frame is None:
            return
        roi = self._get_roi(vessel_id)
        if roi is None:
            return
        crop = self._crop_roi(frame, roi)
        if crop.size == 0:
            return

        vs = self._vessel_states[vessel_id]
        vs.state = STATE_ARMED
        vs.arm_time = time.monotonic()
        vs.mix_start_time = 0.0
        vs.mixing_time = None
        vs.mixing_entered = False
        vs.peak_delta_e = 0.0
        vs.last_pink_fraction = 0.0
        vs.pink_ever_triggered = False
        vs.delta_e_buffer.clear()
        vs.both_met_consecutive = 0
        vs.both_met_first_time = None
        vs.pipetting_delay = None
        vs.ts_elapsed.clear()
        vs.ts_mean_a.clear()
        vs.ts_delta_e.clear()
        vs.ts_pink_fraction.clear()

        self._pool.set_reference(vessel_id, crop)
        ref_a = self._pool.get_reference_mean_a(vessel_id)
        vs.reference_mean_a = ref_a if ref_a is not None else 0.0

        # If reference already has significant pink, use pink-fraction completion detection
        ref_pink = self._pool.get_reference_pink_fraction(vessel_id)
        if ref_pink is not None and ref_pink >= PINK_FRACTION_TRIGGER:
            vs.pink_ever_triggered = True
            logger.info("Vessel %d: reference is pink (%.3f) — pink completion active", vessel_id, ref_pink)

        self._cards[vessel_id].reset_sparkline()
        self._cards[vessel_id].set_state(STATE_ARMED)
        self._results_bar.update_vessel(vessel_id, STATE_ARMED)
        logger.info("Vessel %d: ARMED (ref a*=%.2f, ref pink=%.3f)",
                    vessel_id, vs.reference_mean_a, ref_pink or 0.0)

    def _reset_vessel(self, vessel_id: int) -> None:
        vs = self._vessel_states[vessel_id]
        vs.state = STATE_IDLE
        vs.ts_elapsed.clear()
        vs.ts_mean_a.clear()
        vs.ts_delta_e.clear()
        vs.ts_pink_fraction.clear()
        self._pool.remove(vessel_id)
        self._cards[vessel_id].reset_sparkline()
        self._cards[vessel_id].set_state(STATE_IDLE)
        self._results_bar.update_vessel(vessel_id, STATE_IDLE)

    def _key_reset_all(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        mixing = [
            VESSEL_LABELS.get(v, f"V{v}") for v, vs in self._vessel_states.items()
            if vs.state == STATE_MIXING
        ]
        if mixing:
            reply = QMessageBox.question(
                self, "Reset All",
                f"Vessel(s) {', '.join(mixing)} are currently mixing. Reset all anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        for vid, vs in self._vessel_states.items():
            if vs.active:
                vs.state = STATE_IDLE
                vs.ts_elapsed.clear()
                vs.ts_mean_a.clear()
                vs.ts_delta_e.clear()
                vs.ts_pink_fraction.clear()
                self._pool.remove(vid)
                self._cards[vid].reset_sparkline()
                self._cards[vid].set_state(STATE_IDLE)
                self._results_bar.update_vessel(vid, STATE_IDLE)

    def _export_csv(self) -> None:
        default_name = f"mixing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", default_name,
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # Timeseries section
                writer.writerow(["vessel_id", "label", "elapsed_s", "mean_a_star", "delta_e", "pink_fraction"])
                for vid, vs in self._vessel_states.items():
                    if not vs.active or not vs.ts_elapsed:
                        continue
                    label = VESSEL_LABELS.get(vid, f"V{vid}")
                    for i in range(len(vs.ts_elapsed)):
                        pk = vs.ts_pink_fraction[i] if i < len(vs.ts_pink_fraction) else ""
                        writer.writerow([
                            vid, label,
                            f"{vs.ts_elapsed[i]:.3f}",
                            f"{vs.ts_mean_a[i]:.4f}",
                            f"{vs.ts_delta_e[i]:.4f}",
                            f"{pk:.4f}" if isinstance(pk, float) else pk,
                        ])

                # Summary section
                writer.writerow([])
                writer.writerow(["# Summary"])
                writer.writerow(["vessel_id", "label", "state", "pipetting_delay_s", "mixing_time_s"])
                for vid, vs in self._vessel_states.items():
                    if not vs.active:
                        continue
                    label = VESSEL_LABELS.get(vid, f"V{vid}")
                    delay = f"{vs.pipetting_delay:.2f}" if vs.pipetting_delay is not None else ""
                    mt = f"{vs.mixing_time:.2f}" if vs.mixing_time is not None else ""
                    writer.writerow([vid, label, vs.state, delay, mt])

            win = self.window()
            if win is not self:
                win.setWindowTitle(f"Exported: {Path(path).name} — Kineticolor")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Could not write file:\n{exc}")

    def _on_calibrate_again(self) -> None:
        self.recalibrate_requested.emit()
