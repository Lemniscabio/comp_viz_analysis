"""Single application window with calibration and monitor panels on a QStackedWidget."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtWidgets import QMainWindow, QStackedWidget, QMessageBox
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QCloseEvent

from .common.constants import DEFAULT_CONFIG_PATH
from .common.roi_config import load_config, ConfigValidationError
from .calibration.calibration_window import CalibrationWindow
from .monitor.monitor_window import MonitorWindow

logger = logging.getLogger(__name__)

_PAGE_CALIBRATION = 0


class MainWindow(QMainWindow):
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        super().__init__()
        self._config_path = config_path
        self._mon_panel: MonitorWindow | None = None

        self.setMinimumSize(1000, 760)
        self.resize(1400, 800)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Calibration panel always lives at index 0
        self._cal_panel = CalibrationWindow(config_path)
        self._cal_panel.next_clicked.connect(self._show_monitor)
        self._stack.addWidget(self._cal_panel)

        # Go straight to monitor if a valid config already exists
        if config_path.exists():
            self._show_monitor()
        else:
            self.setWindowTitle("Vessel Calibration — Kineticolor")

    # ------------------------------------------------------------------
    # Page transitions
    # ------------------------------------------------------------------

    def _show_monitor(self) -> None:
        try:
            config = load_config(self._config_path)
        except (FileNotFoundError, ConfigValidationError) as exc:
            QMessageBox.critical(self, "Config Error", f"Cannot load config:\n{exc}")
            return

        # Release the calibration camera BEFORE the capture thread opens the same device.
        # DirectShow silently returns black frames when a device is opened twice.
        self._cal_panel.cleanup()

        # Tear down any existing monitor panel
        if self._mon_panel is not None:
            self._mon_panel.stop_monitoring()
            self._stack.removeWidget(self._mon_panel)
            self._mon_panel.deleteLater()

        self._mon_panel = MonitorWindow(config)
        self._mon_panel.recalibrate_requested.connect(self._show_calibration)
        self._stack.addWidget(self._mon_panel)
        self._stack.setCurrentWidget(self._mon_panel)
        self.setWindowTitle("Live Mixing Monitor — Kineticolor")
        # Give keyboard focus to the monitor panel for key shortcuts
        self._mon_panel.setFocus()

    def _show_calibration(self) -> None:
        mixing = self._mon_panel.mixing_vessel_labels() if self._mon_panel else []
        msg = (
            "This will stop monitoring and delete the current calibration.\n\n"
            f"Vessel(s) {', '.join(mixing)} are still mixing.\n\n"
            if mixing else
            "This will stop monitoring and delete the current calibration.\n\n"
        )
        reply = QMessageBox.question(
            self, "Calibrate Again",
            msg + "Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        if self._mon_panel:
            self._mon_panel.stop_monitoring()

        # Delete the saved config so calibration starts fresh
        if self._config_path.exists():
            self._config_path.unlink()

        self._cal_panel.reset()
        self._stack.setCurrentWidget(self._cal_panel)
        self.setWindowTitle("Vessel Calibration — Kineticolor")
        # Delay camera open so DirectShow/Camo has time to fully release the device
        # that the capture thread just closed. Immediate re-open returns black frames.
        QTimer.singleShot(1200, self._cal_panel.start_camera)

    # ------------------------------------------------------------------
    # Close event
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        # Check for active mixing before allowing close
        if self._mon_panel and self._mon_panel.is_mixing_active():
            labels = ", ".join(self._mon_panel.mixing_vessel_labels())
            reply = QMessageBox.question(
                self, "Quit",
                f"Vessel(s) {labels} are currently being monitored. Quit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        # Check for unsaved calibration changes (only relevant if on calibration page)
        if (self._stack.currentWidget() is self._cal_panel
                and self._cal_panel.has_unsaved_rois()):
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "Unsaved calibration changes. Save before closing?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                self._cal_panel.save_config_now()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        # Clean shutdown
        if self._mon_panel:
            self._mon_panel.stop_monitoring()
        self._cal_panel.cleanup()
        super().closeEvent(event)
