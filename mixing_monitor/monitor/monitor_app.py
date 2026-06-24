"""Entry point for the Live Mixing Monitor app.

Usage:
    python -m mixing_monitor.monitor.monitor_app [--config path/to/vessel_rois.json]

If no config is found, shows an error dialog with a button to launch the
calibration app.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton

from ..common.constants import DEFAULT_CONFIG_PATH
from ..common.roi_config import load_config, ConfigValidationError
from .monitor_window import MonitorWindow


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Mixing Monitor — Kineticolor")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to vessel_rois.json (default: mixing_monitor/config/vessel_rois.json)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Live Mixing Monitor — Kineticolor")
    app.setStyle("Fusion")

    # Load config
    config = None
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        _show_no_config_dialog(str(args.config))
        sys.exit(1)
    except ConfigValidationError as exc:
        QMessageBox.critical(None, "Config Error", f"Cannot load config:\n{exc}")
        sys.exit(1)

    window = MonitorWindow(config)
    window.show()
    sys.exit(app.exec())


def _show_no_config_dialog(config_path: str) -> None:
    box = QMessageBox()
    box.setWindowTitle("No Config Found")
    box.setIcon(QMessageBox.Icon.Warning)
    box.setText(
        f"No vessel config found at:\n{config_path}\n\n"
        "Run the Calibration app first to set up your vessel ROIs."
    )
    cal_btn = box.addButton("Open Calibration", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    box.exec()

    if box.clickedButton() == cal_btn:
        subprocess.Popen(
            [sys.executable, "-m", "mixing_monitor.calibration.calibration_app"],
            cwd=str(Path(__file__).parent.parent.parent),
        )


if __name__ == "__main__":
    main()
