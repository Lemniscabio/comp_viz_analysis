"""Entry point for the Vessel Calibration app.

Usage:
    python -m mixing_monitor.calibration.calibration_app [--config path/to/vessel_rois.json]
"""

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from ..common.constants import DEFAULT_CONFIG_PATH
from .calibration_window import CalibrationWindow


def main() -> None:
    parser = argparse.ArgumentParser(description="Vessel Calibration — Kineticolor")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to vessel_rois.json (default: mixing_monitor/config/vessel_rois.json)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Vessel Calibration — Kineticolor")
    app.setStyle("Fusion")

    window = CalibrationWindow(config_path=args.config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
