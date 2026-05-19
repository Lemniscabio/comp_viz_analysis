"""Single entry point for the Kineticolor Mixing Monitor.

Usage:
    python -m mixing_monitor.main_app [--config path/to/vessel_rois.json]

Opens calibration if no config exists; goes straight to the monitor if one does.
"""

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from .common.constants import DEFAULT_CONFIG_PATH
from .main_window import MainWindow


def main() -> None:
    parser = argparse.ArgumentParser(description="Kineticolor Mixing Monitor")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to vessel_rois.json (default: mixing_monitor/config/vessel_rois.json)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Kineticolor Mixing Monitor")
    app.setStyle("Fusion")

    window = MainWindow(config_path=args.config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
