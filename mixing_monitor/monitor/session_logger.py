"""CSV writer for session events and per-vessel timeseries."""

from __future__ import annotations

import csv
import logging
import threading
from datetime import datetime
from pathlib import Path

from ..common.constants import RESULTS_DIR

logger = logging.getLogger(__name__)

_EVENTS_COLUMNS = [
    "timestamp", "vessel_id", "vessel_label",
    "event", "mixing_time_s",
    "mean_a_star_ref", "mean_a_star_final", "mean_delta_e_final", "notes",
]

_TIMESERIES_COLUMNS = [
    "elapsed_s", "vessel_id", "vessel_label", "mean_a_star", "mean_delta_e",
]


class SessionLogger:
    """Creates and writes session CSV files on app launch."""

    def __init__(self, results_dir: Path = RESULTS_DIR) -> None:
        self._lock = threading.Lock()
        self._results_dir = results_dir
        self._results_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_ts = ts
        self._events_path = results_dir / f"session_{ts}.csv"
        self._ts_paths: dict[int, Path] = {}
        self._ts_writers: dict[int, csv.writer] = {}
        self._ts_files: dict[int, object] = {}

        # Write events file header
        self._events_fh = self._events_path.open("w", newline="", encoding="utf-8")
        self._events_writer = csv.DictWriter(self._events_fh, fieldnames=_EVENTS_COLUMNS)
        self._events_writer.writeheader()
        self._events_fh.flush()
        logger.info("Session log: %s", self._events_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_event(
        self,
        vessel_id: int,
        vessel_label: str,
        event: str,
        mixing_time_s: float | None = None,
        mean_a_star_ref: float | None = None,
        mean_a_star_final: float | None = None,
        mean_delta_e_final: float | None = None,
        notes: str = "",
    ) -> None:
        """Append one event row to the session events CSV."""
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "vessel_id": vessel_id,
            "vessel_label": vessel_label,
            "event": event,
            "mixing_time_s": f"{mixing_time_s:.3f}" if mixing_time_s is not None else "",
            "mean_a_star_ref": f"{mean_a_star_ref:.3f}" if mean_a_star_ref is not None else "",
            "mean_a_star_final": f"{mean_a_star_final:.3f}" if mean_a_star_final is not None else "",
            "mean_delta_e_final": f"{mean_delta_e_final:.3f}" if mean_delta_e_final is not None else "",
            "notes": notes,
        }
        with self._lock:
            self._events_writer.writerow(row)
            self._events_fh.flush()

    def log_timeseries(
        self,
        vessel_id: int,
        vessel_label: str,
        elapsed_s: float,
        mean_a_star: float,
        mean_delta_e: float,
    ) -> None:
        """Append one timeseries row to the per-vessel CSV (created on first call)."""
        with self._lock:
            if vessel_id not in self._ts_writers:
                path = self._results_dir / f"session_{self._session_ts}_{vessel_label}_timeseries.csv"
                fh = path.open("w", newline="", encoding="utf-8")
                writer = csv.writer(fh)
                writer.writerow(_TIMESERIES_COLUMNS)
                self._ts_paths[vessel_id] = path
                self._ts_files[vessel_id] = fh
                self._ts_writers[vessel_id] = writer
            self._ts_writers[vessel_id].writerow([
                f"{elapsed_s:.3f}", vessel_id, vessel_label,
                f"{mean_a_star:.4f}", f"{mean_delta_e:.4f}",
            ])
            self._ts_files[vessel_id].flush()

    def close(self) -> None:
        with self._lock:
            self._events_fh.close()
            for fh in self._ts_files.values():
                fh.close()
