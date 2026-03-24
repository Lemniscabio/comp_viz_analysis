"""Data exporter: CSV and XLSX output for time series results."""
from __future__ import annotations
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger("kineticolor")

COLUMNS = [
    "frame_number", "timestamp",
    "grand_delta_e", "contact_perimeter",
    "contrast", "homogeneity", "energy",
    "variance_r", "variance_g", "variance_b",
    "variance_l", "variance_a", "variance_b_star",
    "variance_delta_e",
]


class DataExporter:
    """Exporter for time series metric data to CSV and XLSX formats."""

    def export(self, results: List[Dict[str, Any]], output_path: Union[Path, str],
               fmt: str = "csv") -> None:
        """Export results to CSV or XLSX.

        Args:
            results: List of result dictionaries with metric data.
            output_path: Path to output file.
            fmt: Export format ('csv' or 'xlsx').

        Raises:
            ValueError: If format is not supported.
        """
        output_path = Path(output_path)
        if fmt == "csv":
            self._export_csv(results, output_path)
        elif fmt == "xlsx":
            self._export_xlsx(results, output_path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        logger.info(f"Exported {len(results)} rows to {output_path}")

    def _export_csv(self, results: List[Dict[str, Any]], path: Path) -> None:
        """Export results to CSV file.

        Args:
            results: List of result dictionaries.
            path: Output file path.
        """
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    def _export_xlsx(self, results: List[Dict[str, Any]], path: Path) -> None:
        """Export results to XLSX file.

        Args:
            results: List of result dictionaries.
            path: Output file path.
        """
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Metrics"
        ws.append(COLUMNS)
        for row in results:
            ws.append([row.get(col) for col in COLUMNS])
        wb.save(path)
