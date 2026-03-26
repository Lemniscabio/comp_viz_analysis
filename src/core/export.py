"""Data exporter: CSV and XLSX output for time series results."""
from __future__ import annotations
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger("kineticolor")

COLUMNS = [
    "frame_number", "timestamp",
    "grand_delta_e", "normalized_delta_e",
    "contact_perimeter",
    "contrast", "homogeneity", "energy",
    "variance_r", "variance_g", "variance_b",
    "variance_l", "variance_a", "variance_b_star",
    "variance_delta_e",
]


class DataExporter:
    """Exporter for time series metric data to CSV and XLSX formats."""

    def _add_normalized_delta_e(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add normalized_delta_e column (0-1 range, divided by max)."""
        if not results:
            return results
        max_de = max(r.get("grand_delta_e", 0) for r in results)
        if max_de <= 0:
            max_de = 1.0
        enriched = []
        for row in results:
            new_row = dict(row)
            new_row["normalized_delta_e"] = row.get("grand_delta_e", 0) / max_de
            enriched.append(new_row)
        return enriched

    def export(self, results: List[Dict[str, Any]], output_path: Union[Path, str],
               fmt: str = "csv") -> None:
        output_path = Path(output_path)
        enriched = self._add_normalized_delta_e(results)
        if fmt == "csv":
            self._export_csv(enriched, output_path)
        elif fmt == "xlsx":
            self._export_xlsx(enriched, output_path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        logger.info(f"Exported {len(enriched)} rows to {output_path}")

    def _export_csv(self, results: List[Dict[str, Any]], path: Path) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    def _export_xlsx(self, results: List[Dict[str, Any]], path: Path) -> None:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Metrics"
        ws.append(COLUMNS)
        for row in results:
            ws.append([row.get(col) for col in COLUMNS])
        wb.save(path)
