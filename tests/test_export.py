"""Tests for data exporter."""
from __future__ import annotations
import csv
from pathlib import Path
from src.core.export import DataExporter

SAMPLE_RESULTS = [
    {
        "frame_number": 0, "timestamp": 0.0,
        "grand_delta_e": 0.0, "contact_perimeter": 0,
        "contrast": 0.0, "homogeneity": 1.0, "energy": 1.0,
        "variance_r": 0.0, "variance_g": 0.0, "variance_b": 0.0,
        "variance_l": 0.0, "variance_a": 0.0, "variance_b_star": 0.0,
        "variance_delta_e": 0.0,
    },
    {
        "frame_number": 1, "timestamp": 0.033,
        "grand_delta_e": 5.2, "contact_perimeter": 42,
        "contrast": 3.1, "homogeneity": 0.8, "energy": 0.6,
        "variance_r": 10.0, "variance_g": 8.0, "variance_b": 12.0,
        "variance_l": 5.0, "variance_a": 2.0, "variance_b_star": 3.0,
        "variance_delta_e": 4.5,
    },
]

class TestDataExporter:
    def test_export_csv(self, tmp_path):
        out = tmp_path / "results.csv"
        exporter = DataExporter()
        exporter.export(SAMPLE_RESULTS, out, fmt="csv")
        assert out.exists()
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["frame_number"] == "0"
        assert rows[1]["grand_delta_e"] == "5.2"

    def test_export_xlsx(self, tmp_path):
        out = tmp_path / "results.xlsx"
        exporter = DataExporter()
        exporter.export(SAMPLE_RESULTS, out, fmt="xlsx")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_results(self, tmp_path):
        out = tmp_path / "empty.csv"
        exporter = DataExporter()
        exporter.export([], out, fmt="csv")
        assert out.exists()
