"""Tests for src/core/batch.py."""
from __future__ import annotations

import json
from pathlib import Path

from src.core.batch import (
    BATCH_CSV_COLUMNS,
    write_batch_config,
    write_summary_row,
)
from src.core.mixing_time import MixingTimeResult


def test_summary_row_has_all_columns(tmp_path):
    csv = tmp_path / "summary.csv"
    r = MixingTimeResult(t_start_s=1.0, levels=(0.90, 0.95, 0.99))
    r.t_mix = {0.90: 5.0, 0.95: 8.0, 0.99: 11.0}
    r.t_deltaE = {0.95: 6.0}
    r.t_spatial = {0.95: 7.0}
    r.t_texture = {0.95: 8.0}
    write_summary_row(
        csv, video_file="a.mp4", fps=30.0, duration_s=20.0,
        frame_count=600, roi=(0, 0, 100, 100), result=r, append=False,
    )
    text = csv.read_text()
    header = text.split("\n")[0]
    for col in BATCH_CSV_COLUMNS:
        assert col in header, f"missing column: {col}"
    # Row should be present
    assert "a.mp4" in text


def test_summary_row_appends(tmp_path):
    csv = tmp_path / "summary.csv"
    r = MixingTimeResult(t_start_s=0.0, levels=(0.90, 0.95, 0.99))
    write_summary_row(csv, video_file="a.mp4", fps=30.0, duration_s=10.0,
                      frame_count=300, roi=None, result=r, append=False)
    write_summary_row(csv, video_file="b.mp4", fps=30.0, duration_s=10.0,
                      frame_count=300, roi=None, result=r, append=True)
    lines = csv.read_text().strip().split("\n")
    # 1 header + 2 rows
    assert len(lines) == 3
    assert "a.mp4" in lines[1]
    assert "b.mp4" in lines[2]


def test_batch_config_json_roundtrip(tmp_path):
    out = tmp_path / "batch_config.json"
    write_batch_config(
        out, roi=(1, 2, 3, 4), mask_present=True,
        grid=(5, 5), params=None,
    )
    data = json.loads(out.read_text())
    assert data["roi"] == [1, 2, 3, 4]
    assert data["mask_present"] is True
    assert data["grid_rows"] == 5
    assert data["grid_cols"] == 5
