"""Tests for Delta E metric."""
from __future__ import annotations
import numpy as np
from src.core.metrics.delta_e import DeltaEMetric

class TestDeltaE:
    def test_identical_frames_zero(self):
        metric = DeltaEMetric(grid_rows=5, grid_cols=5)
        lab = np.full((100, 100, 3), [50.0, 0.0, 0.0])
        result = metric.compute(lab, lab)
        assert result["grand_delta_e"] == 0.0

    def test_known_delta_e(self):
        metric = DeltaEMetric(grid_rows=5, grid_cols=5)
        frame = np.full((10, 10, 3), [50.0, 10.0, 0.0])
        ref = np.full((10, 10, 3), [50.0, 0.0, 0.0])
        result = metric.compute(frame, ref)
        assert abs(result["grand_delta_e"] - 10.0) < 0.01

    def test_pixel_delta_e_shape(self):
        metric = DeltaEMetric(grid_rows=2, grid_cols=2)
        lab = np.random.rand(20, 30, 3) * 100
        ref = np.random.rand(20, 30, 3) * 100
        result = metric.compute(lab, ref)
        assert result["pixel_delta_e"].shape == (20, 30)

    def test_row_col_averages(self):
        metric = DeltaEMetric(grid_rows=5, grid_cols=5)
        lab = np.random.rand(100, 100, 3) * 100
        ref = np.random.rand(100, 100, 3) * 100
        result = metric.compute(lab, ref)
        assert result["row_avg"].shape == (5,)
        assert result["col_avg"].shape == (5,)
        assert result["cell_avg"].shape == (25,)

    def test_with_mask(self):
        metric = DeltaEMetric(grid_rows=1, grid_cols=1)
        frame = np.full((10, 10, 3), [80.0, 0.0, 0.0])
        ref = np.full((10, 10, 3), [50.0, 0.0, 0.0])
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0, 0] = 1
        result = metric.compute(frame, ref, mask)
        assert abs(result["grand_delta_e"] - 30.0) < 0.01
