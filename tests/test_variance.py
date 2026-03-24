"""Tests for variance metric."""
from __future__ import annotations
import numpy as np
from src.core.metrics.variance import VarianceMetric

class TestVariance:
    def test_uniform_zero(self):
        metric = VarianceMetric(grid_rows=5, grid_cols=5)
        rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
        lab = np.full((100, 100, 3), [50.0, 0.0, 0.0])
        cell_delta_e = np.zeros(25)
        result = metric.compute_variance(rgb, lab, cell_delta_e)
        assert result["variance_r"] == 0.0
        assert result["variance_g"] == 0.0
        assert result["variance_b"] == 0.0
        assert result["variance_l"] == 0.0
        assert result["variance_a"] == 0.0
        assert result["variance_b_star"] == 0.0
        assert result["variance_delta_e"] == 0.0

    def test_different_quadrants_high(self):
        metric = VarianceMetric(grid_rows=2, grid_cols=2)
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:50, :50] = [255, 0, 0]
        rgb[:50, 50:] = [0, 255, 0]
        rgb[50:, :50] = [0, 0, 255]
        rgb[50:, 50:] = [255, 255, 0]
        lab = np.random.rand(100, 100, 3) * 50
        cell_delta_e = np.array([10.0, 20.0, 30.0, 40.0])
        result = metric.compute_variance(rgb, lab, cell_delta_e)
        assert result["variance_r"] > 0
        assert result["variance_delta_e"] > 0

    def test_with_mask(self):
        metric = VarianceMetric(grid_rows=2, grid_cols=2)
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:50, :50] = [255, 0, 0]
        rgb[:50, 50:] = [100, 100, 100]
        rgb[50:, :50] = [100, 100, 100]
        rgb[50:, 50:] = [100, 100, 100]
        lab = np.full((100, 100, 3), [50.0, 0.0, 0.0])
        cell_delta_e = np.array([np.nan, 5.0, 5.0, 5.0])
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[:50, :50] = 0
        result = metric.compute_variance(rgb, lab, cell_delta_e, mask)
        assert result["variance_r"] == 0.0
