"""Tests for homogeneity metric."""
from __future__ import annotations

import numpy as np

from src.core.metrics.homogeneity import HomogeneityMetric
from src.core.metrics.glcm import GLCMBuilder


class TestHomogeneity:
    def test_uniform_max(self):
        """A perfectly uniform image has homogeneity = 1.0 (all mass on diagonal)."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        gray = np.full((20, 20), 100, dtype=np.uint8)
        glcm = builder.build(gray)
        metric = HomogeneityMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert abs(result["homogeneity"] - 1.0) < 1e-10

    def test_noisy_lower(self):
        """Random noise produces homogeneity well below 0.5."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 0))
        np.random.seed(42)
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        glcm = builder.build(gray)
        metric = HomogeneityMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert result["homogeneity"] < 0.5

    def test_raises_without_glcm(self):
        """compute() raises ValueError when glcm is not provided."""
        metric = HomogeneityMetric()
        gray = np.full((10, 10), 50, dtype=np.uint8)
        try:
            metric.compute(gray, gray)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_returns_dict_with_homogeneity_key(self):
        """Return value is a dict containing the 'homogeneity' key."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        gray = np.full((10, 10), 200, dtype=np.uint8)
        glcm = builder.build(gray)
        metric = HomogeneityMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert "homogeneity" in result
        assert isinstance(result["homogeneity"], float)

    def test_bounded_zero_to_one(self):
        """Homogeneity is in [0, 1] for any valid normalized GLCM."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        np.random.seed(7)
        gray = np.random.randint(0, 256, (40, 40), dtype=np.uint8)
        glcm = builder.build(gray)
        metric = HomogeneityMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert 0.0 <= result["homogeneity"] <= 1.0
