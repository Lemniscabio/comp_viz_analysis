"""Tests for contrast metric."""
from __future__ import annotations

import numpy as np

from src.core.metrics.contrast import ContrastMetric
from src.core.metrics.glcm import GLCMBuilder


class TestContrast:
    def test_uniform_zero(self):
        """A perfectly uniform image has no contrast between pixel pairs."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        gray = np.full((20, 20), 100, dtype=np.uint8)
        glcm = builder.build(gray)
        metric = ContrastMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert result["contrast"] == 0.0

    def test_checkerboard_high(self):
        """Alternating columns of 0 and 255 produce high contrast."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 0))
        gray = np.zeros((20, 20), dtype=np.uint8)
        gray[:, 1::2] = 255
        glcm = builder.build(gray)
        metric = ContrastMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert result["contrast"] > 100

    def test_raises_without_glcm(self):
        """compute() raises ValueError when glcm is not provided."""
        metric = ContrastMetric()
        gray = np.full((10, 10), 50, dtype=np.uint8)
        try:
            metric.compute(gray, gray)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_returns_dict_with_contrast_key(self):
        """Return value is a dict containing the 'contrast' key."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        gray = np.full((10, 10), 80, dtype=np.uint8)
        glcm = builder.build(gray)
        metric = ContrastMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert "contrast" in result
        assert isinstance(result["contrast"], float)

    def test_non_negative(self):
        """Contrast is always >= 0."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        np.random.seed(0)
        gray = np.random.randint(0, 256, (30, 30), dtype=np.uint8)
        glcm = builder.build(gray)
        metric = ContrastMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert result["contrast"] >= 0.0
