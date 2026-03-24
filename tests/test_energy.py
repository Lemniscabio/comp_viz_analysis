"""Tests for energy (ASM) metric."""
from __future__ import annotations

import numpy as np

from src.core.metrics.energy import EnergyMetric
from src.core.metrics.glcm import GLCMBuilder


class TestEnergy:
    def test_uniform_max(self):
        """A perfectly uniform image has energy = 1.0 (all mass in one GLCM cell)."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        gray = np.full((20, 20), 100, dtype=np.uint8)
        glcm = builder.build(gray)
        metric = EnergyMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert abs(result["energy"] - 1.0) < 1e-10

    def test_random_low(self):
        """Random noise produces energy well below 0.1."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 0))
        np.random.seed(42)
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        glcm = builder.build(gray)
        metric = EnergyMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert result["energy"] < 0.1

    def test_raises_without_glcm(self):
        """compute() raises ValueError when glcm is not provided."""
        metric = EnergyMetric()
        gray = np.full((10, 10), 50, dtype=np.uint8)
        try:
            metric.compute(gray, gray)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_returns_dict_with_energy_key(self):
        """Return value is a dict containing the 'energy' key."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        gray = np.full((10, 10), 50, dtype=np.uint8)
        glcm = builder.build(gray)
        metric = EnergyMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert "energy" in result
        assert isinstance(result["energy"], float)

    def test_bounded_zero_to_one(self):
        """Energy is in (0, 1] for any valid normalized GLCM."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        np.random.seed(3)
        gray = np.random.randint(0, 256, (40, 40), dtype=np.uint8)
        glcm = builder.build(gray)
        metric = EnergyMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert 0.0 < result["energy"] <= 1.0
