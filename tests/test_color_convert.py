"""Tests for color conversion utilities."""

from __future__ import annotations

import numpy as np
from src.utils.color_convert import rgb_to_lab, lab_to_rgb


class TestRgbToLab:
    def test_black(self):
        """Pure black RGB -> L*=0, a*~0, b*~0."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        lab = rgb_to_lab(img)
        assert lab.shape == (10, 10, 3)
        assert np.allclose(lab[:, :, 0], 0, atol=0.5)

    def test_white(self):
        """Pure white RGB -> L*=100, a*~0, b*~0."""
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        lab = rgb_to_lab(img)
        assert np.allclose(lab[:, :, 0], 100, atol=0.5)

    def test_output_dtype(self):
        """Output should be float64."""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        lab = rgb_to_lab(img)
        assert lab.dtype == np.float64


class TestLabToRgb:
    def test_roundtrip(self):
        """RGB -> LAB -> RGB should be close to original."""
        img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        lab = rgb_to_lab(img)
        recovered = lab_to_rgb(lab)
        assert recovered.dtype == np.uint8
        assert np.allclose(recovered, img, atol=2)
