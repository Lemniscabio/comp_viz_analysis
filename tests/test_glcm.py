"""Tests for GLCM builder."""
from __future__ import annotations
import numpy as np
from src.core.metrics.glcm import GLCMBuilder

class TestGLCMBuilder:
    def test_uniform_image(self):
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        gray = np.full((10, 10), 128, dtype=np.uint8)
        glcm = builder.build(gray)
        assert glcm.shape == (16, 16)
        assert abs(np.sum(glcm) - 1.0) < 1e-10
        level = 128 * 16 // 256
        assert glcm[level, level] == 1.0

    def test_normalized(self):
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        glcm = builder.build(gray)
        assert abs(np.sum(glcm) - 1.0) < 1e-10

    def test_mask_excludes_pairs(self):
        builder = GLCMBuilder(gray_levels=16, offset=(1, 0))
        gray = np.zeros((5, 5), dtype=np.uint8)
        gray[:, 2:] = 255
        mask = np.ones((5, 5), dtype=np.uint8)
        mask[:, 2] = 0
        glcm = builder.build(gray, mask)
        assert abs(np.sum(glcm) - 1.0) < 1e-10
        off_diag = np.sum(glcm) - np.trace(glcm)
        assert abs(off_diag) < 1e-10

    def test_fully_masked_returns_zeros(self):
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        gray = np.full((10, 10), 100, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        glcm = builder.build(gray, mask)
        assert np.sum(glcm) == 0.0
