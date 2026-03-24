"""Tests for contact metric."""
from __future__ import annotations
import numpy as np
from src.core.metrics.contact import ContactMetric


class TestContact:
    def test_uniform_image_zero(self) -> None:
        """Uniform image should have zero contact perimeter."""
        metric = ContactMetric(threshold=128)
        gray = np.full((10, 10), 200, dtype=np.uint8)
        result = metric.compute(gray, gray)
        assert result["contact_perimeter"] == 0

    def test_half_black_white(self) -> None:
        """Half-black, half-white image should have contact along the edge."""
        metric = ContactMetric(threshold=128)
        gray = np.zeros((10, 10), dtype=np.uint8)
        gray[:, 5:] = 255
        result = metric.compute(gray, gray)
        assert result["contact_perimeter"] == 10

    def test_checkerboard_high(self) -> None:
        """Checkerboard pattern should have high contact perimeter."""
        metric = ContactMetric(threshold=128)
        gray = np.zeros((4, 4), dtype=np.uint8)
        gray[0, 1] = 255
        gray[0, 3] = 255
        gray[1, 0] = 255
        gray[1, 2] = 255
        gray[2, 1] = 255
        gray[2, 3] = 255
        gray[3, 0] = 255
        gray[3, 2] = 255
        result = metric.compute(gray, gray)
        assert result["contact_perimeter"] > 0

    def test_mask_excludes_boundary(self) -> None:
        """Mask should exclude contact at masked boundaries."""
        metric = ContactMetric(threshold=128)
        gray = np.zeros((10, 10), dtype=np.uint8)
        gray[:, 5:] = 255
        mask = np.ones((10, 10), dtype=np.uint8)
        mask[:, 4:6] = 0
        result = metric.compute(gray, gray, mask)
        assert result["contact_perimeter"] == 0
