"""Tests for grid analyzer."""

from __future__ import annotations

import numpy as np
from src.core.grid_analyzer import GridAnalyzer


class TestGridAnalyzer:
    def test_cell_coordinates(self):
        """5x5 grid on a 100x100 image gives 20x20 cells."""
        ga = GridAnalyzer(rows=5, cols=5)
        cells = ga.get_cell_coords(height=100, width=100)
        assert len(cells) == 25
        assert cells[0] == (0, 0, 20, 20)  # (y, x, h, w)
        assert cells[24] == (80, 80, 20, 20)

    def test_cell_averages_uniform(self):
        """Uniform image: all cell averages should be equal."""
        ga = GridAnalyzer(rows=5, cols=5)
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        avgs = ga.compute_cell_averages(img)
        assert avgs.shape == (25, 3)
        assert np.allclose(avgs, 128.0)

    def test_cell_averages_different_quadrants(self):
        """Image with 4 distinct quadrants should produce different averages."""
        ga = GridAnalyzer(rows=2, cols=2)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :50] = [255, 0, 0]
        img[:50, 50:] = [0, 255, 0]
        img[50:, :50] = [0, 0, 255]
        img[50:, 50:] = [255, 255, 0]
        avgs = ga.compute_cell_averages(img)
        assert avgs.shape == (4, 3)
        assert np.allclose(avgs[0], [255, 0, 0])
        assert np.allclose(avgs[1], [0, 255, 0])

    def test_masked_cell_flagging(self):
        """Cells >50% masked are flagged invalid."""
        ga = GridAnalyzer(rows=2, cols=2)
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[:50, :50] = 0  # fully mask top-left cell
        valid = ga.get_valid_cells(mask, height=100, width=100)
        assert valid[0] == False
        assert valid[1] == True
        assert valid[2] == True
        assert valid[3] == True

    def test_cell_averages_with_mask(self):
        """Masked pixels excluded from cell average computation."""
        ga = GridAnalyzer(rows=1, cols=1)
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        img[5:, :] = 200
        mask = np.ones((10, 10), dtype=np.uint8)
        mask[5:, :] = 0  # mask out the 200 region
        avgs = ga.compute_cell_averages(img, mask)
        assert np.allclose(avgs[0], 100.0)
