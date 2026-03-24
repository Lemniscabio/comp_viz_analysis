"""Grid analyzer: divides ROI into N×N cells for spatial analysis."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class GridAnalyzer:
    """Divides an image region into a grid of cells."""

    def __init__(self, rows: int = 5, cols: int = 5) -> None:
        self._rows = rows
        self._cols = cols

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    def get_cell_coords(
        self, height: int, width: int
    ) -> List[Tuple[int, int, int, int]]:
        """Get (y, x, h, w) coordinates for each cell. Row-major order."""
        cell_h = height // self._rows
        cell_w = width // self._cols
        cells = []
        for r in range(self._rows):
            for c in range(self._cols):
                y = r * cell_h
                x = c * cell_w
                h = cell_h if r < self._rows - 1 else height - y
                w = cell_w if c < self._cols - 1 else width - x
                cells.append((y, x, h, w))
        return cells

    def compute_cell_averages(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute average color per cell.

        Args:
            image: (H, W, C) image array.
            mask: (H, W) uint8, 1=keep, 0=exclude. None means keep all.

        Returns:
            (N_cells, C) array of per-cell average values.
        """
        h, w = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1
        cells = self.get_cell_coords(h, w)
        averages = np.zeros((len(cells), channels), dtype=np.float64)

        for i, (cy, cx, ch, cw) in enumerate(cells):
            cell_data = image[cy : cy + ch, cx : cx + cw]
            if mask is not None:
                cell_mask = mask[cy : cy + ch, cx : cx + cw]
                valid = cell_mask > 0
                if not np.any(valid):
                    averages[i] = np.nan
                    continue
                if cell_data.ndim == 3:
                    averages[i] = np.mean(cell_data[valid], axis=0)
                else:
                    averages[i, 0] = np.mean(cell_data[valid])
            else:
                if cell_data.ndim == 3:
                    averages[i] = np.mean(cell_data.reshape(-1, channels), axis=0)
                else:
                    averages[i, 0] = np.mean(cell_data)

        return averages

    def get_valid_cells(
        self, mask: np.ndarray, height: int, width: int
    ) -> List[bool]:
        """Determine which cells are valid (>50% unmasked).

        Args:
            mask: (H, W) uint8, 1=keep, 0=exclude.
            height: Height of the image region.
            width: Width of the image region.

        Returns:
            List of booleans, one per cell, True if >50% of pixels are unmasked.
        """
        cells = self.get_cell_coords(height, width)
        valid = []
        for cy, cx, ch, cw in cells:
            cell_mask = mask[cy : cy + ch, cx : cx + cw]
            fraction_unmasked = np.mean(cell_mask > 0)
            valid.append(fraction_unmasked > 0.5)
        return valid
