"""Delta E metric: perceptually uniform color distance from reference frame."""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from src.core.grid_analyzer import GridAnalyzer
from src.core.metrics.base_metric import BaseMetric


class DeltaEMetric(BaseMetric):
    """
    Delta E metric computes perceptually uniform color distance from a reference frame.

    Operates on CIE-L*a*b* frames. Per-pixel ΔE is the Euclidean distance in L*a*b*
    space between the current frame and the reference frame. Grand ΔE is the mean
    across all (valid) pixels. Spatially resolved averages are computed over an N×M
    grid of cells.
    """

    def __init__(self, grid_rows: int = 5, grid_cols: int = 5) -> None:
        """
        Initialize DeltaEMetric.

        Args:
            grid_rows: Number of rows in the spatial grid.
            grid_cols: Number of columns in the spatial grid.
        """
        self._grid = GridAnalyzer(rows=grid_rows, cols=grid_cols)

    def compute(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute Delta E metric for a single frame.

        Args:
            frame: Current frame in CIE-L*a*b* color space (H, W, 3), float64.
            reference_frame: Reference frame in CIE-L*a*b* color space (H, W, 3).
            mask: Optional (H, W) uint8 array where 1=keep, 0=exclude.

        Returns:
            Dict with keys:
                - "grand_delta_e": float -- mean ΔE across all valid pixels.
                - "pixel_delta_e": np.ndarray (H, W) -- per-pixel ΔE values.
                - "row_avg": np.ndarray (grid_rows,) -- per-row mean ΔE.
                - "col_avg": np.ndarray (grid_cols,) -- per-column mean ΔE.
                - "cell_avg": np.ndarray (grid_rows * grid_cols,) -- per-cell mean ΔE,
                              row-major order. NaN for fully masked cells.
        """
        diff = frame.astype(np.float64) - reference_frame.astype(np.float64)
        pixel_de = np.sqrt(np.sum(diff ** 2, axis=2))

        if mask is not None:
            valid = mask > 0
            grand = float(np.mean(pixel_de[valid])) if np.any(valid) else 0.0
        else:
            grand = float(np.mean(pixel_de))

        h, w = pixel_de.shape
        cells = self._grid.get_cell_coords(h, w)
        n_rows, n_cols = self._grid.rows, self._grid.cols
        row_avg = np.zeros(n_rows, dtype=np.float64)
        col_avg = np.zeros(n_cols, dtype=np.float64)
        cell_avg = np.zeros(len(cells), dtype=np.float64)

        for i, (cy, cx, ch, cw) in enumerate(cells):
            cell_de = pixel_de[cy : cy + ch, cx : cx + cw]
            if mask is not None:
                cell_mask = mask[cy : cy + ch, cx : cx + cw]
                cell_valid = cell_mask > 0
                cell_avg[i] = (
                    float(np.mean(cell_de[cell_valid]))
                    if np.any(cell_valid)
                    else np.nan
                )
            else:
                cell_avg[i] = float(np.mean(cell_de))

        cell_avg_grid = cell_avg.reshape(n_rows, n_cols)
        for r in range(n_rows):
            row_vals = cell_avg_grid[r][~np.isnan(cell_avg_grid[r])]
            row_avg[r] = float(np.mean(row_vals)) if len(row_vals) > 0 else np.nan
        for c in range(n_cols):
            col_vals = cell_avg_grid[:, c][~np.isnan(cell_avg_grid[:, c])]
            col_avg[c] = float(np.mean(col_vals)) if len(col_vals) > 0 else np.nan

        return {
            "grand_delta_e": grand,
            "pixel_delta_e": pixel_de,
            "row_avg": row_avg,
            "col_avg": col_avg,
            "cell_avg": cell_avg,
        }
