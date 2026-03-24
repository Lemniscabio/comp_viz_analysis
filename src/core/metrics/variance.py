"""Variance metric: spatial variation of cell-averaged colors."""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from src.core.grid_analyzer import GridAnalyzer
from src.core.metrics.base_metric import BaseMetric


class VarianceMetric(BaseMetric):
    """Computes per-channel cell-based color variance across an N×N grid."""

    def __init__(self, grid_rows: int = 5, grid_cols: int = 5) -> None:
        self._grid = GridAnalyzer(rows=grid_rows, cols=grid_cols)

    def compute(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Not called directly — use compute_variance instead.

        Intentional interface deviation: Variance needs precomputed per-cell delta-E
        values from DeltaEMetric, so it can't follow the standard compute() signature.
        AnalysisEngine calls compute_variance() explicitly after Delta E runs.
        """
        raise NotImplementedError("Use compute_variance() with precomputed data")

    def compute_variance(
        self,
        rgb_frame: np.ndarray,
        lab_frame: np.ndarray,
        cell_delta_e: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute per-channel variance of cell-averaged colors.

        Args:
            rgb_frame: (H, W, 3) uint8 RGB image.
            lab_frame: (H, W, 3) float CIE-L*a*b* image.
            cell_delta_e: (N_cells,) array of per-cell ΔE values. NaN = invalid cell.
            mask: (H, W) uint8 mask, 1=keep, 0=exclude. None means keep all.

        Returns:
            Dict with keys: variance_r, variance_g, variance_b,
                            variance_l, variance_a, variance_b_star,
                            variance_delta_e.
        """
        rgb_avgs = self._grid.compute_cell_averages(rgb_frame, mask)
        lab_avgs = self._grid.compute_cell_averages(lab_frame, mask)
        h, w = rgb_frame.shape[:2]

        if mask is not None:
            valid_cells = self._grid.get_valid_cells(mask, h, w)
            valid_mask = np.array(valid_cells, dtype=bool)
        else:
            valid_mask = np.ones(len(rgb_avgs), dtype=bool)

        # Also exclude cells where delta_e is NaN
        valid_de = ~np.isnan(cell_delta_e)
        valid_mask = valid_mask & valid_de

        def _var(arr: np.ndarray) -> float:
            vals = arr[valid_mask]
            if len(vals) < 2:
                return 0.0
            return float(np.var(vals))

        return {
            "variance_r": _var(rgb_avgs[:, 0]),
            "variance_g": _var(rgb_avgs[:, 1]),
            "variance_b": _var(rgb_avgs[:, 2]),
            "variance_l": _var(lab_avgs[:, 0]),
            "variance_a": _var(lab_avgs[:, 1]),
            "variance_b_star": _var(lab_avgs[:, 2]),
            "variance_delta_e": _var(cell_delta_e),
        }
