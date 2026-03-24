"""Homogeneity metric from GLCM."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.core.metrics.base_metric import BaseMetric


class HomogeneityMetric(BaseMetric):
    """Computes homogeneity from a precomputed normalized GLCM.

    H = Σᵢ Σⱼ p_ij / (1 + |i - j|)

    High values (→ 1) indicate a uniform/homogeneous image (mixing complete).
    Low values indicate a heterogeneous image (mixing in progress).
    """

    def compute(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
        glcm: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute homogeneity from a precomputed GLCM.

        Args:
            frame: Current frame (unused directly; GLCM encodes texture).
            reference_frame: Reference frame (unused directly).
            mask: Optional exclusion mask (handled upstream in GLCM building).
            glcm: Precomputed normalized GLCM of shape (N, N). Required.

        Returns:
            Dict with key ``"homogeneity"`` (float in [0, 1]).

        Raises:
            ValueError: If glcm is None.
        """
        if glcm is None:
            raise ValueError("HomogeneityMetric requires a precomputed GLCM")
        n = glcm.shape[0]
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        homogeneity = float(np.sum(glcm / (1 + np.abs(i - j))))
        return {"homogeneity": homogeneity}
