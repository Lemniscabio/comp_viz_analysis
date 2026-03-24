"""Contrast metric from GLCM."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.core.metrics.base_metric import BaseMetric


class ContrastMetric(BaseMetric):
    """Computes contrast from a precomputed normalized GLCM.

    Contrast = Σᵢ Σⱼ |i - j|² · p_ij

    High values indicate a heterogeneous (visually varied) image.
    Approaches 0 as the image becomes uniform (mixing complete).
    """

    def compute(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
        glcm: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute contrast from a precomputed GLCM.

        Args:
            frame: Current frame (unused directly; GLCM encodes texture).
            reference_frame: Reference frame (unused directly).
            mask: Optional exclusion mask (handled upstream in GLCM building).
            glcm: Precomputed normalized GLCM of shape (N, N). Required.

        Returns:
            Dict with key ``"contrast"`` (float).

        Raises:
            ValueError: If glcm is None.
        """
        if glcm is None:
            raise ValueError("ContrastMetric requires a precomputed GLCM")
        n = glcm.shape[0]
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        contrast = float(np.sum((i - j) ** 2 * glcm))
        return {"contrast": contrast}
