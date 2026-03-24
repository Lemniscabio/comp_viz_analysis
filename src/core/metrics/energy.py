"""Energy (Angular Second Moment / ASM) metric from GLCM."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.core.metrics.base_metric import BaseMetric


class EnergyMetric(BaseMetric):
    """Computes energy (ASM) from a precomputed normalized GLCM.

    ASM = Σᵢ Σⱼ (p_ij)²

    High values indicate large uniform regions (single dominant gray level).
    Low values indicate random or highly varied textures.
    Reaches maximum (1.0) when the image is perfectly uniform.
    """

    def compute(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
        glcm: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute energy (ASM) from a precomputed GLCM.

        Args:
            frame: Current frame (unused directly; GLCM encodes texture).
            reference_frame: Reference frame (unused directly).
            mask: Optional exclusion mask (handled upstream in GLCM building).
            glcm: Precomputed normalized GLCM of shape (N, N). Required.

        Returns:
            Dict with key ``"energy"`` (float in (0, 1]).

        Raises:
            ValueError: If glcm is None.
        """
        if glcm is None:
            raise ValueError("EnergyMetric requires a precomputed GLCM")
        energy = float(np.sum(glcm ** 2))
        return {"energy": energy}
