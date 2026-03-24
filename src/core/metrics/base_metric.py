"""Abstract base class for all metrics."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseMetric(ABC):
    """Base class for all mixing metrics."""

    @abstractmethod
    def compute(
        self, frame: np.ndarray, reference_frame: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute metric for a single frame.

        Args:
            frame: Current frame (RGB or grayscale).
            reference_frame: Reference frame for comparison.
            mask: Optional binary mask (1 = valid, 0 = masked).

        Returns:
            Dict with metric results.
        """
        ...
