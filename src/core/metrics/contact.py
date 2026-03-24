"""Contact metric: binary threshold perimeter between light/dark regions."""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from src.core.metrics.base_metric import BaseMetric


class ContactMetric(BaseMetric):
    """
    Contact metric measures the perimeter of contact between light and dark regions.

    Detects edges where binary (thresholded) pixels change value.
    Uses 4-connectivity (horizontal and vertical neighbors only).
    """

    def __init__(self, threshold: int = 128) -> None:
        """
        Initialize ContactMetric.

        Args:
            threshold: Grayscale threshold for binary classification (0-255).
                      Pixels >= threshold are white (1), below are black (0).
        """
        self._threshold = threshold

    def compute(
        self, frame: np.ndarray, reference_frame: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute contact perimeter for a frame.

        Args:
            frame: Grayscale frame (H x W).
            reference_frame: Reference frame (unused for contact metric).
            mask: Optional binary mask (1 = valid, 0 = masked).
                  Only contacts with both neighbors valid count.

        Returns:
            Dict with key "contact_perimeter" containing edge count (int).
        """
        # Convert frame to binary using threshold
        binary = (frame >= self._threshold).astype(np.uint8)

        # If no mask provided, all pixels are valid
        if mask is None:
            mask = np.ones_like(binary, dtype=np.uint8)

        # Horizontal edges (left-right transitions)
        h_left = binary[:, :-1]
        h_right = binary[:, 1:]
        m_left = mask[:, :-1]
        m_right = mask[:, 1:]
        h_contact = np.sum(
            (h_left != h_right) & (m_left > 0) & (m_right > 0)
        )

        # Vertical edges (top-bottom transitions)
        v_top = binary[:-1, :]
        v_bottom = binary[1:, :]
        m_top = mask[:-1, :]
        m_bottom = mask[1:, :]
        v_contact = np.sum(
            (v_top != v_bottom) & (m_top > 0) & (m_bottom > 0)
        )

        return {"contact_perimeter": int(h_contact + v_contact)}
