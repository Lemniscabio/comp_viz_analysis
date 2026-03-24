"""GLCM (Gray-Level Co-occurrence Matrix) builder."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


class GLCMBuilder:
    """Builds a normalized Gray-Level Co-occurrence Matrix (GLCM) for a grayscale image.

    The GLCM is shared by the Contrast, Homogeneity, and Energy metrics — it is
    computed once per frame and passed to each of those consumers.

    Args:
        gray_levels: Number of quantized gray levels (default 16).
        offset: Pixel pair offset as (dx, dy) — default (1, 1) means one pixel
            right and one pixel down.
    """

    def __init__(self, gray_levels: int = 16, offset: Tuple[int, int] = (1, 1)) -> None:
        self._gray_levels = gray_levels
        self._dx, self._dy = offset

    @property
    def gray_levels(self) -> int:
        """Number of quantized gray levels."""
        return self._gray_levels

    def build(self, grayscale: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Build and return the normalized GLCM for a grayscale image.

        Args:
            grayscale: 2-D uint8 array (H x W) of grayscale pixel values (0–255).
            mask: Optional 2-D uint8 array (H x W). Pixels with value 0 are
                excluded; both pixels of a pair must be unmasked to be counted.

        Returns:
            Normalized GLCM as a float64 array of shape (gray_levels, gray_levels).
            All values sum to 1.0, or all zeros if no valid pairs exist.
        """
        quantized = (grayscale.astype(np.int32) * self._gray_levels // 256).clip(
            0, self._gray_levels - 1
        )
        h, w = quantized.shape
        dx, dy = self._dx, self._dy

        # Build source/target slices for the given offset.
        if dy >= 0:
            src_rows = slice(0, h - dy) if dy > 0 else slice(0, h)
            tgt_rows = slice(dy, h) if dy > 0 else slice(0, h)
        else:
            src_rows = slice(-dy, h)
            tgt_rows = slice(0, h + dy)

        if dx >= 0:
            src_cols = slice(0, w - dx) if dx > 0 else slice(0, w)
            tgt_cols = slice(dx, w) if dx > 0 else slice(0, w)
        else:
            src_cols = slice(-dx, w)
            tgt_cols = slice(0, w + dx)

        src = quantized[src_rows, src_cols].ravel()
        tgt = quantized[tgt_rows, tgt_cols].ravel()

        if mask is not None:
            m_src = mask[src_rows, src_cols].ravel()
            m_tgt = mask[tgt_rows, tgt_cols].ravel()
            valid = (m_src > 0) & (m_tgt > 0)
            src = src[valid]
            tgt = tgt[valid]

        glcm = np.zeros((self._gray_levels, self._gray_levels), dtype=np.float64)
        if len(src) == 0:
            return glcm

        np.add.at(glcm, (src, tgt), 1)
        total = np.sum(glcm)
        if total > 0:
            glcm /= total
        return glcm
