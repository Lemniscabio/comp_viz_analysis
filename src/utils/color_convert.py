"""RGB <-> CIE-L*a*b* color space conversion helpers."""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab, lab2rgb


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 image to CIE-L*a*b* float64.

    Args:
        image: RGB image with shape (H, W, 3) and dtype uint8.

    Returns:
        CIE-L*a*b* image with shape (H, W, 3) and dtype float64.
    """
    return rgb2lab(image)


def lab_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a CIE-L*a*b* float64 image to RGB uint8.

    Args:
        image: CIE-L*a*b* image with shape (H, W, 3) and dtype float64.

    Returns:
        RGB image with shape (H, W, 3) and dtype uint8, clipped to [0, 255].
    """
    rgb_float = lab2rgb(image)
    return np.clip(rgb_float * 255, 0, 255).astype(np.uint8)
