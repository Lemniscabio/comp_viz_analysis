"""Frame processing: ROI extraction, mask application, color space conversion."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.color_convert import rgb_to_lab

logger = logging.getLogger("kineticolor")


class FrameProcessor:
    """Handles ROI cropping, exclusion mask application, and color conversion."""

    def __init__(self, brightness_change_threshold: float = 0.2) -> None:
        self._brightness_change_threshold = brightness_change_threshold
        self._prev_brightness: Optional[float] = None

    def crop_to_roi(
        self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Crop frame to region of interest.

        Args:
            frame: (H, W, 3) image array.
            roi: (x, y, width, height) or None for full frame.

        Returns:
            Cropped image array.
        """
        if roi is None:
            return frame.copy()
        x, y, w, h = roi
        return frame[y : y + h, x : x + w].copy()

    def apply_mask(
        self, frame: np.ndarray, mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Apply exclusion mask to frame. Masked-out pixels become 0.

        Args:
            frame: (H, W, 3) image array.
            mask: (H, W) uint8 array where 1=keep, 0=exclude. None means keep all.

        Returns:
            Masked image array.
        """
        if mask is None:
            return frame.copy()
        return frame * mask[:, :, np.newaxis]

    def to_lab(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to CIE-L*a*b*.

        Args:
            frame: (H, W, 3) BGR image array.

        Returns:
            (H, W, 3) CIE-L*a*b* float64 array.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_to_lab(rgb)

    def to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to grayscale.

        Args:
            frame: (H, W, 3) BGR image array.

        Returns:
            (H, W) uint8 grayscale array.
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def check_brightness(
        self, frame: np.ndarray, mask: Optional[np.ndarray]
    ) -> None:
        """Check for drastic brightness changes between consecutive frames.

        Logs a WARNING if average brightness changes by more than the threshold
        (relative change). The threshold is a fraction, e.g. 0.2 = 20%.

        Args:
            frame: (H, W, 3) or (H, W) image array.
            mask: Optional (H, W) uint8 array where 1=valid pixels, 0=excluded.
        """
        if frame.ndim == 3:
            gray = np.mean(frame, axis=2)
        else:
            gray = frame.astype(np.float64)

        if mask is not None:
            valid = mask > 0
            if not np.any(valid):
                return
            avg_brightness = float(np.mean(gray[valid]))
        else:
            avg_brightness = float(np.mean(gray))

        if self._prev_brightness is not None and self._prev_brightness > 0:
            change = abs(avg_brightness - self._prev_brightness) / self._prev_brightness
            if change > self._brightness_change_threshold:
                logger.warning(
                    f"Brightness changed by {change:.1%} "
                    f"({self._prev_brightness:.1f} -> {avg_brightness:.1f}). "
                    f"Check lighting consistency."
                )

        self._prev_brightness = avg_brightness
