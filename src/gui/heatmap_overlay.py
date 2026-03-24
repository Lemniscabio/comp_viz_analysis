"""Heatmap overlay: renders per-pixel delta-E as a color-mapped overlay."""
from __future__ import annotations

import cv2
import numpy as np


def create_heatmap_overlay(
    frame_bgr: np.ndarray,
    pixel_delta_e: np.ndarray,
    opacity: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a delta-E heatmap onto a BGR frame.

    Args:
        frame_bgr: (H, W, 3) BGR uint8 frame.
        pixel_delta_e: (H, W) float64 per-pixel delta-E values.
        opacity: Blend factor for heatmap (0=frame only, 1=heatmap only).
        colormap: OpenCV colormap constant.

    Returns:
        (H, W, 3) BGR uint8 blended frame.
    """
    de_max = np.max(pixel_delta_e) if np.max(pixel_delta_e) > 0 else 1.0
    normalized = np.clip(pixel_delta_e / de_max * 255, 0, 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(normalized, colormap)

    if heatmap.shape[:2] != frame_bgr.shape[:2]:
        heatmap = cv2.resize(heatmap, (frame_bgr.shape[1], frame_bgr.shape[0]))

    blended = cv2.addWeighted(frame_bgr, 1.0 - opacity, heatmap, opacity, 0)
    return blended
