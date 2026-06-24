"""Per-vessel colorimetric analysis.

Pure OpenCV + NumPy — no GUI imports, no src/ imports.
Reimplements the Lab conversion and Delta-E math locally (~10 lines).
"""

import numpy as np
import cv2

from .constants import PINK_A_STAR_THRESHOLD

_RESIZE_SCALE = 0.25  # crop is downsampled to 25% before Lab/Delta-E (10× throughput)


class VesselAnalyzer:
    """Computes mixing metrics for a single vessel crop.

    All computation is stateless per-frame; state (reference, reference mean a*)
    is captured at construction time.

    Delta-E is computed on a 25%-scaled copy of the crop. mean a* is also
    computed on the scaled copy — the difference from full-res is negligible
    (<0.1 unit) because both metrics are spatial means.
    """

    def __init__(self, reference_frame: np.ndarray) -> None:
        """
        Args:
            reference_frame: BGR crop of the vessel at the reference (arm) time.
        """
        ref_small = _resize_quarter(reference_frame)
        self._ref_lab = _bgr_to_lab_float32(ref_small)
        self._ref_mean_a = float(self._ref_lab[:, :, 1].mean())

    def analyze(self, frame_bgr: np.ndarray) -> dict:
        """Analyze a single frame crop.

        Returns:
            {
                "mean_a_star":   float,  mean of a* channel
                "mean_delta_e":  float,  grand Delta-E from reference
                "pink_fraction": float,  fraction of pixels with a* > PINK_A_STAR_THRESHOLD
            }
        """
        small = _resize_quarter(frame_bgr)
        lab = _bgr_to_lab_float32(small)
        mean_a = float(lab[:, :, 1].mean())
        diff = lab - self._ref_lab
        delta_e = float(np.sqrt((diff ** 2).sum(axis=2)).mean())
        pink_fraction = float((lab[:, :, 1] > PINK_A_STAR_THRESHOLD).mean())
        return {"mean_a_star": mean_a, "mean_delta_e": delta_e, "pink_fraction": pink_fraction}

    @property
    def reference_mean_a(self) -> float:
        return self._ref_mean_a

    @property
    def reference_pink_fraction(self) -> float:
        return float((self._ref_lab[:, :, 1] > PINK_A_STAR_THRESHOLD).mean())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resize_quarter(frame_bgr: np.ndarray) -> np.ndarray:
    """Downsample a BGR frame to 25% of its original dimensions."""
    return cv2.resize(frame_bgr, None, fx=_RESIZE_SCALE, fy=_RESIZE_SCALE, interpolation=cv2.INTER_AREA)


def _bgr_to_lab_float32(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert a uint8 BGR frame to float32 CIE-L*a*b*.

    OpenCV's COLOR_BGR2Lab on uint8 input encodes Lab as:
        L* in [0, 255]  (maps from [0, 100])
        a* in [0, 255]  (maps from [-128, 127])
        b* in [0, 255]  (maps from [-128, 127])

    We undo this encoding to recover the standard Lab values so that
    Delta-E has its conventional perceptual meaning.
    """
    lab_encoded = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab_encoded[:, :, 0] * (100.0 / 255.0)
    a = lab_encoded[:, :, 1] - 128.0
    b = lab_encoded[:, :, 2] - 128.0
    return np.stack([L, a, b], axis=2)
