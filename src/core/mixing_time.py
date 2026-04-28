"""Automated mixing-time quantification from per-frame metric time series.

T_mix,L = max(T_deltaE,L, T_spatial,L, T_texture,L) where L in {0.90, 0.95, 0.99}.

All public functions accept time arrays in seconds and return times in seconds
relative to t_start. NaN means "could not compute".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


# ---------- defaults (override via MixingTimeParams) -----------------------
DEFAULT_SMOOTH_WINDOW_S = 1.5
DEFAULT_TAIL_FRACTION = 0.20
DEFAULT_HOLD_DURATION_S = 2.0
DEFAULT_HOLD_FRACTION = 0.05
DEFAULT_TAIL_SLOPE_MAX = 0.05
DEFAULT_DELTAE_MIN_AMPLITUDE = 3.0
DEFAULT_CELL_DELTAE_MIN_AMPLITUDE = 2.0
DEFAULT_TEXTURE_NOISE_MULT = 3.0
DEFAULT_START_SIGMA_MULT = 3.0
DEFAULT_START_RANGE_FRAC = 0.03
DEFAULT_START_HOLD_S = 0.75
DEFAULT_BASELINE_FRAC = 0.08
DEFAULT_BASELINE_MAX_S = 2.0


def _frames_per_second(t: np.ndarray) -> float:
    if len(t) < 2:
        return 1.0
    dt = np.median(np.diff(t))
    return 1.0 / dt if dt > 0 else 1.0


def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def smooth_series(
    t: np.ndarray,
    y: np.ndarray,
    window_s: float = DEFAULT_SMOOTH_WINDOW_S,
    method: str = "median_savgol",
) -> np.ndarray:
    """Robust two-stage smoother.

    Stage 1: median filter (spike suppression, NaN-tolerant via interpolation).
    Stage 2: Savitzky-Golay (smooth derivative) when method == 'median_savgol'.
    """
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 5:
        return y.copy()

    # NaN-fill for filter stability — interpolate over interior NaNs.
    if np.any(~np.isfinite(y)):
        valid = np.isfinite(y)
        if not np.any(valid):
            return y.copy()
        y = np.interp(np.arange(len(y)), np.where(valid)[0], y[valid])

    fps = _frames_per_second(np.asarray(t, dtype=np.float64))
    win = _odd(int(round(window_s * fps)))
    win = min(win, _odd(len(y) // 2 - 1) if len(y) > 6 else 3)

    stage1 = median_filter(y, size=win, mode="nearest")
    if method == "rolling_median":
        return stage1
    poly = min(3, win - 1)
    return savgol_filter(stage1, window_length=win, polyorder=poly, mode="interp")
