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


def robust_std(x: np.ndarray) -> float:
    """Median absolute deviation scaled to match Gaussian std."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def detect_start_time(
    t: np.ndarray,
    grand_de: np.ndarray,
    *,
    sigma_mult: float = DEFAULT_START_SIGMA_MULT,
    range_frac: float = DEFAULT_START_RANGE_FRAC,
    hold_s: float = DEFAULT_START_HOLD_S,
    baseline_frac: float = DEFAULT_BASELINE_FRAC,
    baseline_max_s: float = DEFAULT_BASELINE_MAX_S,
    smooth_window_s: float = DEFAULT_SMOOTH_WINDOW_S,
) -> float:
    """First sustained departure of grand Delta-E from its baseline.

    Baseline = first min(baseline_frac * duration, baseline_max_s).
    Returns t[0] if no significant change is detected.
    """
    t = np.asarray(t, dtype=np.float64)
    grand_de = np.asarray(grand_de, dtype=np.float64)
    if len(t) < 5:
        return float(t[0]) if len(t) else 0.0

    y = smooth_series(t, grand_de, window_s=smooth_window_s)
    duration = t[-1] - t[0]
    base_window = min(max(baseline_frac * duration, 0.5), baseline_max_s)
    base_end_idx = int(np.searchsorted(t, t[0] + base_window))
    base_end_idx = max(base_end_idx, 5)
    base_end_idx = min(base_end_idx, len(y) - 2)

    baseline_med = float(np.median(y[:base_end_idx]))
    sigma = robust_std(np.diff(grand_de[:base_end_idx]))
    if sigma <= 0:
        sigma = robust_std(grand_de[:base_end_idx]) or 1e-6

    total_range = float(np.nanmax(y) - np.nanmin(y))
    threshold = max(sigma_mult * sigma, range_frac * total_range)

    fps = _frames_per_second(t)
    hold_frames = max(1, int(round(hold_s * fps)))
    deviated = np.abs(y - baseline_med) > threshold

    # First index where `deviated` is True for `hold_frames` consecutive samples.
    if not np.any(deviated):
        return float(t[0])
    run = 0
    for i in range(base_end_idx, len(deviated)):
        if deviated[i]:
            run += 1
            if run >= hold_frames:
                return float(t[i - hold_frames + 1])
        else:
            run = 0
    return float(t[0])


@dataclass
class PlateauEstimate:
    y_inf: float
    amplitude: float
    tail_slope: float            # raw slope (units/s)
    normalized_tail_slope: float # |slope| * tail_duration / amplitude
    stable: bool                 # normalized_tail_slope <= threshold
    tail_start_idx: int


def estimate_plateau(
    t: np.ndarray,
    y: np.ndarray,
    *,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    slope_max: float = DEFAULT_TAIL_SLOPE_MAX,
) -> PlateauEstimate:
    """Estimate y_inf as median of final tail, plus a stability check.

    amplitude is max |y - y_inf| over the trace.
    Stability: |a| * tail_duration / amplitude <= slope_max.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(t)
    if n < 5:
        return PlateauEstimate(float(y[-1]) if n else 0.0, 0.0, 0.0, 0.0, False, 0)

    tail_n = max(3, int(round(tail_fraction * n)))
    tail_start = n - tail_n
    y_inf = float(np.median(y[tail_start:]))
    amplitude = float(np.nanmax(np.abs(y - y_inf)))
    if amplitude <= 0:
        return PlateauEstimate(y_inf, 0.0, 0.0, 0.0, False, tail_start)

    a, _b = np.polyfit(t[tail_start:], y[tail_start:], 1)
    tail_duration = t[-1] - t[tail_start]
    norm_slope = abs(a) * tail_duration / amplitude
    return PlateauEstimate(
        y_inf=y_inf,
        amplitude=amplitude,
        tail_slope=float(a),
        normalized_tail_slope=float(norm_slope),
        stable=bool(norm_slope <= slope_max),
        tail_start_idx=tail_start,
    )


def compute_plateau_time(
    t: np.ndarray,
    y: np.ndarray,
    *,
    level: float,
    t_start: float,
    mode: str = "monotonic",                   # or "non_monotonic"
    smooth_window_s: float = DEFAULT_SMOOTH_WINDOW_S,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    slope_max: float = DEFAULT_TAIL_SLOPE_MAX,
    hold_duration_s: Optional[float] = None,
    hold_fraction: float = DEFAULT_HOLD_FRACTION,
    min_amplitude: float = 0.0,
    return_diagnostics: bool = False,
):
    """First sustained time after the dominant excursion where R(t) <= 1 - level.

    R(t) = |y - y_inf| / amplitude.

    For non_monotonic metrics (Contact/Contrast/Variance), search starts at
    argmax(R) so the algorithm cannot pick t=0 when y already starts near y_inf.
    Returns NaN if plateau is unstable, signal is weak, or video ends before
    threshold is satisfied for hold_duration_s.

    Returns time relative to t_start (seconds). If return_diagnostics, also
    returns a dict.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(t) < 5:
        return (np.nan, {}) if return_diagnostics else np.nan

    ys = smooth_series(t, y, window_s=smooth_window_s)
    plateau = estimate_plateau(t, ys, tail_fraction=tail_fraction, slope_max=slope_max)

    diag: Dict[str, float] = {
        "y_inf": plateau.y_inf,
        "amplitude": plateau.amplitude,
        "tail_slope": plateau.tail_slope,
        "normalized_tail_slope": plateau.normalized_tail_slope,
        "stable": plateau.stable,
        "weak_signal": plateau.amplitude < min_amplitude,
        "no_plateau": not plateau.stable,
    }

    if not plateau.stable or plateau.amplitude < max(min_amplitude, 1e-12):
        return (np.nan, diag) if return_diagnostics else np.nan

    R = np.abs(ys - plateau.y_inf) / plateau.amplitude
    threshold = 1.0 - level

    start_idx = int(np.searchsorted(t, t_start))
    if mode == "non_monotonic":
        seg = R[start_idx:]
        if len(seg) == 0:
            return (np.nan, diag) if return_diagnostics else np.nan
        peak_offset = int(np.argmax(seg))
        search_start = start_idx + peak_offset
        diag["t_peak"] = float(t[search_start])
    else:
        search_start = start_idx

    duration_after = t[-1] - t[search_start] if search_start < len(t) else 0.0
    hold_s = hold_duration_s if hold_duration_s is not None else max(
        DEFAULT_HOLD_DURATION_S, hold_fraction * duration_after
    )
    fps = _frames_per_second(t)
    hold_frames = max(1, int(round(hold_s * fps)))

    below = R <= threshold
    run = 0
    for i in range(search_start, len(R)):
        if below[i]:
            run += 1
            if run >= hold_frames:
                t_abs = t[i - hold_frames + 1]
                rel = float(t_abs - t_start)
                diag["t_abs"] = float(t_abs)
                return (rel, diag) if return_diagnostics else rel
        else:
            run = 0

    return (np.nan, diag) if return_diagnostics else np.nan
