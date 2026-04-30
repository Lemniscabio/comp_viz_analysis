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


def compute_spatial_time(
    t: np.ndarray,
    cell_de_series: np.ndarray,         # shape (n_frames, n_cells), NaN ok
    *,
    level: float,
    t_start: float,
    smooth_window_s: float = DEFAULT_SMOOTH_WINDOW_S,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    slope_max: float = DEFAULT_TAIL_SLOPE_MAX,
    hold_duration_s: Optional[float] = None,
    hold_fraction: float = DEFAULT_HOLD_FRACTION,
    min_cell_amplitude: float = DEFAULT_CELL_DELTAE_MIN_AMPLITUDE,
    slow_percentile: float = 5.0,
) -> Dict[str, float]:
    """Spatial mixing time: max(T_cell, T_var).

    T_cell: first sustained time the slow-percentile of normalized cell progress
            P_j(t) reaches `level`.
    T_var:  first sustained time after argmax(V) that V(t) <= (1-level)^2.
    """
    t = np.asarray(t, dtype=np.float64)
    cells = np.asarray(cell_de_series, dtype=np.float64)
    if cells.ndim != 2 or cells.shape[0] != len(t):
        return {"t_cell": np.nan, "t_variance": np.nan, "t_spatial": np.nan,
                "n_valid_cells": 0}

    n_frames, n_cells = cells.shape
    fps = _frames_per_second(t)

    P = np.full_like(cells, np.nan)
    valid_cells: List[int] = []
    for j in range(n_cells):
        col = cells[:, j]
        finite_frac = np.mean(np.isfinite(col))
        if finite_frac < 0.5:
            continue
        cs = smooth_series(t, col, window_s=smooth_window_s)
        plateau = estimate_plateau(t, cs, tail_fraction=tail_fraction,
                                   slope_max=slope_max)
        if plateau.amplitude < min_cell_amplitude:
            continue
        D0 = float(np.nanmedian(cs[: max(3, int(0.05 * n_frames))]))
        denom = plateau.y_inf - D0
        if abs(denom) < 1e-9:
            continue
        Pj = (cs - D0) / denom
        Pj = np.clip(Pj, 0.0, 1.0)
        P[:, j] = Pj
        valid_cells.append(j)

    if len(valid_cells) < 2:
        return {"t_cell": np.nan, "t_variance": np.nan, "t_spatial": np.nan,
                "n_valid_cells": len(valid_cells)}

    Pv = P[:, valid_cells]
    P_slow = np.nanpercentile(Pv, slow_percentile, axis=1, method="weibull")
    V = np.nanvar(Pv, axis=1)

    duration_after = max(t[-1] - t_start, 1.0)
    hold_s = hold_duration_s if hold_duration_s is not None else max(
        DEFAULT_HOLD_DURATION_S, hold_fraction * duration_after
    )
    hold_frames = max(1, int(round(hold_s * fps)))
    start_idx = int(np.searchsorted(t, t_start))

    above = P_slow >= level
    t_cell = np.nan
    run = 0
    for i in range(start_idx, len(P_slow)):
        if above[i]:
            run += 1
            if run >= hold_frames:
                t_cell = float(t[i - hold_frames + 1] - t_start)
                break
        else:
            run = 0

    seg = V[start_idx:]
    if len(seg) == 0 or not np.any(np.isfinite(seg)):
        t_var = np.nan
    else:
        peak_off = int(np.nanargmax(seg))
        search_start = start_idx + peak_off
        var_thresh = (1.0 - level) ** 2
        t_var = np.nan
        run = 0
        for i in range(search_start, len(V)):
            if np.isfinite(V[i]) and V[i] <= var_thresh:
                run += 1
                if run >= hold_frames:
                    t_var = float(t[i - hold_frames + 1] - t_start)
                    break
            else:
                run = 0

    candidates = [c for c in (t_cell, t_var) if np.isfinite(c)]
    t_spatial = max(candidates) if candidates else np.nan

    return {
        "t_cell": float(t_cell) if np.isfinite(t_cell) else np.nan,
        "t_variance": float(t_var) if np.isfinite(t_var) else np.nan,
        "t_spatial": float(t_spatial) if np.isfinite(t_spatial) else np.nan,
        "n_valid_cells": len(valid_cells),
    }


METHOD_DEFAULT = "default"           # T_mix = max(ΔE, Spatial, Texture)
METHOD_DELTAE_ONLY = "deltaE_only"   # T_mix = T_deltaE
METHOD_TOP5 = "top5_per_frame"       # mean of top-5 cells/frame, normalized first-crossing
METHOD_ALLCELLS = "allcells_per_frame"  # mean of all valid cells/frame, normalized first-crossing


@dataclass
class MixingTimeParams:
    levels: Tuple[float, ...] = (0.90, 0.95, 0.99)
    auto_detect_start: bool = False
    manual_t_start_s: Optional[float] = 0.0
    smooth_window_s: float = DEFAULT_SMOOTH_WINDOW_S
    tail_fraction: float = DEFAULT_TAIL_FRACTION
    hold_duration_s: float = DEFAULT_HOLD_DURATION_S
    deltaE_min_amplitude: float = DEFAULT_DELTAE_MIN_AMPLITUDE
    cell_min_amplitude: float = DEFAULT_CELL_DELTAE_MIN_AMPLITUDE
    include_energy_homogeneity: bool = False
    start_sigma_mult: float = DEFAULT_START_SIGMA_MULT
    method: str = METHOD_DEFAULT


@dataclass
class MixingTimeResult:
    t_start_s: float
    levels: Tuple[float, ...]
    t_deltaE: Dict[float, float] = field(default_factory=dict)
    t_cell: Dict[float, float] = field(default_factory=dict)
    t_variance: Dict[float, float] = field(default_factory=dict)
    t_spatial: Dict[float, float] = field(default_factory=dict)
    t_contact: Dict[float, float] = field(default_factory=dict)
    t_contrast: Dict[float, float] = field(default_factory=dict)
    t_energy: Dict[float, float] = field(default_factory=dict)
    t_homogeneity: Dict[float, float] = field(default_factory=dict)
    t_texture: Dict[float, float] = field(default_factory=dict)
    t_mix: Dict[float, float] = field(default_factory=dict)
    amplitudes: Dict[str, float] = field(default_factory=dict)
    tail_slopes: Dict[str, float] = field(default_factory=dict)
    n_valid_cells: int = 0
    status: str = "ok"
    notes: List[str] = field(default_factory=list)
    confidence: str = "high"

    @property
    def t_mix_95(self) -> float:
        return self.t_mix.get(0.95, float("nan"))

    @property
    def t_mix_90(self) -> float:
        return self.t_mix.get(0.90, float("nan"))

    @property
    def t_mix_99(self) -> float:
        return self.t_mix.get(0.99, float("nan"))


def _stack_cells(rows: List[dict]) -> np.ndarray:
    arrs = [np.asarray(r["cell_avg"], dtype=np.float64) for r in rows
            if "cell_avg" in r and r["cell_avg"] is not None]
    if not arrs:
        return np.zeros((0, 0))
    n = max(a.size for a in arrs)
    out = np.full((len(arrs), n), np.nan)
    for i, a in enumerate(arrs):
        out[i, : a.size] = a
    return out


def compute_mixing_time(
    results: List[dict],
    params: Optional[MixingTimeParams] = None,
) -> MixingTimeResult:
    """Compute T_mix and component times from engine.results.

    Inputs are absolute timestamps; outputs are relative to detected (or manual)
    t_start. Returns a fully populated MixingTimeResult, with NaNs and notes
    for components that could not be computed.
    """
    params = params or MixingTimeParams()
    notes: List[str] = []

    if len(results) < 10:
        return MixingTimeResult(
            t_start_s=0.0, levels=params.levels, status="too_few_frames",
            notes=["fewer than 10 frames — cannot compute mixing time"],
            confidence="low",
        )

    t = np.array([r["timestamp"] for r in results], dtype=np.float64)
    grand = np.array([r["grand_delta_e"] for r in results], dtype=np.float64)
    contact = np.array([r["contact_perimeter"] for r in results], dtype=np.float64)
    contrast = np.array([r["contrast"] for r in results], dtype=np.float64)
    energy = np.array([r["energy"] for r in results], dtype=np.float64)
    homog = np.array([r["homogeneity"] for r in results], dtype=np.float64)

    if params.auto_detect_start:
        t_start = detect_start_time(
            t, grand,
            sigma_mult=params.start_sigma_mult,
            smooth_window_s=params.smooth_window_s,
        )
    else:
        t_start = params.manual_t_start_s if params.manual_t_start_s is not None else float(t[0])

    res = MixingTimeResult(t_start_s=t_start, levels=params.levels)
    cells = _stack_cells(results)

    grand_smoothed = smooth_series(t, grand, window_s=params.smooth_window_s)
    plat_de = estimate_plateau(t, grand_smoothed, tail_fraction=params.tail_fraction)
    res.amplitudes["grand_delta_e"] = float(plat_de.amplitude)
    plat_contact = estimate_plateau(
        t, smooth_series(t, contact, window_s=params.smooth_window_s),
        tail_fraction=params.tail_fraction,
    )
    plat_contrast = estimate_plateau(
        t, smooth_series(t, contrast, window_s=params.smooth_window_s),
        tail_fraction=params.tail_fraction,
    )
    res.amplitudes["contact"] = float(plat_contact.amplitude)
    res.amplitudes["contrast"] = float(plat_contrast.amplitude)
    res.tail_slopes["grand_delta_e"] = plat_de.tail_slope
    res.tail_slopes["contact"] = plat_contact.tail_slope
    res.tail_slopes["contrast"] = plat_contrast.tail_slope

    if not plat_de.stable:
        notes.append("Grand Delta-E tail not stable — video may end before plateau")

    for L in params.levels:
        t_de = compute_plateau_time(
            t, grand, level=L, t_start=t_start, mode="monotonic",
            smooth_window_s=params.smooth_window_s,
            tail_fraction=params.tail_fraction,
            hold_duration_s=params.hold_duration_s,
            min_amplitude=params.deltaE_min_amplitude,
        )
        res.t_deltaE[L] = float(t_de)

        spatial = compute_spatial_time(
            t, cells, level=L, t_start=t_start,
            smooth_window_s=params.smooth_window_s,
            tail_fraction=params.tail_fraction,
            hold_duration_s=params.hold_duration_s,
            min_cell_amplitude=params.cell_min_amplitude,
        )
        res.t_cell[L] = spatial["t_cell"]
        res.t_variance[L] = spatial["t_variance"]
        res.t_spatial[L] = spatial["t_spatial"]
        res.n_valid_cells = spatial["n_valid_cells"]

        t_contact = compute_plateau_time(
            t, contact, level=L, t_start=t_start, mode="non_monotonic",
            smooth_window_s=params.smooth_window_s,
            tail_fraction=params.tail_fraction,
            hold_duration_s=params.hold_duration_s,
        )
        t_contrast = compute_plateau_time(
            t, contrast, level=L, t_start=t_start, mode="non_monotonic",
            smooth_window_s=params.smooth_window_s,
            tail_fraction=params.tail_fraction,
            hold_duration_s=params.hold_duration_s,
        )
        t_energy = compute_plateau_time(
            t, energy, level=L, t_start=t_start, mode="non_monotonic",
            smooth_window_s=params.smooth_window_s,
            tail_fraction=params.tail_fraction,
            hold_duration_s=params.hold_duration_s,
        )
        t_homog = compute_plateau_time(
            t, homog, level=L, t_start=t_start, mode="non_monotonic",
            smooth_window_s=params.smooth_window_s,
            tail_fraction=params.tail_fraction,
            hold_duration_s=params.hold_duration_s,
        )
        res.t_contact[L] = float(t_contact)
        res.t_contrast[L] = float(t_contrast)
        res.t_energy[L] = float(t_energy)
        res.t_homogeneity[L] = float(t_homog)

        gates = [t_contact, t_contrast]
        if params.include_energy_homogeneity:
            gates += [t_energy, t_homog]
        finite_gates = [g for g in gates if np.isfinite(g)]
        t_texture = max(finite_gates) if finite_gates else np.nan
        res.t_texture[L] = float(t_texture)

        components = [t_de, spatial["t_spatial"], t_texture]
        finite = [c for c in components if np.isfinite(c)]
        if not finite:
            res.t_mix[L] = float("nan")
        else:
            res.t_mix[L] = float(max(finite))
            if len(finite) < len(components):
                notes.append(f"L={L}: some components NaN — t_mix is max of available")

    L = 0.95
    bulk_ok = np.isfinite(res.t_deltaE.get(L, np.nan))
    spatial_ok = np.isfinite(res.t_spatial.get(L, np.nan))
    texture_ok = np.isfinite(res.t_texture.get(L, np.nan))
    plateau_ok = plat_de.stable
    n_ok = sum([bulk_ok, spatial_ok, texture_ok])

    if n_ok == 3 and plateau_ok and res.n_valid_cells >= 6:
        finite = [res.t_deltaE[L], res.t_spatial[L], res.t_texture[L]]
        spread = (max(finite) - min(finite)) / max(finite) if max(finite) > 0 else 0
        if spread < 0.5:
            res.confidence = "high"
        else:
            res.confidence = "medium"
            notes.append("components disagree by >50% — confidence reduced")
    elif n_ok >= 2 and plateau_ok:
        res.confidence = "medium"
    else:
        res.confidence = "low"

    if not plateau_ok:
        res.status = "no stable plateau"

    # ---- method dispatch: optionally override t_mix with an alternative rule
    if params.method == METHOD_DELTAE_ONLY:
        res.t_mix = dict(res.t_deltaE)
        res.status = "ΔE-only"
        amp = res.amplitudes.get("grand_delta_e", 0.0)
        if not np.isfinite(res.t_mix.get(0.95, np.nan)):
            res.confidence = "low"
        elif amp >= 3.0:
            res.confidence = "high"
        elif amp >= 1.5:
            res.confidence = "medium"
        else:
            res.confidence = "low"
    elif params.method in (METHOD_TOP5, METHOD_ALLCELLS):
        K = 5 if params.method == METHOD_TOP5 else None
        out = _compute_cells_first_crossing(results, params.levels, K=K)
        res.t_mix = out["t_mix"]
        res.t_deltaE = dict(out["t_mix"])  # so dashed ΔE95 marker aligns
        res.amplitudes["grand_delta_e"] = out["peak"]
        if K == 5:
            res.status = "top5-per-frame, normalized first-crossing"
        else:
            res.status = "all-cells-avg per-frame, normalized first-crossing"
        peak = out["peak"]
        if not np.isfinite(res.t_mix.get(0.95, np.nan)):
            res.confidence = "low"
        elif peak >= 3.0:
            res.confidence = "high"
        elif peak >= 1.5:
            res.confidence = "medium"
        else:
            res.confidence = "low"

    res.notes = notes
    return res


def _compute_cells_first_crossing(
    results: List[dict],
    levels: Tuple[float, ...],
    K: Optional[int] = 5,
) -> Dict[str, object]:
    """Per-frame cell-averaging metric, normalized, first-crossing.

    K=5: mean of the top-5 cell ΔE values per frame
    K=None: mean of all valid (finite) cell ΔE values per frame
    Then normalize by the peak of the resulting time series and return
    the first time the normalized signal crosses each level.
    """
    t = np.array([r["timestamp"] for r in results], dtype=np.float64)
    cells = np.array(
        [np.asarray(r["cell_avg"], dtype=np.float64) for r in results]
    )  # (n_frames, n_cells)
    n_frames = cells.shape[0]
    series = np.full(n_frames, np.nan, dtype=np.float64)
    for i in range(n_frames):
        row = cells[i]
        finite = row[np.isfinite(row)]
        if len(finite) == 0:
            continue
        if K is not None and len(finite) >= K:
            series[i] = np.partition(finite, -K)[-K:].mean()
        else:
            series[i] = finite.mean()

    finite = series[np.isfinite(series)]
    peak = float(finite.max()) if len(finite) else 0.0
    t_mix: Dict[float, float] = {L: float("nan") for L in levels}
    if peak > 0:
        norm = series / peak
        for L in levels:
            mask = np.isfinite(norm) & (norm >= L)
            idx = np.where(mask)[0]
            t_mix[L] = float(t[idx[0]]) if len(idx) else float("nan")
    return {"t_mix": t_mix, "peak": peak}
