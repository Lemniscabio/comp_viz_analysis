# Automated Mixing-Time Quantification & Batch Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a defensible, automated mixing-time metric `T_mix,95 = max(T_deltaE,95, T_spatial,95, T_texture,95)` to the existing Kineticolor GUI, with per-component diagnostics, vertical-line markers on plots, configurable settings, and a batch-analysis mode that processes many videos and produces a summary CSV.

**Architecture:** Add a new pure-numpy module `src/core/mixing_time.py` containing all timing math (smoothing, start detection, plateau estimation, residual-to-plateau timing, spatial/texture/bulk component times, final aggregation, confidence). Engine stores per-cell Delta-E time series so spatial timing can run post-hoc. A new `MixingTimeResult` dataclass is computed in `AnalysisEngine.finalize()` and consumed by the GUI for display and plot markers, by `DataExporter` for export, and by a new `BatchWorker` driving a folder of videos with a `BatchDialog` UI.

**Tech Stack:** Python, numpy, scipy (signal.medfilt, savgol_filter), PyQt6, pyqtgraph, OpenCV, dataclasses, csv, json. No new heavy deps.

---

## File Structure

**New:**
- `src/core/mixing_time.py` — algorithms + `MixingTimeResult` dataclass
- `src/core/batch.py` — `BatchAnalyzer`, summary-CSV writer, JSON config dump
- `src/gui/batch_dialog.py` — file picker + progress dialog for batch
- `src/gui/batch_worker.py` — `QThread` driving `BatchAnalyzer` over many videos
- `src/gui/mixing_results_panel.py` — formatted results panel widget
- `tests/test_mixing_time.py` — unit tests for algorithm correctness
- `tests/test_batch.py` — batch driver smoke test

**Modified:**
- `src/core/analysis_engine.py` — store cell_avg per frame; add `finalize()` to compute MixingTimeResult
- `src/core/export.py` — accept optional `mixing_result` and write extra columns / per-video detail CSV
- `src/gui/main_window.py` — add results panel dock, batch menu action, wire post-analysis hook
- `src/gui/controls_panel.py` — new mixing-time settings (collapsed/advanced) + Batch Analyze button
- `src/gui/plots_panel.py` — replace ad-hoc 0.95 marker with `set_mixing_result()` for solid main + dashed component lines on every plot
- `scripts/batch_analyze.py` — replace stub `_compute_mixing_times` with real call into `mixing_time`

---

## Conventions

- Tests use `pytest`, located in `tests/`, run with `pytest tests/<file> -v`.
- All times in seconds, relative to `t_start` in **reported** values; absolute timestamps preserved in raw arrays.
- All thresholds carry sensible defaults (constants module-level) but accept overrides via a `MixingTimeParams` dataclass.
- Use `np.nan` for "could not compute"; never silently substitute 0.
- Commit after each task.

---

## Task 1: Engine stores per-frame cell Delta-E

**Why:** Spatial timing needs per-cell traces. Currently engine stores only flat scalars; cell_avg is emitted live but discarded after the frame.

**Files:**
- Modify: `src/core/analysis_engine.py` (`process_frame`, around lines 162–172)
- Test: `tests/test_analysis_engine.py` (extend existing)

- [ ] **Step 1: Test that engine.results rows contain `cell_avg` after process_frame**

```python
# tests/test_analysis_engine.py — add this test
def test_results_row_contains_cell_avg(small_engine_config, synthetic_frame):
    engine = AnalysisEngine(small_engine_config)
    engine.process_frame(synthetic_frame, 0, 0.0)
    engine.process_frame(synthetic_frame, 1, 0.1)
    assert "cell_avg" in engine.results[0]
    assert len(engine.results[0]["cell_avg"]) == (
        small_engine_config["grid_rows"] * small_engine_config["grid_cols"]
    )
```

- [ ] **Step 2: Run test — expect FAIL (`cell_avg` not in stored row)**

Run: `pytest tests/test_analysis_engine.py::test_results_row_contains_cell_avg -v`

- [ ] **Step 3: Add `cell_avg` (and `row_avg`, `col_avg`) into `stored_row`**

In `src/core/analysis_engine.py`, replace the `stored_row` assembly (around line 162):

```python
stored_row: Dict[str, Any] = {
    "frame_number": frame_number,
    "timestamp": timestamp,
    "grand_delta_e": de_result["grand_delta_e"],
    "contact_perimeter": contact_result["contact_perimeter"],
    "contrast": self._last_glcm_results["contrast"],
    "homogeneity": self._last_glcm_results["homogeneity"],
    "energy": self._last_glcm_results["energy"],
    # Persist spatial Delta-E so post-hoc mixing-time analysis can run.
    "cell_avg": np.asarray(de_result["cell_avg"], dtype=np.float64).copy(),
    "row_avg": np.asarray(de_result["row_avg"], dtype=np.float64).copy(),
    "col_avg": np.asarray(de_result["col_avg"], dtype=np.float64).copy(),
}
stored_row.update(var_result)
self._results.append(stored_row)
```

- [ ] **Step 4: Run test — expect PASS**

Run: `pytest tests/test_analysis_engine.py -v`

- [ ] **Step 5: Run existing export tests to confirm CSV ignores extra fields**

Run: `pytest tests/test_export.py -v`. Expected: PASS (DataExporter uses `extrasaction="ignore"`).

- [ ] **Step 6: Commit**

```bash
git add src/core/analysis_engine.py tests/test_analysis_engine.py
git commit -m "feat: persist per-frame cell/row/col Delta-E in engine results"
```

---

## Task 2: `mixing_time.py` — smoothing primitive

**Files:**
- Create: `src/core/mixing_time.py`
- Test: `tests/test_mixing_time.py`

- [ ] **Step 1: Write failing test for `smooth_series`**

```python
# tests/test_mixing_time.py
import numpy as np
import pytest
from src.core.mixing_time import smooth_series


def test_smooth_series_suppresses_spikes():
    t = np.linspace(0, 10, 1001)  # 100 Hz
    y = np.sin(t).copy()
    y[500] = 100.0  # huge spike
    out = smooth_series(t, y, window_s=1.0)
    assert abs(out[500] - np.sin(t[500])) < 0.5
    assert np.allclose(out[:50], y[:50], atol=0.1)


def test_smooth_series_handles_nan():
    t = np.linspace(0, 5, 51)
    y = np.full_like(t, 1.0)
    y[10] = np.nan
    out = smooth_series(t, y, window_s=1.0)
    assert np.all(np.isfinite(out))


def test_smooth_series_short_input_returns_input():
    t = np.array([0.0, 0.1])
    y = np.array([1.0, 2.0])
    out = smooth_series(t, y, window_s=1.0)
    assert np.allclose(out, y)
```

- [ ] **Step 2: Run tests — expect FAIL (module missing)**

Run: `pytest tests/test_mixing_time.py -v`

- [ ] **Step 3: Implement `smooth_series` and module skeleton**

Create `src/core/mixing_time.py`:

```python
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

    Stage 1: rolling/median filter (spike suppression, NaN-tolerant via fill).
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
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_mixing_time.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/core/mixing_time.py tests/test_mixing_time.py
git commit -m "feat: add smooth_series for mixing-time analysis"
```

---

## Task 3: Robust std + start-time detection

**Files:**
- Modify: `src/core/mixing_time.py`
- Test: `tests/test_mixing_time.py`

- [ ] **Step 1: Add failing tests**

```python
# tests/test_mixing_time.py — append
from src.core.mixing_time import detect_start_time, robust_std


def test_robust_std_resists_outlier():
    x = np.concatenate([np.random.RandomState(0).randn(1000) * 0.1, [50.0]])
    assert 0.05 < robust_std(x) < 0.2


def test_detect_start_time_step_change():
    t = np.linspace(0, 20, 2001)
    y = np.where(t < 5.0, 0.0, 30.0 * (1 - np.exp(-(t - 5.0))))
    t_start = detect_start_time(t, y)
    assert 4.7 < t_start < 5.5


def test_detect_start_time_no_change_returns_zero():
    t = np.linspace(0, 10, 1001)
    y = np.random.RandomState(0).randn(len(t)) * 0.01
    t_start = detect_start_time(t, y)
    assert t_start == t[0]
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_mixing_time.py::test_robust_std_resists_outlier -v`

- [ ] **Step 3: Implement**

Append to `src/core/mixing_time.py`:

```python
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
    if len(t) < 5:
        return float(t[0]) if len(t) else 0.0

    y = smooth_series(t, grand_de, window_s=smooth_window_s)
    duration = t[-1] - t[0]
    base_window = min(max(baseline_frac * duration, 0.5), baseline_max_s)
    base_end_idx = int(np.searchsorted(t, t[0] + base_window))
    base_end_idx = max(base_end_idx, 5)
    base_end_idx = min(base_end_idx, len(y) - 2)

    baseline_med = float(np.median(y[:base_end_idx]))
    sigma = robust_std(np.diff(y[:base_end_idx]))
    if sigma <= 0:
        sigma = robust_std(y[:base_end_idx]) or 1e-6

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
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_mixing_time.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/core/mixing_time.py tests/test_mixing_time.py
git commit -m "feat: add robust_std and detect_start_time"
```

---

## Task 4: Plateau estimation + tail stability

- [ ] **Step 1: Tests**

```python
# tests/test_mixing_time.py — append
from src.core.mixing_time import estimate_plateau


def test_plateau_stable_curve():
    t = np.linspace(0, 30, 3001)
    y = 30 * (1 - np.exp(-t / 3.0))
    res = estimate_plateau(t, y, tail_fraction=0.2)
    assert abs(res.y_inf - 30.0) < 0.5
    assert res.stable is True


def test_plateau_unstable_when_still_rising():
    t = np.linspace(0, 5, 501)
    y = 5 * t  # linear, never plateaus
    res = estimate_plateau(t, y, tail_fraction=0.2)
    assert res.stable is False
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

Append to `mixing_time.py`:

```python
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

    amplitude is max |y - y_inf| over the trace (after smoothing upstream).
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
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add -p && git commit -m "feat: add estimate_plateau with tail-slope stability check"
```

---

## Task 5: Generic residual-to-plateau timing

- [ ] **Step 1: Tests covering the four critical regimes**

```python
# tests/test_mixing_time.py — append
from src.core.mixing_time import compute_plateau_time


def _exp_curve(tau, t_end=30.0, fs=100):
    t = np.linspace(0, t_end, int(t_end * fs) + 1)
    y = 30 * (1 - np.exp(-t / tau))
    return t, y


def test_monotonic_T95_close_to_3tau():
    t, y = _exp_curve(2.0)
    out = compute_plateau_time(t, y, level=0.95, t_start=0.0, mode="monotonic")
    # 1 - exp(-T/tau) >= 0.95 -> T >= 3*tau = 6.0
    assert 5.5 < out < 7.5


def test_non_monotonic_returns_after_excursion():
    t = np.linspace(0, 30, 3001)
    # contact-like: start at 5, peak to 25 at t=8, settle back to 5 by t=20
    y = 5 + 20 * np.exp(-((t - 8) ** 2) / (2 * 1.5 ** 2))
    out = compute_plateau_time(t, y, level=0.95, t_start=0.0, mode="non_monotonic")
    assert out > 9.0  # MUST be after peak, not at t=0


def test_brief_false_crossing_ignored_by_hold():
    t = np.linspace(0, 30, 3001)
    y = 30 * (1 - np.exp(-t / 2.0))
    # Inject a brief dip at t=4 that crosses below threshold (~28.5 for level=0.95)
    y[399:401] = 35.0  # spike above plateau
    out = compute_plateau_time(t, y, level=0.95, t_start=0.0, mode="monotonic",
                               hold_duration_s=2.0)
    assert out > 5.5  # honest 95% time, not the spike


def test_weak_signal_returns_nan():
    t = np.linspace(0, 30, 3001)
    y = 0.05 * np.sin(t) + 100.0  # tiny amplitude
    out = compute_plateau_time(t, y, level=0.95, t_start=0.0, mode="monotonic",
                               min_amplitude=3.0)
    assert np.isnan(out)


def test_no_plateau_returns_nan():
    t = np.linspace(0, 5, 501)
    y = 5 * t
    out = compute_plateau_time(t, y, level=0.95, t_start=0.0, mode="monotonic")
    assert np.isnan(out)
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

Append to `mixing_time.py`:

```python
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

    # Where to start the search.
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
```

- [ ] **Step 4: Run all mixing_time tests — PASS**

- [ ] **Step 5: Commit**

```bash
git add -p && git commit -m "feat: add compute_plateau_time with monotonic/non-monotonic modes"
```

---

## Task 6: Spatial component (cell-progress + cell variance)

- [ ] **Step 1: Tests**

```python
# tests/test_mixing_time.py — append
from src.core.mixing_time import compute_spatial_time


def test_spatial_time_synced_cells():
    t = np.linspace(0, 30, 3001)
    n_cells = 25
    cells = np.zeros((len(t), n_cells))
    for j in range(n_cells):
        tau = 2.0 + 0.05 * j
        cells[:, j] = 30 * (1 - np.exp(-t / tau))
    out = compute_spatial_time(t, cells, level=0.95, t_start=0.0)
    assert np.isfinite(out["t_spatial"])
    assert out["t_spatial"] > 5.0


def test_spatial_time_one_lagging_cell_dominates():
    t = np.linspace(0, 30, 3001)
    n_cells = 25
    cells = np.tile((30 * (1 - np.exp(-t / 2.0)))[:, None], (1, n_cells))
    cells[:, 0] = 30 * (1 - np.exp(-t / 8.0))  # one slow cell
    out = compute_spatial_time(t, cells, level=0.95, t_start=0.0)
    # Slow cell needs ~3*8=24s to reach 95%
    assert out["t_cell"] > 18.0
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

```python
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
    P_slow = np.nanpercentile(Pv, slow_percentile, axis=1)
    V = np.nanvar(Pv, axis=1)

    duration_after = max(t[-1] - t_start, 1.0)
    hold_s = hold_duration_s if hold_duration_s is not None else max(
        DEFAULT_HOLD_DURATION_S, hold_fraction * duration_after
    )
    hold_frames = max(1, int(round(hold_s * fps)))
    start_idx = int(np.searchsorted(t, t_start))

    # T_cell: first time P_slow >= level for hold_frames.
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

    # T_variance: search after argmax(V) since V starts low, peaks, falls.
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
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add -p && git commit -m "feat: add compute_spatial_time using cell progress and variance"
```

---

## Task 7: Top-level `compute_mixing_time`, params dataclass, confidence

- [ ] **Step 1: Tests**

```python
# tests/test_mixing_time.py — append
from src.core.mixing_time import (
    MixingTimeParams, MixingTimeResult, compute_mixing_time,
)


def _synth_results(t_end=30.0, fs=30, n_cells=25, lag_per_cell=0.0):
    """Build a list of dict rows shaped like AnalysisEngine.results."""
    t = np.linspace(0, t_end, int(t_end * fs) + 1)
    rows = []
    for i, ti in enumerate(t):
        cell_avg = np.array(
            [30 * (1 - np.exp(-(ti) / (2.0 + lag_per_cell * j))) for j in range(n_cells)]
        )
        rows.append({
            "frame_number": i,
            "timestamp": float(ti),
            "grand_delta_e": float(np.mean(cell_avg)),
            "contact_perimeter": float(5 + 20 * np.exp(-((ti - 8) ** 2) / 4.5)),
            "contrast": float(2 + 8 * np.exp(-((ti - 6) ** 2) / 6.0)),
            "homogeneity": 0.9, "energy": 0.5,
            "variance_delta_e": 0.0,
            "cell_avg": cell_avg,
            "row_avg": np.zeros(5), "col_avg": np.zeros(5),
        })
    return rows


def test_compute_mixing_time_returns_max_of_components():
    rows = _synth_results(lag_per_cell=0.05)
    res = compute_mixing_time(rows, MixingTimeParams())
    assert isinstance(res, MixingTimeResult)
    assert np.isfinite(res.t_mix_95)
    assert res.t_mix_95 >= res.t_deltaE_95
    assert res.t_mix_95 >= res.t_spatial_95
    assert res.t_mix_95 >= res.t_texture_95
    assert res.confidence in {"high", "medium", "low"}


def test_compute_mixing_time_video_too_short_marks_no_plateau():
    rows = _synth_results(t_end=4.0)  # never plateaus
    res = compute_mixing_time(rows, MixingTimeParams())
    assert "no stable plateau" in res.status.lower() or np.isnan(res.t_mix_95)
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

```python
@dataclass
class MixingTimeParams:
    levels: Tuple[float, ...] = (0.90, 0.95, 0.99)
    auto_detect_start: bool = True
    manual_t_start_s: Optional[float] = None
    smooth_window_s: float = DEFAULT_SMOOTH_WINDOW_S
    tail_fraction: float = DEFAULT_TAIL_FRACTION
    hold_duration_s: float = DEFAULT_HOLD_DURATION_S
    deltaE_min_amplitude: float = DEFAULT_DELTAE_MIN_AMPLITUDE
    cell_min_amplitude: float = DEFAULT_CELL_DELTAE_MIN_AMPLITUDE
    include_energy_homogeneity: bool = False
    start_sigma_mult: float = DEFAULT_START_SIGMA_MULT


@dataclass
class MixingTimeResult:
    t_start_s: float
    levels: Tuple[float, ...]
    # per-level component times (relative to t_start), indexed [level]
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
    """Compute T_mix and all component times from engine.results.

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

    # Start time
    if params.auto_detect_start:
        t_start = detect_start_time(
            t, grand,
            sigma_mult=params.start_sigma_mult,
            smooth_window_s=params.smooth_window_s,
        )
    else:
        t_start = params.manual_t_start_s if params.manual_t_start_s is not None else float(t[0])

    res = MixingTimeResult(t_start_s=t_start, levels=params.levels)

    # Per-level component times.
    cells = _stack_cells(results)

    grand_smoothed = smooth_series(t, grand, window_s=params.smooth_window_s)
    res.amplitudes["grand_delta_e"] = float(
        estimate_plateau(t, grand_smoothed,
                         tail_fraction=params.tail_fraction).amplitude
    )
    res.amplitudes["contact"] = float(
        estimate_plateau(t, smooth_series(t, contact, window_s=params.smooth_window_s),
                         tail_fraction=params.tail_fraction).amplitude
    )
    res.amplitudes["contrast"] = float(
        estimate_plateau(t, smooth_series(t, contrast, window_s=params.smooth_window_s),
                         tail_fraction=params.tail_fraction).amplitude
    )

    plat_de = estimate_plateau(t, grand_smoothed, tail_fraction=params.tail_fraction)
    res.tail_slopes["grand_delta_e"] = plat_de.tail_slope
    if not plat_de.stable:
        notes.append("Grand Delta-E tail not stable — video may end before plateau")

    for L in params.levels:
        # Bulk
        t_de = compute_plateau_time(
            t, grand, level=L, t_start=t_start, mode="monotonic",
            smooth_window_s=params.smooth_window_s,
            tail_fraction=params.tail_fraction,
            hold_duration_s=params.hold_duration_s,
            min_amplitude=params.deltaE_min_amplitude,
        )
        res.t_deltaE[L] = float(t_de)

        # Spatial
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

        # Texture (Contact + Contrast always; Energy/Homogeneity optional)
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
        elif len(finite) < len(components):
            res.t_mix[L] = float(max(finite))
            notes.append(f"L={L}: some components NaN — t_mix is max of available")
        else:
            res.t_mix[L] = float(max(finite))

    # Confidence
    L = 0.95
    bulk_ok = np.isfinite(res.t_deltaE.get(L, np.nan))
    spatial_ok = np.isfinite(res.t_spatial.get(L, np.nan))
    texture_ok = np.isfinite(res.t_texture.get(L, np.nan))
    plateau_ok = plat_de.stable
    n_ok = sum([bulk_ok, spatial_ok, texture_ok])

    if n_ok == 3 and plateau_ok and res.n_valid_cells >= 6:
        # check inter-component spread
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

    res.notes = notes
    return res
```

- [ ] **Step 4: Run all `tests/test_mixing_time.py` — PASS**

- [ ] **Step 5: Commit**

```bash
git add -p && git commit -m "feat: add compute_mixing_time aggregator with confidence scoring"
```

---

## Task 8: Engine `finalize()` and integration

**Files:**
- Modify: `src/core/analysis_engine.py`
- Test: `tests/test_analysis_engine.py`

- [ ] **Step 1: Test**

```python
# tests/test_analysis_engine.py — append
def test_engine_finalize_returns_mixing_result(small_engine_config, sequence_of_frames):
    engine = AnalysisEngine(small_engine_config)
    for i, f in enumerate(sequence_of_frames):
        engine.process_frame(f, i, i / 30.0)
    result = engine.finalize()
    assert hasattr(result, "t_mix")
    assert 0.95 in result.t_mix
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement**

In `src/core/analysis_engine.py`, add at top of file:

```python
from src.core.mixing_time import (
    MixingTimeParams, MixingTimeResult, compute_mixing_time,
)
```

Add method to class:

```python
def finalize(
    self, params: Optional[MixingTimeParams] = None
) -> MixingTimeResult:
    """Compute the mixing-time summary for the captured time series."""
    return compute_mixing_time(self._results, params or MixingTimeParams())
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add -p && git commit -m "feat: AnalysisEngine.finalize computes MixingTimeResult"
```

---

## Task 9: Mixing-results panel widget

**Files:**
- Create: `src/gui/mixing_results_panel.py`

- [ ] **Step 1: Implement widget**

```python
"""Mixing-time results panel: T_mix,95 plus per-component breakdown."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from src.core.mixing_time import MixingTimeResult


def _fmt(x: float) -> str:
    return "—" if x is None or x != x else f"{x:.2f} s"  # NaN check via x!=x


class MixingResultsPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        self._title = QLabel("Mixing Time")
        self._title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self._title)

        self._main = QLabel("T_mix,95 = —")
        self._main.setStyleSheet("font-size: 22px; color: #2d7d46;")
        layout.addWidget(self._main)

        self._sub = QLabel("T_mix,90 = —    T_mix,99 = —")
        layout.addWidget(self._sub)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine); layout.addWidget(sep)

        grid = QGridLayout()
        layout.addLayout(grid)
        self._labels = {}
        rows = [
            ("Bulk Delta-E T95",        "t_deltaE_95"),
            ("Spatial T95",             "t_spatial_95"),
            ("Texture T95",             "t_texture_95"),
            ("Contact T95",             "t_contact_95"),
            ("Contrast T95",            "t_contrast_95"),
            ("Cell slow-region T95",    "t_cell_95"),
            ("Cell variance T95",       "t_variance_95"),
        ]
        for r, (label, key) in enumerate(rows):
            grid.addWidget(QLabel(label), r, 0)
            v = QLabel("—")
            grid.addWidget(v, r, 1)
            self._labels[key] = v

        self._status = QLabel("status: —")
        self._status.setStyleSheet("color: gray;")
        layout.addWidget(self._status)
        self._confidence = QLabel("confidence: —")
        layout.addWidget(self._confidence)
        self._t_start = QLabel("t_start: —")
        layout.addWidget(self._t_start)
        layout.addStretch()

    def clear(self) -> None:
        self._main.setText("T_mix,95 = —")
        self._sub.setText("T_mix,90 = —    T_mix,99 = —")
        for v in self._labels.values():
            v.setText("—")
        self._status.setText("status: —")
        self._confidence.setText("confidence: —")
        self._t_start.setText("t_start: —")

    def set_result(self, r: MixingTimeResult) -> None:
        self._main.setText(f"T_mix,95 = {_fmt(r.t_mix.get(0.95))}")
        self._sub.setText(
            f"T_mix,90 = {_fmt(r.t_mix.get(0.90))}    "
            f"T_mix,99 = {_fmt(r.t_mix.get(0.99))}"
        )
        L = 0.95
        m = {
            "t_deltaE_95":   r.t_deltaE.get(L),
            "t_spatial_95":  r.t_spatial.get(L),
            "t_texture_95":  r.t_texture.get(L),
            "t_contact_95":  r.t_contact.get(L),
            "t_contrast_95": r.t_contrast.get(L),
            "t_cell_95":     r.t_cell.get(L),
            "t_variance_95": r.t_variance.get(L),
        }
        for k, v in m.items():
            self._labels[k].setText(_fmt(v))
        self._status.setText(f"status: {r.status}")
        color = {"high": "#2d7d46", "medium": "#cc8400", "low": "#c0392b"}.get(
            r.confidence, "gray"
        )
        self._confidence.setText(f"confidence: {r.confidence}")
        self._confidence.setStyleSheet(f"color: {color};")
        self._t_start.setText(f"t_start: {r.t_start_s:.2f} s")
```

- [ ] **Step 2: Smoke import**

Run: `python -c "from src.gui.mixing_results_panel import MixingResultsPanel; print('ok')"`

- [ ] **Step 3: Commit**

```bash
git add src/gui/mixing_results_panel.py
git commit -m "feat: MixingResultsPanel widget for displaying T_mix breakdown"
```

---

## Task 10: Plot markers (solid main + dashed components)

**Files:**
- Modify: `src/gui/plots_panel.py`

- [ ] **Step 1: Add `set_mixing_result(result)` method on PlotsPanel**

Replace the ad-hoc 0.95 marker logic. In `src/gui/plots_panel.py`, add this method to `PlotsPanel`:

```python
def set_mixing_result(self, result) -> None:
    """Draw vertical T_mix lines on every metric plot.

    result: MixingTimeResult or None to clear.
    """
    # Lazy create marker storage on first call.
    if not hasattr(self, "_mix_markers"):
        self._mix_markers = []
        self._mix_text = {}

    for line in self._mix_markers:
        for plot in self._all_metric_plots():
            plot.removeItem(line)
    self._mix_markers.clear()
    for plot, item in list(self._mix_text.items()):
        plot.removeItem(item)
    self._mix_text.clear()

    if result is None:
        return

    import pyqtgraph as pg
    main_t = result.t_mix.get(0.95)
    if main_t is None or main_t != main_t:  # NaN
        return
    abs_t = result.t_start_s + main_t

    component_times = {
        "ΔE95": result.t_deltaE.get(0.95),
        "Spatial95": result.t_spatial.get(0.95),
        "Texture95": result.t_texture.get(0.95),
    }
    for plot in self._all_metric_plots():
        # main solid line + label
        line = pg.InfiniteLine(
            pos=abs_t, angle=90,
            pen=pg.mkPen("w", width=2, style=pg.QtCore.Qt.PenStyle.SolidLine),
        )
        plot.addItem(line)
        self._mix_markers.append(line)
        text = pg.TextItem(f"Tmix95 = {main_t:.2f} s", color="w", anchor=(0, 1))
        text.setPos(abs_t, 0)
        plot.addItem(text)
        self._mix_text[plot] = text
        # component dashed lines
        for label, ct in component_times.items():
            if ct is None or ct != ct:
                continue
            comp_line = pg.InfiniteLine(
                pos=result.t_start_s + ct, angle=90,
                pen=pg.mkPen("y", width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
            )
            plot.addItem(comp_line)
            self._mix_markers.append(comp_line)


def _all_metric_plots(self):
    return [
        self._plot_de, self._plot_contrast, self._plot_energy,
        self._plot_homogeneity, self._plot_contact,
    ]
```

Also: in `clear_data`, after clearing curves, call `self.set_mixing_result(None)`.

- [ ] **Step 2: Manual smoke (no automated UI test)**

Run app: `python -m src.main`. Open a video, run analysis, verify lines appear.

- [ ] **Step 3: Commit**

```bash
git add src/gui/plots_panel.py
git commit -m "feat: PlotsPanel.set_mixing_result draws Tmix markers on all plots"
```

---

## Task 11: Wire engine.finalize → main_window after analysis

**Files:**
- Modify: `src/gui/main_window.py`

- [ ] **Step 1: Add docked results panel**

In `MainWindow._setup_ui`, after the controls dock:

```python
from src.gui.mixing_results_panel import MixingResultsPanel
self._mixing_panel = MixingResultsPanel()
self._mixing_dock = QDockWidget("Mixing Time", self)
self._mixing_dock.setWidget(self._mixing_panel)
self._mixing_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
self._mixing_dock.setMinimumWidth(280)
self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._mixing_dock)
```

- [ ] **Step 2: Compute on analysis finished**

In `_on_analysis_finished` (around line 315), after `self._set_state(AppState.CONFIGURED)`:

```python
if self._worker and self._worker.engine:
    params = self._controls.get_mixing_params()  # Task 12
    result = self._worker.engine.finalize(params)
    self._mixing_panel.set_result(result)
    self._plots_panel.set_mixing_result(result)
    self._latest_mixing_result = result
else:
    self._latest_mixing_result = None
```

In `_on_video_selected`, also call `self._mixing_panel.clear()` and `self._plots_panel.set_mixing_result(None)`. Add `self._latest_mixing_result = None` to `__init__`.

- [ ] **Step 3: Smoke run, commit**

```bash
git add src/gui/main_window.py
git commit -m "feat: compute mixing-time after analysis and display in GUI"
```

---

## Task 12: Mixing-time settings on ControlsPanel

**Files:**
- Modify: `src/gui/controls_panel.py`

- [ ] **Step 1: Add advanced row (collapsed by default — actually inline since GUI is already dense)**

Add a new `QHBoxLayout` row 4 to `ControlsPanel.__init__`, after row 3:

```python
row4 = QHBoxLayout()
from PyQt6.QtWidgets import QCheckBox, QDoubleSpinBox

row4.addWidget(QLabel("Auto t_start:"))
self._chk_auto_start = QCheckBox()
self._chk_auto_start.setChecked(True)
self._chk_auto_start.setToolTip("Automatically detect when mixing begins")
row4.addWidget(self._chk_auto_start)

row4.addWidget(QLabel("Manual t_start (s):"))
self._spin_manual_start = QDoubleSpinBox()
self._spin_manual_start.setRange(0.0, 1e6); self._spin_manual_start.setValue(0.0)
self._spin_manual_start.setEnabled(False)
self._chk_auto_start.toggled.connect(
    lambda on: self._spin_manual_start.setEnabled(not on)
)
row4.addWidget(self._spin_manual_start)

row4.addWidget(QLabel("Smooth window (s):"))
self._spin_smooth = QDoubleSpinBox()
self._spin_smooth.setRange(0.1, 10.0); self._spin_smooth.setSingleStep(0.1)
self._spin_smooth.setValue(1.5)
row4.addWidget(self._spin_smooth)

row4.addWidget(QLabel("Tail frac:"))
self._spin_tail = QDoubleSpinBox()
self._spin_tail.setRange(0.05, 0.5); self._spin_tail.setSingleStep(0.01)
self._spin_tail.setValue(0.20)
row4.addWidget(self._spin_tail)

row4.addWidget(QLabel("Hold (s):"))
self._spin_hold = QDoubleSpinBox()
self._spin_hold.setRange(0.5, 30.0); self._spin_hold.setSingleStep(0.5)
self._spin_hold.setValue(2.0)
row4.addWidget(self._spin_hold)

self._chk_include_eh = QCheckBox("Include E/H in texture")
self._chk_include_eh.setToolTip("Include Energy & Homogeneity in texture gate")
row4.addWidget(self._chk_include_eh)

self._btn_batch = QPushButton("Batch Analyze Videos")
self._btn_batch.setToolTip("Run analysis on multiple videos with current ROI/mask")
row4.addWidget(self._btn_batch)

row4.addStretch()
main_layout.addLayout(row4)
```

Add signal:
```python
batch_requested = pyqtSignal()
# in __init__: self._btn_batch.clicked.connect(lambda: self.batch_requested.emit())
```

Add method:

```python
def get_mixing_params(self):
    from src.core.mixing_time import MixingTimeParams
    return MixingTimeParams(
        auto_detect_start=self._chk_auto_start.isChecked(),
        manual_t_start_s=(
            None if self._chk_auto_start.isChecked()
            else float(self._spin_manual_start.value())
        ),
        smooth_window_s=float(self._spin_smooth.value()),
        tail_fraction=float(self._spin_tail.value()),
        hold_duration_s=float(self._spin_hold.value()),
        include_energy_homogeneity=self._chk_include_eh.isChecked(),
    )
```

- [ ] **Step 2: Wire in main_window**

In `_connect_signals`: `ctrl.batch_requested.connect(self._on_batch_requested)` (stub the method to `pass` for now).

- [ ] **Step 3: Smoke run, commit**

```bash
git add -p && git commit -m "feat: mixing-time settings + batch button on ControlsPanel"
```

---

## Task 13: Batch summary CSV writer + JSON config

**Files:**
- Create: `src/core/batch.py`
- Test: `tests/test_batch.py`

- [ ] **Step 1: Test**

```python
# tests/test_batch.py
import json
from pathlib import Path
import numpy as np
from src.core.batch import write_summary_row, BATCH_CSV_COLUMNS, write_batch_config
from src.core.mixing_time import MixingTimeResult


def test_summary_row_has_all_columns(tmp_path):
    csv = tmp_path / "summary.csv"
    r = MixingTimeResult(t_start_s=1.0, levels=(0.90, 0.95, 0.99))
    r.t_mix = {0.90: 5.0, 0.95: 8.0, 0.99: 11.0}
    r.t_deltaE = {0.95: 6.0}; r.t_spatial = {0.95: 7.0}; r.t_texture = {0.95: 8.0}
    write_summary_row(csv, video_file="a.mp4", fps=30.0, duration_s=20.0,
                      frame_count=600, roi=(0, 0, 100, 100), result=r,
                      append=False)
    text = csv.read_text()
    for col in BATCH_CSV_COLUMNS:
        assert col in text.split("\n")[0]


def test_batch_config_json_roundtrip(tmp_path):
    out = tmp_path / "batch_config.json"
    write_batch_config(out, roi=(1, 2, 3, 4), mask_present=True,
                       grid=(5, 5), params=None)
    data = json.loads(out.read_text())
    assert data["roi"] == [1, 2, 3, 4]
    assert data["grid_rows"] == 5
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement**

```python
# src/core/batch.py
"""Batch analysis: per-video summary CSV and JSON config dump."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

from src.core.mixing_time import MixingTimeParams, MixingTimeResult

BATCH_CSV_COLUMNS = [
    "video_file", "status", "confidence", "fps", "duration_s", "frame_count",
    "roi_x", "roi_y", "roi_w", "roi_h", "t_start_s",
    "t_deltaE_90_s", "t_deltaE_95_s", "t_deltaE_99_s",
    "t_cell_90_s", "t_cell_95_s", "t_cell_99_s",
    "t_variance_90_s", "t_variance_95_s", "t_variance_99_s",
    "t_spatial_90_s", "t_spatial_95_s", "t_spatial_99_s",
    "t_contact_90_s", "t_contact_95_s", "t_contact_99_s",
    "t_contrast_90_s", "t_contrast_95_s", "t_contrast_99_s",
    "t_texture_90_s", "t_texture_95_s", "t_texture_99_s",
    "t_mix_90_s", "t_mix_95_s", "t_mix_99_s",
    "grand_deltaE_amplitude", "contact_amplitude", "contrast_amplitude",
    "final_tail_slope_deltaE", "final_tail_slope_contact",
    "final_tail_slope_contrast", "notes",
]


def _g(d, k):
    v = d.get(k)
    return "" if v is None or v != v else f"{v:.4f}"  # NaN -> ""


def _row_for(video_file, fps, duration_s, frame_count, roi, result: MixingTimeResult):
    rx, ry, rw, rh = roi if roi is not None else ("", "", "", "")
    row = {
        "video_file": video_file, "status": result.status,
        "confidence": result.confidence,
        "fps": f"{fps:.3f}" if fps else "",
        "duration_s": f"{duration_s:.3f}" if duration_s else "",
        "frame_count": frame_count,
        "roi_x": rx, "roi_y": ry, "roi_w": rw, "roi_h": rh,
        "t_start_s": f"{result.t_start_s:.4f}",
    }
    levels = (0.90, 0.95, 0.99)
    for L, suffix in zip(levels, ("90", "95", "99")):
        row[f"t_deltaE_{suffix}_s"]   = _g(result.t_deltaE,   L)
        row[f"t_cell_{suffix}_s"]     = _g(result.t_cell,     L)
        row[f"t_variance_{suffix}_s"] = _g(result.t_variance, L)
        row[f"t_spatial_{suffix}_s"]  = _g(result.t_spatial,  L)
        row[f"t_contact_{suffix}_s"]  = _g(result.t_contact,  L)
        row[f"t_contrast_{suffix}_s"] = _g(result.t_contrast, L)
        row[f"t_texture_{suffix}_s"]  = _g(result.t_texture,  L)
        row[f"t_mix_{suffix}_s"]      = _g(result.t_mix,      L)
    row["grand_deltaE_amplitude"] = _g(result.amplitudes, "grand_delta_e")
    row["contact_amplitude"]      = _g(result.amplitudes, "contact")
    row["contrast_amplitude"]     = _g(result.amplitudes, "contrast")
    row["final_tail_slope_deltaE"]   = _g(result.tail_slopes, "grand_delta_e")
    row["final_tail_slope_contact"]  = _g(result.tail_slopes, "contact")
    row["final_tail_slope_contrast"] = _g(result.tail_slopes, "contrast")
    row["notes"] = " | ".join(result.notes)
    return row


def write_summary_row(
    csv_path: Path,
    *,
    video_file: str,
    fps: float,
    duration_s: float,
    frame_count: int,
    roi: Optional[Tuple[int, int, int, int]],
    result: MixingTimeResult,
    append: bool = True,
) -> None:
    csv_path = Path(csv_path)
    write_header = not append or not csv_path.exists()
    mode = "a" if append and csv_path.exists() else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BATCH_CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(_row_for(video_file, fps, duration_s, frame_count, roi, result))


def write_batch_config(
    out_path: Path,
    *,
    roi: Optional[Tuple[int, int, int, int]],
    mask_present: bool,
    grid: Tuple[int, int],
    params: Optional[MixingTimeParams],
) -> None:
    payload = {
        "roi": list(roi) if roi else None,
        "mask_present": mask_present,
        "grid_rows": grid[0], "grid_cols": grid[1],
        "mixing_params": asdict(params) if params else None,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/core/batch.py tests/test_batch.py
git commit -m "feat: batch summary CSV writer and JSON config dump"
```

---

## Task 14: Batch worker (QThread) processing many videos

**Files:**
- Create: `src/gui/batch_worker.py`

- [ ] **Step 1: Implement**

```python
"""Background QThread that runs AnalysisEngine over a list of videos."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from src.core.analysis_engine import AnalysisEngine
from src.core.batch import write_summary_row, write_batch_config
from src.core.export import DataExporter
from src.core.mixing_time import MixingTimeParams
from src.core.video_reader import VideoReader


class BatchWorker(QThread):
    progress = pyqtSignal(int, int, str)            # current_idx, total, current_filename
    video_done = pyqtSignal(str, str)               # filename, status
    finished_all = pyqtSignal(str)                  # summary_csv_path
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        videos: List[Path],
        output_dir: Path,
        config: dict,
        roi: Optional[Tuple[int, int, int, int]],
        mask: Optional[np.ndarray],
        params: MixingTimeParams,
        scale_roi: bool,
        export_per_video_csv: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._videos = videos
        self._out = Path(output_dir)
        self._config = config
        self._roi = roi
        self._mask = mask
        self._params = params
        self._scale_roi = scale_roi
        self._per_video = export_per_video_csv
        self._stop = False
        self._template_size: Optional[Tuple[int, int]] = None  # (W, H)

    def stop(self) -> None:
        self._stop = True

    def _scaled_roi_mask(self, frame_size):
        """Return (roi, mask) scaled to current frame size if needed."""
        if self._roi is None and self._mask is None:
            return None, None
        if self._template_size is None:
            self._template_size = frame_size
            return self._roi, self._mask
        sw, sh = self._template_size
        cw, ch = frame_size
        if (sw, sh) == (cw, ch):
            return self._roi, self._mask
        if not self._scale_roi:
            return None, None  # caller will skip
        sx, sy = cw / sw, ch / sh
        new_roi = None
        if self._roi is not None:
            x, y, w, h = self._roi
            new_roi = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
        new_mask = None
        if self._mask is not None:
            new_mask = cv2.resize(self._mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
        return new_roi, new_mask

    def run(self) -> None:
        try:
            self._out.mkdir(parents=True, exist_ok=True)
            summary = self._out / "batch_summary.csv"
            if summary.exists():
                summary.unlink()
            write_batch_config(
                self._out / "batch_analysis_config.json",
                roi=self._roi,
                mask_present=self._mask is not None,
                grid=(self._config["grid_rows"], self._config["grid_cols"]),
                params=self._params,
            )

            total = len(self._videos)
            for i, video in enumerate(self._videos, 1):
                if self._stop:
                    break
                self.progress.emit(i, total, video.name)
                try:
                    reader = VideoReader(
                        path=str(video),
                        frame_skip=self._config["frame_skip"],
                        fps_override=self._config.get("video_fps_override"),
                    )
                    frame_size = (reader.width, reader.height)
                    roi, mask = self._scaled_roi_mask(frame_size)
                    if (self._roi is not None and roi is None) and not self._scale_roi:
                        self.video_done.emit(video.name, "skipped (size mismatch)")
                        reader.release()
                        continue

                    engine = AnalysisEngine(self._config)
                    duration = reader.frame_count / max(reader.fps, 1e-6)
                    for fn, frame in reader:
                        if self._stop:
                            break
                        engine.process_frame(frame, fn, reader.timestamp(fn),
                                             roi=roi, mask=mask)
                    reader.release()

                    result = engine.finalize(self._params)
                    write_summary_row(
                        summary, video_file=video.name, fps=reader.fps,
                        duration_s=duration, frame_count=len(engine.results),
                        roi=roi, result=result, append=True,
                    )
                    if self._per_video:
                        DataExporter().export(
                            engine.results, self._out / f"{video.stem}_metrics.csv",
                            fmt="csv",
                        )
                    self.video_done.emit(video.name, f"ok ({result.confidence})")
                except Exception as e:
                    self.video_done.emit(video.name, f"error: {e}")
                    continue

            self.finished_all.emit(str(summary))
        except Exception as e:
            self.error_occurred.emit(str(e))
```

- [ ] **Step 2: Smoke import**

Run: `python -c "from src.gui.batch_worker import BatchWorker; print('ok')"`

- [ ] **Step 3: Commit**

```bash
git add src/gui/batch_worker.py
git commit -m "feat: BatchWorker QThread for multi-video analysis"
```

---

## Task 15: Batch dialog (file picker + progress)

**Files:**
- Create: `src/gui/batch_dialog.py`

- [ ] **Step 1: Implement**

```python
"""Modal batch-analysis dialog: pick videos, output dir, scale option, run."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QProgressBar, QPushButton, QVBoxLayout,
)

from src.core.mixing_time import MixingTimeParams
from src.gui.batch_worker import BatchWorker

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


class BatchDialog(QDialog):
    def __init__(
        self,
        config: dict,
        roi,
        mask: Optional[np.ndarray],
        params: MixingTimeParams,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Analyze Videos")
        self.setMinimumSize(560, 480)

        self._config = config
        self._roi = roi
        self._mask = mask
        self._params = params
        self._videos: List[Path] = []
        self._output_dir: Optional[Path] = None
        self._worker: Optional[BatchWorker] = None

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Videos:"))
        self._list = QListWidget()
        layout.addWidget(self._list)

        row = QHBoxLayout()
        btn_add_files = QPushButton("Add Files…")
        btn_add_files.clicked.connect(self._on_add_files)
        btn_add_folder = QPushButton("Add Folder…")
        btn_add_folder.clicked.connect(self._on_add_folder)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._on_clear)
        row.addWidget(btn_add_files); row.addWidget(btn_add_folder); row.addWidget(btn_clear)
        layout.addLayout(row)

        row2 = QHBoxLayout()
        btn_out = QPushButton("Choose Output Dir…")
        btn_out.clicked.connect(self._on_choose_out)
        self._lbl_out = QLabel("(no output dir)")
        row2.addWidget(btn_out); row2.addWidget(self._lbl_out, 1)
        layout.addLayout(row2)

        self._chk_scale = QCheckBox("Scale ROI/mask if video size differs")
        self._chk_scale.setChecked(True)
        layout.addWidget(self._chk_scale)

        self._chk_per_video = QCheckBox("Export per-video metrics CSV")
        self._chk_per_video.setChecked(True)
        layout.addWidget(self._chk_per_video)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._lbl_status = QLabel("")
        layout.addWidget(self._lbl_status)

        row3 = QHBoxLayout()
        self._btn_run = QPushButton("Run")
        self._btn_run.setStyleSheet(
            "QPushButton:enabled { background-color: #2d7d46; color: white; "
            "font-weight: bold; padding: 6px 16px; }"
        )
        self._btn_run.clicked.connect(self._on_run)
        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.reject)
        row3.addStretch(); row3.addWidget(self._btn_run); row3.addWidget(self._btn_close)
        layout.addLayout(row3)

    # --- pickers ---
    def _on_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Videos", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.m4v);;All (*)",
        )
        for f in files:
            self._videos.append(Path(f))
            self._list.addItem(f)

    def _on_add_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not d:
            return
        for p in sorted(Path(d).iterdir()):
            if p.suffix.lower() in VIDEO_EXTS:
                self._videos.append(p)
                self._list.addItem(str(p))

    def _on_clear(self) -> None:
        self._videos.clear()
        self._list.clear()

    def _on_choose_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output Directory")
        if d:
            self._output_dir = Path(d)
            self._lbl_out.setText(d)

    # --- run ---
    def _on_run(self) -> None:
        if not self._videos:
            self._lbl_status.setText("No videos selected")
            return
        if self._output_dir is None:
            self._lbl_status.setText("Choose an output directory")
            return
        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setMaximum(len(self._videos))
        self._worker = BatchWorker(
            videos=self._videos,
            output_dir=self._output_dir,
            config=self._config,
            roi=self._roi,
            mask=self._mask,
            params=self._params,
            scale_roi=self._chk_scale.isChecked(),
            export_per_video_csv=self._chk_per_video.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.video_done.connect(self._on_video_done)
        self._worker.finished_all.connect(self._on_finished_all)
        self._worker.error_occurred.connect(
            lambda e: self._lbl_status.setText(f"Error: {e}")
        )
        self._worker.start()

    def _on_progress(self, i: int, total: int, name: str) -> None:
        self._progress.setValue(i)
        self._lbl_status.setText(f"[{i}/{total}] {name}")

    def _on_video_done(self, name: str, status: str) -> None:
        self._lbl_status.setText(f"{name}: {status}")

    def _on_finished_all(self, summary_csv: str) -> None:
        self._btn_run.setEnabled(True)
        self._lbl_status.setText(f"Done. Summary: {summary_csv}")

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        super().closeEvent(event)
```

- [ ] **Step 2: Wire from main_window**

Replace the stub `_on_batch_requested` in `MainWindow`:

```python
def _on_batch_requested(self) -> None:
    if not self._video_path:
        QMessageBox.information(self, "Batch", "Open one video first to set ROI/mask/grid template.")
        return
    config = self._controls.get_config()
    params = self._controls.get_mixing_params()
    roi = self._video_panel.selector.roi
    mask = self._video_panel.selector.mask
    dlg = BatchDialog(config=config, roi=roi, mask=mask, params=params, parent=self)
    dlg.exec()
```

Add at the top of `main_window.py`:
```python
from src.gui.batch_dialog import BatchDialog
```

- [ ] **Step 3: Smoke run; commit**

```bash
git add -p && git commit -m "feat: BatchDialog UI wired into main window"
```

---

## Task 16: Single-video Export now includes mixing-time block

**Files:**
- Modify: `src/core/export.py`, `src/gui/main_window.py`

- [ ] **Step 1: Test**

```python
# tests/test_export.py — add
def test_export_with_mixing_result_writes_summary_csv(tmp_path):
    from src.core.mixing_time import MixingTimeResult
    from src.core.export import DataExporter
    res = MixingTimeResult(t_start_s=0.5, levels=(0.90, 0.95, 0.99))
    res.t_mix = {0.95: 7.0, 0.90: 5.0, 0.99: 9.0}
    res.t_deltaE = {0.95: 6.0}
    res.t_spatial = {0.95: 7.0}
    res.t_texture = {0.95: 8.0}
    out = tmp_path / "results.csv"
    DataExporter().export(
        [{"frame_number": 0, "timestamp": 0.0, "grand_delta_e": 1.0,
          "contact_perimeter": 1.0, "contrast": 1.0, "homogeneity": 1.0,
          "energy": 1.0, "variance_r": 0, "variance_g": 0, "variance_b": 0,
          "variance_l": 0, "variance_a": 0, "variance_b_star": 0,
          "variance_delta_e": 0}],
        out, fmt="csv", mixing_result=res,
    )
    summary = tmp_path / "results_mixing_summary.csv"
    assert summary.exists()
    assert "t_mix_95_s" in summary.read_text().split("\n")[0]
```

- [ ] **Step 2: Modify `DataExporter.export` to accept `mixing_result`**

In `src/core/export.py`:

```python
def export(
    self, results, output_path, fmt: str = "csv",
    mixing_result=None,
) -> None:
    output_path = Path(output_path)
    enriched = self._add_normalized_delta_e(results)
    if fmt == "csv":
        self._export_csv(enriched, output_path)
    elif fmt == "xlsx":
        self._export_xlsx(enriched, output_path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    if mixing_result is not None:
        from src.core.batch import write_summary_row
        summary_path = output_path.with_name(output_path.stem + "_mixing_summary.csv")
        write_summary_row(
            summary_path, video_file=output_path.name,
            fps=0.0, duration_s=0.0, frame_count=len(results), roi=None,
            result=mixing_result, append=False,
        )
    logger.info(f"Exported {len(enriched)} rows to {output_path}")
```

- [ ] **Step 3: Wire in `main_window._on_export`**

Pass `mixing_result=self._latest_mixing_result` into `exporter.export(...)`.

- [ ] **Step 4: Run all tests; commit**

```bash
pytest tests/test_export.py -v
git add -p && git commit -m "feat: export writes companion mixing summary CSV"
```

---

## Task 17: Update `scripts/batch_analyze.py` to use real mixing time

**Files:**
- Modify: `scripts/batch_analyze.py`

- [ ] **Step 1: Replace stub**

In `scripts/batch_analyze.py`, replace `_compute_mixing_times` body:

```python
def _compute_mixing_times(results: list) -> dict:
    from src.core.mixing_time import MixingTimeParams, compute_mixing_time
    r = compute_mixing_time(results, MixingTimeParams())
    return {"t_90": r.t_mix.get(0.90), "t_95": r.t_mix.get(0.95),
            "t_99": r.t_mix.get(0.99), "result": r}
```

Adjust `_prepend_mixing_header` to also write the rule:

```python
def _prepend_mixing_header(csv_path: Path, mix: dict) -> None:
    fmt = lambda v: "NaN" if v is None or v != v else f"{v:.3f}"
    header = (
        f"# mixing_time_t90 = {fmt(mix.get('t_90'))}\n"
        f"# mixing_time_t95 = {fmt(mix.get('t_95'))}\n"
        f"# mixing_time_t99 = {fmt(mix.get('t_99'))}\n"
        f"# rule = max(T_deltaE, T_spatial, T_texture)\n"
    )
    csv_path.write_text(header + csv_path.read_text())
```

- [ ] **Step 2: Smoke run on a sample video; commit**

```bash
git add scripts/batch_analyze.py
git commit -m "feat: batch_analyze.py uses compute_mixing_time"
```

---

## Task 18: Final verification

- [ ] **Step 1: All tests pass**

Run: `pytest tests/ -v`
Expected: all green.

- [ ] **Step 2: Manual GUI smoke**

Run: `python -m src.main`
Verify:
- single-video flow still works (open → ROI → start → export)
- after analysis, `MixingResultsPanel` shows values
- vertical T_mix,95 line appears on every plot
- Batch button opens dialog; selecting a folder + output dir + Run produces `batch_summary.csv` and `batch_analysis_config.json`
- per-video CSV files appear in output dir

- [ ] **Step 3: Final commit if any small fixes needed**

```bash
git add -p && git commit -m "chore: post-integration polish"
```

---

## Self-Review

**Coverage check:**
- T_mix,95 = max(deltaE, spatial, texture): Task 7 ✓
- 90/95/99 levels: Task 7 (loops over `params.levels`) ✓
- smoothing primitive: Task 2 ✓
- start detection: Task 3 ✓
- plateau + tail-slope: Task 4 ✓
- monotonic vs non-monotonic, hold window: Task 5 (with hold_frames) ✓
- spatial cell + variance: Task 6 ✓
- texture composition + optional E/H: Task 7 (params.include_energy_homogeneity) ✓
- confidence scoring: Task 7 ✓
- GUI results panel: Task 9, 11 ✓
- vertical-line markers (solid main + dashed components): Task 10 ✓
- settings controls: Task 12 ✓
- batch dialog/worker: Task 14, 15 ✓
- summary CSV columns (full list verified): Task 13 ✓
- per-video detail CSV: Task 14 ✓
- batch JSON config: Task 13 ✓
- single-video Export still works AND adds mixing summary: Task 16 ✓
- ROI scaling for differing video sizes: Task 14 (`_scaled_roi_mask`) ✓
- background thread, progress: Task 14, 15 ✓
- weak signal / no-plateau / video too short: Tasks 4, 5, 7 (NaN + status) ✓
- defensive against false-zero on Contact/Contrast: Task 5 (`mode="non_monotonic"`, `argmax` after t_start) ✓
- tests for monotonic/non-monotonic/false-crossing/weak/no-plateau: Task 5 ✓

**Type/name consistency:** `MixingTimeResult.t_mix.get(0.95)` used throughout; `compute_mixing_time` is the single entry point; `set_mixing_result` consistent in plots panel + main window.

**No placeholders:** all code blocks contain executable code.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-28-mixing-time-and-batch-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
