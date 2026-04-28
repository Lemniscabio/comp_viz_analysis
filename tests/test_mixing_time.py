"""Tests for src/core/mixing_time.py."""
from __future__ import annotations

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
    # contact-like: start at 5, peak ~25 around t=8, settle back to ~5
    y = 5 + 20 * np.exp(-((t - 8) ** 2) / (2 * 1.5 ** 2))
    out = compute_plateau_time(t, y, level=0.95, t_start=0.0, mode="non_monotonic")
    assert out > 9.0


def test_brief_false_crossing_ignored_by_hold():
    t = np.linspace(0, 30, 3001)
    y = 30 * (1 - np.exp(-t / 2.0))
    y[399:401] = 35.0  # brief spike above plateau
    out = compute_plateau_time(
        t, y, level=0.95, t_start=0.0, mode="monotonic", hold_duration_s=2.0
    )
    assert out > 5.5


def test_weak_signal_returns_nan():
    t = np.linspace(0, 30, 3001)
    y = 0.05 * np.sin(t) + 100.0
    out = compute_plateau_time(
        t, y, level=0.95, t_start=0.0, mode="monotonic", min_amplitude=3.0
    )
    assert np.isnan(out)


def test_no_plateau_returns_nan():
    t = np.linspace(0, 5, 501)
    y = 5 * t
    out = compute_plateau_time(t, y, level=0.95, t_start=0.0, mode="monotonic")
    assert np.isnan(out)


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
    # t_mix_95 must be >= each finite component (use NaN-safe max behavior)
    components = [res.t_deltaE.get(0.95), res.t_spatial.get(0.95), res.t_texture.get(0.95)]
    finite = [c for c in components if np.isfinite(c)]
    assert finite, "expected at least one finite component"
    assert res.t_mix_95 >= max(finite) - 1e-6
    assert res.confidence in {"high", "medium", "low"}


def test_compute_mixing_time_video_too_short_marks_no_plateau():
    rows = _synth_results(t_end=4.0)  # never plateaus
    res = compute_mixing_time(rows, MixingTimeParams())
    assert "no stable plateau" in res.status.lower() or not np.isfinite(res.t_mix_95)
