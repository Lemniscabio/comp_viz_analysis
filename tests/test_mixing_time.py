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
