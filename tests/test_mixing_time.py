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
