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
