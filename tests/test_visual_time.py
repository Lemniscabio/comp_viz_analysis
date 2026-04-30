"""Tests for src/core/visual_time.py."""
from __future__ import annotations

import math
from unittest.mock import patch

from src.core.visual_time import read_visual_time


class _FakeRun:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


def _patch_mdls(stdout: str, returncode: int = 0):
    return patch(
        "src.core.visual_time.subprocess.run",
        return_value=_FakeRun(stdout, returncode),
    )


def test_bare_float():
    with _patch_mdls("8.04\n"):
        assert read_visual_time("dummy.mp4") == 8.04


def test_bare_float_with_trailing_s():
    with _patch_mdls("8.04 s\n"):
        assert read_visual_time("dummy.mp4") == 8.04


def test_key_value():
    with _patch_mdls("visual_time=12.5\n"):
        assert read_visual_time("dummy.mp4") == 12.5


def test_key_value_with_extra_text():
    with _patch_mdls("visual_time=7.2; clean=yes\n"):
        assert read_visual_time("dummy.mp4") == 7.2


def test_null_returns_nan():
    with _patch_mdls("(null)\n"):
        assert math.isnan(read_visual_time("dummy.mp4"))


def test_empty_returns_nan():
    with _patch_mdls(""):
        assert math.isnan(read_visual_time("dummy.mp4"))


def test_unparseable_returns_nan():
    with _patch_mdls("not a number\n"):
        assert math.isnan(read_visual_time("dummy.mp4"))


def test_mdls_failure_returns_nan():
    with _patch_mdls("", returncode=1):
        assert math.isnan(read_visual_time("dummy.mp4"))
