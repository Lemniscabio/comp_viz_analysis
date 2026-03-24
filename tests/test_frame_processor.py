"""Tests for frame processor."""

from __future__ import annotations

import logging
import numpy as np
import pytest
from src.core.frame_processor import FrameProcessor


class TestCropToRoi:
    def test_crop_dimensions(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        fp = FrameProcessor()
        cropped = fp.crop_to_roi(frame, (10, 20, 50, 30))  # x, y, w, h
        assert cropped.shape == (30, 50, 3)

    def test_crop_content(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:20, 10:20] = 255
        fp = FrameProcessor()
        cropped = fp.crop_to_roi(frame, (10, 10, 10, 10))
        assert np.all(cropped == 255)

    def test_none_roi_returns_full_frame(self):
        frame = np.zeros((50, 60, 3), dtype=np.uint8)
        fp = FrameProcessor()
        cropped = fp.crop_to_roi(frame, None)
        assert cropped.shape == frame.shape


class TestApplyMask:
    def test_mask_zeros_pixels(self):
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.uint8)
        mask[5:, :] = 0  # mask out bottom half
        fp = FrameProcessor()
        result = fp.apply_mask(frame, mask)
        assert np.all(result[5:, :] == 0)
        assert np.all(result[:5, :] == 128)

    def test_none_mask_returns_copy(self):
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        fp = FrameProcessor()
        result = fp.apply_mask(frame, None)
        assert np.array_equal(result, frame)


class TestBrightnessCheck:
    def test_no_warning_on_stable_brightness(self, caplog):
        fp = FrameProcessor(brightness_change_threshold=0.2)
        frame1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 105, dtype=np.uint8)
        fp.check_brightness(frame1, None)
        with caplog.at_level(logging.WARNING):
            fp.check_brightness(frame2, None)
        assert "brightness" not in caplog.text.lower()

    def test_warning_on_drastic_change(self, caplog):
        fp = FrameProcessor(brightness_change_threshold=0.2)
        frame1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 200, dtype=np.uint8)
        fp.check_brightness(frame1, None)
        with caplog.at_level(logging.WARNING):
            fp.check_brightness(frame2, None)
        assert "brightness" in caplog.text.lower()

    def test_mask_excluded_from_brightness(self):
        fp = FrameProcessor(brightness_change_threshold=0.2)
        frame = np.full((10, 10, 3), 200, dtype=np.uint8)
        frame[5:, :] = 0
        mask = np.ones((10, 10), dtype=np.uint8)
        mask[5:, :] = 0
        fp.check_brightness(frame, mask)
        # Brightness should reflect only unmasked region (value 200)
        assert abs(fp._prev_brightness - 200.0) < 1.0
