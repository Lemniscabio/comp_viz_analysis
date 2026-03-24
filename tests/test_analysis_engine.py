"""Tests for analysis engine."""
from __future__ import annotations
import numpy as np
from src.core.analysis_engine import AnalysisEngine
from src.utils.config_loader import DEFAULT_CONFIG

class TestAnalysisEngine:
    def _make_config(self, **overrides):
        config = DEFAULT_CONFIG.copy()
        config.update(overrides)
        return config

    def test_set_reference_frame(self):
        config = self._make_config()
        engine = AnalysisEngine(config)
        ref = np.full((50, 50, 3), 128, dtype=np.uint8)
        engine.set_reference_frame_data(ref)
        assert engine._reference_frame is not None

    def test_process_single_frame(self):
        config = self._make_config()
        engine = AnalysisEngine(config)
        ref = np.full((50, 50, 3), 100, dtype=np.uint8)
        frame = np.full((50, 50, 3), 150, dtype=np.uint8)
        engine.set_reference_frame_data(ref)
        engine.process_frame(frame, frame_number=1, timestamp=0.033)
        assert len(engine.results) == 1
        row = engine.results[0]
        assert row["frame_number"] == 1
        assert row["timestamp"] == 0.033
        assert "grand_delta_e" in row
        assert "contact_perimeter" in row
        assert "contrast" in row
        assert "homogeneity" in row
        assert "energy" in row
        assert "variance_r" in row

    def test_glcm_frame_skip(self):
        """When glcm_frame_skip=2, GLCM metrics hold value on non-GLCM frames."""
        config = self._make_config(glcm_frame_skip=2)
        engine = AnalysisEngine(config)
        ref = np.full((50, 50, 3), 100, dtype=np.uint8)
        engine.set_reference_frame_data(ref)
        # Frame 0: uniform -> contrast=0. This is a GLCM frame.
        frame1 = np.full((50, 50, 3), 120, dtype=np.uint8)
        engine.process_frame(frame1, frame_number=0, timestamp=0.0)
        assert engine.results[0]["contrast"] == 0.0
        # Frame 1: half-and-half -> would have high contrast if recomputed.
        # But glcm_frame_skip=2, so GLCM is NOT recomputed. Value held from frame 0.
        frame2 = np.zeros((50, 50, 3), dtype=np.uint8)
        frame2[:, 25:] = 255
        engine.process_frame(frame2, frame_number=1, timestamp=0.033)
        # Contrast should still be 0 (held), not the high value from the split frame
        assert engine.results[1]["contrast"] == 0.0
        # Frame 2: another GLCM frame. Now GLCM is recomputed on a new frame.
        frame3 = np.zeros((50, 50, 3), dtype=np.uint8)
        frame3[:, 25:] = 200
        engine.process_frame(frame3, frame_number=2, timestamp=0.066)
        # Now contrast should be recomputed and be > 0
        assert engine.results[2]["contrast"] > 0
