"""Integration test: end-to-end analysis on a synthetic video."""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from src.core.analysis_engine import AnalysisEngine
from src.core.export import DataExporter
from src.utils.config_loader import load_config


def _create_synthetic_video(path: Path, n_frames: int = 10) -> None:
    """Create a simple video that transitions from blue to green."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (100, 100))
    for i in range(n_frames):
        t = i / (n_frames - 1)
        color = np.array([255 * (1 - t), 255 * t, 0], dtype=np.uint8)  # BGR
        frame = np.full((100, 100, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestIntegration:
    def test_end_to_end(self, tmp_path):
        """Full pipeline: synthetic video -> analysis -> CSV export."""
        video_path = tmp_path / "test.mp4"
        _create_synthetic_video(video_path, n_frames=5)

        config = load_config()
        engine = AnalysisEngine(config)

        cap = cv2.VideoCapture(str(video_path))
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            engine.process_frame(frame, frame_num, frame_num / 30.0)
            frame_num += 1
        cap.release()

        assert len(engine.results) == 5
        # First frame is reference -> delta_e should be 0
        assert engine.results[0]["grand_delta_e"] == 0.0
        # Later frames should have increasing delta_e
        assert engine.results[-1]["grand_delta_e"] > engine.results[1]["grand_delta_e"]

        # Export CSV
        csv_path = tmp_path / "results.csv"
        exporter = DataExporter()
        exporter.export(engine.results, csv_path, fmt="csv")
        assert csv_path.exists()

        # Export XLSX
        xlsx_path = tmp_path / "results.xlsx"
        exporter.export(engine.results, xlsx_path, fmt="xlsx")
        assert xlsx_path.exists()
