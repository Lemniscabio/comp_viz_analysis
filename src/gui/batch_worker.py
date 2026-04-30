"""Background QThread that runs AnalysisEngine over a list of videos."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from src.core.analysis_engine import AnalysisEngine
from src.core.batch import write_summary_row
from src.core.export import DataExporter
from src.core.mixing_time import MixingTimeParams
from src.core.video_reader import VideoReader
from src.core.visual_time import read_visual_time


class BatchWorker(QThread):
    """Runs AnalysisEngine on each video and aggregates results."""

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
        self._per_video = bool(export_per_video_csv)
        self._stop = False
        self._template_size: Optional[Tuple[int, int]] = None  # (W, H)

    def stop(self) -> None:
        self._stop = True

    # --- ROI / mask scaling ------------------------------------------------
    def _scaled_roi_mask(self, frame_size: Tuple[int, int]):
        """Return (roi, mask) appropriate for a video of frame_size = (W, H).

        First call locks the template size to that video's frame_size;
        subsequent videos either reuse (same size), get scaled (if scale_roi),
        or the caller treats (None, None) as 'skip due to size mismatch' when
        a template ROI/mask is set but scaling is disabled.
        """
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
            return None, None  # caller skips

        sx, sy = cw / sw, ch / sh
        new_roi = None
        if self._roi is not None:
            x, y, w, h = self._roi
            new_roi = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
        new_mask = None
        if self._mask is not None:
            new_mask = cv2.resize(self._mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
        return new_roi, new_mask

    # --- main loop ---------------------------------------------------------
    def run(self) -> None:
        try:
            self._out.mkdir(parents=True, exist_ok=True)
            summary = self._out / "batch_summary.csv"
            if summary.exists():
                summary.unlink()

            total = len(self._videos)
            for i, video in enumerate(self._videos, 1):
                if self._stop:
                    break
                self.progress.emit(i, total, video.name)

                reader: Optional[VideoReader] = None
                try:
                    reader = VideoReader(
                        path=str(video),
                        frame_skip=self._config["frame_skip"],
                        fps_override=self._config.get("video_fps_override"),
                    )
                    frame_size = (reader.width, reader.height)
                    roi, mask = self._scaled_roi_mask(frame_size)
                    if (self._roi is not None and roi is None
                            and not self._scale_roi):
                        self.video_done.emit(video.name, "skipped (size mismatch)")
                        reader.release()
                        continue

                    engine = AnalysisEngine(self._config)
                    fps = reader.fps if reader.fps > 0 else 1.0
                    duration = reader.frame_count / fps
                    for fn, frame in reader:
                        if self._stop:
                            break
                        engine.process_frame(
                            frame, fn, reader.timestamp(fn),
                            roi=roi, mask=mask,
                        )

                    result = engine.finalize(self._params)
                    visual_t = read_visual_time(video)
                    write_summary_row(
                        summary,
                        video_file=video.name,
                        fps=fps,
                        duration_s=duration,
                        frame_count=len(engine.results),
                        roi=roi,
                        result=result,
                        visual_t=visual_t,
                        append=True,
                    )
                    if self._per_video:
                        DataExporter().export(
                            engine.results,
                            self._out / f"{video.stem}_metrics.csv",
                            fmt="csv",
                        )
                    self.video_done.emit(video.name, f"ok ({result.confidence})")
                except Exception as e:
                    self.video_done.emit(video.name, f"error: {e}")
                finally:
                    if reader is not None:
                        try:
                            reader.release()
                        except Exception:
                            pass

            self.finished_all.emit(str(summary))
        except Exception as e:
            self.error_occurred.emit(str(e))
