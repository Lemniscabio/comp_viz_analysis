"""CLI entry point for Kineticolor analysis."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from src.core.analysis_engine import AnalysisEngine
from src.core.export import DataExporter
from src.core.video_reader import VideoReader
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kineticolor: Computer vision mixing analysis"
    )
    parser.add_argument("--video", type=str, default=None, help="Path to video file (omit for GUI mode)")
    parser.add_argument("--roi", type=str, default=None, help="ROI as x,y,w,h (default: full frame)")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--output", type=str, default="results.csv", help="Output file path")
    parser.add_argument("--reference-frame", type=int, default=0, help="Reference frame number (default: 0)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # GUI mode: launch when no --video provided
    if not args.video:
        from PyQt6.QtWidgets import QApplication
        from src.gui.main_window import MainWindow

        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())

    # CLI mode
    logger = setup_logger()

    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = load_config()

    roi = None
    if args.roi:
        parts = [int(x.strip()) for x in args.roi.split(",")]
        if len(parts) != 4:
            logger.error("ROI must be x,y,w,h")
            sys.exit(1)
        roi = tuple(parts)

    reader = VideoReader(
        path=args.video,
        frame_skip=config["frame_skip"],
        fps_override=config.get("video_fps_override"),
    )

    engine = AnalysisEngine(config)

    # Set reference frame
    if args.reference_frame > 0:
        ref = reader.get_frame(args.reference_frame)
        if ref is None:
            logger.error(f"Cannot read reference frame {args.reference_frame}")
            sys.exit(1)
        engine.set_reference_frame_data(ref)
        logger.info(f"Using frame {args.reference_frame} as reference")

    total_frames = reader.frame_count
    frame_skip = config["frame_skip"]
    expected_frames = total_frames // frame_skip
    bar_width = 30

    logger.info(f"Starting analysis ({total_frames} frames, skip={frame_skip})...")
    t_start = time.perf_counter()
    processed = 0

    for frame_number, frame in reader:
        timestamp = reader.timestamp(frame_number)
        engine.process_frame(frame, frame_number, timestamp, roi=roi)
        processed += 1

        elapsed = time.perf_counter() - t_start
        fps = processed / elapsed if elapsed > 0 else 0
        if expected_frames > 0:
            pct = processed / expected_frames
            eta = (expected_frames - processed) / fps if fps > 0 else 0
            filled = int(bar_width * pct)
            bar = "=" * filled + ">" + " " * (bar_width - filled - 1)
            sys.stdout.write(
                f"\r  [{bar}] {pct:.0%}  {processed}/{expected_frames} frames  "
                f"{fps:.1f} fps  ETA: {eta:.0f}s   "
            )
            sys.stdout.flush()

    sys.stdout.write("\r" + " " * 80 + "\r")  # clear the progress line
    sys.stdout.flush()

    elapsed_total = time.perf_counter() - t_start
    reader.release()

    # Export
    output_path = Path(args.output)
    fmt = config.get("export_format", "csv")
    if output_path.suffix == ".xlsx":
        fmt = "xlsx"
    elif output_path.suffix == ".csv":
        fmt = "csv"

    exporter = DataExporter()
    exporter.export(engine.results, output_path, fmt=fmt)
    logger.info(
        f"Analysis complete. {len(engine.results)} frames processed "
        f"in {elapsed_total:.1f}s ({len(engine.results)/elapsed_total:.1f} frames/sec)."
    )


if __name__ == "__main__":
    main()
