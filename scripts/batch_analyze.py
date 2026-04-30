#!/usr/bin/env python3
"""Batch-analyze a folder of videos and write a single batch_summary.csv.

For each video <stem>.{mp4,mov,avi,mkv,m4v} in <video_dir>:
  - run the full mixing-time pipeline (AnalysisEngine + finalize)
  - read the user's visual mixing time from the file's macOS Finder Comment
    (Get Info → Comments — type a bare number like `8.04`)
  - append one row to <output_dir>/batch_summary.csv

Output is a single CSV. No per-video CSVs, no PNGs, no log files,
no JSON config. T_mix,95 vs visual time is in the same row for diffing.

Usage:
    python scripts/batch_analyze.py <video_dir> <output_dir> [--config PATH]
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.analysis_engine import AnalysisEngine  # noqa: E402
from src.core.batch import write_summary_row  # noqa: E402
from src.core.mixing_time import MixingTimeParams  # noqa: E402
from src.core.video_reader import VideoReader  # noqa: E402
from src.core.visual_time import read_visual_time  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def find_videos(video_dir: Path) -> List[Path]:
    return sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def analyze_one(video_path: Path, summary_csv: Path, config: dict) -> int:
    """Run engine on one video and append a row to summary_csv. Returns n_frames."""
    reader = VideoReader(
        path=str(video_path),
        frame_skip=config["frame_skip"],
        fps_override=config.get("video_fps_override"),
    )
    fps = reader.fps if reader.fps > 0 else 1.0
    duration = reader.frame_count / fps
    expected = max(1, reader.frame_count // max(config["frame_skip"], 1))
    engine = AnalysisEngine(config)

    t0 = time.perf_counter()
    processed = 0
    last_print = 0.0
    bar_width = 24
    for frame_number, frame in reader:
        engine.process_frame(frame, frame_number, reader.timestamp(frame_number))
        processed += 1
        now = time.perf_counter()
        if now - last_print > 0.25:
            elapsed = now - t0
            rate = processed / elapsed if elapsed > 0 else 0
            pct = processed / expected if expected else 0
            eta = (expected - processed) / rate if rate > 0 else 0
            filled = int(bar_width * min(pct, 1.0))
            bar = "=" * filled + ">" + " " * max(bar_width - filled - 1, 0)
            sys.stdout.write(
                f"\r  [{bar}] {processed}/{expected}  {rate:5.1f} fps  ETA {eta:5.0f}s "
            )
            sys.stdout.flush()
            last_print = now
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()
    reader.release()

    if not engine.results:
        raise RuntimeError("no frames produced")

    result = engine.finalize(MixingTimeParams())
    visual_t = read_visual_time(video_path)

    write_summary_row(
        summary_csv,
        video_file=video_path.name,
        fps=fps,
        duration_s=duration,
        frame_count=len(engine.results),
        roi=None,
        result=result,
        visual_t=visual_t,
        append=True,
    )
    return len(engine.results)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-analyze videos -> single summary CSV")
    ap.add_argument("video_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "config" / "default_config.yaml",
    )
    args = ap.parse_args()

    video_dir: Path = args.video_dir.expanduser()
    output_dir: Path = args.output_dir.expanduser()
    config_path: Path = args.config.expanduser()

    if not video_dir.is_dir():
        sys.exit(f"video_dir not found: {video_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(video_dir)
    if not videos:
        sys.exit(f"No videos found in {video_dir}")

    config = load_config(config_path) if config_path.exists() else load_config()

    summary_csv = output_dir / "batch_summary.csv"
    if summary_csv.exists():
        summary_csv.unlink()

    print(f"Found {len(videos)} videos.")
    print(f"Output:       {summary_csv}")
    print(f"Config:       {config_path}")
    print()

    t_start = time.perf_counter()
    done = failed = 0

    for i, src in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {src.name}")
        try:
            n = analyze_one(src, summary_csv, config)
            print(f"  done: {n} frames")
            done += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

    total = time.perf_counter() - t_start
    print()
    print(f"Summary: processed={done}, failed={failed}, total={total:.0f}s")
    print(f"Output: {summary_csv}")


if __name__ == "__main__":
    main()
