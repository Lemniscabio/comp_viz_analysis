#!/usr/bin/env python3
"""Batch-analyze a folder of videos in parallel and write batch_summary.csv.

For each video <stem>.{mp4,mov,avi,mkv,m4v} in <video_dir>:
  - run the full mixing-time pipeline (AnalysisEngine + finalize)
  - read the user's visual mixing time from the file's macOS Finder Comment
    (Get Info → Comments — type a bare number like `8.04` or `1:23`)
  - append one row to <output_dir>/batch_summary.csv

Output is a single CSV. No per-video CSVs, no PNGs, no log files,
no JSON config. T_mix,95 vs visual time is in the same row.

Usage:
    python scripts/batch_analyze.py <video_dir> <output_dir> [--workers N] [--config PATH]

Workers default to min(10, cpu_count). Each worker pins its own BLAS
thread count to 1 so 10 workers don't oversubscribe the cores.
"""
from __future__ import annotations

# IMPORTANT: cap BLAS / OpenMP thread counts BEFORE numpy / cv2 are imported,
# both here and in workers (workers re-execute this module-level code).
import os

for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def find_videos(video_dir: Path) -> List[Path]:
    return sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)


# ---------------------------------------------------------------------------
# Worker (runs in a separate process)
# ---------------------------------------------------------------------------
def _process_video(args: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Worker entry point: run pipeline on one video, return a serializable dict.

    Returning a plain dict (rather than a MixingTimeResult) keeps the
    parent → child IPC simple and avoids dataclass-pickling surprises.
    """
    video_path_str, config = args
    video_path = Path(video_path_str)

    # Imports go inside the worker so each process pays the import cost
    # only once it's busy. This also makes the parent's startup snappy.
    from src.core.analysis_engine import AnalysisEngine
    from src.core.mixing_time import MixingTimeParams
    from src.core.video_reader import VideoReader
    from src.core.visual_time import read_visual_time

    name = video_path.name
    print(f"  ▶ start    {name}", flush=True)

    t0 = time.perf_counter()
    reader = VideoReader(
        path=str(video_path),
        frame_skip=config["frame_skip"],
        fps_override=config.get("video_fps_override"),
    )
    fps = reader.fps if reader.fps > 0 else 1.0
    duration = reader.frame_count / fps
    expected = max(1, reader.frame_count // max(config["frame_skip"], 1))
    engine = AnalysisEngine(config)
    last_tick = t0
    processed = 0
    try:
        for frame_number, frame in reader:
            engine.process_frame(frame, frame_number, reader.timestamp(frame_number))
            processed += 1
            now = time.perf_counter()
            if now - last_tick >= 2.0:
                rate = processed / (now - t0) if now > t0 else 0
                pct = 100.0 * processed / expected
                eta = (expected - processed) / rate if rate > 0 else 0
                print(
                    f"  ⋯ tick     {name}  "
                    f"{processed}/{expected} ({pct:4.1f}%)  "
                    f"{rate:5.1f} fps  ETA {eta:4.0f}s",
                    flush=True,
                )
                last_tick = now
    finally:
        reader.release()

    if not engine.results:
        return {
            "video_path": str(video_path), "ok": False,
            "error": "no frames produced", "elapsed_s": time.perf_counter() - t0,
        }

    result = engine.finalize(MixingTimeParams())
    visual_t = read_visual_time(video_path)

    return {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "ok": True,
        "fps": fps,
        "duration_s": duration,
        "frame_count": len(engine.results),
        "elapsed_s": time.perf_counter() - t0,
        "visual_t": visual_t,
        "result": asdict(result),  # plain-dict serialization
    }


def _safe_process_video(args: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Catch exceptions in the worker so one bad video doesn't kill the pool."""
    try:
        return _process_video(args)
    except Exception as e:
        return {
            "video_path": args[0],
            "video_name": Path(args[0]).name,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "elapsed_s": 0.0,
        }


# ---------------------------------------------------------------------------
# Parent: write rows as workers finish
# ---------------------------------------------------------------------------
def _result_from_dict(d: Dict[str, Any]):
    """Reconstruct a minimal MixingTimeResult-shaped object from asdict() output."""
    from src.core.mixing_time import MixingTimeResult

    # Dict[float, float] keys survive pickling but only when going through
    # asdict()/dict() round-trips on the same Python; nothing fancy needed.
    return MixingTimeResult(**d)


def _write_row(summary_csv: Path, payload: Dict[str, Any]) -> None:
    from src.core.batch import write_summary_row

    result = _result_from_dict(payload["result"])
    write_summary_row(
        summary_csv,
        video_file=payload["video_name"],
        fps=payload["fps"],
        duration_s=payload["duration_s"],
        frame_count=payload["frame_count"],
        roi=None,
        result=result,
        visual_t=payload["visual_t"],
        append=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-analyze videos -> single summary CSV")
    ap.add_argument("video_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument(
        "--workers", type=int, default=min(10, os.cpu_count() or 4),
        help="Parallel worker processes. Default: min(10, cpu_count).",
    )
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

    # Imports here keep the worker spawn-import cost predictable.
    from src.utils.config_loader import load_config
    config = load_config(config_path) if config_path.exists() else load_config()

    summary_csv = output_dir / "batch_summary.csv"
    if summary_csv.exists():
        summary_csv.unlink()

    workers = max(1, min(args.workers, len(videos)))
    print(f"Found {len(videos)} videos.")
    print(f"Workers:      {workers} (BLAS threads/worker = 1)")
    print(f"Output:       {summary_csv}")
    print(f"Config:       {config_path}")
    print()

    work_items = [(str(v), config) for v in videos]
    t_start = time.perf_counter()
    done = failed = 0

    # 'spawn' is the only safe start method on macOS for our stack
    # (OpenCV + numpy + threads + Qt-adjacent code). 'fork' can deadlock.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for i, payload in enumerate(
            pool.imap_unordered(_safe_process_video, work_items), 1
        ):
            name = payload.get("video_name") or Path(payload.get("video_path", "?")).name
            elapsed = payload.get("elapsed_s", 0.0)
            if payload.get("ok"):
                _write_row(summary_csv, payload)
                conf = payload["result"].get("confidence", "?")
                tmix95 = payload["result"].get("t_mix", {}).get(0.95)
                tmix_str = f"T95={tmix95:.2f}s" if isinstance(tmix95, (int, float)) and tmix95 == tmix95 else "T95=NaN"
                print(f"[{i}/{len(videos)}] {name}: ok ({conf}) {tmix_str} in {elapsed:.1f}s")
                done += 1
            else:
                print(f"[{i}/{len(videos)}] {name}: FAILED ({payload.get('error')}) in {elapsed:.1f}s")
                if payload.get("traceback"):
                    sys.stderr.write(payload["traceback"] + "\n")
                failed += 1
            sys.stdout.flush()

    total = time.perf_counter() - t_start
    print()
    print(f"Summary: processed={done}, failed={failed}, total={total:.0f}s")
    print(f"Output: {summary_csv}")


if __name__ == "__main__":
    # Required on macOS when using 'spawn'; harmless elsewhere.
    mp.freeze_support()
    main()
