#!/usr/bin/env python3
"""Batch-analyze short videos using a per-frame top-5-cell ΔE metric.

For each frame, sort the 25 cell ΔE values and take the mean of the
top 5. That gives a "top5_grand_delta_e" time series. Normalize it
to [0, 1] (divide by its peak), then T_mix,L = first time the
normalized value reaches L (0.90 / 0.95 / 0.99).

Filter: only videos with duration < 40 s are processed (skipped
otherwise, with a notice).

No plateau check, no hold window, no smoothing. Pure "first crossing".

Output: <output_dir>/batch_summary_top5.csv (same schema as batch_summary.csv
so the existing dashboard works).

Usage:
    python scripts/batch_analyze_top5.py <video_dir> <output_dir> [--workers N] [--max-duration 40]
"""
from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse
import math
import multiprocessing as mp
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


# ---------------------------------------------------------------------------
# Filter videos by duration (probes each file briefly)
# ---------------------------------------------------------------------------
def find_videos_under(video_dir: Path, max_duration_s: float):
    """Return [(path, duration), ...] for videos shorter than max_duration_s.

    Also returns a list of (path, duration) skipped videos for reporting.
    """
    import cv2
    kept, skipped = [], []
    for p in sorted(video_dir.iterdir()):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            skipped.append((p, float("nan")))
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        dur = n / fps if fps > 0 else float("nan")
        if dur <= max_duration_s:
            kept.append((p, dur))
        else:
            skipped.append((p, dur))
    return kept, skipped


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _process_video(args: Tuple[str, Dict[str, Any], Any]) -> Dict[str, Any]:
    video_path_str, config, prog_q = args
    video_path = Path(video_path_str)
    name = video_path.name

    import numpy as np
    from src.core.analysis_engine import AnalysisEngine
    from src.core.mixing_time import MixingTimeResult
    from src.core.video_reader import VideoReader
    from src.core.visual_time import read_visual_time

    prog_q.put(("start", name))

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
            if now - last_tick >= 0.3:
                rate = processed / (now - t0) if now > t0 else 0
                prog_q.put(("progress", name, processed, expected, rate))
                last_tick = now
    finally:
        reader.release()

    if not engine.results:
        return {
            "video_path": str(video_path), "video_name": name, "ok": False,
            "error": "no frames produced", "elapsed_s": time.perf_counter() - t0,
        }

    # ---- Top-5 metric ------------------------------------------------------
    times = np.array([r["timestamp"] for r in engine.results], dtype=np.float64)
    n_frames = len(engine.results)
    cells = np.array(
        [np.asarray(r["cell_avg"], dtype=np.float64) for r in engine.results]
    )  # shape (n_frames, n_cells), NaN allowed

    K = 5
    top5 = np.full(n_frames, np.nan, dtype=np.float64)
    for i in range(n_frames):
        row = cells[i]
        finite = row[np.isfinite(row)]
        if len(finite) == 0:
            continue
        if len(finite) >= K:
            top5[i] = np.partition(finite, -K)[-K:].mean()
        else:
            top5[i] = finite.mean()

    # Normalize by peak
    finite_top5 = top5[np.isfinite(top5)]
    peak = float(finite_top5.max()) if len(finite_top5) else 0.0
    if peak <= 0:
        return {
            "video_path": str(video_path), "video_name": name, "ok": False,
            "error": "top-5 peak is zero or negative",
            "elapsed_s": time.perf_counter() - t0,
        }
    norm = top5 / peak

    # First-crossing for each level
    def first_cross(level: float) -> float:
        mask = np.isfinite(norm) & (norm >= level)
        idx = np.where(mask)[0]
        return float(times[idx[0]]) if len(idx) else float("nan")

    t90 = first_cross(0.90)
    t95 = first_cross(0.95)
    t99 = first_cross(0.99)

    # Build a MixingTimeResult-compatible payload so the existing
    # write_summary_row/CSV schema/dashboard all keep working.
    result = MixingTimeResult(
        t_start_s=0.0,
        levels=(0.90, 0.95, 0.99),
        t_mix={0.90: t90, 0.95: t95, 0.99: t99},
        t_deltaE={0.90: t90, 0.95: t95, 0.99: t99},  # method = pure ΔE-style
        amplitudes={"grand_delta_e": peak, "contact": 0.0, "contrast": 0.0},
        tail_slopes={"grand_delta_e": 0.0, "contact": 0.0, "contrast": 0.0},
        n_valid_cells=int(np.isfinite(cells).any(axis=0).sum()),
        status="top5-per-frame, normalized first-crossing",
        confidence="high" if peak >= 3.0 else ("medium" if peak >= 1.5 else "low"),
    )

    visual_t = read_visual_time(video_path)

    return {
        "video_path": str(video_path),
        "video_name": name,
        "ok": True,
        "fps": fps,
        "duration_s": duration,
        "frame_count": len(engine.results),
        "elapsed_s": time.perf_counter() - t0,
        "visual_t": visual_t,
        "result": asdict(result),
    }


def _safe_process_video(args):
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
# Live in-place renderer (same as batch_analyze.py)
# ---------------------------------------------------------------------------
class LiveRenderer(threading.Thread):
    def __init__(self, q, total: int):
        super().__init__(daemon=True)
        self._q = q; self._total = total
        self._slots: Dict[str, Tuple[int, int, float]] = {}
        self._completed = 0; self._lines_drawn = 0
        self._tty = sys.stdout.isatty()

    def _erase_block(self):
        if self._lines_drawn and self._tty:
            sys.stdout.write(f"\033[{self._lines_drawn}F\033[J")
        self._lines_drawn = 0

    def _draw_block(self):
        running = sorted(self._slots.keys())
        sys.stdout.write(
            f"─── {self._completed} done / {self._total}  ({len(running)} running) ───\n"
        )
        for nm in running:
            p, e, r = self._slots[nm]
            pct = (100.0 * p / e) if e else 0.0
            disp = nm if len(nm) <= 40 else nm[:37] + "..."
            sys.stdout.write(
                f"  ▶ {disp:<40}  {p:>5}/{e:<5} ({pct:5.1f}%)  {r:5.1f} fps\n"
            )
        self._lines_drawn = 1 + len(running)
        sys.stdout.flush()

    def run(self):
        last_redraw = 0.0
        while True:
            try:
                ev = self._q.get(timeout=0.15)
            except Empty:
                ev = None
            need_redraw = False
            if ev is not None:
                kind = ev[0]
                if kind == "stop":
                    self._erase_block(); sys.stdout.flush(); return
                elif kind == "start":
                    self._slots[ev[1]] = (0, 0, 0.0); need_redraw = True
                elif kind == "progress":
                    _, name, p, e, r = ev
                    self._slots[name] = (p, e, r); need_redraw = True
                elif kind == "done":
                    _, name, log_line = ev
                    self._slots.pop(name, None); self._completed += 1
                    self._erase_block()
                    sys.stdout.write(log_line + "\n")
                    self._draw_block()
                    last_redraw = time.perf_counter()
                    continue
            now = time.perf_counter()
            if need_redraw and (now - last_redraw) > 0.2:
                self._erase_block(); self._draw_block(); last_redraw = now


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------
def _write_row(summary_csv: Path, payload: Dict[str, Any]) -> None:
    from src.core.batch import write_summary_row
    from src.core.mixing_time import MixingTimeResult

    result = MixingTimeResult(**payload["result"])
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Top-5-cell ΔE per frame, normalize, first-crossing of 0.95"
    )
    ap.add_argument("video_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument(
        "--workers", type=int, default=min(10, os.cpu_count() or 4),
        help="Parallel worker processes. Default: min(10, cpu_count).",
    )
    ap.add_argument(
        "--max-duration", type=float, default=40.0,
        help="Skip videos longer than this many seconds. Default: 40.",
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

    kept, skipped = find_videos_under(video_dir, args.max_duration)
    if not kept:
        sys.exit(
            f"No videos with duration <= {args.max_duration}s found in {video_dir}"
        )

    from src.utils.config_loader import load_config
    config = load_config(config_path) if config_path.exists() else load_config()

    summary_csv = output_dir / "batch_summary_top5.csv"
    if summary_csv.exists():
        summary_csv.unlink()

    workers = max(1, min(args.workers, len(kept)))
    print(f"Mode:         TOP-5 CELLS PER FRAME (normalized first-crossing)")
    print(f"Duration filter: <= {args.max_duration:.0f} s")
    print(f"Found:        {len(kept)} videos kept, {len(skipped)} skipped")
    if skipped:
        for p, d in skipped[:6]:
            print(f"  skipped: {p.name} ({d:.1f}s)")
        if len(skipped) > 6:
            print(f"  ... and {len(skipped) - 6} more")
    print(f"Workers:      {workers} (BLAS threads/worker = 1)")
    print(f"Output:       {summary_csv}")
    print(f"Config:       {config_path}")
    print()

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    prog_q = manager.Queue()
    renderer = LiveRenderer(prog_q, total=len(kept))
    renderer.start()

    work_items = [(str(p), config, prog_q) for p, _ in kept]
    t_start = time.perf_counter()
    done = failed = 0

    try:
        with ctx.Pool(processes=workers) as pool:
            for i, payload in enumerate(
                pool.imap_unordered(_safe_process_video, work_items), 1
            ):
                name = payload.get("video_name") or "?"
                elapsed = payload.get("elapsed_s", 0.0)
                if payload.get("ok"):
                    _write_row(summary_csv, payload)
                    conf = payload["result"].get("confidence", "?")
                    tmix95 = payload["result"].get("t_mix", {}).get(0.95)
                    tmix_str = (
                        f"T95={tmix95:.2f}s"
                        if isinstance(tmix95, (int, float)) and tmix95 == tmix95
                        else "T95=NaN"
                    )
                    line = f"[{i}/{len(kept)}] {name}: ok ({conf}) {tmix_str} in {elapsed:.1f}s"
                    done += 1
                else:
                    line = (
                        f"[{i}/{len(kept)}] {name}: FAILED "
                        f"({payload.get('error')}) in {elapsed:.1f}s"
                    )
                    if payload.get("traceback"):
                        sys.stderr.write(payload["traceback"] + "\n")
                    failed += 1
                prog_q.put(("done", name, line))
    finally:
        prog_q.put(("stop",))
        renderer.join(timeout=2)

    total = time.perf_counter() - t_start
    print()
    print(f"Summary: processed={done}, failed={failed}, total={total:.0f}s")
    print(f"Output: {summary_csv}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
