#!/usr/bin/env python3
"""Batch-analyze short videos with BOTH top-5 and all-cells-avg metrics.

For each frame: compute two scalar series from the per-cell ΔE values:
  - top5_de(t):  mean of the top-5 cells that frame
  - all25_de(t): mean of all valid (finite) cells that frame
Each series is normalized by its own peak; T_mix,L is the first time
the normalized signal crosses L (no smoothing, no hold).

Filter: only videos with duration <= --max-duration (default 40 s)
are processed.

Output: <output_dir>/batch_summary_cells.csv
Columns include both methods' values + visual time + deltas, so you
can compare them side-by-side in the same row.

Usage:
    python scripts/batch_analyze_cells.py <video_dir> <output_dir> [--workers N] [--max-duration 40]
"""
from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import math
import multiprocessing as mp
import sys
import threading
import time
import traceback
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

CSV_COLUMNS = [
    "video_file", "status", "fps", "duration_s", "frame_count",
    "visual_t_mix_s",
    # top-5
    "top5_peak", "top5_t_mix_90_s", "top5_t_mix_95_s", "top5_t_mix_99_s",
    "top5_delta_95_s", "top5_abs_delta_95_s",
    # all 25
    "all_peak", "all_t_mix_90_s", "all_t_mix_95_s", "all_t_mix_99_s",
    "all_delta_95_s", "all_abs_delta_95_s",
    "n_valid_cells", "notes",
]


def find_videos_under(video_dir: Path, max_duration_s: float):
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
        (kept if dur <= max_duration_s else skipped).append((p, dur))
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
    from src.core.mixing_time import _compute_cells_first_crossing
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

    visual_t = read_visual_time(video_path)
    levels = (0.90, 0.95, 0.99)
    top5 = _compute_cells_first_crossing(engine.results, levels, K=5)
    all25 = _compute_cells_first_crossing(engine.results, levels, K=None)

    cells = np.array(
        [np.asarray(r["cell_avg"], dtype=np.float64) for r in engine.results]
    )
    n_valid_cells = int(np.isfinite(cells).any(axis=0).sum())

    return {
        "video_path": str(video_path),
        "video_name": name,
        "ok": True,
        "fps": fps,
        "duration_s": duration,
        "frame_count": len(engine.results),
        "elapsed_s": time.perf_counter() - t0,
        "visual_t": visual_t,
        "top5": top5,
        "all25": all25,
        "n_valid_cells": n_valid_cells,
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
# Live in-place renderer
# ---------------------------------------------------------------------------
class LiveRenderer(threading.Thread):
    def __init__(self, q, total: int):
        super().__init__(daemon=True)
        self._q = q; self._total = total
        self._slots: Dict[str, Tuple[int, int, float]] = {}
        self._completed = 0; self._lines_drawn = 0
        self._tty = sys.stdout.isatty()

    def _erase(self):
        if self._lines_drawn and self._tty:
            sys.stdout.write(f"\033[{self._lines_drawn}F\033[J")
        self._lines_drawn = 0

    def _draw(self):
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
        last = 0.0
        while True:
            try:
                ev = self._q.get(timeout=0.15)
            except Empty:
                ev = None
            need = False
            if ev is not None:
                k = ev[0]
                if k == "stop":
                    self._erase(); sys.stdout.flush(); return
                elif k == "start":
                    self._slots[ev[1]] = (0, 0, 0.0); need = True
                elif k == "progress":
                    _, n, p, e, r = ev
                    self._slots[n] = (p, e, r); need = True
                elif k == "done":
                    _, n, line = ev
                    self._slots.pop(n, None); self._completed += 1
                    self._erase(); sys.stdout.write(line + "\n"); self._draw()
                    last = time.perf_counter()
                    continue
            now = time.perf_counter()
            if need and (now - last) > 0.2:
                self._erase(); self._draw(); last = now


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------
def _fmt(v: Optional[float], dp: int = 4) -> str:
    if v is None:
        return ""
    try:
        if v != v:
            return ""
        return f"{float(v):.{dp}f}"
    except (TypeError, ValueError):
        return ""


def _write_row(csv_path: Path, payload: Dict[str, Any]) -> None:
    visual = payload["visual_t"]
    t5 = payload["top5"]["t_mix"]; p5 = payload["top5"]["peak"]
    ta = payload["all25"]["t_mix"]; pa = payload["all25"]["peak"]

    def _delta(t95: float, vis: float):
        if vis is None or vis != vis or t95 is None or t95 != t95:
            return float("nan"), float("nan")
        return t95 - vis, abs(t95 - vis)

    d5, ad5 = _delta(t5.get(0.95, float("nan")), visual)
    da, ada = _delta(ta.get(0.95, float("nan")), visual)

    row = {
        "video_file": payload["video_name"],
        "status": "ok",
        "fps": _fmt(payload["fps"], 3),
        "duration_s": _fmt(payload["duration_s"], 3),
        "frame_count": payload["frame_count"],
        "visual_t_mix_s": _fmt(visual),
        "top5_peak": _fmt(p5, 3),
        "top5_t_mix_90_s": _fmt(t5.get(0.90)),
        "top5_t_mix_95_s": _fmt(t5.get(0.95)),
        "top5_t_mix_99_s": _fmt(t5.get(0.99)),
        "top5_delta_95_s": _fmt(d5),
        "top5_abs_delta_95_s": _fmt(ad5),
        "all_peak": _fmt(pa, 3),
        "all_t_mix_90_s": _fmt(ta.get(0.90)),
        "all_t_mix_95_s": _fmt(ta.get(0.95)),
        "all_t_mix_99_s": _fmt(ta.get(0.99)),
        "all_delta_95_s": _fmt(da),
        "all_abs_delta_95_s": _fmt(ada),
        "n_valid_cells": payload.get("n_valid_cells", ""),
        "notes": "",
    }
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Side-by-side top-5 and all-cells-avg per-frame metrics"
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
            f"No videos with duration <= {args.max_duration}s in {video_dir}"
        )

    from src.utils.config_loader import load_config
    config = load_config(config_path) if config_path.exists() else load_config()

    summary_csv = output_dir / "batch_summary_cells.csv"
    if summary_csv.exists():
        summary_csv.unlink()

    workers = max(1, min(args.workers, len(kept)))
    print(f"Mode:         TOP-5 + ALL-CELLS AVG (per-frame, normalized)")
    print(f"Duration filter: <= {args.max_duration:.0f} s")
    print(f"Found:        {len(kept)} kept, {len(skipped)} skipped")
    if skipped:
        for p, d in skipped[:6]:
            print(f"  skipped: {p.name} ({d:.1f}s)")
        if len(skipped) > 6:
            print(f"  ... and {len(skipped) - 6} more")
    print(f"Workers:      {workers}")
    print(f"Output:       {summary_csv}")
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
                    t5 = payload["top5"]["t_mix"].get(0.95)
                    ta = payload["all25"]["t_mix"].get(0.95)
                    t5s = f"{t5:.2f}" if isinstance(t5, (int, float)) and t5 == t5 else "NaN"
                    tas = f"{ta:.2f}" if isinstance(ta, (int, float)) and ta == ta else "NaN"
                    line = (
                        f"[{i}/{len(kept)}] {name}: ok  "
                        f"top5={t5s}s  all={tas}s  in {elapsed:.1f}s"
                    )
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
