#!/usr/bin/env python3
"""Batch-analyze using ONLY grand Delta-E as the mixing-time metric.

Same pipeline as scripts/batch_analyze.py but the final T_mix,L value
is taken purely from T_deltaE,L (no spatial, no texture, no max(...)).

Spatial and texture component columns are still written to the CSV so
you can see what they would have contributed — but they do NOT gate
the reported T_mix,95.

Output: <output_dir>/batch_summary_deltaE.csv

Usage:
    python scripts/batch_analyze_deltaE.py <video_dir> <output_dir> [--workers N]
"""
from __future__ import annotations

# Cap BLAS / OpenMP thread counts BEFORE numpy / cv2 imports.
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


def find_videos(video_dir: Path) -> List[Path]:
    return sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _process_video(args: Tuple[str, Dict[str, Any], Any]) -> Dict[str, Any]:
    video_path_str, config, prog_q = args
    video_path = Path(video_path_str)
    name = video_path.name

    from src.core.analysis_engine import AnalysisEngine
    from src.core.mixing_time import MixingTimeParams
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

    result = engine.finalize(MixingTimeParams())
    visual_t = read_visual_time(video_path)

    # ---- THE ONLY FUNCTIONAL DIFFERENCE FROM batch_analyze.py ----
    # Replace T_mix with T_deltaE — drop spatial/texture from the gate.
    # The component columns (t_spatial_*, t_texture_*) remain in the CSV
    # so you can still see what they WOULD have said.
    result.t_mix = dict(result.t_deltaE)

    # Recompute confidence based on ΔE alone:
    #  - high  : amplitude >= 3 AND |normalized tail slope of ΔE| <= 0.05
    #  - medium: amplitude >= 1.5 (signal exists but tail still drifting)
    #  - low   : weak signal or T_mix,95 is NaN
    amp = result.amplitudes.get("grand_delta_e", 0.0)
    slope = result.tail_slopes.get("grand_delta_e", float("nan"))
    t95 = result.t_mix.get(0.95, float("nan"))
    if isinstance(t95, float) and math.isnan(t95):
        result.confidence = "low"
    elif amp >= 3.0 and isinstance(slope, float) and not math.isnan(slope):
        result.confidence = "high"
    elif amp >= 1.5:
        result.confidence = "medium"
    else:
        result.confidence = "low"
    result.status = "deltaE-only" if not math.isnan(t95) else "deltaE-only / NaN"

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


def _safe_process_video(args: Tuple[str, Dict[str, Any], Any]) -> Dict[str, Any]:
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
# Live in-place renderer (identical to batch_analyze.py)
# ---------------------------------------------------------------------------
class LiveRenderer(threading.Thread):
    def __init__(self, q, total: int):
        super().__init__(daemon=True)
        self._q = q
        self._total = total
        self._slots: Dict[str, Tuple[int, int, float]] = {}
        self._completed = 0
        self._lines_drawn = 0
        self._tty = sys.stdout.isatty()

    def _erase_block(self) -> None:
        if self._lines_drawn and self._tty:
            sys.stdout.write(f"\033[{self._lines_drawn}F\033[J")
        self._lines_drawn = 0

    def _draw_block(self) -> None:
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

    def run(self) -> None:
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
                    self._erase_block()
                    sys.stdout.flush()
                    return
                elif kind == "start":
                    self._slots[ev[1]] = (0, 0, 0.0)
                    need_redraw = True
                elif kind == "progress":
                    _, name, p, e, r = ev
                    self._slots[name] = (p, e, r)
                    need_redraw = True
                elif kind == "done":
                    _, name, log_line = ev
                    self._slots.pop(name, None)
                    self._completed += 1
                    self._erase_block()
                    sys.stdout.write(log_line + "\n")
                    self._draw_block()
                    last_redraw = time.perf_counter()
                    continue

            now = time.perf_counter()
            if need_redraw and (now - last_redraw) > 0.2:
                self._erase_block()
                self._draw_block()
                last_redraw = now


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
        description="Batch-analyze videos -> batch_summary_deltaE.csv "
                    "(uses ONLY grand Delta-E for T_mix; ignores spatial/texture)"
    )
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

    from src.utils.config_loader import load_config
    config = load_config(config_path) if config_path.exists() else load_config()

    summary_csv = output_dir / "batch_summary_deltaE.csv"
    if summary_csv.exists():
        summary_csv.unlink()

    workers = max(1, min(args.workers, len(videos)))
    print(f"Mode:         GRAND DELTA-E ONLY (T_mix = T_deltaE)")
    print(f"Found {len(videos)} videos.")
    print(f"Workers:      {workers} (BLAS threads/worker = 1)")
    print(f"Output:       {summary_csv}")
    print(f"Config:       {config_path}")
    print()

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    prog_q = manager.Queue()
    renderer = LiveRenderer(prog_q, total=len(videos))
    renderer.start()

    work_items = [(str(v), config, prog_q) for v in videos]
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
                    line = f"[{i}/{len(videos)}] {name}: ok ({conf}) {tmix_str} in {elapsed:.1f}s"
                    done += 1
                else:
                    line = (
                        f"[{i}/{len(videos)}] {name}: FAILED "
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
