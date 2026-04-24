#!/usr/bin/env python3
"""Batch-analyze a folder of videos through Kineticolor.

For each video <stem>.{mp4,mov,avi,mkv,m4v} in <video_dir>, writes:
    <output_dir>/<stem>.csv   — per-frame metrics time series
    <output_dir>/<stem>.png   — summary plot (grand ΔE + normalized ΔE)
    <output_dir>/<stem>.log   — per-video log

Usage:
    python scripts/batch_analyze.py <video_dir> <output_dir> [--config PATH]

Design choices (all four concerns from the brief):

  1. Sequential, CPU-saturating.
     One video at a time. numpy / skimage / OpenCV already parallelize
     across cores via BLAS/OpenMP, so a single Python process pins the
     machine. Running multiple in parallel would thrash without gain.

  2. CSV + PNG with matching stem names.
     Both land in <output_dir> named after the input video.

  3. Mixing-time values in CSV — currently STUBBED.
     A commented block is prepended to each CSV as placeholder:
         # mixing_time_t90 = TBD
         # mixing_time_t95 = TBD
         # mixing_time_t99 = TBD
     Once we pick a quantification rule (grand ΔE plateau vs max-cell
     plateau vs max of both), fill these in via _compute_mixing_times().

  4. One-at-a-time persistence / resumable.
     CSV + PNG flushed to disk before moving to the next video. If a
     video fails, its partial CSV is deleted so the retry is clean. On
     a re-run, videos with an existing CSV are skipped.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Project imports — script must be run from the project root (or anywhere,
# since we fix sys.path below).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.analysis_engine import AnalysisEngine  # noqa: E402
from src.core.export import DataExporter  # noqa: E402
from src.core.video_reader import VideoReader  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


# ---------------------------------------------------------------------------
# Mixing-time stub
# ---------------------------------------------------------------------------
def _compute_mixing_times(results: list) -> dict:
    """Return t_90 / t_95 / t_99 once a quantification rule is chosen.

    Options on the table:
      (a) first t where normalized grand ΔE >= threshold
      (b) first t where max(cell ΔE) / max(cell ΔE plateau) >= threshold
      (c) max(t_threshold_grand, t_threshold_maxcell)

    Until we pick one, return placeholders so the CSV header block is
    structurally present.
    """
    # --- uncomment once a rule is chosen ---
    # import numpy as np
    # t   = np.array([r["timestamp"] for r in results])
    # g   = np.array([r["grand_delta_e"] for r in results])
    # gn  = g / (g.max() if g.max() > 0 else 1.0)
    # def first_cross(thresh: float) -> float | None:
    #     idx = int(np.argmax(gn >= thresh))
    #     return float(t[idx]) if gn[idx] >= thresh else None
    # return {"t_90": first_cross(0.90),
    #         "t_95": first_cross(0.95),
    #         "t_99": first_cross(0.99)}
    return {"t_90": None, "t_95": None, "t_99": None}


def _prepend_mixing_header(csv_path: Path, mix: dict) -> None:
    """Prepend a commented block with mixing-time placeholders to the CSV."""
    fmt = lambda v: "TBD" if v is None else f"{v:.3f}"  # noqa: E731
    header = (
        f"# mixing_time_t90 = {fmt(mix.get('t_90'))}\n"
        f"# mixing_time_t95 = {fmt(mix.get('t_95'))}\n"
        f"# mixing_time_t99 = {fmt(mix.get('t_99'))}\n"
        f"# quantification_rule = TBD  "
        f"(grand_delta_e / max_cell_delta_e / combined)\n"
    )
    existing = csv_path.read_text()
    csv_path.write_text(header + existing)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _write_plot(results: list, png_path: Path, title: str) -> None:
    t = np.array([r["timestamp"] for r in results], dtype=float)
    grand = np.array([r.get("grand_delta_e", np.nan) for r in results], dtype=float)
    g_max = float(np.nanmax(grand)) if np.isfinite(grand).any() and np.nanmax(grand) > 0 else 1.0
    norm = grand / g_max

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t, grand, color="#1f77b4", linewidth=1.2)
    ax1.set_ylabel("Grand ΔE")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    ax2.plot(t, norm, color="#d62728", linewidth=1.2)
    for thr in (0.90, 0.95, 0.99):
        ax2.axhline(thr, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax2.set_ylabel("Normalized ΔE")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # --- once mixing-time rule chosen, draw vertical markers here ---

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single-video runner
# ---------------------------------------------------------------------------
def analyze_one(video_path: Path, csv_path: Path, png_path: Path, config: dict) -> Tuple[int, float]:
    """Run engine on one video, write CSV + PNG. Returns (n_frames, seconds)."""
    reader = VideoReader(
        path=str(video_path),
        frame_skip=config["frame_skip"],
        fps_override=config.get("video_fps_override"),
    )
    engine = AnalysisEngine(config)

    t0 = time.perf_counter()
    for frame_number, frame in reader:
        timestamp = reader.timestamp(frame_number)
        engine.process_frame(frame, frame_number, timestamp)
    elapsed = time.perf_counter() - t0
    reader.release()

    results = engine.results
    if not results:
        raise RuntimeError("no frames produced")

    # 1. write the normal CSV (per-frame metrics)
    fmt = "xlsx" if csv_path.suffix == ".xlsx" else "csv"
    DataExporter().export(results, csv_path, fmt=fmt)

    # 2. prepend mixing-time placeholder header (concern #3)
    if fmt == "csv":
        _prepend_mixing_header(csv_path, _compute_mixing_times(results))

    # 3. summary plot
    _write_plot(results, png_path, title=video_path.stem)

    return len(results), elapsed


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------
def find_videos(video_dir: Path) -> List[Path]:
    return sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-analyze a folder of videos")
    ap.add_argument("video_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "default_config.yaml")
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

    print(f"Found {len(videos)} videos.")
    print(f"Output dir:   {output_dir}")
    print(f"Config:       {config_path}")
    print()

    t_start = time.perf_counter()
    done = skipped = failed = 0

    for i, src in enumerate(videos, 1):
        stem = src.stem
        csv_path = output_dir / f"{stem}.csv"
        png_path = output_dir / f"{stem}.png"
        log_path = output_dir / f"{stem}.log"

        print(f"[{i}/{len(videos)}] {src.name}")

        # Concern #4: resume — skip if CSV already exists.
        if csv_path.exists():
            print(f"  (skip: {csv_path.name} already exists)")
            skipped += 1
            continue

        # Per-video logger that writes to its own .log file.
        fh = logging.FileHandler(log_path, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root = logging.getLogger("kineticolor")
        root.setLevel(logging.DEBUG)
        root.addHandler(fh)

        try:
            n, dt = analyze_one(src, csv_path, png_path, config)
            print(f"  done: {n} frames in {dt:.1f}s  ({n/dt:.1f} fps)")
            print(f"  -> {csv_path.name}, {png_path.name}")
            done += 1
        except Exception as e:
            print(f"  FAILED: {e}  (see {log_path.name})")
            with open(log_path, "a") as f:
                traceback.print_exc(file=f)
            # Remove any partial outputs so the next run retries cleanly.
            csv_path.unlink(missing_ok=True)
            png_path.unlink(missing_ok=True)
            failed += 1
        finally:
            root.removeHandler(fh)
            fh.close()

    total = time.perf_counter() - t_start
    print()
    print(f"Summary: processed={done}, skipped={skipped}, failed={failed}, total={total:.0f}s")
    print(f"Output in: {output_dir}")


if __name__ == "__main__":
    main()
