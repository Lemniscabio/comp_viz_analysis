"""Cloud Run Job entrypoint: analyze ONE video selected by CLOUD_RUN_TASK_INDEX.

Pipeline (main-based): download -> ffmpeg 480p -> main AnalysisEngine (all 6 metrics)
-> main DataExporter CSV -> level_times() -> results JSON. Updates Firestore.

Env vars (set at execution time): BUCKET, JOB_ID, CLOUD_RUN_TASK_INDEX (injected).
"""
from __future__ import annotations

import os
# Let numpy/OpenCV/scikit-image use ALL CPU cores the task was allocated.
# KC_THREADS is set at deploy time to match --cpu; falls back to detected cores.
# MUST run before numpy/cv2/skimage are imported.
_threads = os.environ.get("KC_THREADS") or str(os.cpu_count() or 4)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _threads)

import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.worker.levels import normalized_delta_e, level_times

# Channels rendered in the UI. grand_delta_e drives the main view; the rest fill the hidden panel.
SERIES_KEYS = ["grand_delta_e", "contact_perimeter", "contrast", "homogeneity",
               "energy", "variance_delta_e"]


def select_video(manifest: Dict[str, Any], task_index: int) -> Dict[str, Any]:
    for entry in manifest["videos"]:
        if entry["idx"] == task_index:
            return entry
    raise IndexError(f"no video with idx={task_index} in manifest")


def ffmpeg_480p_cmd(src: str, dst: str) -> List[str]:
    # scale to 480p height, width auto to keep aspect & stay even; no crop (vessel-agnostic).
    return ["ffmpeg", "-y", "-i", src, "-vf", "scale=-2:480", "-c:a", "copy", dst]


def _downsample(seq: Sequence[float], max_points: int) -> List[float]:
    n = len(seq)
    if n <= max_points:
        return list(seq)
    step = n / max_points
    return [seq[int(i * step)] for i in range(max_points)]


def results_doc(results: List[Dict[str, Any]], duration_s: float, fps: float,
                max_points: int = 500) -> Dict[str, Any]:
    timestamps = [r["timestamp"] for r in results]
    grand = [r.get("grand_delta_e", 0.0) for r in results]
    norm = normalized_delta_e(grand)
    levels = level_times(timestamps, grand, levels=(0.90, 0.95, 0.99))
    series: Dict[str, List[float]] = {"timestamp": _downsample(timestamps, max_points),
                                      "normalized_delta_e": _downsample(norm, max_points)}
    for k in SERIES_KEYS:
        series[k] = _downsample([r.get(k, 0.0) for r in results], max_points)
    return {
        "duration_s": duration_s, "fps": fps, "frame_count": len(results),
        "levels": {f"{L:.2f}": t for L, t in levels.items()},
        "series": series,
    }


def main() -> None:
    from google.cloud import storage, firestore
    from src.core.analysis_engine import AnalysisEngine
    from src.core.video_reader import VideoReader
    from src.core.export import DataExporter
    from src.utils.config_loader import load_config

    bucket_name = os.environ["BUCKET"]
    job_id = os.environ["JOB_ID"]
    task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))

    gcs = storage.Client()
    bucket = gcs.bucket(bucket_name)
    fs = firestore.Client()
    job_ref = fs.collection("jobs").document(job_id)

    manifest = json.loads(bucket.blob(f"jobs/{job_id}/manifest.json").download_as_text())
    entry = select_video(manifest, task_index)
    idx, filename, object_path = entry["idx"], entry["filename"], entry["object_path"]
    stem = Path(filename).stem

    _set_video(job_ref, idx, {"status": "running", "error": None})
    config = load_config()  # repo default_config.yaml — all six metrics, unchanged from main

    try:
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / filename
            small = Path(td) / f"480p_{stem}.mp4"
            bucket.blob(object_path).download_to_filename(str(raw))
            subprocess.run(ffmpeg_480p_cmd(str(raw), str(small)), check=True,
                           capture_output=True)

            reader = VideoReader(path=str(small), frame_skip=config["frame_skip"],
                                 fps_override=config.get("video_fps_override"))
            fps = reader.fps if reader.fps > 0 else 1.0
            duration = reader.frame_count / fps
            engine = AnalysisEngine(config)
            try:
                for frame_number, frame in reader:
                    engine.process_frame(frame, frame_number, reader.timestamp(frame_number))
            finally:
                reader.release()
            if not engine.results:
                raise RuntimeError("no frames produced")

            csv_local = Path(td) / "results.csv"
            DataExporter().export(engine.results, csv_local, fmt="csv")
            bucket.blob(f"jobs/{job_id}/results/{idx}__{stem}.csv").upload_from_filename(str(csv_local))

            doc = results_doc(engine.results, duration_s=duration, fps=fps)
            bucket.blob(f"jobs/{job_id}/results/{idx}.json").upload_from_string(
                json.dumps(doc), content_type="application/json")

        lv = doc["levels"]
        _set_video(job_ref, idx, {
            "status": "done", "error": None, "duration_s": duration,
            "t_mix_90_s": lv.get("0.90"), "t_mix_95_s": lv.get("0.95"),
            "t_mix_99_s": lv.get("0.99"),
        })
    except subprocess.CalledProcessError as e:
        traceback.print_exc()
        _set_video(job_ref, idx, {"status": "failed",
                                  "error": f"ffmpeg failed: {e.stderr.decode()[:300]}"})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _set_video(job_ref, idx, {"status": "failed", "error": f"{type(e).__name__}: {e}"})

    _maybe_finalize_job(job_ref)


def _set_video(job_ref, idx: int, patch: Dict[str, Any]) -> None:
    from google.cloud import firestore

    @firestore.transactional
    def _txn(txn):
        data = job_ref.get(transaction=txn).to_dict() or {}
        videos = data.get("videos", [])
        for v in videos:
            if v.get("idx") == idx:
                v.update(patch)
        txn.update(job_ref, {"videos": videos})

    _txn(job_ref._client.transaction())


def _maybe_finalize_job(job_ref) -> None:
    from google.cloud import firestore

    @firestore.transactional
    def _txn(txn):
        data = job_ref.get(transaction=txn).to_dict() or {}
        videos = data.get("videos", [])
        if any(v.get("status") in ("pending", "running") for v in videos):
            return
        any_fail = any(v.get("status") == "failed" for v in videos)
        txn.update(job_ref, {"status": "failed" if any_fail else "done"})

    _txn(job_ref._client.transaction())


if __name__ == "__main__":
    main()
