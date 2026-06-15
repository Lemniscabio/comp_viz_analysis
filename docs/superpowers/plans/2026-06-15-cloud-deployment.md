# Kineticolor Cloud Deployment Implementation Plan (main-based)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy Kineticolor as a web app where a signed-in `@lemnisca.bio` user uploads videos, the analysis runs server-side on Google Cloud (independent of the browser), and each video's **ΔE graph + 0.90/0.95/0.99 mixing times** appear automatically — with the other five metric graphs available behind a hidden/expandable panel, and short paper-grounded hover tooltips explaining each.

**Architecture:** A stateless **Cloud Run service** (FastAPI + bundled React SPA) handles Google OAuth (domain-gated to `lemnisca.bio`), mints keyless V4 signed URLs so the browser uploads videos straight to a **GCS bucket**, and triggers a **Cloud Run Job** (`kineticolor-worker`) with one task per video. Each worker task downscales the video to 480p (ffmpeg), runs **main's `AnalysisEngine`** (all six metrics), computes the mixing times with the exact level-crossing logic lifted from main, writes a results JSON + CSV to GCS, and updates a **Firestore** job record. The SPA polls the backend for status and renders the results. Security/platform patterns (OAuth domain gate, signed-URL uploads, Workload Identity Federation CI/CD, least-privilege service accounts) are reused from `NEW_GCP_OF_SCRIPTS/phase3-run-app`; the heavyweight Cloud Batch runtime is replaced by right-sized Cloud Run Jobs.

**CRITICAL — source of truth:** This plan builds **only from the `main` branch**. It takes **zero code from `feat/mixing-time`** (`mixing_time.py`, `batch.py`, `visual_time.py`, `batch_analyze_*` are off-limits). The single mixing-time computation used here is lifted verbatim from main's `src/gui/plots_panel.py:_update_mixing_marker`:
```python
max_val = float(np.max(raw)) if np.max(raw) > 0 else 1.0
norm = raw / max_val                 # normalize grand ΔE by its own max
idx = np.argmax(norm >= 0.95)        # FIRST frame index where normalized ΔE ≥ level
t_mix = float(t[idx])                # ...its timestamp = mixing time (guard if never reached)
```
Generalized to levels {0.90, 0.95, 0.99}. `grand_delta_e` is produced per frame by main's `AnalysisEngine`; `normalized_delta_e = grand_delta_e/max` is what main's `DataExporter` already adds.

**Tech Stack:** Python 3.12, FastAPI, `google-cloud-storage`, `google-cloud-run` (run_v2), `google-cloud-firestore`, `google-auth`, ffmpeg; React + Vite + TypeScript + `uplot` (fast canvas charts); Docker (multi-stage); GitHub Actions + Workload Identity Federation; GCP Cloud Run, Cloud Run Jobs, GCS, Firestore, Artifact Registry.

**Conventions (rename freely, stay consistent):**
- GCP project id: `kineticolor-cloud` · Region: `us-central1` · Bucket: `kineticolor-videos`
- Artifact Registry repo: `kineticolor` · Cloud Run service: `kineticolor-app` · Cloud Run Job: `kineticolor-worker`
- Service accounts: `kc-backend@` `kc-worker@` `kc-ci-deployer@` · Allowed domain: `lemnisca.bio`
- All new code lives under a new top-level `web/` directory, on a new branch off `main`.

**GCS object layout (single bucket):**
```
jobs/<job_id>/manifest.json                # video object paths, written at allocate
jobs/<job_id>/inputs/<idx>__<filename>     # uploaded videos (browser → signed PUT)
jobs/<job_id>/results/<idx>.json           # per-video: all metric series + level times + duration
jobs/<job_id>/results/<idx>__<stem>.csv    # per-video time-series CSV (main's DataExporter)
```

**Firestore data model (collection `jobs`, doc id = `<job_id>`):** lightweight only — series live in GCS JSON (Firestore docs cap at 1 MiB).
```
{ job_id, owner_email, created_at, status, video_count,
  videos: [ { idx, filename, object_path, status, duration_s,
              t_mix_90_s, t_mix_95_s, t_mix_99_s, error } ] }
```
`status` ∈ `allocated | submitted | running | done | failed`. Per-video `status` ∈ `pending | running | done | failed`.

---

## File Structure

```
web/
  worker/
    levels.py              # PURE: normalized ΔE + level-crossing times (lifted from main)
    worker.py              # Cloud Run Job entrypoint: ffmpeg 480p -> main engine -> results
    Dockerfile             # ffmpeg + main src/ engine + worker
    requirements.txt
  backend/
    main.py                # FastAPI app, serves SPA static + /api routes
    config.py              # env-var settings
    auth.py                # Google ID-token verification + hd==lemnisca.bio gate
    gcs.py                 # signed URLs + safe filename (keyless)
    firestore_store.py     # job record CRUD
    runner.py              # trigger Cloud Run Job execution with overrides
    routes_jobs.py         # allocate / submit / status / result endpoints
    schemas.py
    Dockerfile             # multi-stage: build SPA, then python serving
    requirements.txt
  frontend/
    package.json
    vite.config.ts
    index.html
    src/
      main.tsx
      App.tsx
      lib/auth.ts          # Google Sign-In + token storage
      lib/api.ts           # typed API client
      lib/upload.ts        # bounded-concurrency upload pool
      lib/tooltips.ts      # paper-grounded metric explanations
      components/DeltaEChart.tsx   # uplot ΔE graph + 0.90/0.95/0.99 markers
      components/MetricChart.tsx   # generic uplot line chart (hidden panel)
      components/InfoHover.tsx     # hover tooltip
      views/UploadView.tsx
      views/ResultView.tsx
  infra/
    setup.sh               # one-time: APIs, bucket, SAs, IAM, AR, Firestore, WIF
    cors.json
    deploy.sh              # manual fallback deploy (also creates the worker Job)
  .github/workflows/deploy.yml
  tests/
    test_levels.py
    test_worker.py
    test_auth.py
    test_gcs.py
    test_firestore_store.py
    test_runner.py
    test_routes_jobs.py
```

---

## Phase 0 — Branch off main, scaffold, deps

### Task 0: Create the web branch off main and the directory layout

**Files:**
- Create: `web/__init__.py`, `web/worker/__init__.py`, `web/backend/__init__.py`, `web/tests/__init__.py`
- Create: `web/worker/requirements.txt`, `web/backend/requirements.txt`

- [ ] **Step 1: Branch from main (NOT from feat/mixing-time), preserving the plan doc**

```bash
cd "/Users/kartikey/Desktop/work_products/comp_viz_code:files/comp_viz_analysis"
# stash the untracked plan doc so it survives the checkout
git stash push -u -- docs/superpowers/plans/2026-06-15-cloud-deployment.md 2>/dev/null || true
git checkout main
git checkout -b feat/web-deploy
git stash pop 2>/dev/null || true
git log --oneline -1   # confirm we are at main's HEAD (fe16c94 or later), NOT a feat/mixing-time commit
```
Expected: HEAD is main's latest commit. `src/core/mixing_time.py` must NOT exist:
```bash
test ! -f src/core/mixing_time.py && echo "GOOD: no mixing_time.py (on main base)" || echo "STOP: feat-branch code present"
```

- [ ] **Step 2: Create package markers**

```bash
mkdir -p web/worker web/backend web/tests web/frontend/src web/infra web/.github/workflows
touch web/__init__.py web/worker/__init__.py web/backend/__init__.py web/tests/__init__.py
```

- [ ] **Step 3: Worker requirements (main engine deps, headless, + ffmpeg installed via Docker apt)**

`web/worker/requirements.txt`:
```
numpy>=1.24
opencv-python-headless>=4.8
scikit-image>=0.21
scipy>=1.11
pyyaml>=6.0
openpyxl>=3.1
google-cloud-storage>=2.16
google-cloud-firestore>=2.16
```

- [ ] **Step 4: Backend requirements**

`web/backend/requirements.txt`:
```
fastapi>=0.110
uvicorn[standard]>=0.29
google-auth>=2.29
google-cloud-storage>=2.16
google-cloud-firestore>=2.16
google-cloud-run>=0.10
pydantic>=2.6
pytest>=7.4
httpx>=0.27
```

- [ ] **Step 5: Commit**

```bash
git add web/__init__.py web/worker web/backend web/tests docs/superpowers/plans/2026-06-15-cloud-deployment.md
git commit -m "chore: scaffold web/ deployment on main base"
```

---

## Phase 1 — Worker (Cloud Run Job)

### Task 1: Level-crossing logic (lifted verbatim from main)

**Files:**
- Create: `web/worker/levels.py`
- Test: `web/tests/test_levels.py`

- [ ] **Step 1: Write the failing test**

`web/tests/test_levels.py`:
```python
import math
from web.worker.levels import normalized_delta_e, level_times


def test_normalized_divides_by_max():
    assert normalized_delta_e([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_normalized_all_zero_safe():
    assert normalized_delta_e([0.0, 0.0]) == [0.0, 0.0]


def test_level_times_first_crossing_matches_main_logic():
    # grand ΔE rising 0->10; normalized = [0,.2,.5,.9,.95,1.0]
    grand = [0.0, 2.0, 5.0, 9.0, 9.5, 10.0]
    t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    out = level_times(t, grand, levels=(0.90, 0.95, 0.99))
    assert out[0.90] == 3.0   # first idx where norm>=0.90
    assert out[0.95] == 4.0
    assert out[0.99] == 5.0


def test_level_times_not_reached_is_none():
    grand = [0.0, 1.0, 2.0]   # normalized maxes at 1.0 only at last; 0.99 reached, but test a never-case
    t = [0.0, 1.0, 2.0]
    out = level_times(t, grand, levels=(0.99,))
    # norm = [0,0.5,1.0] -> 0.99 first reached at idx 2
    assert out[0.99] == 2.0
    # a level above the curve's max-normalized (impossible >1) -> None
    out2 = level_times(t, grand, levels=(1.01,))
    assert out2[1.01] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/kartikey/Desktop/work_products/comp_viz_code:files/comp_viz_analysis" && python -m pytest web/tests/test_levels.py -v`
Expected: FAIL (`No module named 'web.worker.levels'`)

- [ ] **Step 3: Write the implementation (mirrors main's `np.argmax(norm >= L)` + guard)**

`web/worker/levels.py`:
```python
"""Mixing-time level crossings — lifted verbatim from main's
src/gui/plots_panel.py:_update_mixing_marker. NOTHING from feat/mixing-time.

mixing time at level L = first timestamp where (grand ΔE / max grand ΔE) >= L.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence


def normalized_delta_e(grand: Sequence[float]) -> List[float]:
    m = max(grand) if grand and max(grand) > 0 else 1.0
    return [g / m for g in grand]


def level_times(
    timestamps: Sequence[float],
    grand_delta_e: Sequence[float],
    levels: Sequence[float] = (0.90, 0.95, 0.99),
) -> Dict[float, Optional[float]]:
    norm = normalized_delta_e(grand_delta_e)
    out: Dict[float, Optional[float]] = {}
    for L in levels:
        idx = next((i for i, v in enumerate(norm) if v >= L), None)  # == np.argmax + guard
        out[L] = float(timestamps[idx]) if idx is not None else None
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest web/tests/test_levels.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add web/worker/levels.py web/tests/test_levels.py
git commit -m "feat: mixing-time level crossings lifted from main's plots_panel logic"
```

### Task 2: Worker entrypoint — ffmpeg 480p → main engine → results

**Files:**
- Create: `web/worker/worker.py`
- Test: `web/tests/test_worker.py`

- [ ] **Step 1: Write the failing test** (pure helpers: manifest selection, ffmpeg command, results-doc assembly)

`web/tests/test_worker.py`:
```python
import web.worker.worker as w


def test_select_video_by_index():
    manifest = {"videos": [
        {"idx": 0, "filename": "a.mp4", "object_path": "jobs/J/inputs/0__a.mp4"},
        {"idx": 1, "filename": "b.mp4", "object_path": "jobs/J/inputs/1__b.mp4"},
    ]}
    assert w.select_video(manifest, 1)["filename"] == "b.mp4"


def test_ffmpeg_480p_cmd_scales_height_keeps_aspect():
    cmd = w.ffmpeg_480p_cmd("in.mp4", "out.mp4")
    assert "ffmpeg" in cmd[0]
    joined = " ".join(cmd)
    assert "scale=-2:480" in joined          # 480p height, width auto-even, aspect preserved
    assert "in.mp4" in joined and "out.mp4" in joined


def test_results_doc_shape_includes_series_and_levels():
    results = [
        {"frame_number": 0, "timestamp": 0.0, "grand_delta_e": 0.0, "contact_perimeter": 1,
         "contrast": 0.5, "homogeneity": 0.5, "energy": 0.5, "variance_delta_e": 1.0},
        {"frame_number": 1, "timestamp": 1.0, "grand_delta_e": 10.0, "contact_perimeter": 0,
         "contrast": 0.0, "homogeneity": 1.0, "energy": 1.0, "variance_delta_e": 0.0},
    ]
    doc = w.results_doc(results, duration_s=1.0, fps=1.0, max_points=500)
    assert doc["levels"]["0.95"] == 1.0
    assert doc["series"]["grand_delta_e"] == [0.0, 10.0]
    assert doc["series"]["normalized_delta_e"] == [0.0, 1.0]
    # all six metric channels present for the hidden panel
    for key in ("contact_perimeter", "contrast", "homogeneity", "energy", "variance_delta_e"):
        assert key in doc["series"]
    assert doc["duration_s"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest web/tests/test_worker.py -v`
Expected: FAIL (`No module named 'web.worker.worker'`)

- [ ] **Step 3: Write the worker**

`web/worker/worker.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest web/tests/test_worker.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Verify main's engine emits the metric keys this worker reads**

Run: `python -c "import yaml; from src.utils.config_loader import load_config; from src.core.analysis_engine import AnalysisEngine; e=AnalysisEngine(load_config()); print('engine ok')"`
Then confirm the per-frame result keys match `SERIES_KEYS`. Inspect `src/core/analysis_engine.py` for the dict keys it stores per frame (look for where `grand_delta_e`, `contrast`, `homogeneity`, `energy`, `contact_perimeter`, `variance_delta_e` are written). If a key name differs on main, update `SERIES_KEYS` and the `results_doc` test to match the real names. (This is the one place the worker depends on main's engine internals — verify, don't assume.)

- [ ] **Step 6: Commit**

```bash
git add web/worker/worker.py web/tests/test_worker.py
git commit -m "feat: worker — ffmpeg 480p + main engine + results JSON/CSV + Firestore"
```

### Task 3: Worker Dockerfile (with ffmpeg)

**Files:**
- Create: `web/worker/Dockerfile`

- [ ] **Step 1: Write the Dockerfile**

`web/worker/Dockerfile`:
```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY web/worker/requirements.txt ./req.txt
RUN pip install --no-cache-dir -r req.txt

# main engine + config + worker package
COPY src/ ./src/
COPY config/ ./config/
COPY web/__init__.py ./web/__init__.py
COPY web/worker/ ./web/worker/

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "web.worker.worker"]
```

- [ ] **Step 2: Build locally**

Run: `cd "/Users/kartikey/Desktop/work_products/comp_viz_code:files/comp_viz_analysis" && docker build -f web/worker/Dockerfile -t kineticolor-worker:local .`
Expected: build succeeds.

- [ ] **Step 3: Verify ffmpeg + import inside the image**

Run: `docker run --rm kineticolor-worker:local sh -c "ffmpeg -version | head -1 && python -c 'import web.worker.worker; print(\"import ok\")'"`
Expected: prints an ffmpeg version line and `import ok`.

- [ ] **Step 4: Commit**

```bash
git add web/worker/Dockerfile
git commit -m "feat: worker container (ffmpeg + main engine)"
```

---

## Phase 2 — Backend platform modules

### Task 4: Settings

**Files:**
- Create: `web/backend/config.py`
- Test: `web/tests/test_config_backend.py`

- [ ] **Step 1: Write the failing test**

`web/tests/test_config_backend.py`:
```python
import web.backend.config as c


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("KC_PROJECT", "proj-x")
    monkeypatch.setenv("KC_BUCKET", "bucket-x")
    monkeypatch.setenv("KC_OAUTH_CLIENT_ID", "cid")
    monkeypatch.setenv("KC_ALLOWED_DOMAIN", "lemnisca.bio")
    s = c.Settings.from_env()
    assert s.project == "proj-x" and s.bucket == "bucket-x"
    assert s.allowed_domain == "lemnisca.bio"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest web/tests/test_config_backend.py -v`
Expected: FAIL (`No module named 'web.backend.config'`)

- [ ] **Step 3: Write the implementation**

`web/backend/config.py`:
```python
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    project: str
    region: str
    bucket: str
    oauth_client_id: str
    allowed_domain: str
    worker_job: str
    backend_sa: str
    dev_no_auth: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            project=os.environ.get("KC_PROJECT", ""),
            region=os.environ.get("KC_REGION", "us-central1"),
            bucket=os.environ.get("KC_BUCKET", ""),
            oauth_client_id=os.environ.get("KC_OAUTH_CLIENT_ID", ""),
            allowed_domain=os.environ.get("KC_ALLOWED_DOMAIN", "lemnisca.bio"),
            worker_job=os.environ.get("KC_WORKER_JOB", "kineticolor-worker"),
            backend_sa=os.environ.get("KC_BACKEND_SA", ""),
            dev_no_auth=os.environ.get("KC_DEV_NO_AUTH", "") == "1",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest web/tests/test_config_backend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add web/backend/config.py web/tests/test_config_backend.py
git commit -m "feat: backend settings from env"
```

### Task 5: Auth — Google ID token + domain gate

**Files:**
- Create: `web/backend/auth.py`
- Test: `web/tests/test_auth.py`

- [ ] **Step 1: Write the failing test**

`web/tests/test_auth.py`:
```python
import pytest
import web.backend.auth as a


def test_accepts_domain_account():
    u = a.user_from_idinfo({"email": "x@lemnisca.bio", "email_verified": True,
                            "hd": "lemnisca.bio", "sub": "1"}, "lemnisca.bio")
    assert u.email == "x@lemnisca.bio"


def test_rejects_unverified():
    with pytest.raises(PermissionError):
        a.user_from_idinfo({"email": "x@lemnisca.bio", "email_verified": False,
                            "hd": "lemnisca.bio"}, "lemnisca.bio")


def test_rejects_wrong_domain():
    with pytest.raises(PermissionError):
        a.user_from_idinfo({"email": "x@gmail.com", "email_verified": True, "hd": None},
                           "lemnisca.bio")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest web/tests/test_auth.py -v`
Expected: FAIL (`No module named 'web.backend.auth'`)

- [ ] **Step 3: Write the implementation**

`web/backend/auth.py`:
```python
from __future__ import annotations
from dataclasses import dataclass

from fastapi import Header, HTTPException
from google.auth.transport import requests as g_requests
from google.oauth2 import id_token

from web.backend.config import Settings


@dataclass(frozen=True)
class User:
    email: str
    sub: str


def user_from_idinfo(idinfo: dict, allowed_domain: str) -> User:
    if not idinfo.get("email_verified"):
        raise PermissionError("email not verified")
    if idinfo.get("hd") != allowed_domain:
        raise PermissionError(f"not a {allowed_domain} Workspace account (hd={idinfo.get('hd')!r})")
    return User(email=idinfo.get("email", ""), sub=idinfo.get("sub", ""))


def make_auth_dependency(settings: Settings):
    def current_user(authorization: str = Header(default="")) -> User:
        if settings.dev_no_auth:
            return User(email="dev@lemnisca.bio", sub="dev")
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        token = authorization.split(" ", 1)[1]
        try:
            idinfo = id_token.verify_oauth2_token(token, g_requests.Request(),
                                                  settings.oauth_client_id)
            return user_from_idinfo(idinfo, settings.allowed_domain)
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(401, f"invalid token: {e}")

    return current_user
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest web/tests/test_auth.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add web/backend/auth.py web/tests/test_auth.py
git commit -m "feat: backend auth with lemnisca.bio domain gate"
```

### Task 6: GCS helper — keyless signed URLs + safe filenames

**Files:**
- Create: `web/backend/gcs.py`
- Test: `web/tests/test_gcs.py`

- [ ] **Step 1: Write the failing test**

`web/tests/test_gcs.py`:
```python
import pytest
import web.backend.gcs as g


def test_safe_filename_rejects_traversal():
    with pytest.raises(ValueError):
        g.safe_filename("../etc/passwd")
    with pytest.raises(ValueError):
        g.safe_filename("/abs/x.mp4")


def test_safe_filename_rejects_non_video():
    with pytest.raises(ValueError):
        g.safe_filename("notes.txt")


def test_safe_filename_keeps_basename():
    assert g.safe_filename("clip 01.mp4") == "clip 01.mp4"


def test_input_object_path():
    assert g.input_object_path("J1", 2, "a.mp4") == "jobs/J1/inputs/2__a.mp4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest web/tests/test_gcs.py -v`
Expected: FAIL (`No module named 'web.backend.gcs'`)

- [ ] **Step 3: Write the implementation**

`web/backend/gcs.py`:
```python
from __future__ import annotations
import datetime as dt
from pathlib import PurePosixPath

import google.auth
import google.auth.transport.requests
from google.cloud import storage

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def safe_filename(name: str) -> str:
    p = PurePosixPath(name)
    if name != p.name or name in (".", "..") or name.startswith("/"):
        raise ValueError(f"unsafe filename: {name!r}")
    if p.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"unsupported video type: {name!r}")
    return name


def input_object_path(job_id: str, idx: int, filename: str) -> str:
    return f"jobs/{job_id}/inputs/{idx}__{filename}"


class GcsService:
    def __init__(self, bucket_name: str, signer_email: str):
        self._signer_email = signer_email
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)

    def _token(self) -> str:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(google.auth.transport.requests.Request())
        return creds.token

    def _signed(self, object_path: str, method: str) -> str:
        return self._bucket.blob(object_path).generate_signed_url(
            version="v4", expiration=dt.timedelta(minutes=30), method=method,
            service_account_email=self._signer_email, access_token=self._token())

    def signed_put_url(self, object_path: str) -> str:
        return self._signed(object_path, "PUT")

    def signed_get_url(self, object_path: str) -> str:
        return self._signed(object_path, "GET")

    def upload_json(self, object_path: str, data: bytes) -> None:
        self._bucket.blob(object_path).upload_from_string(data, content_type="application/json")

    def exists(self, object_path: str) -> bool:
        return self._bucket.blob(object_path).exists()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest web/tests/test_gcs.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add web/backend/gcs.py web/tests/test_gcs.py
git commit -m "feat: keyless signed URLs + safe filename validation"
```

### Task 7: Firestore store

**Files:**
- Create: `web/backend/firestore_store.py`
- Test: `web/tests/test_firestore_store.py`

- [ ] **Step 1: Write the failing test**

`web/tests/test_firestore_store.py`:
```python
import web.backend.firestore_store as fsx


def test_new_job_record_shape():
    rec = fsx.new_job_record("J1", "x@lemnisca.bio", ["a.mp4", "b.mp4"], "2026-06-15T00:00:00Z")
    assert rec["status"] == "allocated" and rec["video_count"] == 2
    assert rec["videos"][0] == {
        "idx": 0, "filename": "a.mp4", "object_path": "jobs/J1/inputs/0__a.mp4",
        "status": "pending", "duration_s": None,
        "t_mix_90_s": None, "t_mix_95_s": None, "t_mix_99_s": None, "error": None,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest web/tests/test_firestore_store.py -v`
Expected: FAIL (`No module named 'web.backend.firestore_store'`)

- [ ] **Step 3: Write the implementation**

`web/backend/firestore_store.py`:
```python
from __future__ import annotations
from typing import Any, Dict, List

from web.backend.gcs import input_object_path


def new_job_record(job_id: str, owner_email: str, files: List[str], created_at: str) -> Dict[str, Any]:
    videos = [
        {"idx": i, "filename": fn, "object_path": input_object_path(job_id, i, fn),
         "status": "pending", "duration_s": None,
         "t_mix_90_s": None, "t_mix_95_s": None, "t_mix_99_s": None, "error": None}
        for i, fn in enumerate(files)
    ]
    return {"job_id": job_id, "owner_email": owner_email, "created_at": created_at,
            "status": "allocated", "video_count": len(files), "videos": videos}


class FirestoreStore:
    def __init__(self):
        from google.cloud import firestore
        self._col = firestore.Client().collection("jobs")

    def create(self, record: Dict[str, Any]) -> None:
        self._col.document(record["job_id"]).set(record)

    def get(self, job_id: str) -> Dict[str, Any] | None:
        snap = self._col.document(job_id).get()
        return snap.to_dict() if snap.exists else None

    def set_status(self, job_id: str, status: str) -> None:
        self._col.document(job_id).update({"status": status})

    def list_for_owner(self, owner_email: str) -> List[Dict[str, Any]]:
        return [d.to_dict() for d in self._col.where("owner_email", "==", owner_email).stream()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest web/tests/test_firestore_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add web/backend/firestore_store.py web/tests/test_firestore_store.py
git commit -m "feat: Firestore job-record store"
```

### Task 8: Runner — trigger Cloud Run Job execution

**Files:**
- Create: `web/backend/runner.py`
- Test: `web/tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

`web/tests/test_runner.py`:
```python
import pytest
import web.backend.runner as r


def test_build_overrides_sets_task_count_and_env():
    ov = r.build_overrides("J1", "b", 3)
    assert ov["task_count"] == 3
    env = {e["name"]: e["value"] for e in ov["container_overrides"][0]["env"]}
    assert env["JOB_ID"] == "J1" and env["BUCKET"] == "b"


def test_build_overrides_caps_video_count():
    with pytest.raises(ValueError):
        r.build_overrides("J1", "b", r.MAX_TASKS + 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest web/tests/test_runner.py -v`
Expected: FAIL (`No module named 'web.backend.runner'`)

- [ ] **Step 3: Write the implementation**

`web/backend/runner.py`:
```python
from __future__ import annotations
from typing import Any, Dict

MAX_TASKS = 50  # cap fan-out to protect quota/cost


def build_overrides(job_id: str, bucket: str, video_count: int) -> Dict[str, Any]:
    if video_count < 1 or video_count > MAX_TASKS:
        raise ValueError(f"video_count {video_count} out of range 1..{MAX_TASKS}")
    return {"task_count": video_count, "container_overrides": [
        {"env": [{"name": "JOB_ID", "value": job_id}, {"name": "BUCKET", "value": bucket}]}]}


class JobRunner:
    def __init__(self, project: str, region: str, job_name: str):
        from google.cloud import run_v2
        self._client = run_v2.JobsClient()
        self._job_path = f"projects/{project}/locations/{region}/jobs/{job_name}"

    def trigger(self, job_id: str, bucket: str, video_count: int) -> str:
        from google.cloud import run_v2
        ov = build_overrides(job_id, bucket, video_count)
        overrides = run_v2.RunJobRequest.Overrides(
            task_count=ov["task_count"],
            container_overrides=[run_v2.RunJobRequest.Overrides.ContainerOverride(
                env=[run_v2.EnvVar(name=e["name"], value=e["value"])
                     for e in ov["container_overrides"][0]["env"]])])
        op = self._client.run_job(request=run_v2.RunJobRequest(name=self._job_path, overrides=overrides))
        return op.metadata.name if op.metadata else self._job_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest web/tests/test_runner.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add web/backend/runner.py web/tests/test_runner.py
git commit -m "feat: Cloud Run Job runner with task-count cap + env overrides"
```

---

## Phase 3 — Backend API + app assembly

### Task 9: Schemas

**Files:**
- Create: `web/backend/schemas.py`

- [ ] **Step 1: Write the schemas**

`web/backend/schemas.py`:
```python
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class AllocateReq(BaseModel):
    files: List[str]


class UploadTarget(BaseModel):
    idx: int
    filename: str
    object_path: str
    url: str


class AllocateResp(BaseModel):
    job_id: str
    uploads: List[UploadTarget]


class SubmitReq(BaseModel):
    job_id: str


class VideoStatus(BaseModel):
    idx: int
    filename: str
    status: str
    duration_s: Optional[float] = None
    t_mix_90_s: Optional[float] = None
    t_mix_95_s: Optional[float] = None
    t_mix_99_s: Optional[float] = None
    error: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    video_count: int
    videos: List[VideoStatus]
```

- [ ] **Step 2: Verify import**

Run: `python -c "import web.backend.schemas as s; print(s.JobStatus.__name__)"`
Expected: prints `JobStatus`

- [ ] **Step 3: Commit**

```bash
git add web/backend/schemas.py
git commit -m "feat: API schemas"
```

### Task 10: Job routes

**Files:**
- Create: `web/backend/routes_jobs.py`
- Test: `web/tests/test_routes_jobs.py`

- [ ] **Step 1: Write the failing test**

`web/tests/test_routes_jobs.py`:
```python
from fastapi.testclient import TestClient
import web.backend.main as m


class FakeGcs:
    def __init__(self): self.objects = set(); self.json = {}
    def signed_put_url(self, p): return f"https://put/{p}"
    def signed_get_url(self, p): return f"https://get/{p}"
    def upload_json(self, p, d): self.json[p] = d
    def exists(self, p): return p in self.objects


class FakeStore:
    def __init__(self): self.db = {}
    def create(self, rec): self.db[rec["job_id"]] = rec
    def get(self, jid): return self.db.get(jid)
    def set_status(self, jid, s): self.db[jid]["status"] = s
    def list_for_owner(self, e): return [r for r in self.db.values() if r["owner_email"] == e]


class FakeRunner:
    def __init__(self): self.triggered = []
    def trigger(self, jid, bucket, n): self.triggered.append((jid, n)); return "exec-1"


def make_client():
    app = m.create_app(dev_no_auth=True)
    gcs, store, runner = FakeGcs(), FakeStore(), FakeRunner()
    app.dependency_overrides[m.get_gcs] = lambda: gcs
    app.dependency_overrides[m.get_store] = lambda: store
    app.dependency_overrides[m.get_runner] = lambda: runner
    return TestClient(app), gcs, store, runner


def test_allocate_returns_urls_and_record():
    client, gcs, store, _ = make_client()
    body = client.post("/api/jobs:allocate", json={"files": ["a.mp4", "b.mp4"]}).json()
    assert len(body["uploads"]) == 2
    assert body["uploads"][0]["url"].startswith("https://put/")
    assert store.get(body["job_id"])["video_count"] == 2


def test_submit_fails_if_inputs_missing():
    client, gcs, store, runner = make_client()
    jid = client.post("/api/jobs:allocate", json={"files": ["a.mp4"]}).json()["job_id"]
    assert client.post("/api/jobs:submit", json={"job_id": jid}).status_code == 400


def test_submit_triggers_when_inputs_present():
    client, gcs, store, runner = make_client()
    body = client.post("/api/jobs:allocate", json={"files": ["a.mp4"]}).json()
    gcs.objects.add(body["uploads"][0]["object_path"])
    assert client.post("/api/jobs:submit", json={"job_id": body["job_id"]}).status_code == 200
    assert runner.triggered == [(body["job_id"], 1)]
    assert store.get(body["job_id"])["status"] == "submitted"


def test_rejects_unsafe_filename():
    client, *_ = make_client()
    assert client.post("/api/jobs:allocate", json={"files": ["../x.mp4"]}).status_code == 400


def test_result_url_only_when_done():
    client, gcs, store, _ = make_client()
    body = client.post("/api/jobs:allocate", json={"files": ["a.mp4"]}).json()
    jid = body["job_id"]
    # not done yet -> 404
    assert client.get(f"/api/jobs/{jid}/result/0").status_code == 404
    store.db[jid]["videos"][0]["status"] = "done"
    r = client.get(f"/api/jobs/{jid}/result/0").json()
    assert r["url"].startswith("https://get/")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest web/tests/test_routes_jobs.py -v`
Expected: FAIL (`No module named 'web.backend.main'`) — main.py is Task 11; this test drives both.

- [ ] **Step 3: Write the routes**

`web/backend/routes_jobs.py`:
```python
from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from web.backend.auth import User
from web.backend.gcs import safe_filename
from web.backend.firestore_store import new_job_record
from web.backend.schemas import (AllocateReq, AllocateResp, UploadTarget, SubmitReq,
                                 JobStatus, VideoStatus)


def build_router(get_gcs, get_store, get_runner, current_user, settings):
    router = APIRouter(prefix="/api")

    @router.post("/jobs:allocate", response_model=AllocateResp)
    def allocate(req: AllocateReq, user: User = Depends(current_user),
                 gcs=Depends(get_gcs), store=Depends(get_store)):
        if not req.files:
            raise HTTPException(400, "no files")
        try:
            files = [safe_filename(f) for f in req.files]
        except ValueError as e:
            raise HTTPException(400, str(e))
        job_id = uuid.uuid4().hex[:12]
        record = new_job_record(job_id, user.email, files,
                                datetime.now(timezone.utc).isoformat())
        gcs.upload_json(f"jobs/{job_id}/manifest.json",
                        json.dumps({"videos": record["videos"]}).encode())
        store.create(record)
        uploads = [UploadTarget(idx=v["idx"], filename=v["filename"],
                                object_path=v["object_path"],
                                url=gcs.signed_put_url(v["object_path"]))
                   for v in record["videos"]]
        return AllocateResp(job_id=job_id, uploads=uploads)

    @router.post("/jobs:submit", response_model=JobStatus)
    def submit(req: SubmitReq, user: User = Depends(current_user),
               gcs=Depends(get_gcs), store=Depends(get_store), runner=Depends(get_runner)):
        rec = store.get(req.job_id)
        if not rec or rec["owner_email"] != user.email:
            raise HTTPException(404, "job not found")
        missing = [v["filename"] for v in rec["videos"] if not gcs.exists(v["object_path"])]
        if missing:
            raise HTTPException(400, f"inputs not uploaded: {missing}")
        runner.trigger(req.job_id, settings.bucket, rec["video_count"])
        store.set_status(req.job_id, "submitted")
        return _status(store.get(req.job_id))

    @router.get("/jobs/{job_id}", response_model=JobStatus)
    def status(job_id: str, user: User = Depends(current_user), store=Depends(get_store)):
        rec = store.get(job_id)
        if not rec or rec["owner_email"] != user.email:
            raise HTTPException(404, "job not found")
        return _status(rec)

    @router.get("/jobs/{job_id}/result/{idx}")
    def result_url(job_id: str, idx: int, user: User = Depends(current_user),
                   gcs=Depends(get_gcs), store=Depends(get_store)):
        rec = store.get(job_id)
        if not rec or rec["owner_email"] != user.email:
            raise HTTPException(404, "job not found")
        v = next((x for x in rec["videos"] if x["idx"] == idx), None)
        if not v or v["status"] != "done":
            raise HTTPException(404, "result not ready")
        return {"url": gcs.signed_get_url(f"jobs/{job_id}/results/{idx}.json")}

    return router


def _status(rec) -> JobStatus:
    keys = ("idx", "filename", "status", "duration_s",
            "t_mix_90_s", "t_mix_95_s", "t_mix_99_s", "error")
    return JobStatus(job_id=rec["job_id"], status=rec["status"], video_count=rec["video_count"],
                     videos=[VideoStatus(**{k: v.get(k) for k in keys}) for v in rec["videos"]])
```

- [ ] **Step 4: Defer run** — passes after Task 11 creates `main.py`.

- [ ] **Step 5: Commit**

```bash
git add web/backend/routes_jobs.py web/tests/test_routes_jobs.py
git commit -m "feat: allocate/submit/status/result routes"
```

### Task 11: FastAPI app + SPA static serving

**Files:**
- Create: `web/backend/main.py`

- [ ] **Step 1: Write the app**

`web/backend/main.py`:
```python
from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from web.backend.config import Settings
from web.backend.auth import make_auth_dependency
from web.backend.routes_jobs import build_router

_STATIC = Path(__file__).parent / "static"


def get_gcs():
    from web.backend.gcs import GcsService
    s = Settings.from_env()
    return GcsService(s.bucket, s.backend_sa)


def get_store():
    from web.backend.firestore_store import FirestoreStore
    return FirestoreStore()


def get_runner():
    from web.backend.runner import JobRunner
    s = Settings.from_env()
    return JobRunner(s.project, s.region, s.worker_job)


def create_app(dev_no_auth: bool | None = None) -> FastAPI:
    settings = Settings.from_env()
    if dev_no_auth is not None:
        settings = settings.__class__(**{**settings.__dict__, "dev_no_auth": dev_no_auth})
    app = FastAPI(title="Kineticolor Cloud")
    app.include_router(build_router(get_gcs, get_store, get_runner,
                                    make_auth_dependency(settings), settings))

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    if _STATIC.exists():
        app.mount("/assets", StaticFiles(directory=_STATIC / "assets"), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            cand = _STATIC / full_path
            return FileResponse(cand if full_path and cand.is_file() else _STATIC / "index.html")

    return app


app = create_app()
```

- [ ] **Step 2: Run the route tests**

Run: `python -m pytest web/tests/test_routes_jobs.py -v`
Expected: PASS (5 passed)

- [ ] **Step 3: Run the full backend suite**

Run: `python -m pytest web/tests/ -v`
Expected: all pass.

- [ ] **Step 4: Smoke test**

Run: `KC_DEV_NO_AUTH=1 KC_BUCKET=x python -c "from web.backend.main import create_app; from fastapi.testclient import TestClient; print(TestClient(create_app()).get('/healthz').json())"`
Expected: `{'ok': True}`

- [ ] **Step 5: Commit**

```bash
git add web/backend/main.py
git commit -m "feat: FastAPI app + SPA static serving"
```

---

## Phase 4 — Frontend SPA

### Task 12: Scaffolding — auth, api, upload, paper tooltips

**Files:**
- Create: `web/frontend/package.json`, `vite.config.ts`, `index.html`, `src/main.tsx`, `src/lib/auth.ts`, `src/lib/api.ts`, `src/lib/upload.ts`, `src/lib/tooltips.ts`

- [ ] **Step 1: package.json** (uplot for fast canvas charts)

`web/frontend/package.json`:
```json
{
  "name": "kineticolor-frontend",
  "private": true,
  "type": "module",
  "scripts": { "dev": "vite", "build": "tsc -b && vite build", "preview": "vite preview" },
  "dependencies": { "react": "^18.3.1", "react-dom": "^18.3.1", "uplot": "^1.6.30" },
  "devDependencies": {
    "@types/react": "^18.3.3", "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1", "typescript": "^5.5.3", "vite": "^5.3.4"
  }
}
```

- [ ] **Step 2: vite.config.ts** (proxy to localhost only — no hardcoded deployed backend)

`web/frontend/vite.config.ts`:
```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
  server: { proxy: { "/api": "http://localhost:8080", "/healthz": "http://localhost:8080" } },
});
```

- [ ] **Step 3: index.html**

`web/frontend/index.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Kineticolor</title>
    <script src="https://accounts.google.com/gsi/client" async defer></script>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 4: src/lib/auth.ts**

`web/frontend/src/lib/auth.ts`:
```typescript
const CLIENT_ID = import.meta.env.VITE_OAUTH_CLIENT_ID as string;
const KEY = "kc_id_token";

export const getToken = () => sessionStorage.getItem(KEY);
export const setToken = (t: string) => sessionStorage.setItem(KEY, t);
export const clearToken = () => sessionStorage.removeItem(KEY);

export function renderSignIn(el: HTMLElement, onToken: (t: string) => void) {
  // @ts-expect-error GSI global
  google.accounts.id.initialize({
    client_id: CLIENT_ID, hosted_domain: "lemnisca.bio",
    callback: (r: { credential: string }) => { setToken(r.credential); onToken(r.credential); },
  });
  // @ts-expect-error GSI global
  google.accounts.id.renderButton(el, { theme: "outline", size: "large" });
}
```

- [ ] **Step 5: src/lib/api.ts**

`web/frontend/src/lib/api.ts`:
```typescript
import { getToken, clearToken } from "./auth";

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  headers.set("Content-Type", "application/json");
  const r = await fetch(path, { ...init, headers });
  if (r.status === 401) { clearToken(); throw new Error("session expired"); }
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json() as Promise<T>;
}

export interface UploadTarget { idx: number; filename: string; object_path: string; url: string; }
export interface AllocateResp { job_id: string; uploads: UploadTarget[]; }
export interface VideoStatus {
  idx: number; filename: string; status: string; duration_s: number | null;
  t_mix_90_s: number | null; t_mix_95_s: number | null; t_mix_99_s: number | null; error: string | null;
}
export interface JobStatus { job_id: string; status: string; video_count: number; videos: VideoStatus[]; }
export interface ResultDoc {
  duration_s: number; fps: number; frame_count: number;
  levels: Record<string, number | null>;
  series: Record<string, number[]>;
}

export const api = {
  allocate: (files: string[]) => req<AllocateResp>("/api/jobs:allocate",
    { method: "POST", body: JSON.stringify({ files }) }),
  submit: (job_id: string) => req<JobStatus>("/api/jobs:submit",
    { method: "POST", body: JSON.stringify({ job_id }) }),
  status: (job_id: string) => req<JobStatus>(`/api/jobs/${job_id}`),
  resultUrl: (job_id: string, idx: number) => req<{ url: string }>(`/api/jobs/${job_id}/result/${idx}`),
  fetchResult: async (signedUrl: string) => (await fetch(signedUrl)).json() as Promise<ResultDoc>,
};
```

- [ ] **Step 6: src/lib/upload.ts**

`web/frontend/src/lib/upload.ts`:
```typescript
export async function uploadFile(url: string, file: File, retries = 3): Promise<void> {
  for (let attempt = 0; ; attempt++) {
    try {
      const r = await fetch(url, { method: "PUT", body: file });
      if (!r.ok) throw new Error(`PUT ${r.status}`);
      return;
    } catch (e) {
      if (attempt >= retries) throw e;
      await new Promise((res) => setTimeout(res, 500 * (attempt + 1)));
    }
  }
}

export async function uploadAll(targets: { url: string; file: File }[],
                                concurrency = 6, onProgress?: (done: number) => void): Promise<void> {
  let done = 0, next = 0;
  async function worker() {
    while (next < targets.length) {
      const i = next++;
      await uploadFile(targets[i].url, targets[i].file);
      onProgress?.(++done);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, targets.length) }, worker));
}
```

- [ ] **Step 7: src/lib/tooltips.ts** — paper-grounded explanations (Barrington et al., Org. Process Res. Dev. 2022, 26, 3073−3088)

`web/frontend/src/lib/tooltips.ts`:
```typescript
// Short hover explanations grounded in the Kineticolor paper.
// Each: how it's calculated + how to read it.
export interface MetricInfo { key: string; label: string; how: string; read: string; }

export const DELTA_E_INFO: MetricInfo = {
  key: "normalized_delta_e",
  label: "ΔE (color change)",
  how: "ΔE is the straight-line (Euclidean) distance in CIE-L*a*b* color space between each frame's average color and the reference frame (t=0). Shown normalized 0–1 (divided by its maximum).",
  read: "Rises as the mixture's color changes, then flattens (plateaus) once color stops changing — i.e. mixing is complete. The 0.90 / 0.95 / 0.99 markers are the first times ΔE reaches 90 / 95 / 99% of its final value. ΔE perception: <1 invisible, 2–10 visible at a glance, ~100 opposite colors.",
};

export const METRIC_INFO: Record<string, MetricInfo> = {
  contact_perimeter: {
    key: "contact_perimeter", label: "Contact",
    how: "Each frame is made black/white by a grayscale threshold; Contact is the total perimeter between black and white regions.",
    read: "Peaks during the mixing transition (mixed and unmixed regions coexist) and decays toward ~0 as the vessel becomes uniform.",
  },
  contrast: {
    key: "contrast", label: "Contrast (GLCM)",
    how: "Computed from the Gray-Level Co-occurrence Matrix: the gray-level difference across neighbouring pixel pairs.",
    read: "Highest when the image is most visibly heterogeneous; falls toward zero as the mixture becomes homogeneous.",
  },
  homogeneity: {
    key: "homogeneity", label: "Homogeneity (GLCM)",
    how: "From the GLCM: how close the matrix is to diagonal (neighbouring pixels sharing the same gray level).",
    read: "Increases as pixels become similar — higher means more thoroughly mixed.",
  },
  energy: {
    key: "energy", label: "Energy / ASM (GLCM)",
    how: "Angular Second Moment from the GLCM — the amount of single 'block' color (sum of squared probabilities).",
    read: "Low for noisy/heterogeneous frames, rises toward its maximum (1) as the frame becomes one uniform color.",
  },
  variance_delta_e: {
    key: "variance_delta_e", label: "Variance (by cell)",
    how: "The region is split into a 5×5 grid; this is the variance of average ΔE across those cells.",
    read: "High when different areas of the vessel differ in color; drops as mixing evens them out.",
  },
};
```

- [ ] **Step 8: main.tsx**

`web/frontend/src/main.tsx`:
```tsx
import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";

createRoot(document.getElementById("root")!).render(<App />);
```

- [ ] **Step 9: Install + typecheck** (App/components added next task)

Run: `cd "/Users/kartikey/Desktop/work_products/comp_viz_code:files/comp_viz_analysis/web/frontend" && npm install`
Expected: install succeeds.

- [ ] **Step 10: Commit**

```bash
git add web/frontend/package.json web/frontend/vite.config.ts web/frontend/index.html web/frontend/src/lib web/frontend/src/main.tsx
git commit -m "feat: SPA scaffolding — auth, api, upload, paper tooltips"
```

### Task 13: Chart + hover components

**Files:**
- Create: `web/frontend/src/components/InfoHover.tsx`, `DeltaEChart.tsx`, `MetricChart.tsx`

- [ ] **Step 1: InfoHover.tsx** (the ⓘ hover tooltip)

`web/frontend/src/components/InfoHover.tsx`:
```tsx
import React, { useState } from "react";
import type { MetricInfo } from "../lib/tooltips";

export function InfoHover({ info }: { info: MetricInfo }) {
  const [open, setOpen] = useState(false);
  return (
    <span style={{ position: "relative", marginLeft: 6 }}
          onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <span style={{ cursor: "help", border: "1px solid #888", borderRadius: "50%",
                     fontSize: 11, padding: "0 5px", color: "#555" }}>i</span>
      {open && (
        <div style={{ position: "absolute", zIndex: 10, top: "1.4em", left: 0, width: 300,
                      background: "#fff", border: "1px solid #ccc", borderRadius: 6,
                      padding: 10, boxShadow: "0 4px 12px rgba(0,0,0,.15)", fontSize: 12,
                      lineHeight: 1.4, textAlign: "left", fontWeight: 400 }}>
          <div style={{ marginBottom: 6 }}><b>How it's calculated.</b> {info.how}</div>
          <div><b>How to read it.</b> {info.read}</div>
        </div>
      )}
    </span>
  );
}
```

- [ ] **Step 2: DeltaEChart.tsx** (normalized ΔE + 0.90/0.95/0.99 markers, auto-rendered)

`web/frontend/src/components/DeltaEChart.tsx`:
```tsx
import React, { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { ResultDoc } from "../lib/api";

const LEVEL_COLORS: Record<string, string> = { "0.90": "#f59e0b", "0.95": "#10b981", "0.99": "#3b82f6" };

export function DeltaEChart({ result }: { result: ResultDoc }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const t = result.series.timestamp;
    const y = result.series.normalized_delta_e;
    const opts: uPlot.Options = {
      width: 640, height: 320, title: "Normalized ΔE vs time",
      scales: { x: { time: false }, y: { range: [0, 1.05] } },
      axes: [{ label: "Time (s)" }, { label: "Normalized ΔE (0–1)" }],
      series: [{}, { label: "ΔE", stroke: "#7c3aed", width: 2 }],
      hooks: {
        draw: [(u) => {
          // vertical markers at each reached level
          for (const [lvl, tx] of Object.entries(result.levels)) {
            if (tx == null) continue;
            const cx = u.valToPos(tx, "x", true);
            u.ctx.save();
            u.ctx.strokeStyle = LEVEL_COLORS[lvl] ?? "#999";
            u.ctx.setLineDash([4, 3]);
            u.ctx.beginPath();
            u.ctx.moveTo(cx, u.bbox.top);
            u.ctx.lineTo(cx, u.bbox.top + u.bbox.height);
            u.ctx.stroke();
            u.ctx.restore();
          }
        }],
      },
    };
    const plot = new uPlot(opts, [t, y], ref.current);
    return () => plot.destroy();
  }, [result]);
  return <div ref={ref} />;
}
```

- [ ] **Step 3: MetricChart.tsx** (generic single-series line, used in the hidden panel)

`web/frontend/src/components/MetricChart.tsx`:
```tsx
import React, { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

export function MetricChart({ t, y, label }: { t: number[]; y: number[]; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const opts: uPlot.Options = {
      width: 480, height: 220, title: label,
      scales: { x: { time: false } }, axes: [{ label: "Time (s)" }, {}],
      series: [{}, { label, stroke: "#2563eb", width: 1.5 }],
    };
    const plot = new uPlot(opts, [t, y], ref.current);
    return () => plot.destroy();
  }, [t, y, label]);
  return <div ref={ref} />;
}
```

- [ ] **Step 4: Typecheck**

Run: `cd "/Users/kartikey/Desktop/work_products/comp_viz_code:files/comp_viz_analysis/web/frontend" && npx tsc --noEmit`
Expected: no errors (Upload/Result views + App added next task — if tsc complains about missing App imports, continue to Task 14 then re-run).

- [ ] **Step 5: Commit**

```bash
git add web/frontend/src/components
git commit -m "feat: ΔE chart with level markers, metric chart, info hover"
```

### Task 14: Views + App (upload, auto-rendered result, hidden metrics panel)

**Files:**
- Create: `web/frontend/src/views/UploadView.tsx`, `src/views/ResultView.tsx`, `src/App.tsx`

- [ ] **Step 1: UploadView.tsx**

`web/frontend/src/views/UploadView.tsx`:
```tsx
import React, { useState } from "react";
import { api } from "../lib/api";
import { uploadAll } from "../lib/upload";

export function UploadView({ onSubmitted }: { onSubmitted: (jobId: string) => void }) {
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true); setError(null); setProgress(0);
    try {
      const alloc = await api.allocate(files.map((f) => f.name));
      const byName = new Map(files.map((f) => [f.name, f]));
      await uploadAll(alloc.uploads.map((u) => ({ url: u.url, file: byName.get(u.filename)! })),
                      6, setProgress);
      await api.submit(alloc.job_id);
      onSubmitted(alloc.job_id);
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  }

  return (
    <div>
      <h2>Upload videos</h2>
      <input type="file" multiple accept="video/*"
             onChange={(e) => setFiles(Array.from(e.target.files ?? []))} />
      <p>{files.length} file(s) selected</p>
      <button disabled={busy || files.length === 0} onClick={run}>
        {busy ? `Uploading ${progress}/${files.length}…` : "Analyze"}
      </button>
      {error && <p style={{ color: "crimson" }}>{error}</p>}
    </div>
  );
}
```

- [ ] **Step 2: ResultView.tsx** — polls status; for each done video auto-fetches the result JSON and renders ΔE chart + times; other 5 graphs behind a collapsed `<details>`. Mixing numbers flagged when duration > 60s.

`web/frontend/src/views/ResultView.tsx`:
```tsx
import React, { useEffect, useState } from "react";
import { api, JobStatus, ResultDoc, VideoStatus } from "../lib/api";
import { DeltaEChart } from "../components/DeltaEChart";
import { MetricChart } from "../components/MetricChart";
import { InfoHover } from "../components/InfoHover";
import { DELTA_E_INFO, METRIC_INFO } from "../lib/tooltips";

const DURATION_CAP_S = 60;
const fmt = (v: number | null) => (v == null ? "—" : `${v.toFixed(2)} s`);

function VideoResult({ jobId, v }: { jobId: string; v: VideoStatus }) {
  const [doc, setDoc] = useState<ResultDoc | null>(null);
  useEffect(() => {
    if (v.status !== "done") return;
    (async () => {
      const { url } = await api.resultUrl(jobId, v.idx);
      setDoc(await api.fetchResult(url));
    })();
  }, [jobId, v.idx, v.status]);

  if (v.status === "failed") return <div><b>{v.filename}</b> — failed: {v.error}</div>;
  if (v.status !== "done" || !doc) return <div><b>{v.filename}</b> — {v.status}…</div>;

  const longVideo = (v.duration_s ?? 0) > DURATION_CAP_S;
  const t = doc.series.timestamp;

  return (
    <div style={{ borderTop: "1px solid #eee", padding: "16px 0" }}>
      <h3>{v.filename} <InfoHover info={DELTA_E_INFO} /></h3>
      <DeltaEChart result={doc} />
      <div style={{ margin: "8px 0" }}>
        <b>Mixing time</b>{" "}
        <span style={{ color: "#f59e0b" }}>90%: {fmt(v.t_mix_90_s)}</span>{"  "}
        <span style={{ color: "#10b981" }}>95%: {fmt(v.t_mix_95_s)}</span>{"  "}
        <span style={{ color: "#3b82f6" }}>99%: {fmt(v.t_mix_99_s)}</span>
        {longVideo && (
          <div style={{ color: "#b45309", fontSize: 12, marginTop: 4 }}>
            ℹ Heads up: this clip is {v.duration_s?.toFixed(0)}s. Mixing-time numbers are most
            reliable for short clips (≤{DURATION_CAP_S}s); for long, highly viscous, or
            dead-zone-prone reactions, sanity-check them against the ΔE curve. The graph itself is valid.
          </div>
        )}
      </div>
      <details>
        <summary style={{ cursor: "pointer" }}>Other metrics (contact, contrast, homogeneity, energy, variance)</summary>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 16, marginTop: 12 }}>
          {Object.values(METRIC_INFO).map((info) => (
            <div key={info.key}>
              <div style={{ fontSize: 13, fontWeight: 600 }}>{info.label}<InfoHover info={info} /></div>
              <MetricChart t={t} y={doc.series[info.key] ?? []} label={info.label} />
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}

export function ResultView({ jobId }: { jobId: string }) {
  const [job, setJob] = useState<JobStatus | null>(null);
  useEffect(() => {
    let alive = true;
    async function tick() {
      try {
        const s = await api.status(jobId);
        if (!alive) return;
        setJob(s);
        if (s.status !== "done" && s.status !== "failed") setTimeout(tick, 4000);
      } catch { if (alive) setTimeout(tick, 4000); }
    }
    tick();
    return () => { alive = false; };
  }, [jobId]);

  if (!job) return <p>Loading…</p>;
  return (
    <div>
      <h2>Job {job.job_id} — {job.status}</h2>
      {job.videos.map((v) => <VideoResult key={v.idx} jobId={job.job_id} v={v} />)}
    </div>
  );
}
```

- [ ] **Step 3: App.tsx**

`web/frontend/src/App.tsx`:
```tsx
import React, { useEffect, useRef, useState } from "react";
import { getToken, renderSignIn } from "./lib/auth";
import { UploadView } from "./views/UploadView";
import { ResultView } from "./views/ResultView";

export function App() {
  const [signedIn, setSignedIn] = useState(!!getToken());
  const [jobId, setJobId] = useState<string | null>(null);
  const btnRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!signedIn && btnRef.current) renderSignIn(btnRef.current, () => setSignedIn(true));
  }, [signedIn]);

  if (!signedIn) {
    return (
      <div style={{ maxWidth: 640, margin: "4rem auto", fontFamily: "system-ui" }}>
        <h1>Kineticolor</h1>
        <p>Sign in with your <b>@lemnisca.bio</b> account.</p>
        <div ref={btnRef} />
      </div>
    );
  }
  return (
    <div style={{ maxWidth: 900, margin: "2rem auto", fontFamily: "system-ui" }}>
      <h1>Kineticolor — Mixing-Time Analysis</h1>
      <UploadView onSubmitted={setJobId} />
      {jobId && <ResultView jobId={jobId} />}
    </div>
  );
}
```

- [ ] **Step 4: Build the SPA**

Run: `cd "/Users/kartikey/Desktop/work_products/comp_viz_code:files/comp_viz_analysis/web/frontend" && VITE_OAUTH_CLIENT_ID=dummy npm run build`
Expected: `dist/index.html` + `dist/assets/*`, no TS errors.

- [ ] **Step 5: Commit**

```bash
git add web/frontend/src/views web/frontend/src/App.tsx
git commit -m "feat: upload + auto-rendered ΔE result, mixing times, hidden metrics panel"
```

---

## Phase 5 — Backend container

### Task 15: Multi-stage backend Dockerfile

**Files:**
- Create: `web/backend/Dockerfile`

- [ ] **Step 1: Write the Dockerfile**

`web/backend/Dockerfile`:
```dockerfile
# --- Stage 1: build SPA ---
FROM node:20-slim AS fe
ARG VITE_OAUTH_CLIENT_ID
WORKDIR /fe
COPY web/frontend/package.json web/frontend/package-lock.json* ./
RUN npm install
COPY web/frontend/ ./
RUN VITE_OAUTH_CLIENT_ID=${VITE_OAUTH_CLIENT_ID} npm run build

# --- Stage 2: python backend serving the SPA ---
FROM python:3.12-slim
WORKDIR /app
COPY web/backend/requirements.txt ./req.txt
RUN pip install --no-cache-dir -r req.txt
COPY web/__init__.py ./web/__init__.py
COPY web/backend/ ./web/backend/
COPY --from=fe /fe/dist ./web/backend/static
ENV PYTHONUNBUFFERED=1
CMD ["sh", "-c", "uvicorn web.backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
```

- [ ] **Step 2: Build + run + healthz**

Run:
```bash
cd "/Users/kartikey/Desktop/work_products/comp_viz_code:files/comp_viz_analysis"
docker build -f web/backend/Dockerfile --build-arg VITE_OAUTH_CLIENT_ID=dummy -t kineticolor-app:local .
docker run -d --rm -p 8080:8080 -e KC_DEV_NO_AUTH=1 -e KC_BUCKET=x --name kc kineticolor-app:local
sleep 3 && curl -s localhost:8080/healthz && docker stop kc
```
Expected: prints `{"ok":true}`

- [ ] **Step 3: Commit**

```bash
git add web/backend/Dockerfile
git commit -m "feat: multi-stage backend image (SPA build + uvicorn)"
```

---

## Phase 6 — Infrastructure (one-time GCP setup)

### Task 16: CORS + setup script

**Files:**
- Create: `web/infra/cors.json`, `web/infra/setup.sh`

- [ ] **Step 1: cors.json**

`web/infra/cors.json`:
```json
[
  { "origin": ["https://kineticolor-app-REPLACEME.run.app", "http://localhost:5173"],
    "method": ["GET", "PUT"], "responseHeader": ["Content-Type"], "maxAgeSeconds": 3600 }
]
```

- [ ] **Step 2: setup.sh** (`set -euo pipefail`, least privilege, WIF)

`web/infra/setup.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT="${KC_PROJECT:-kineticolor-cloud}"
REGION="${KC_REGION:-us-central1}"
BUCKET="${KC_BUCKET:-kineticolor-videos}"
REPO="kineticolor"
BACKEND_SA="kc-backend@${PROJECT}.iam.gserviceaccount.com"
WORKER_SA="kc-worker@${PROJECT}.iam.gserviceaccount.com"
CI_SA="kc-ci-deployer@${PROJECT}.iam.gserviceaccount.com"
GITHUB_REPO="${KC_GITHUB_REPO:-Lemniscabio/comp_viz_analysis}"

gcloud config set project "$PROJECT"

gcloud services enable run.googleapis.com storage.googleapis.com firestore.googleapis.com \
  artifactregistry.googleapis.com iamcredentials.googleapis.com logging.googleapis.com

gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" 2>/dev/null || true
gcloud firestore databases create --location="$REGION" 2>/dev/null || true
gcloud storage buckets create "gs://${BUCKET}" --location="$REGION" --uniform-bucket-level-access 2>/dev/null || true

gcloud storage buckets update "gs://${BUCKET}" --lifecycle-file=/dev/stdin <<'LIFECYCLE'
{"rule":[
  {"action":{"type":"SetStorageClass","storageClass":"NEARLINE"},"condition":{"age":60,"matchesPrefix":["jobs/"]}},
  {"action":{"type":"Delete"},"condition":{"age":365,"matchesPrefix":["jobs/"]}}
]}
LIFECYCLE

for sa in kc-backend kc-worker kc-ci-deployer; do
  gcloud iam service-accounts create "$sa" 2>/dev/null || true
done

gcloud storage buckets update "gs://${BUCKET}" --cors-file="$(dirname "$0")/cors.json"

# Backend: sign URLs as itself, bucket-scoped storage, Firestore, trigger worker job
gcloud iam service-accounts add-iam-policy-binding "$BACKEND_SA" \
  --member="serviceAccount:${BACKEND_SA}" --role="roles/iam.serviceAccountTokenCreator"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${BACKEND_SA}" --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${BACKEND_SA}" --role="roles/datastore.user"
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${BACKEND_SA}" --role="roles/run.developer"
gcloud iam service-accounts add-iam-policy-binding "$WORKER_SA" \
  --member="serviceAccount:${BACKEND_SA}" --role="roles/iam.serviceAccountUser"

# Worker: bucket-scoped storage, Firestore, AR read, logging
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${WORKER_SA}" --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${WORKER_SA}" --role="roles/datastore.user"
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${WORKER_SA}" --role="roles/artifactregistry.reader"
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${WORKER_SA}" --role="roles/logging.logWriter"

# Workload Identity Federation for GitHub Actions
gcloud iam workload-identity-pools create kc-github-pool --location=global 2>/dev/null || true
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global --workload-identity-pool=kc-github-pool \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='${GITHUB_REPO}'" 2>/dev/null || true
POOL_ID="$(gcloud iam workload-identity-pools describe kc-github-pool --location=global --format='value(name)')"
gcloud iam service-accounts add-iam-policy-binding "$CI_SA" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_REPO}"
gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:${CI_SA}" --role="roles/run.admin"
gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:${CI_SA}" --role="roles/artifactregistry.writer"
gcloud iam service-accounts add-iam-policy-binding "$BACKEND_SA" --member="serviceAccount:${CI_SA}" --role="roles/iam.serviceAccountUser"
gcloud iam service-accounts add-iam-policy-binding "$WORKER_SA" --member="serviceAccount:${CI_SA}" --role="roles/iam.serviceAccountUser"

echo "DONE. Create an OAuth Web client id, set GitHub repo Variables OAUTH_CLIENT_ID and WIF_PROVIDER:"
gcloud iam workload-identity-pools providers describe github-provider \
  --location=global --workload-identity-pool=kc-github-pool --format='value(name)'
```

- [ ] **Step 3: Lint**

Run: `bash -n web/infra/setup.sh`
Expected: no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add web/infra/cors.json web/infra/setup.sh
git commit -m "feat: one-time GCP setup (least privilege, WIF) + CORS"
```

### Task 17: Manual deploy script

**Files:**
- Create: `web/infra/deploy.sh`

- [ ] **Step 1: deploy.sh** (builds both images, creates the worker Job + backend service)

`web/infra/deploy.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT="${KC_PROJECT:-kineticolor-cloud}"
REGION="${KC_REGION:-us-central1}"
BUCKET="${KC_BUCKET:-kineticolor-videos}"
REPO="kineticolor"
OAUTH_CLIENT_ID="${KC_OAUTH_CLIENT_ID:?set KC_OAUTH_CLIENT_ID}"
BACKEND_SA="kc-backend@${PROJECT}.iam.gserviceaccount.com"
WORKER_SA="kc-worker@${PROJECT}.iam.gserviceaccount.com"
AR="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}"
SHA="$(git rev-parse --short HEAD)"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

docker buildx build --platform linux/amd64 -f web/worker/Dockerfile \
  -t "${AR}/worker:${SHA}" -t "${AR}/worker:latest" --push .
docker buildx build --platform linux/amd64 -f web/backend/Dockerfile \
  --build-arg VITE_OAUTH_CLIENT_ID="${OAUTH_CLIENT_ID}" \
  -t "${AR}/backend:${SHA}" -t "${AR}/backend:latest" --push .

gcloud run jobs deploy kineticolor-worker --image "${AR}/worker:${SHA}" --region "$REGION" \
  --service-account "$WORKER_SA" --cpu 4 --memory 4Gi --task-timeout 3600 --max-retries 0 \
  --set-env-vars "KC_BUCKET=${BUCKET},KC_THREADS=4" --project "$PROJECT"

gcloud run deploy kineticolor-app --image "${AR}/backend:${SHA}" --region "$REGION" \
  --service-account "$BACKEND_SA" --allow-unauthenticated --cpu 1 --memory 512Mi \
  --set-env-vars "KC_PROJECT=${PROJECT},KC_REGION=${REGION},KC_BUCKET=${BUCKET},KC_OAUTH_CLIENT_ID=${OAUTH_CLIENT_ID},KC_ALLOWED_DOMAIN=lemnisca.bio,KC_WORKER_JOB=kineticolor-worker,KC_BACKEND_SA=${BACKEND_SA}" \
  --project "$PROJECT"

gcloud run services describe kineticolor-app --region "$REGION" --format='value(status.url)'
```

- [ ] **Step 2: Lint**

Run: `bash -n web/infra/deploy.sh`
Expected: no syntax errors.

- [ ] **Step 3: Commit**

```bash
git add web/infra/deploy.sh
git commit -m "feat: manual deploy (worker Job + backend service)"
```

---

## Phase 7 — CI/CD

### Task 18: Deploy workflow

**Files:**
- Create: `.github/workflows/deploy.yml` (repo root — that's where GitHub looks)

- [ ] **Step 1: Write the workflow**

`.github/workflows/deploy.yml`:
```yaml
name: deploy
on:
  push: { branches: [main, feat/web-deploy] }
  pull_request: {}

permissions: { contents: read, id-token: write }

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r web/backend/requirements.txt -r web/worker/requirements.txt
      - run: python -m pytest web/tests/ -v
      - uses: actions/setup-node@v4
        with: { node-version: "20" }
      - run: cd web/frontend && npm install && VITE_OAUTH_CLIENT_ID=ci npm run build

  deploy:
    needs: test
    if: github.ref == 'refs/heads/feat/web-deploy'   # change to main once merged
    runs-on: ubuntu-latest
    env:
      PROJECT: kineticolor-cloud
      REGION: us-central1
      BUCKET: kineticolor-videos
      AR: us-central1-docker.pkg.dev/kineticolor-cloud/kineticolor
    steps:
      - uses: actions/checkout@v4
      - name: Guard OAUTH_CLIENT_ID
        run: |
          if [ -z "${{ vars.OAUTH_CLIENT_ID }}" ]; then echo "::error::OAUTH_CLIENT_ID Variable missing"; exit 1; fi
      - uses: google-github-actions/auth@v2
        with:
          project_id: kineticolor-cloud
          workload_identity_provider: ${{ vars.WIF_PROVIDER }}
          service_account: kc-ci-deployer@kineticolor-cloud.iam.gserviceaccount.com
      - uses: google-github-actions/setup-gcloud@v2
      - run: gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
      - name: Build + push
        run: |
          SHA=$(git rev-parse --short HEAD)
          docker buildx build --platform linux/amd64 -f web/worker/Dockerfile \
            -t $AR/worker:$SHA -t $AR/worker:latest --push .
          docker buildx build --platform linux/amd64 -f web/backend/Dockerfile \
            --build-arg VITE_OAUTH_CLIENT_ID=${{ vars.OAUTH_CLIENT_ID }} \
            -t $AR/backend:$SHA -t $AR/backend:latest --push .
      - name: Deploy worker Job
        run: |
          SHA=$(git rev-parse --short HEAD)
          gcloud run jobs deploy kineticolor-worker --image $AR/worker:$SHA --region $REGION \
            --service-account kc-worker@$PROJECT.iam.gserviceaccount.com \
            --cpu 4 --memory 4Gi --task-timeout 3600 --max-retries 0 --set-env-vars KC_BUCKET=$BUCKET,KC_THREADS=4
      - name: Deploy backend service
        run: |
          SHA=$(git rev-parse --short HEAD)
          gcloud run deploy kineticolor-app --image $AR/backend:$SHA --region $REGION \
            --service-account kc-backend@$PROJECT.iam.gserviceaccount.com --allow-unauthenticated \
            --set-env-vars KC_PROJECT=$PROJECT,KC_REGION=$REGION,KC_BUCKET=$BUCKET,KC_OAUTH_CLIENT_ID=${{ vars.OAUTH_CLIENT_ID }},KC_ALLOWED_DOMAIN=lemnisca.bio,KC_WORKER_JOB=kineticolor-worker,KC_BACKEND_SA=kc-backend@$PROJECT.iam.gserviceaccount.com
```

- [ ] **Step 2: Validate YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/deploy.yml')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/deploy.yml
git commit -m "ci: WIF build/test/deploy to Cloud Run"
```

---

## Phase 8 — Deploy, verify, document

### Task 19: First deploy + E2E smoke test

- [ ] **Step 1:** `bash web/infra/setup.sh` (project Owner). Note the printed WIF provider.
- [ ] **Step 2:** Create an OAuth Web client id (GCP Console → Credentials). Add the eventual service URL + `http://localhost:5173` to authorized JS origins.
- [ ] **Step 3:** Set GitHub repo Variables `OAUTH_CLIENT_ID`, `WIF_PROVIDER`.
- [ ] **Step 4:** `KC_OAUTH_CLIENT_ID=<id> bash web/infra/deploy.sh` → prints service URL.
- [ ] **Step 5:** Update `web/infra/cors.json` origin to the real URL, re-run the CORS line; update OAuth authorized origins.
- [ ] **Step 6:** Open the URL, sign in with `@lemnisca.bio`, upload 2 short clips (one ≤60s, one >60s), Analyze. Confirm: ΔE graph + 0.90/0.95/0.99 markers render automatically; the >60s clip shows the "indicative" warning; the "Other metrics" panel expands to 5 graphs with working ⓘ hovers; a non-lemnisca account is rejected.
Expected: all of the above hold.
- [ ] **Step 7:** Commit the cors.json origin update.

```bash
git add web/infra/cors.json && git commit -m "chore: set production CORS origin"
```

### Task 20: README + note about the feat/mixing-time branch

**Files:**
- Create: `web/README.md`
- Modify: `README.md` (repo root, on this branch)

- [ ] **Step 1:** Write `web/README.md` — architecture, GCS layout, the env vars (`KC_PROJECT`, `KC_REGION`, `KC_BUCKET`, `KC_OAUTH_CLIENT_ID`, `KC_ALLOWED_DOMAIN`, `KC_WORKER_JOB`, `KC_BACKEND_SA`, `KC_DEV_NO_AUTH`), one-time setup, deploy (CI + manual), local dev (`KC_DEV_NO_AUTH=1 uvicorn web.backend.main:app` + `cd web/frontend && npm run dev`), and the explicit statement that the analysis logic is main's ΔE engine + the lifted level-crossing function (nothing from feat/mixing-time).

- [ ] **Step 2:** Add a short note near the top of the root `README.md`:
> **Branches:** `feat/mixing-time` contains an experimental multi-metric mixing-time quantifier (kept for reference; **reason: _<USER TO FILL IN>_**). The cloud/web deployment on `feat/web-deploy` deliberately uses only this `main` branch's ΔE logic.

(Leave the reason as a placeholder — the user will supply it.)

- [ ] **Step 3: Commit**

```bash
git add web/README.md README.md
git commit -m "docs: web deployment README + branch note"
```

---

## Carried-over security guardrails (from `NEW_GCP_OF_SCRIPTS/REVIEW_FINDINGS.md`)

- **F-001** Submit verifies every input exists before triggering (Task 10).
- **F-003 / F-006** Worker Job `--task-timeout 3600`, `--max-retries 0` (Tasks 17/18).
- **F-004** `runner.MAX_TASKS=50` caps fan-out (Task 8).
- **F-009** `safe_filename` rejects traversal/abs paths + non-video types (Task 6).
- **F-014** Shell scripts use `set -euo pipefail` (Tasks 16/17).
- **F-016** CI guards missing `OAUTH_CLIENT_ID` (Task 18).
- **F-017** Bucket-scoped `storage.objectAdmin`, not project-wide (Task 16).
- **F-019** No hardcoded dev proxy to a deployed backend (Task 12).
- **F-021** ID token in `sessionStorage`, not `localStorage` (Task 12).

---

## Self-Review

**Spec coverage:** built from `main` only, zero `feat/mixing-time` code (Task 0 guard + Task 1 lifted logic) ✓; n-video upload via signed URLs (Tasks 6, 10, 12) ✓; server-side compute independent of client = Cloud Run Jobs (Tasks 2, 8, 17) ✓; 480p downscale on upload (Task 2 ffmpeg) ✓; ΔE graph + 0.90/0.95/0.99 auto-rendered, no button (Tasks 13, 14) ✓; all 6 metrics kept, 5 behind a hidden panel (Tasks 2 results_doc, 14 `<details>`) ✓; paper-grounded hover tooltips (Tasks 12, 13) ✓; <30–60s accuracy caveat (Task 14 `DURATION_CAP_S`) ✓; lemnisca.bio gate (Task 5) ✓; reuse of NEW_GCP_OF_SCRIPTS security patterns (Tasks 5, 6, 16, 18) ✓; feat/mixing-time kept + README note (Task 20) ✓.

**Type consistency:** per-video keys (`status`, `duration_s`, `t_mix_90_s/95_s/99_s`, `error`, `idx`, `filename`, `object_path`) match across `new_job_record`, worker `_set_video`, `schemas.VideoStatus`, `routes_jobs._status`, and the React `VideoStatus`. Results-doc shape (`levels` keyed `"0.90"/"0.95"/"0.99"` strings; `series` with `timestamp`, `normalized_delta_e`, and the 6 metric keys) matches between worker `results_doc`, the `ResultDoc` TS interface, `DeltaEChart`, `MetricChart`, and `tooltips.ts` (`METRIC_INFO` keys == `SERIES_KEYS` minus `grand_delta_e`). `MAX_TASKS` (runner) vs `video_count` aligned.

**One verification dependency:** Task 2 Step 5 verifies main's `AnalysisEngine` per-frame result dict actually uses the keys in `SERIES_KEYS` (`grand_delta_e`, `contact_perimeter`, `contrast`, `homogeneity`, `energy`, `variance_delta_e`) — these come from main's `export.py` COLUMNS, but the engine's internal dict keys must be confirmed before relying on them. If any differ, update `SERIES_KEYS` + the `results_doc` test.
