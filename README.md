# Kineticolor

Computer-vision measurement of **mixing time** in transparent lab reactors. Instead of a pH probe (which lags 10–20 s), Kineticolor watches a video of the vessel and measures how long the liquid takes to become visually homogeneous after a reagent is added.

It does **not** understand chemistry. It tracks **colour change from a reference frame** (frame 0). When the colour stops changing — the metrics plateau — mixing is complete.

> Based on: Barrington, Dickinson, McGuire, Yan & Reid, *"Computer Vision for Kinetic Analysis of Lab- and Process-Scale Mixing Phenomena"*, Org. Process Res. Dev. 2022, 26, 3073–3088. The paper is included in this repo: [`docs/computer-vision-for-kinetic-analysis-of-lab-and-process-scale-mixing-phenomena.pdf`](docs/computer-vision-for-kinetic-analysis-of-lab-and-process-scale-mixing-phenomena.pdf).

---

## The core measurement

For each video frame, the engine computes **six complementary metrics** inside a region of interest. The primary one is **ΔE** — the perceptual colour distance (CIE‑L\*a\*b\* Euclidean distance) of each frame's average colour from the reference frame.

**Mixing time** is read off the ΔE curve:

1. Normalise: `norm(t) = ΔE(t) / max(ΔE)` → a 0→1 curve.
2. The mixing time at level **L** is the **first timestamp where `norm(t) ≥ L`**.
3. Kineticolor reports **L = 0.90, 0.95, 0.99** (90/95/99 % of the way to the colour plateau).

> **Accuracy note:** the ΔE *graph* is valid for any clip length, but the numeric mixing times are most reliable for **short clips (≤ ~60 s)**. For long, highly viscous, or dead‑zone‑prone reactions, treat the numbers as indicative and sanity‑check them against the curve.

The other five metrics (Contact, GLCM Contrast/Homogeneity/Energy, cell Variance) are computed too and shown for context; see `config/default_config.yaml` for tunables (grid size, GLCM levels, thresholds, frame skip).

---

## Two ways to run it

Kineticolor has **two interfaces that share the same analysis engine** (`src/core/`):

| | Desktop app | Cloud web app |
| --- | --- | --- |
| Where | `src/` (PyQt6 GUI + CLI) | `web/` (deployed multi‑user product) |
| Who | a single analyst on their machine | any `@lemnisca.bio` user, in a browser |
| Use | live ROI selection, real‑time plots, CSV export | upload videos → analyse server‑side → view results |
| Runs on | your computer | Google Cloud (Cloud Run) |

`src/core/` has **zero dependency on the GUI** — it's a standalone, headless library that both the desktop CLI and the cloud worker import.

---

## A. Desktop app

**Install**
```bash
pip install -r requirements.txt
```

**GUI** (live feed / video playback, ROI drawing, real‑time metric plots):
```bash
python -m src.main
```

**Headless / CLI** (analyse one video → CSV time series):
```bash
python -m src.main --video path/to/clip.mp4 --output results.csv --config config/default_config.yaml
```

**Batch** (folder of videos → one summary CSV, parallel):
```bash
python scripts/batch_analyze.py <video_dir> <output_dir> --workers 8
```

---

## B. Cloud web app (`web/`)

A signed‑in `@lemnisca.bio` user uploads videos, the analysis runs server‑side (independent of their machine), and per‑video ΔE graphs + mixing times come back. Live on Google Cloud project **`mixinlab`**.

### Architecture

```
Browser (React SPA, Google Sign-In)
   │  resumable, chunked uploads via signed URLs (direct to GCS)
   ▼
Cloud Run service  "kineticolor-app"   (FastAPI + bundled SPA)
   │  - verifies Google ID token, enforces hd == lemnisca.bio
   │  - RBAC (pending → admin grants role); per-owner data scoping
   │  - writes video/run metadata to Firestore
   │  - triggers the worker Job (one task per selected video)
   ▼
Cloud Run Job  "kineticolor-worker"   (the headless engine in a container)
   - task N downloads video N from GCS → ffmpeg 480p → AnalysisEngine
   - writes results JSON + CSV to GCS, updates the run's Firestore doc
   ▼
GCS bucket  "mixinlab-videos"   +   Firestore (kc_users / kc_videos / kc_runs)
```

- **Compute:** analysis runs as a **Cloud Run Job** (one parallel task per video), fully decoupled from the browser — closing the tab doesn't stop it. Status is tracked in Firestore; the SPA polls.
- **Uploads:** browser → GCS directly via **keyless V4 signed URLs**, using **resumable, chunked** transfers (16 MiB chunks) that resume on connection drops.
- **Auth:** Google OAuth (consent screen is **External** because the project lives under the `pushkar-org` org, not `lemnisca.bio`). Access is restricted by the **backend**, which verifies the ID token's `hd == lemnisca.bio` claim.
- **RBAC:** Firestore `kc_users`. Roles `admin / runner / viewer`, statuses `pending / active / disabled`. A new user who signs in is **pending** (no access) until an **admin** grants them a role. Seed admins (auto‑provisioned): `kartikey.attri@lemnisca.bio`, `laalchand.kumawat@lemnisca.bio`. Users see only their own videos/runs; admins can view across users and manage access.

### The five screens
1. **Upload** — drag‑and‑drop; videos persist to GCS grouped by date (upload ≠ analyse).
2. **Select** — pick already‑uploaded videos (grouped by date) → **Run analysis**.
3. **Status** — runs in progress: Running / Completed.
4. **Results** — per‑video normalised ΔE graph with 0.90/0.95/0.99 markers, the mixing‑time numbers, and the other five metrics behind an expandable panel (hover ⓘ for paper‑grounded explanations).
5. **Profile** — your run + upload history; for admins, a user‑management panel (approve pending users, assign roles).

### Local development

The backend serves the API; Vite serves the SPA with hot reload and proxies `/api` to it.

> **Note:** this repo's folder path contains a `:` which breaks npm's PATH lookup. Run Vite **directly via node** (not `npm run dev` / `npm run build`).

```bash
# 1) one-time, so the local backend can reach Firestore/GCS
gcloud auth application-default login

# 2) backend on :8080  (from repo root)
KC_DEV_NO_AUTH=1 KC_PROJECT=mixinlab KC_REGION=us-central1 KC_BUCKET=mixinlab-videos \
KC_BACKEND_SA=kc-backend@mixinlab.iam.gserviceaccount.com \
<venv>/bin/python -m uvicorn web.backend.main:app --port 8080 --reload

# 3) frontend on :5173  (from web/frontend)
VITE_OAUTH_CLIENT_ID="<oauth-web-client-id>" node node_modules/vite/bin/vite.js
```
`KC_DEV_NO_AUTH=1` skips token verification (returns a dev admin). Open http://localhost:5173.

Python tests:
```bash
<venv>/bin/python -m pytest web/tests/ -q
```

### Deploy

One script bootstraps everything for project `mixinlab`:
```bash
# first run prints how to create the OAuth Web client, then stops
bash web/infra/bootstrap-mixinlab.sh
# after creating the client, re-run with it to build + deploy + set CORS
KC_OAUTH_CLIENT_ID="<client-id>" bash web/infra/bootstrap-mixinlab.sh
```
It enables APIs, creates the bucket / Firestore / least‑privilege service accounts / Workload Identity Federation, builds & pushes the worker + backend images, deploys both, and sets the bucket CORS. The **only manual step** is creating the OAuth Web client in the console (consent screen **External**, published to "In production").

**CI/CD:** pushing to `main` triggers `.github/workflows/deploy.yml` — it runs the test suite, then (via WIF, no stored keys) builds + deploys. Requires one repo **Variable**: `OAUTH_CLIENT_ID`.

---

## Repository layout

```
src/                 Desktop app + the shared analysis engine
  core/              Headless engine: metrics/, analysis_engine.py, export.py, video_reader.py  (NO GUI deps)
  gui/               PyQt6 interface
  utils/             config loader, logging, colour helpers
  main.py            CLI / GUI entry point
config/              default_config.yaml (tunable parameters)
scripts/             batch_analyze.py, downscale helpers
tests/               engine + metric tests (pytest)
web/                 Cloud web app
  worker/            Cloud Run Job: ffmpeg + engine + level-crossing logic
  backend/           FastAPI: auth, rbac, videos, runs, admin, signed URLs
  frontend/          React + Vite + TypeScript SPA (5 screens)
  infra/             bootstrap-mixinlab.sh, setup.sh, deploy.sh
docs/                The source paper (PDF) + docs/superpowers/plans/ implementation plans (build history)
.github/workflows/   deploy.yml (test + WIF deploy on push to main)
```

### Key env vars (web)
`KC_PROJECT` · `KC_REGION` · `KC_BUCKET` · `KC_OAUTH_CLIENT_ID` · `KC_ALLOWED_DOMAIN` · `KC_WORKER_JOB` · `KC_BACKEND_SA` · `KC_SEED_ADMINS` · `KC_THREADS` (worker CPU) · `KC_DEV_NO_AUTH` (local only).

---

## Branches

- **`main`** — canonical. The desktop app + the deployed cloud web app.
- **`feat/mixing-time`** — experimental: tests around cropping/downscaling videos and quantifying mixing time via several alternative methods. Kept for reference only; `main` is authoritative.
