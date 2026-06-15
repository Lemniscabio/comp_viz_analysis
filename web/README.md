# Kineticolor Web Deployment

Kineticolor runs as a React single-page application served by a FastAPI backend on Cloud Run. Users authenticate with Google OAuth and must belong to the configured domain. The browser uploads videos directly to Google Cloud Storage through signed URLs, then the backend starts a Cloud Run Job with one task per video. Each worker downscales its video to 480p, runs the analysis, writes JSON and CSV results to Cloud Storage, and updates the job record in Firestore. The frontend polls the backend and renders the results when processing completes.

## Architecture

- `web/frontend`: React, Vite, TypeScript, and uPlot user interface.
- `web/backend`: FastAPI API, Google OAuth domain gate, signed URLs, Firestore job records, Cloud Run Job triggering, and built SPA hosting.
- `web/worker`: Cloud Run Job worker with ffmpeg and the Kineticolor analysis engine.
- `web/infra`: one-time GCP setup and manual deployment scripts.
- `.github/workflows/deploy.yml`: tests, builds, and deploys through Workload Identity Federation.

The analysis logic is main's ΔE engine plus the level-crossing function lifted from main's `src/gui/plots_panel.py:_update_mixing_marker`, generalized to levels 0.90, 0.95, and 0.99. Nothing is taken from `feat/mixing-time`.

## Cloud Storage Layout

```text
jobs/<job_id>/manifest.json
jobs/<job_id>/inputs/<idx>__<filename>
jobs/<job_id>/results/<idx>.json
jobs/<job_id>/results/<idx>__<stem>.csv
```

Firestore stores only job status and per-video summary fields. Metric series remain in Cloud Storage.

## Environment Variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `KC_PROJECT` | GCP project ID | empty in the app; `kineticolor-cloud` in scripts |
| `KC_REGION` | Cloud Run and Artifact Registry region | `us-central1` |
| `KC_BUCKET` | Cloud Storage bucket | empty in the app; `kineticolor-videos` in scripts |
| `KC_OAUTH_CLIENT_ID` | Google OAuth Web client ID | empty; required by deployment |
| `KC_ALLOWED_DOMAIN` | Allowed Google Workspace domain | `lemnisca.bio` |
| `KC_WORKER_JOB` | Cloud Run worker Job name | `kineticolor-worker` |
| `KC_BACKEND_SA` | Backend service account email used for keyless signed URLs | empty |
| `KC_DEV_NO_AUTH` | Set to `1` to bypass backend authentication for local development | disabled |

`web/infra/setup.sh` also accepts `KC_GITHUB_REPO` to configure the GitHub Workload Identity Federation repository binding.

## One-Time Setup

Run the setup script as a principal that can enable APIs and create GCP resources:

```bash
bash web/infra/setup.sh
```

Then create a Google OAuth Web client, add the local and deployed origins, and configure the GitHub repository variables `OAUTH_CLIENT_ID` and `WIF_PROVIDER`. Replace the placeholder Cloud Run origin in `web/infra/cors.json` after the first deployment and apply the bucket CORS configuration again.

## Deployment

The GitHub Actions workflow tests the backend and worker, builds the frontend, builds both container images, pushes them to Artifact Registry, and deploys the worker Job and backend service. Deployment currently runs from `feat/web-deploy`; change the workflow condition to `main` after merging.

For a manual deployment:

```bash
KC_OAUTH_CLIENT_ID=<oauth-client-id> bash web/infra/deploy.sh
```

The script builds and pushes both images, deploys the Cloud Run Job and service, and prints the service URL.

## Local Development

Start the backend from the repository root:

```bash
KC_DEV_NO_AUTH=1 uvicorn web.backend.main:app
```

In another terminal, start the Vite development server:

```bash
cd web/frontend
npm install
npm run dev
```

Vite proxies `/api` and `/healthz` to `http://localhost:8080`.
