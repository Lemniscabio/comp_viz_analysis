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
