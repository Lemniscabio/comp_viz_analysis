#!/usr/bin/env bash
set -euo pipefail

PROJECT="${KC_PROJECT:-mixinlab}"
REGION="${KC_REGION:-us-central1}"
BUCKET="${KC_BUCKET:-mixinlab-videos}"
REPO="kineticolor"
OAUTH_CLIENT_ID="${KC_OAUTH_CLIENT_ID:?set KC_OAUTH_CLIENT_ID}"
# Normalize paste artifacts: a bare client id, NOT a URL. Strip scheme + trailing slash.
OAUTH_CLIENT_ID="${OAUTH_CLIENT_ID#http://}"
OAUTH_CLIENT_ID="${OAUTH_CLIENT_ID#https://}"
OAUTH_CLIENT_ID="${OAUTH_CLIENT_ID%/}"
case "$OAUTH_CLIENT_ID" in
  *.apps.googleusercontent.com) : ;;
  *) echo "ERROR: KC_OAUTH_CLIENT_ID must end in .apps.googleusercontent.com (got: $OAUTH_CLIENT_ID)" >&2; exit 1 ;;
esac
echo "Using OAuth client id: $OAUTH_CLIENT_ID"
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
# seed admins contains commas -> set with a non-comma delimiter
gcloud run services update kineticolor-app --region "$REGION" --project "$PROJECT" \
  --update-env-vars "^@@^KC_SEED_ADMINS=kartikey.attri@lemnisca.bio,laalchand.kumawat@lemnisca.bio"

gcloud run services describe kineticolor-app --region "$REGION" --format='value(status.url)'
