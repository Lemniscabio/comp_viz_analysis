#!/usr/bin/env bash
set -euo pipefail

PROJECT="${KC_PROJECT:-mixinlab}"
REGION="${KC_REGION:-us-central1}"
BUCKET="${KC_BUCKET:-mixinlab-videos}"
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

# NOTE: CORS is intentionally NOT set here. The real Cloud Run URLs don't exist
# until the first deploy, and bootstrap Phase 3 is the single source of truth for
# the bucket CORS (correct URLs + resumable headers). Setting a placeholder here
# would clobber the good CORS every time setup.sh runs.

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
