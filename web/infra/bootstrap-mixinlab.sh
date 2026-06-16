#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot GCP bootstrap for the Kineticolor web app on project "mixinlab".
#
# What it does:
#   Phase 1  enable APIs + create infra (Artifact Registry, Firestore, GCS
#            bucket w/ lifecycle+CORS, service accounts, least-privilege IAM,
#            Workload Identity Federation for CI)        -> runs setup.sh
#   Phase 2  build+push the worker and backend images, deploy the Cloud Run
#            Job + Cloud Run service                     -> runs deploy.sh
#   Phase 3  point the bucket CORS at the live service URL
#
# Usage:
#   # First run (no OAuth client yet) — does Phase 1, then prints how to make
#   # the OAuth client and stops:
#   bash web/infra/bootstrap-mixinlab.sh
#
#   # After you create the OAuth Web client id, re-run with it to deploy:
#   KC_OAUTH_CLIENT_ID="1234-abc.apps.googleusercontent.com" \
#     bash web/infra/bootstrap-mixinlab.sh
#
# Prerequisites:
#   * gcloud installed and logged in:  gcloud auth login
#   * Docker Desktop running (for buildx cross-build to linux/amd64)
#   * You have Owner/Editor on project "mixinlab"
#
# Re-runnable: Phase 1 is idempotent; Phase 2 re-pushes images + updates services.
# ---------------------------------------------------------------------------
set -euo pipefail

export KC_PROJECT="mixinlab"
export KC_REGION="${KC_REGION:-us-central1}"
export KC_BUCKET="${KC_BUCKET:-mixinlab-videos}"
export KC_GITHUB_REPO="${KC_GITHUB_REPO:-Lemniscabio/comp_viz_analysis}"

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "############################################################"
echo "# Kineticolor bootstrap"
echo "#   project : $KC_PROJECT"
echo "#   region  : $KC_REGION"
echo "#   bucket  : gs://$KC_BUCKET"
echo "############################################################"

# --- sanity: authenticated + project exists ---
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
  echo "ERROR: not logged in. Run:  gcloud auth login" >&2
  exit 1
fi
gcloud config set project "$KC_PROJECT" >/dev/null

# ===========================================================================
# Phase 1 — infrastructure (idempotent)
# ===========================================================================
echo
echo "== Phase 1: enable APIs + create services (setup.sh) =="
bash "$HERE/setup.sh"

# ===========================================================================
# OAuth gate — the one step that must be done by hand in the console
# ===========================================================================
if [[ -z "${KC_OAUTH_CLIENT_ID:-}" ]]; then
  cat <<EOF

────────────────────────────────────────────────────────────────────────────
ACTION REQUIRED — create an OAuth 2.0 Web client, then re-run this script.

 1. Open: https://console.cloud.google.com/apis/credentials?project=$KC_PROJECT
 2. (first time only) Configure the OAuth consent screen:
       User type: EXTERNAL
         NOTE: this project is in the "pushkar-org" organization (not lemnisca.bio),
         so "Internal" is NOT available — pick External. Access is still restricted
         to lemnisca.bio: the BACKEND verifies the Google ID token's hd==lemnisca.bio
         claim and 403s everyone else. (External consent + app-level hd gate is the
         same pattern used by the NEW_GCP app.)
       Publishing status: PUBLISH / "In production"
         Scopes are only openid/email/profile (non-sensitive), so NO Google app
         verification is required. If left in "Testing", only emails you add as
         Test users can sign in — publishing to production lifts that limit.
 3. + CREATE CREDENTIALS -> OAuth client ID
       Application type : Web application
       Name            : kineticolor-web
       Authorized JavaScript origins:  http://localhost:5173
       (you will add the live Cloud Run URL after the first deploy — printed below)
 4. CREATE, then copy the Client ID (looks like 1234-abc.apps.googleusercontent.com)

 5. Re-run to build + deploy:
       KC_OAUTH_CLIENT_ID="<paste-client-id>" bash web/infra/bootstrap-mixinlab.sh
────────────────────────────────────────────────────────────────────────────
EOF
  exit 0
fi
export KC_OAUTH_CLIENT_ID

# ===========================================================================
# Phase 2 — build + deploy
# ===========================================================================
echo
echo "== Phase 2: configure docker auth + build/push + deploy (deploy.sh) =="
gcloud auth configure-docker "${KC_REGION}-docker.pkg.dev" --quiet
bash "$HERE/deploy.sh"

# ===========================================================================
# Phase 3 — CORS to the live URL(s)
# Cloud Run serves the service on TWO equivalent hostnames: the hash form
# (status.url, *.a.run.app) and the project-number form
# (kineticolor-app-<projectNumber>.<region>.run.app). Browsers send whichever
# you visit as the Origin, so BOTH must be in the bucket CORS allowlist.
# ===========================================================================
URL_HASH="$(gcloud run services describe kineticolor-app --region "$KC_REGION" \
            --format='value(status.url)')"
PNUM="$(gcloud projects describe "$KC_PROJECT" --format='value(projectNumber)')"
URL_PNUM="https://kineticolor-app-${PNUM}.${KC_REGION}.run.app"
echo
echo "== Phase 3: set bucket CORS for both URLs =="
echo "   $URL_HASH"
echo "   $URL_PNUM"
TMP_CORS="$(mktemp)"
cat > "$TMP_CORS" <<EOF
[ { "origin": ["$URL_HASH", "$URL_PNUM", "http://localhost:5173"],
    "method": ["GET", "PUT", "POST"],
    "responseHeader": ["Content-Type", "Location", "Range", "x-goog-resumable", "content-range"],
    "maxAgeSeconds": 3600 } ]
EOF
gcloud storage buckets update "gs://${KC_BUCKET}" --cors-file="$TMP_CORS"
rm -f "$TMP_CORS"

cat <<EOF

✅ DEPLOYED.  The service answers on BOTH of these URLs:
   $URL_HASH
   $URL_PNUM

LAST STEP (one-time) — authorize BOTH URLs for sign-in:
  https://console.cloud.google.com/apis/credentials?project=$KC_PROJECT
  -> open the "kineticolor-web" OAuth client
  -> Authorized JavaScript origins -> ADD both URLs above
  -> SAVE

Then open either URL and sign in with a @lemnisca.bio account.
EOF
