# Kineticolor Web v2 — Screens, RBAC, Resumable Uploads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve the deployed Kineticolor web app into a 5-screen product (Upload → Select → Status → Results → Profile) with decoupled upload/analysis, resumable chunked GCS uploads, and NEW_GCP-style RBAC (pending/active users, admin approval, two hardcoded seed admins).

**Architecture:** Upload and analysis are decoupled. **Uploads** persist videos as durable assets (GCS `uploads/<email>/<date>/...` + Firestore `kc_videos`). **Runs** are a separate step that selects already-uploaded videos and triggers a Cloud Run Job (Firestore `kc_runs`). Access is gated by an RBAC layer (Firestore `kc_users`) copied verbatim from `NEW_GCP_OF_SCRIPTS/phase3-run-app`: seed admins are auto-provisioned, every other lemnisca.bio user is `pending` until an admin grants a role. Uploads use **GCS resumable signed-URL sessions** (chunked, resume-on-drop) to survive the `ERR_HTTP2_PING_FAILED` failures a single PUT hits.

**Tech Stack:** Python 3.12, FastAPI, google-cloud-storage/firestore/run, google-auth; React + Vite 8 + TypeScript + react-router-dom + uplot; GCP Cloud Run + Cloud Run Jobs + GCS + Firestore.

**Builds on:** the existing `web/` app on branch `main` (auth.py, worker engine, Dockerfiles, infra all stay). This plan **replaces** the jobs-based flow (`routes_jobs.py`, the single-screen SPA) with the videos+runs model.

**Conventions:**
- Project `mixinlab`, region `us-central1`, bucket `mixinlab-videos`.
- Firestore collections: `kc_users`, `kc_videos`, `kc_runs`.
- GCS: `uploads/<email>/<YYYY-MM-DD>/<video_id>__<filename>`; `runs/<run_id>/manifest.json`; `runs/<run_id>/results/<video_id>.json` + `.csv`.
- Env: `KC_SEED_ADMINS` (default `kartikey.attri@lemnisca.bio,laalchand.kumawat@lemnisca.bio`).
- Roles: `admin`, `runner`, `viewer`. Statuses: `pending`, `active`, `disabled`.
- venv python for tests: `/Users/kartikey/kineticolor-web-venv/bin/python` (run from repo root).

---

## File Structure

```
web/backend/
  config.py            # MODIFY: add seed_admins
  users.py             # NEW: UserRecord, ROLES/STATUSES, resolve_on_login, FirestoreUserRepository (from NEW_GCP)
  rbac.py              # NEW: current_account, _enforce, require_active/runner/admin
  videos.py            # NEW: VideoRecord, video_id, object paths, FirestoreVideoRepository
  runs.py              # NEW: RunRecord, run-record builder, FirestoreRunRepository
  gcs.py               # MODIFY: add resumable-initiate signed URL; keep signed GET
  runner.py            # MODIFY: env RUN_ID (was JOB_ID); same task-count cap
  schemas.py           # MODIFY: new request/response models
  routes_me.py         # NEW: GET /api/me, /api/me/videos, /api/me/runs
  routes_admin.py      # NEW: GET/POST /api/admin/users, GET /api/admin/runs, /api/admin/videos
  routes_videos.py     # NEW: POST /api/videos:allocate, /{id}:finalize, GET /api/videos
  routes_runs.py       # NEW: POST /api/runs, GET /api/runs, /api/runs/{id}, /api/runs/{id}/result/{video_id}
  main.py              # MODIFY: wire new routers + dep providers; drop routes_jobs
  routes_jobs.py       # DELETE

web/worker/
  worker.py            # MODIFY: read runs/<run_id>/manifest.json, update kc_runs, write runs/<run_id>/results/...

web/frontend/src/
  lib/api.ts           # MODIFY: v2 endpoints + types
  lib/auth.ts          # (unchanged)
  lib/upload.ts        # MODIFY: resumable chunked uploader
  lib/me.ts            # NEW: fetch current account (role/status)
  lib/tooltips.ts      # (unchanged)
  components/ProfileHeader.tsx   # NEW: header w/ user + nav + role
  components/DeltaEChart.tsx     # (unchanged)
  components/MetricChart.tsx     # (unchanged)
  components/InfoHover.tsx       # (unchanged)
  views/UploadView.tsx           # MODIFY: drag-drop + resumable upload
  views/SelectView.tsx           # NEW: date-grouped videos + checkboxes + Run
  views/StatusView.tsx           # NEW: running/completed runs
  views/ResultsView.tsx          # MODIFY (rename from ResultView): run results
  views/ProfileView.tsx          # NEW: my history + admin user-mgmt panel
  views/PendingView.tsx          # NEW: shown to pending/disabled users
  App.tsx                        # MODIFY: router + role gate + header

web/infra/
  bootstrap-mixinlab.sh # MODIFY: CORS adds POST + resumable headers + expose location; deploy sets KC_SEED_ADMINS
  deploy.sh             # MODIFY: pass KC_SEED_ADMINS to backend service
```

---

## Phase 1 — RBAC backend (copied from NEW_GCP)

### Task 1: Config — seed admins

**Files:** Modify `web/backend/config.py`; Test `web/tests/test_config_backend.py`

- [ ] **Step 1: Add failing test**

Append to `web/tests/test_config_backend.py`:
```python
def test_seed_admins_default_and_override(monkeypatch):
    monkeypatch.delenv("KC_SEED_ADMINS", raising=False)
    import importlib, web.backend.config as c
    importlib.reload(c)
    s = c.Settings.from_env()
    assert "kartikey.attri@lemnisca.bio" in s.seed_admins
    assert "laalchand.kumawat@lemnisca.bio" in s.seed_admins
    monkeypatch.setenv("KC_SEED_ADMINS", "A@x.bio, b@x.bio")
    s2 = c.Settings.from_env()
    assert s2.seed_admins == ["a@x.bio", "b@x.bio"]  # lowercased, trimmed
```

- [ ] **Step 2: Run → fail**

Run: `cd "/Users/kartikey/Desktop/work_products/comp_viz_code:files/comp_viz_analysis" && /Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_config_backend.py -v`
Expected: FAIL (`seed_admins` attribute missing).

- [ ] **Step 3: Implement** — edit `web/backend/config.py`: add the field + parser.

Add at top after imports:
```python
def _parse_seed_admins() -> list[str]:
    raw = os.environ.get("KC_SEED_ADMINS",
                         "kartikey.attri@lemnisca.bio,laalchand.kumawat@lemnisca.bio")
    return [e.strip().lower() for e in raw.split(",") if e.strip()]
```
Add `seed_admins: list[str]` to the `Settings` dataclass (use `field(default_factory=list)` import not needed since we set it in `from_env`), and in `from_env` add `seed_admins=_parse_seed_admins(),`. Ensure `from dataclasses import dataclass` already present.

- [ ] **Step 4: Run → pass**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_config_backend.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/backend/config.py web/tests/test_config_backend.py
git commit -m "feat(rbac): seed-admins setting (kartikey + laalchand)"
```

### Task 2: Users core — records, roles, resolve_on_login, repo

**Files:** Create `web/backend/users.py`; Test `web/tests/test_users.py`

- [ ] **Step 1: Write the failing test**

`web/tests/test_users.py`:
```python
import datetime as dt
from web.backend.users import UserRecord, ROLES, STATUSES, resolve_on_login

NOW = dt.datetime(2026, 6, 16, tzinfo=dt.timezone.utc)


def test_seed_admin_becomes_admin_active():
    rec = resolve_on_login("Kartikey.Attri@lemnisca.bio", ["kartikey.attri@lemnisca.bio"], None, NOW)
    assert rec.email == "kartikey.attri@lemnisca.bio"
    assert rec.role == "admin" and rec.status == "active" and rec.decided_by == "seed"


def test_new_user_is_pending():
    rec = resolve_on_login("new@lemnisca.bio", ["a@lemnisca.bio"], None, NOW)
    assert rec.role is None and rec.status == "pending"


def test_existing_non_admin_unchanged():
    existing = UserRecord(email="u@lemnisca.bio", role="runner", status="active", requested_at=NOW)
    rec = resolve_on_login("u@lemnisca.bio", ["a@lemnisca.bio"], existing, NOW)
    assert rec is existing


def test_seed_admin_idempotent():
    existing = UserRecord(email="a@lemnisca.bio", role="admin", status="active", requested_at=NOW,
                          decided_by="seed", decided_at=NOW)
    rec = resolve_on_login("a@lemnisca.bio", ["a@lemnisca.bio"], existing, NOW)
    assert rec is existing


def test_role_status_vocab():
    assert ROLES == {"admin", "runner", "viewer"}
    assert STATUSES == {"pending", "active", "disabled"}
```

- [ ] **Step 2: Run → fail**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_users.py -v`
Expected: FAIL (no module).

- [ ] **Step 3: Implement** — `web/backend/users.py` (verbatim logic from NEW_GCP `core/users.py`, renamed collection):
```python
from __future__ import annotations
import dataclasses
import datetime
from dataclasses import dataclass
from typing import Optional, Protocol

ROLES = {"admin", "runner", "viewer"}
STATUSES = {"pending", "active", "disabled"}


@dataclass
class UserRecord:
    email: str
    role: Optional[str]
    status: str
    requested_at: datetime.datetime
    decided_by: Optional[str] = None
    decided_at: Optional[datetime.datetime] = None


def resolve_on_login(email: str, seed_admins: list[str],
                     existing: Optional[UserRecord], now: datetime.datetime) -> UserRecord:
    email = email.lower()
    if email in seed_admins:
        if existing and existing.role == "admin" and existing.status == "active":
            return existing
        return UserRecord(email=email, role="admin", status="active",
                          requested_at=(existing.requested_at if existing else now),
                          decided_by="seed", decided_at=now)
    if existing is None:
        return UserRecord(email=email, role=None, status="pending", requested_at=now)
    return existing


class UserRepository(Protocol):
    def get(self, email: str) -> Optional[UserRecord]: ...
    def upsert(self, record: UserRecord) -> None: ...
    def list_all(self) -> list[UserRecord]: ...
    def set_decision(self, email: str, role: Optional[str], status: str,
                     decided_by: str, now: datetime.datetime) -> None: ...


class FirestoreUserRepository:
    COLLECTION = "kc_users"

    def __init__(self, client, collection: str = COLLECTION) -> None:
        self._c = client
        self._col = collection

    def _doc(self, email: str):
        return self._c.collection(self._col).document(email.lower())

    def _read(self, d: dict) -> UserRecord:
        return UserRecord(email=d["email"], role=d.get("role"),
                          status=d.get("status", "pending"), requested_at=d.get("requested_at"),
                          decided_by=d.get("decided_by"), decided_at=d.get("decided_at"))

    def get(self, email):
        snap = self._doc(email).get()
        return self._read(snap.to_dict()) if snap.exists else None

    def upsert(self, record):
        self._doc(record.email).set({
            "email": record.email.lower(), "role": record.role, "status": record.status,
            "requested_at": record.requested_at, "decided_by": record.decided_by,
            "decided_at": record.decided_at}, merge=True)

    def list_all(self):
        out = [self._read(s.to_dict()) for s in self._c.collection(self._col).stream()]
        return sorted(out, key=lambda u: u.email)

    def set_decision(self, email, role, status, decided_by, now):
        self._doc(email).set({"role": role, "status": status,
                              "decided_by": decided_by, "decided_at": now}, merge=True)
```

- [ ] **Step 4: Run → pass**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_users.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add web/backend/users.py web/tests/test_users.py
git commit -m "feat(rbac): user records + resolve_on_login + Firestore repo (kc_users)"
```

### Task 3: RBAC dependencies

**Files:** Create `web/backend/rbac.py`; Test `web/tests/test_rbac.py`

- [ ] **Step 1: Write the failing test**

`web/tests/test_rbac.py`:
```python
import datetime as dt
import pytest
from fastapi import HTTPException
from web.backend.users import UserRecord
from web.backend.rbac import _enforce

NOW = dt.datetime(2026, 6, 16, tzinfo=dt.timezone.utc)


def rec(role, status):
    return UserRecord(email="u@lemnisca.bio", role=role, status=status, requested_at=NOW)


def test_active_passes_for_any_active():
    assert _enforce(rec("viewer", "active"), "active").role == "viewer"


def test_inactive_blocked():
    with pytest.raises(HTTPException) as e:
        _enforce(rec("admin", "pending"), "active")
    assert e.value.status_code == 403


def test_runner_requires_runner_or_admin():
    assert _enforce(rec("runner", "active"), "runner")
    assert _enforce(rec("admin", "active"), "runner")
    with pytest.raises(HTTPException):
        _enforce(rec("viewer", "active"), "runner")


def test_admin_requires_admin():
    assert _enforce(rec("admin", "active"), "admin")
    with pytest.raises(HTTPException):
        _enforce(rec("runner", "active"), "admin")
```

- [ ] **Step 2: Run → fail**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_rbac.py -v`
Expected: FAIL (no module).

- [ ] **Step 3: Implement** — `web/backend/rbac.py`:
```python
from __future__ import annotations
import datetime
from fastapi import Depends, HTTPException
from web.backend.users import UserRecord, resolve_on_login


def _enforce(rec: UserRecord, need: str) -> UserRecord:
    if rec.status != "active":
        raise HTTPException(status_code=403, detail=f"access {rec.status}")
    if need == "runner" and rec.role not in ("runner", "admin"):
        raise HTTPException(status_code=403, detail="requires runner role")
    if need == "admin" and rec.role != "admin":
        raise HTTPException(status_code=403, detail="requires admin role")
    return rec


def make_rbac(current_user, get_user_repo, settings):
    """Returns (current_account, require_active, require_runner, require_admin) deps."""
    def current_account(user=Depends(current_user), repo=Depends(get_user_repo)):
        now = datetime.datetime.now(datetime.timezone.utc)
        if settings.dev_no_auth:
            return user, UserRecord(email=user.email, role="admin", status="active", requested_at=now)
        existing = repo.get(user.email)
        resolved = resolve_on_login(user.email, settings.seed_admins, existing, now)
        if resolved != existing:
            repo.upsert(resolved)
        return user, resolved

    def require_active(account=Depends(current_account)):
        _enforce(account[1], "active"); return account

    def require_runner(account=Depends(current_account)):
        _enforce(account[1], "runner"); return account

    def require_admin(account=Depends(current_account)):
        _enforce(account[1], "admin"); return account

    return current_account, require_active, require_runner, require_admin
```

- [ ] **Step 4: Run → pass**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_rbac.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add web/backend/rbac.py web/tests/test_rbac.py
git commit -m "feat(rbac): _enforce + current_account/require_* dependency factory"
```

---

## Phase 2 — Videos (uploads) model + resumable signing

### Task 4: Video records + paths

**Files:** Create `web/backend/videos.py`; Test `web/tests/test_videos.py`

- [ ] **Step 1: Write the failing test**

`web/tests/test_videos.py`:
```python
import datetime as dt
from web.backend.videos import (VideoRecord, video_object_path, new_video_record,
                                 VIDEO_EXTS, safe_video_name)
import pytest

NOW = dt.datetime(2026, 6, 16, 9, 30, tzinfo=dt.timezone.utc)


def test_safe_video_name_rejects_bad():
    with pytest.raises(ValueError):
        safe_video_name("../x.mp4")
    with pytest.raises(ValueError):
        safe_video_name("notes.txt")
    assert safe_video_name("clip 1.mp4") == "clip 1.mp4"


def test_object_path_groups_by_email_and_date():
    p = video_object_path("u@lemnisca.bio", "vid123", "clip.mp4", NOW)
    assert p == "uploads/u@lemnisca.bio/2026-06-16/vid123__clip.mp4"


def test_new_video_record_shape():
    rec = new_video_record("vid123", "u@lemnisca.bio", "clip.mp4", 1234, NOW)
    assert rec.video_id == "vid123"
    assert rec.date == "2026-06-16"
    assert rec.gcs_path == "uploads/u@lemnisca.bio/2026-06-16/vid123__clip.mp4"
    assert rec.size_bytes == 1234 and rec.status == "uploaded"
```

- [ ] **Step 2: Run → fail**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_videos.py -v`
Expected: FAIL (no module).

- [ ] **Step 3: Implement** — `web/backend/videos.py`:
```python
from __future__ import annotations
import dataclasses
import datetime
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def safe_video_name(name: str) -> str:
    p = PurePosixPath(name)
    if name != p.name or name in (".", "..") or name.startswith("/"):
        raise ValueError(f"unsafe filename: {name!r}")
    if p.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"unsupported video type: {name!r}")
    return name


def new_video_id() -> str:
    return uuid.uuid4().hex[:16]


def video_object_path(email: str, video_id: str, filename: str, now: datetime.datetime) -> str:
    return f"uploads/{email.lower()}/{now.date().isoformat()}/{video_id}__{filename}"


@dataclass
class VideoRecord:
    video_id: str
    owner_email: str
    filename: str
    date: str
    gcs_path: str
    size_bytes: int
    uploaded_at: datetime.datetime
    status: str  # "allocated" | "uploaded"


def new_video_record(video_id, owner_email, filename, size_bytes, now) -> VideoRecord:
    return VideoRecord(
        video_id=video_id, owner_email=owner_email.lower(), filename=filename,
        date=now.date().isoformat(),
        gcs_path=video_object_path(owner_email, video_id, filename, now),
        size_bytes=size_bytes, uploaded_at=now, status="uploaded")


class FirestoreVideoRepository:
    COLLECTION = "kc_videos"

    def __init__(self, client, collection: str = COLLECTION):
        self._c = client
        self._col = collection

    def _read(self, d) -> VideoRecord:
        return VideoRecord(**{k: d.get(k) for k in
            ("video_id", "owner_email", "filename", "date", "gcs_path",
             "size_bytes", "uploaded_at", "status")})

    def create(self, rec: VideoRecord):
        self._c.collection(self._col).document(rec.video_id).set(dataclasses.asdict(rec))

    def get(self, video_id) -> Optional[VideoRecord]:
        snap = self._c.collection(self._col).document(video_id).get()
        return self._read(snap.to_dict()) if snap.exists else None

    def list_by_owner(self, email):
        q = self._c.collection(self._col).where("owner_email", "==", email.lower())
        recs = [self._read(s.to_dict()) for s in q.stream()]
        return sorted(recs, key=lambda v: v.uploaded_at or datetime.datetime.min, reverse=True)

    def list_all(self):
        recs = [self._read(s.to_dict()) for s in self._c.collection(self._col).stream()]
        return sorted(recs, key=lambda v: v.uploaded_at or datetime.datetime.min, reverse=True)
```

- [ ] **Step 4: Run → pass**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_videos.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add web/backend/videos.py web/tests/test_videos.py
git commit -m "feat(videos): VideoRecord + date-grouped paths + Firestore repo (kc_videos)"
```

### Task 5: GCS — resumable-initiate signed URL

**Files:** Modify `web/backend/gcs.py`; Test `web/tests/test_gcs.py`

- [ ] **Step 1: Write the failing test** — append to `web/tests/test_gcs.py`:
```python
def test_gcsservice_has_resumable_initiate(monkeypatch):
    import web.backend.gcs as g
    # the method exists and builds a POST signed url with the resumable header
    assert hasattr(g.GcsService, "signed_resumable_initiate_url")
```

- [ ] **Step 2: Run → fail**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_gcs.py -v`
Expected: FAIL (`signed_resumable_initiate_url` missing).

- [ ] **Step 3: Implement** — add method to `GcsService` in `web/backend/gcs.py`:
```python
    def signed_resumable_initiate_url(self, object_path: str, content_type: str = "application/octet-stream") -> str:
        # V4 signed POST that the browser calls with header `x-goog-resumable: start`
        # to open a resumable session; the session URI comes back in the Location header.
        return self._bucket.blob(object_path).generate_signed_url(
            version="v4", expiration=dt.timedelta(hours=2), method="POST",
            headers={"x-goog-resumable": "start", "content-type": content_type},
            service_account_email=self._signer_email, access_token=self._token())
```
(Keep existing `signed_get_url`, `exists`, `upload_json`. Ensure `import datetime as dt` is present — it is.)
Also add a helper to read object size for finalize verification:
```python
    def object_size(self, object_path: str) -> int | None:
        blob = self._bucket.blob(object_path)
        blob.reload()
        return blob.size
```

- [ ] **Step 4: Run → pass**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_gcs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/backend/gcs.py web/tests/test_gcs.py
git commit -m "feat(uploads): resumable-initiate signed URL + object_size helper"
```

---

## Phase 3 — Runs model + worker rewire

### Task 6: Run records

**Files:** Create `web/backend/runs.py`; Test `web/tests/test_runs.py`

- [ ] **Step 1: Write the failing test**

`web/tests/test_runs.py`:
```python
import datetime as dt
from web.backend.videos import VideoRecord
from web.backend.runs import new_run_record, RunRecord

NOW = dt.datetime(2026, 6, 16, tzinfo=dt.timezone.utc)


def vid(i):
    return VideoRecord(video_id=f"v{i}", owner_email="u@lemnisca.bio", filename=f"c{i}.mp4",
                       date="2026-06-16", gcs_path=f"uploads/u@lemnisca.bio/2026-06-16/v{i}__c{i}.mp4",
                       size_bytes=10, uploaded_at=NOW, status="uploaded")


def test_new_run_record_builds_manifest_and_video_status():
    rec = new_run_record("run1", "u@lemnisca.bio", [vid(0), vid(1)], NOW)
    assert rec.run_id == "run1" and rec.status == "submitted" and rec.video_count == 2
    assert rec.videos[0] == {
        "idx": 0, "video_id": "v0", "filename": "c0.mp4",
        "object_path": "uploads/u@lemnisca.bio/2026-06-16/v0__c0.mp4",
        "status": "pending", "duration_s": None,
        "t_mix_90_s": None, "t_mix_95_s": None, "t_mix_99_s": None, "error": None}
```

- [ ] **Step 2: Run → fail**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_runs.py -v`
Expected: FAIL (no module).

- [ ] **Step 3: Implement** — `web/backend/runs.py`:
```python
from __future__ import annotations
import dataclasses
import datetime
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RunRecord:
    run_id: str
    owner_email: str
    created_at: datetime.datetime
    status: str
    video_count: int
    videos: list[dict] = field(default_factory=list)


def new_run_record(run_id, owner_email, video_recs, now) -> RunRecord:
    videos = [{
        "idx": i, "video_id": v.video_id, "filename": v.filename,
        "object_path": v.gcs_path, "status": "pending", "duration_s": None,
        "t_mix_90_s": None, "t_mix_95_s": None, "t_mix_99_s": None, "error": None
    } for i, v in enumerate(video_recs)]
    return RunRecord(run_id=run_id, owner_email=owner_email.lower(), created_at=now,
                     status="submitted", video_count=len(videos), videos=videos)


def manifest_for(rec: RunRecord) -> dict:
    return {"videos": [{"idx": v["idx"], "video_id": v["video_id"],
                        "filename": v["filename"], "object_path": v["object_path"]}
                       for v in rec.videos]}


class FirestoreRunRepository:
    COLLECTION = "kc_runs"

    def __init__(self, client, collection: str = COLLECTION):
        self._c = client
        self._col = collection

    def create(self, rec: RunRecord):
        self._c.collection(self._col).document(rec.run_id).set(dataclasses.asdict(rec))

    def get(self, run_id) -> Optional[dict]:
        snap = self._c.collection(self._col).document(run_id).get()
        return snap.to_dict() if snap.exists else None

    def list_by_owner(self, email):
        q = self._c.collection(self._col).where("owner_email", "==", email.lower())
        recs = [s.to_dict() for s in q.stream()]
        return sorted(recs, key=lambda r: r.get("created_at") or datetime.datetime.min, reverse=True)

    def list_all(self):
        recs = [s.to_dict() for s in self._c.collection(self._col).stream()]
        return sorted(recs, key=lambda r: r.get("created_at") or datetime.datetime.min, reverse=True)
```

- [ ] **Step 4: Run → pass**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_runs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/backend/runs.py web/tests/test_runs.py
git commit -m "feat(runs): RunRecord + manifest builder + Firestore repo (kc_runs)"
```

### Task 7: Worker — key by run_id + video_id

**Files:** Modify `web/worker/worker.py`; Test `web/tests/test_worker.py`

The analysis core (`ffmpeg_480p_cmd`, `analyze_video` via main engine, `results_doc`, `select_video`) is UNCHANGED. Only the I/O wiring changes: read `runs/<RUN_ID>/manifest.json`, write `runs/<RUN_ID>/results/<video_id>.json|csv`, update `kc_runs/<RUN_ID>`.

- [ ] **Step 1: Update the failing test** — change `select_video` test to keep working and add a run-path test. Append to `web/tests/test_worker.py`:
```python
def test_run_result_paths():
    import web.worker.worker as w
    assert w.result_json_path("run1", "vid9", "clip") == "runs/run1/results/vid9.json"
    assert w.result_csv_path("run1", "vid9", "clip") == "runs/run1/results/vid9__clip.csv"
```

- [ ] **Step 2: Run → fail**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_worker.py -v`
Expected: FAIL (`result_json_path` missing).

- [ ] **Step 3: Implement** — in `web/worker/worker.py`:
  1. Add helpers:
```python
def result_json_path(run_id: str, video_id: str, stem: str) -> str:
    return f"runs/{run_id}/results/{video_id}.json"


def result_csv_path(run_id: str, video_id: str, stem: str) -> str:
    return f"runs/{run_id}/results/{video_id}__{stem}.csv"
```
  2. In `main()`: read env `RUN_ID` (not `JOB_ID`); `job_ref = fs.collection("kc_runs").document(run_id)`; read manifest from `runs/{run_id}/manifest.json`; the manifest entry now has `video_id`, `filename`, `object_path` (the uploads path) — download from `object_path` directly (do NOT prepend anything). Write CSV to `result_csv_path(run_id, video_id, stem)` and JSON to `result_json_path(run_id, video_id, stem)`. `_set_video`/`_maybe_finalize_job` operate on the `kc_runs` doc and match by `idx` (unchanged logic). Update `select_video` call to pass `task_index` from `CLOUD_RUN_TASK_INDEX` (unchanged). Replace all `JOB_ID`/`jobs/` references accordingly.

- [ ] **Step 4: Run → pass**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_worker.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/worker/worker.py web/tests/test_worker.py
git commit -m "feat(worker): key analysis by run_id + video_id (runs/ paths, kc_runs)"
```

### Task 8: Runner — RUN_ID env

**Files:** Modify `web/backend/runner.py`; Test `web/tests/test_runner.py`

- [ ] **Step 1: Update test** — replace JOB_ID assertions:
```python
def test_build_overrides_sets_run_id_env():
    import web.backend.runner as r
    ov = r.build_overrides("run1", "b", 3)
    env = {e["name"]: e["value"] for e in ov["container_overrides"][0]["env"]}
    assert env["RUN_ID"] == "run1" and env["BUCKET"] == "b"
```

- [ ] **Step 2: Run → fail**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_runner.py -v`
Expected: FAIL (env key `JOB_ID` not `RUN_ID`).

- [ ] **Step 3: Implement** — in `web/backend/runner.py` `build_overrides`, rename the env entry `{"name": "JOB_ID", ...}` → `{"name": "RUN_ID", "value": run_id}` and rename the param `job_id`→`run_id` in `build_overrides` and `JobRunner.trigger`.

- [ ] **Step 4: Run → pass**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/backend/runner.py web/tests/test_runner.py
git commit -m "refactor(runner): RUN_ID env (was JOB_ID)"
```

---

## Phase 4 — Backend routes + app assembly

### Task 9: Schemas v2

**Files:** Modify `web/backend/schemas.py`

- [ ] **Step 1: Replace** `web/backend/schemas.py` with:
```python
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class FileMeta(BaseModel):
    name: str
    size: int


class AllocateReq(BaseModel):
    files: List[FileMeta]


class UploadTarget(BaseModel):
    video_id: str
    filename: str
    object_path: str
    initiate_url: str   # signed resumable-initiate (POST) URL


class AllocateResp(BaseModel):
    uploads: List[UploadTarget]


class FinalizeReq(BaseModel):
    video_id: str


class VideoOut(BaseModel):
    video_id: str
    filename: str
    date: str
    size_bytes: int
    owner_email: str


class RunReq(BaseModel):
    video_ids: List[str]


class VideoStatus(BaseModel):
    idx: int
    video_id: str
    filename: str
    status: str
    duration_s: Optional[float] = None
    t_mix_90_s: Optional[float] = None
    t_mix_95_s: Optional[float] = None
    t_mix_99_s: Optional[float] = None
    error: Optional[str] = None


class RunStatus(BaseModel):
    run_id: str
    owner_email: str
    status: str
    video_count: int
    videos: List[VideoStatus]


class MeOut(BaseModel):
    email: str
    role: Optional[str]
    status: str


class SetUserReq(BaseModel):
    role: Optional[str] = None
    status: Optional[str] = None


class ManagedUser(BaseModel):
    email: str
    role: Optional[str]
    status: str
    decided_by: Optional[str] = None
```

- [ ] **Step 2: Verify import**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -c "import web.backend.schemas as s; print(s.RunStatus.__name__, s.MeOut.__name__)"`
Expected: `RunStatus MeOut`.

- [ ] **Step 3: Commit**

```bash
git add web/backend/schemas.py
git commit -m "feat(api): v2 schemas (videos, runs, me, admin)"
```

### Task 10: Video routes

**Files:** Create `web/backend/routes_videos.py`

- [ ] **Step 1: Write the route module** `web/backend/routes_videos.py`:
```python
from __future__ import annotations
import datetime
from fastapi import APIRouter, Depends, HTTPException
from web.backend.videos import (safe_video_name, new_video_id, video_object_path,
                                 new_video_record)
from web.backend.schemas import (AllocateReq, AllocateResp, UploadTarget,
                                  FinalizeReq, VideoOut)


def build_videos_router(get_gcs, get_video_repo, require_runner, require_active):
    router = APIRouter(prefix="/api")

    @router.post("/videos:allocate", response_model=AllocateResp)
    def allocate(req: AllocateReq, account=Depends(require_runner), gcs=Depends(get_gcs)):
        if not req.files:
            raise HTTPException(400, "no files")
        email = account[0].email
        now = datetime.datetime.now(datetime.timezone.utc)
        out = []
        for f in req.files:
            try:
                name = safe_video_name(f.name)
            except ValueError as e:
                raise HTTPException(400, str(e))
            vid = new_video_id()
            path = video_object_path(email, vid, name, now)
            out.append(UploadTarget(video_id=vid, filename=name, object_path=path,
                                    initiate_url=gcs.signed_resumable_initiate_url(path)))
        return AllocateResp(uploads=out)

    @router.post("/videos/{video_id}:finalize", response_model=VideoOut)
    def finalize(video_id: str, req: FinalizeReq, account=Depends(require_runner),
                 gcs=Depends(get_gcs), repo=Depends(get_video_repo)):
        email = account[0].email
        now = datetime.datetime.now(datetime.timezone.utc)
        # Reconstruct the path the same way allocate did is not possible (date/name lost),
        # so the client returns it via finalize body is avoided; instead we trust the
        # allocate-time path passed back. Client sends video_id only; we stored nothing yet,
        # so the client MUST also send filename+object_path. Adjust: read from req.
        raise HTTPException(500, "see Step 2 note")  # replaced below

    return router
```
NOTE: finalize needs `filename` + `object_path` + `size` to write the record (allocate didn't persist anything). Update `FinalizeReq` in `schemas.py` to:
```python
class FinalizeReq(BaseModel):
    video_id: str
    filename: str
    object_path: str
    size_bytes: int
```
Then implement finalize body:
```python
    @router.post("/videos/{video_id}:finalize", response_model=VideoOut)
    def finalize(video_id: str, req: FinalizeReq, account=Depends(require_runner),
                 gcs=Depends(get_gcs), repo=Depends(get_video_repo)):
        email = account[0].email
        now = datetime.datetime.now(datetime.timezone.utc)
        if not req.object_path.startswith(f"uploads/{email.lower()}/"):
            raise HTTPException(400, "object_path does not belong to caller")
        size = gcs.object_size(req.object_path)
        if size is None:
            raise HTTPException(400, "upload not found in storage")
        rec = new_video_record(video_id, email, safe_video_name(req.filename), size, now)
        rec.gcs_path = req.object_path  # preserve the allocate-time dated path
        rec.date = req.object_path.split("/")[2]
        repo.create(rec)
        return VideoOut(video_id=rec.video_id, filename=rec.filename, date=rec.date,
                        size_bytes=rec.size_bytes, owner_email=rec.owner_email)

    @router.get("/videos")
    def list_videos(account=Depends(require_active), repo=Depends(get_video_repo)):
        recs = repo.list_by_owner(account[0].email)
        return {"videos": [{"video_id": v.video_id, "filename": v.filename, "date": v.date,
                            "size_bytes": v.size_bytes, "owner_email": v.owner_email}
                           for v in recs]}
```

- [ ] **Step 2: Update `schemas.py`** `FinalizeReq` to the 4-field version above. Commit happens after Task 13's app wiring lets tests run.

- [ ] **Step 3: Commit**

```bash
git add web/backend/routes_videos.py web/backend/schemas.py
git commit -m "feat(api): video allocate/finalize/list routes (resumable, per-owner)"
```

### Task 11: Run routes

**Files:** Create `web/backend/routes_runs.py`

- [ ] **Step 1: Write** `web/backend/routes_runs.py`:
```python
from __future__ import annotations
import datetime, json, uuid
from fastapi import APIRouter, Depends, HTTPException
from web.backend.runs import new_run_record, manifest_for
from web.backend.runner import MAX_TASKS
from web.backend.schemas import RunReq, RunStatus, VideoStatus


def build_runs_router(get_gcs, get_video_repo, get_run_repo, get_runner,
                      require_runner, require_active, settings):
    router = APIRouter(prefix="/api")

    @router.post("/runs", response_model=RunStatus)
    def create_run(req: RunReq, account=Depends(require_runner), gcs=Depends(get_gcs),
                   vrepo=Depends(get_video_repo), rrepo=Depends(get_run_repo),
                   runner=Depends(get_runner)):
        email = account[0].email
        ids = list(dict.fromkeys(req.video_ids))  # dedupe, keep order
        if not ids:
            raise HTTPException(400, "no videos selected")
        if len(ids) > MAX_TASKS:
            raise HTTPException(400, f"max {MAX_TASKS} videos per run")
        recs = []
        for vid in ids:
            v = vrepo.get(vid)
            if v is None or v.owner_email != email.lower():
                raise HTTPException(404, f"video not found: {vid}")
            recs.append(v)
        run_id = uuid.uuid4().hex[:12]
        now = datetime.datetime.now(datetime.timezone.utc)
        run = new_run_record(run_id, email, recs, now)
        gcs.upload_json(f"runs/{run_id}/manifest.json", json.dumps(manifest_for(run)).encode())
        rrepo.create(run)
        runner.trigger(run_id, settings.bucket, run.video_count)
        return _to_status(rrepo.get(run_id))

    @router.get("/runs/{run_id}", response_model=RunStatus)
    def get_run(run_id: str, account=Depends(require_active), rrepo=Depends(get_run_repo)):
        rec = rrepo.get(run_id)
        if not rec or rec["owner_email"] != account[0].email.lower():
            raise HTTPException(404, "run not found")
        return _to_status(rec)

    @router.get("/runs")
    def list_runs(account=Depends(require_active), rrepo=Depends(get_run_repo)):
        return {"runs": [_to_status(r).model_dump() for r in rrepo.list_by_owner(account[0].email)]}

    @router.get("/runs/{run_id}/result/{video_id}")
    def result_url(run_id: str, video_id: str, account=Depends(require_active),
                   gcs=Depends(get_gcs), rrepo=Depends(get_run_repo)):
        rec = rrepo.get(run_id)
        if not rec or rec["owner_email"] != account[0].email.lower():
            raise HTTPException(404, "run not found")
        v = next((x for x in rec["videos"] if x["video_id"] == video_id), None)
        if not v or v["status"] != "done":
            raise HTTPException(404, "result not ready")
        return {"url": gcs.signed_get_url(f"runs/{run_id}/results/{video_id}.json")}

    return router


def _to_status(rec) -> RunStatus:
    keys = ("idx", "video_id", "filename", "status", "duration_s",
            "t_mix_90_s", "t_mix_95_s", "t_mix_99_s", "error")
    return RunStatus(run_id=rec["run_id"], owner_email=rec["owner_email"],
                     status=rec["status"], video_count=rec["video_count"],
                     videos=[VideoStatus(**{k: v.get(k) for k in keys}) for v in rec["videos"]])
```

- [ ] **Step 2: Commit** (tests run after Task 13)

```bash
git add web/backend/routes_runs.py
git commit -m "feat(api): run create/list/status/result routes (per-owner, capped)"
```

### Task 12: Me + Admin routes

**Files:** Create `web/backend/routes_me.py`, `web/backend/routes_admin.py`

- [ ] **Step 1: Write** `web/backend/routes_me.py`:
```python
from __future__ import annotations
from fastapi import APIRouter, Depends
from web.backend.schemas import MeOut


def build_me_router(current_account, get_video_repo, get_run_repo):
    router = APIRouter(prefix="/api")

    @router.get("/me", response_model=MeOut)
    def me(account=Depends(current_account)):
        rec = account[1]
        return MeOut(email=rec.email, role=rec.role, status=rec.status)

    @router.get("/me/videos")
    def my_videos(account=Depends(current_account), vrepo=Depends(get_video_repo)):
        recs = vrepo.list_by_owner(account[0].email)
        return {"videos": [{"video_id": v.video_id, "filename": v.filename, "date": v.date,
                            "size_bytes": v.size_bytes} for v in recs]}

    @router.get("/me/runs")
    def my_runs(account=Depends(current_account), rrepo=Depends(get_run_repo)):
        return {"runs": rrepo.list_by_owner(account[0].email)}

    return router
```

- [ ] **Step 2: Write** `web/backend/routes_admin.py` (verbatim guards from NEW_GCP):
```python
from __future__ import annotations
import dataclasses, datetime
from fastapi import APIRouter, Depends, HTTPException
from web.backend.users import ROLES, STATUSES
from web.backend.schemas import SetUserReq


def build_admin_router(require_admin, get_user_repo, get_run_repo, get_video_repo, settings):
    router = APIRouter(prefix="/api")

    @router.get("/admin/users")
    def list_users(account=Depends(require_admin), repo=Depends(get_user_repo)):
        return {"users": [dataclasses.asdict(u) for u in repo.list_all()]}

    @router.post("/admin/users/{email}")
    def set_user(email: str, req: SetUserReq, account=Depends(require_admin),
                 repo=Depends(get_user_repo)):
        caller = account[0]
        email = email.lower()
        if req.role is not None and req.role not in ROLES:
            raise HTTPException(400, f"bad role {req.role}")
        if req.status is not None and req.status not in STATUSES:
            raise HTTPException(400, f"bad status {req.status}")
        target = repo.get(email)
        if target is None:
            raise HTTPException(404, "unknown user")
        demote = req.role is not None and req.role != "admin"
        disable = req.status == "disabled"
        if email == caller.email.lower() and (disable or demote):
            raise HTTPException(400, "cannot disable or demote yourself")
        if email in settings.seed_admins and (disable or demote):
            raise HTTPException(400, "cannot demote/disable a seed admin")
        now = datetime.datetime.now(datetime.timezone.utc)
        repo.set_decision(email, role=req.role if req.role is not None else target.role,
                          status=req.status if req.status is not None else target.status,
                          decided_by=caller.email, now=now)
        return dataclasses.asdict(repo.get(email))

    @router.get("/admin/runs")
    def admin_runs(user: str | None = None, account=Depends(require_admin),
                   rrepo=Depends(get_run_repo)):
        recs = rrepo.list_by_owner(user) if user else rrepo.list_all()
        return {"runs": recs}

    @router.get("/admin/videos")
    def admin_videos(user: str | None = None, account=Depends(require_admin),
                     vrepo=Depends(get_video_repo)):
        recs = vrepo.list_by_owner(user) if user else vrepo.list_all()
        return {"videos": [dataclasses.asdict(v) for v in recs]}

    return router
```

- [ ] **Step 3: Commit** (tests after Task 13)

```bash
git add web/backend/routes_me.py web/backend/routes_admin.py
git commit -m "feat(api): /me + admin user/runs/videos routes (NEW_GCP guards)"
```

### Task 13: App wiring + delete jobs routes

**Files:** Modify `web/backend/main.py`; Delete `web/backend/routes_jobs.py`; Test `web/tests/test_routes_v2.py`

- [ ] **Step 1: Write the failing integration test** `web/tests/test_routes_v2.py`:
```python
from fastapi.testclient import TestClient
import web.backend.main as m


class FakeGcs:
    def __init__(self): self.objects={}; self.json={}
    def signed_resumable_initiate_url(self, p, content_type="application/octet-stream"): return f"https://init/{p}"
    def signed_get_url(self, p): return f"https://get/{p}"
    def upload_json(self, p, d): self.json[p]=d
    def object_size(self, p): return self.objects.get(p)
    def exists(self, p): return p in self.objects


class FakeVideos:
    def __init__(self): self.db={}
    def create(self, r): self.db[r.video_id]=r
    def get(self, v): return self.db.get(v)
    def list_by_owner(self, e): return [r for r in self.db.values() if r.owner_email==e.lower()]
    def list_all(self): return list(self.db.values())


class FakeRuns:
    def __init__(self): self.db={}
    def create(self, r):
        import dataclasses; self.db[r.run_id]=dataclasses.asdict(r)
    def get(self, r): return self.db.get(r)
    def list_by_owner(self, e): return [x for x in self.db.values() if x["owner_email"]==e.lower()]
    def list_all(self): return list(self.db.values())


class FakeRunner:
    def __init__(self): self.triggered=[]
    def trigger(self, run_id, bucket, n): self.triggered.append((run_id, n))


def client():
    app = m.create_app(dev_no_auth=True)
    g,v,r,rn = FakeGcs(),FakeVideos(),FakeRuns(),FakeRunner()
    app.dependency_overrides[m.get_gcs]=lambda: g
    app.dependency_overrides[m.get_video_repo]=lambda: v
    app.dependency_overrides[m.get_run_repo]=lambda: r
    app.dependency_overrides[m.get_runner]=lambda: rn
    return TestClient(app), g, v, r, rn


def test_me_is_admin_in_dev():
    c,*_ = client()
    assert c.get("/api/me").json()["role"] == "admin"


def test_allocate_returns_initiate_urls():
    c,g,v,r,rn = client()
    body = c.post("/api/videos:allocate", json={"files":[{"name":"a.mp4","size":10}]}).json()
    assert body["uploads"][0]["initiate_url"].startswith("https://init/")


def test_finalize_then_run_triggers_worker():
    c,g,v,r,rn = client()
    a = c.post("/api/videos:allocate", json={"files":[{"name":"a.mp4","size":10}]}).json()
    u = a["uploads"][0]
    g.objects[u["object_path"]] = 10  # simulate uploaded bytes
    fin = c.post(f"/api/videos/{u['video_id']}:finalize",
                 json={"video_id":u["video_id"],"filename":"a.mp4",
                       "object_path":u["object_path"],"size_bytes":10})
    assert fin.status_code == 200
    run = c.post("/api/runs", json={"video_ids":[u["video_id"]]}).json()
    assert rn.triggered == [(run["run_id"], 1)]
    assert run["status"] == "submitted"


def test_finalize_rejects_foreign_path():
    c,g,v,r,rn = client()
    assert c.post("/api/videos/x:finalize",
                  json={"video_id":"x","filename":"a.mp4",
                        "object_path":"uploads/other@lemnisca.bio/2026-06-16/x__a.mp4",
                        "size_bytes":1}).status_code == 400
```

- [ ] **Step 2: Run → fail**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/test_routes_v2.py -v`
Expected: FAIL (main not wired / get_video_repo missing).

- [ ] **Step 3: Replace `web/backend/main.py`**:
```python
from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from web.backend.config import Settings
from web.backend.auth import make_auth_dependency
from web.backend.rbac import make_rbac
from web.backend.routes_videos import build_videos_router
from web.backend.routes_runs import build_runs_router
from web.backend.routes_me import build_me_router
from web.backend.routes_admin import build_admin_router

_STATIC = Path(__file__).parent / "static"


def get_gcs():
    from web.backend.gcs import GcsService
    s = Settings.from_env(); return GcsService(s.bucket, s.backend_sa)

def get_video_repo():
    from google.cloud import firestore
    from web.backend.videos import FirestoreVideoRepository
    return FirestoreVideoRepository(firestore.Client())

def get_run_repo():
    from google.cloud import firestore
    from web.backend.runs import FirestoreRunRepository
    return FirestoreRunRepository(firestore.Client())

def get_user_repo():
    from google.cloud import firestore
    from web.backend.users import FirestoreUserRepository
    return FirestoreUserRepository(firestore.Client())

def get_runner():
    from web.backend.runner import JobRunner
    s = Settings.from_env(); return JobRunner(s.project, s.region, s.worker_job)


def create_app(dev_no_auth: bool | None = None) -> FastAPI:
    settings = Settings.from_env()
    if dev_no_auth is not None:
        settings = settings.__class__(**{**settings.__dict__, "dev_no_auth": dev_no_auth})
    app = FastAPI(title="Kineticolor Cloud v2")
    current_user = make_auth_dependency(settings)
    current_account, require_active, require_runner, require_admin = make_rbac(
        current_user, get_user_repo, settings)

    app.include_router(build_me_router(current_account, get_video_repo, get_run_repo))
    app.include_router(build_videos_router(get_gcs, get_video_repo, require_runner, require_active))
    app.include_router(build_runs_router(get_gcs, get_video_repo, get_run_repo, get_runner,
                                         require_runner, require_active, settings))
    app.include_router(build_admin_router(require_admin, get_user_repo, get_run_repo,
                                          get_video_repo, settings))

    @app.get("/healthz")
    def healthz(): return {"ok": True}

    if _STATIC.exists():
        app.mount("/assets", StaticFiles(directory=_STATIC / "assets"), name="assets")
        @app.get("/{full_path:path}")
        def spa(full_path: str):
            cand = _STATIC / full_path
            return FileResponse(cand if full_path and cand.is_file() else _STATIC / "index.html")
    return app


app = create_app()
```

- [ ] **Step 4: Delete the old jobs route**

```bash
git rm web/backend/routes_jobs.py web/tests/test_routes_jobs.py
```

- [ ] **Step 5: Run full suite → pass**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/ -q`
Expected: all pass (no references to the deleted jobs routes).

- [ ] **Step 6: Commit**

```bash
git add web/backend/main.py web/tests/test_routes_v2.py
git commit -m "feat(api): wire v2 routers (me/videos/runs/admin), drop jobs flow"
```

---

## Phase 5 — Frontend: resumable uploader + API + role

### Task 14: Resumable chunked uploader

**Files:** Modify `web/frontend/src/lib/upload.ts`

- [ ] **Step 1: Replace** `web/frontend/src/lib/upload.ts`:
```typescript
// GCS resumable upload over a signed initiate URL. Survives connection drops by
// resuming from the last byte the server acknowledges.
const CHUNK = 8 * 1024 * 1024; // 8 MiB

async function openSession(initiateUrl: string, contentType: string): Promise<string> {
  const r = await fetch(initiateUrl, {
    method: "POST",
    headers: { "x-goog-resumable": "start", "content-type": contentType },
  });
  if (!r.ok) throw new Error(`initiate ${r.status}`);
  const loc = r.headers.get("location");
  if (!loc) throw new Error("no resumable session location (CORS must expose 'location')");
  return loc;
}

async function committedOffset(sessionUri: string, total: number): Promise<number> {
  // Query current offset: PUT with empty body + Content-Range: bytes */total
  const r = await fetch(sessionUri, {
    method: "PUT",
    headers: { "content-range": `bytes */${total}` },
  });
  if (r.status === 200 || r.status === 201) return total; // already done
  if (r.status === 308) {
    const range = r.headers.get("range"); // e.g. "bytes=0-8388607"
    if (!range) return 0;
    return parseInt(range.split("-")[1], 10) + 1;
  }
  throw new Error(`status query ${r.status}`);
}

export async function resumableUpload(
  initiateUrl: string, file: File, onProgress?: (sent: number, total: number) => void,
): Promise<void> {
  const total = file.size;
  const contentType = file.type || "application/octet-stream";
  let sessionUri = await openSession(initiateUrl, contentType);
  let offset = 0;
  while (offset < total) {
    const end = Math.min(offset + CHUNK, total);
    const blob = file.slice(offset, end);
    try {
      const r = await fetch(sessionUri, {
        method: "PUT",
        headers: { "content-range": `bytes ${offset}-${end - 1}/${total}` },
        body: blob,
      });
      if (r.status === 308) {
        const range = r.headers.get("range");
        offset = range ? parseInt(range.split("-")[1], 10) + 1 : end;
      } else if (r.status === 200 || r.status === 201) {
        offset = total;
      } else {
        throw new Error(`chunk ${r.status}`);
      }
      onProgress?.(offset, total);
    } catch (e) {
      // connection drop — resync offset from server and retry this chunk
      offset = await committedOffset(sessionUri, total);
      onProgress?.(offset, total);
    }
  }
}

export async function uploadAll(
  items: { initiateUrl: string; file: File }[],
  concurrency = 3,
  onItem?: (i: number, sent: number, total: number) => void,
): Promise<void> {
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const i = next++;
      await resumableUpload(items[i].initiateUrl, items[i].file,
                            (s, t) => onItem?.(i, s, t));
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
}
```

- [ ] **Step 2: Commit**

```bash
git add web/frontend/src/lib/upload.ts
git commit -m "feat(upload): resumable chunked uploader (resume on drop)"
```

### Task 15: API client v2 + me

**Files:** Modify `web/frontend/src/lib/api.ts`; Create `web/frontend/src/lib/me.ts`

- [ ] **Step 1: Replace** `web/frontend/src/lib/api.ts`:
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

export interface UploadTarget { video_id: string; filename: string; object_path: string; initiate_url: string; }
export interface Me { email: string; role: string | null; status: string; }
export interface Video { video_id: string; filename: string; date: string; size_bytes: number; owner_email?: string; }
export interface VideoStatus { idx: number; video_id: string; filename: string; status: string;
  duration_s: number | null; t_mix_90_s: number | null; t_mix_95_s: number | null; t_mix_99_s: number | null; error: string | null; }
export interface RunStatus { run_id: string; owner_email: string; status: string; video_count: number; videos: VideoStatus[]; }
export interface ResultDoc { duration_s: number; fps: number; frame_count: number;
  levels: Record<string, number | null>; series: Record<string, number[]>; }
export interface ManagedUser { email: string; role: string | null; status: string; decided_by: string | null; }

export const api = {
  me: () => req<Me>("/api/me"),
  myVideos: () => req<{ videos: Video[] }>("/api/me/videos"),
  myRuns: () => req<{ runs: any[] }>("/api/me/runs"),
  allocate: (files: { name: string; size: number }[]) =>
    req<{ uploads: UploadTarget[] }>("/api/videos:allocate", { method: "POST", body: JSON.stringify({ files }) }),
  finalize: (t: UploadTarget, size: number) =>
    req<Video>(`/api/videos/${t.video_id}:finalize`, { method: "POST",
      body: JSON.stringify({ video_id: t.video_id, filename: t.filename, object_path: t.object_path, size_bytes: size }) }),
  listVideos: () => req<{ videos: Video[] }>("/api/videos"),
  createRun: (video_ids: string[]) => req<RunStatus>("/api/runs", { method: "POST", body: JSON.stringify({ video_ids }) }),
  listRuns: () => req<{ runs: RunStatus[] }>("/api/runs"),
  run: (id: string) => req<RunStatus>(`/api/runs/${id}`),
  resultUrl: (runId: string, videoId: string) => req<{ url: string }>(`/api/runs/${runId}/result/${videoId}`),
  fetchResult: async (url: string) => (await fetch(url)).json() as Promise<ResultDoc>,
  // admin
  listUsers: () => req<{ users: ManagedUser[] }>("/api/admin/users"),
  setUser: (email: string, body: { role?: string; status?: string }) =>
    req<ManagedUser>(`/api/admin/users/${encodeURIComponent(email)}`, { method: "POST", body: JSON.stringify(body) }),
  adminRuns: (user?: string) => req<{ runs: RunStatus[] }>(`/api/admin/runs${user ? `?user=${encodeURIComponent(user)}` : ""}`),
};
```

- [ ] **Step 2: Create** `web/frontend/src/lib/me.ts`:
```typescript
import { createContext, useContext } from "react";
import type { Me } from "./api";
export const MeContext = createContext<Me | null>(null);
export const useMe = () => useContext(MeContext);
export const isAdmin = (me: Me | null) => me?.role === "admin" && me?.status === "active";
export const canRun = (me: Me | null) => me?.status === "active" && (me?.role === "runner" || me?.role === "admin");
```

- [ ] **Step 3: Commit**

```bash
git add web/frontend/src/lib/api.ts web/frontend/src/lib/me.ts
git commit -m "feat(fe): v2 api client + me/role context"
```

---

## Phase 6 — Frontend: 5 screens + router

### Task 16: Router + add react-router-dom + ProfileHeader + App + Pending gate

**Files:** Modify `web/frontend/package.json` (add `react-router-dom`); Create `web/frontend/src/components/ProfileHeader.tsx`, `web/frontend/src/views/PendingView.tsx`; Modify `web/frontend/src/App.tsx`

- [ ] **Step 1: Add dep**

Run: `cd "/Users/kartikey/Desktop/work_products/comp_viz_code:files/comp_viz_analysis/web/frontend" && npm install react-router-dom@^6`
Expected: installs.

- [ ] **Step 2: Create** `web/frontend/src/components/ProfileHeader.tsx`:
```tsx
import React from "react";
import { Link, useLocation } from "react-router-dom";
import { useMe, isAdmin } from "../lib/me";
import { clearToken } from "../lib/auth";

const tabs = [
  { to: "/upload", label: "Upload" },
  { to: "/select", label: "Select" },
  { to: "/status", label: "Status" },
  { to: "/profile", label: "Profile" },
];

export function ProfileHeader() {
  const me = useMe();
  const loc = useLocation();
  return (
    <header style={{ display: "flex", alignItems: "center", gap: 16, padding: "12px 20px",
                     borderBottom: "1px solid #e5e7eb", position: "sticky", top: 0, background: "#fff", zIndex: 20 }}>
      <strong style={{ fontSize: 16 }}>Kineticolor</strong>
      <nav style={{ display: "flex", gap: 12, flex: 1 }}>
        {tabs.map((t) => (
          <Link key={t.to} to={t.to} style={{ textDecoration: "none",
            fontWeight: loc.pathname.startsWith(t.to) ? 700 : 400,
            color: loc.pathname.startsWith(t.to) ? "#111" : "#666" }}>{t.label}</Link>
        ))}
      </nav>
      <span style={{ fontSize: 13, color: "#444" }}>
        {me?.email} {isAdmin(me) && <em style={{ color: "#7c3aed" }}>(admin)</em>}
      </span>
      <button onClick={() => { clearToken(); location.reload(); }}>Sign out</button>
    </header>
  );
}
```

- [ ] **Step 3: Create** `web/frontend/src/views/PendingView.tsx`:
```tsx
import React from "react";
import { useMe } from "../lib/me";

export function PendingView() {
  const me = useMe();
  return (
    <div style={{ maxWidth: 560, margin: "4rem auto", fontFamily: "system-ui", textAlign: "center" }}>
      <h2>Access pending</h2>
      <p>You're signed in as <b>{me?.email}</b> ({me?.status}).</p>
      <p>An administrator needs to grant you access before you can upload or analyze videos.
         Ask <b>kartikey.attri@lemnisca.bio</b> or <b>laalchand.kumawat@lemnisca.bio</b>.</p>
    </div>
  );
}
```

- [ ] **Step 4: Replace** `web/frontend/src/App.tsx`:
```tsx
import React, { useEffect, useRef, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { getToken, renderSignIn } from "./lib/auth";
import { api, Me } from "./lib/api";
import { MeContext } from "./lib/me";
import { ProfileHeader } from "./components/ProfileHeader";
import { PendingView } from "./views/PendingView";
import { UploadView } from "./views/UploadView";
import { SelectView } from "./views/SelectView";
import { StatusView } from "./views/StatusView";
import { ResultsView } from "./views/ResultsView";
import { ProfileView } from "./views/ProfileView";

export function App() {
  const [signedIn, setSignedIn] = useState(!!getToken());
  const [me, setMe] = useState<Me | null>(null);
  const [loaded, setLoaded] = useState(false);
  const btnRef = useRef<HTMLDivElement>(null);

  useEffect(() => { if (!signedIn && btnRef.current) renderSignIn(btnRef.current, () => setSignedIn(true)); }, [signedIn]);
  useEffect(() => { if (signedIn) api.me().then(setMe).catch(() => setMe(null)).finally(() => setLoaded(true)); }, [signedIn]);

  if (!signedIn)
    return (
      <div style={{ maxWidth: 640, margin: "4rem auto", fontFamily: "system-ui" }}>
        <h1>Kineticolor</h1>
        <p>Sign in with your <b>@lemnisca.bio</b> account.</p>
        <div ref={btnRef} />
      </div>
    );
  if (!loaded) return <p style={{ textAlign: "center", marginTop: 80 }}>Loading…</p>;

  const active = me?.status === "active";
  return (
    <MeContext.Provider value={me}>
      <BrowserRouter>
        {active ? (
          <>
            <ProfileHeader />
            <main style={{ maxWidth: 980, margin: "1.5rem auto", fontFamily: "system-ui", padding: "0 16px" }}>
              <Routes>
                <Route path="/upload" element={<UploadView />} />
                <Route path="/select" element={<SelectView />} />
                <Route path="/status" element={<StatusView />} />
                <Route path="/runs/:runId" element={<ResultsView />} />
                <Route path="/profile" element={<ProfileView />} />
                <Route path="*" element={<Navigate to="/upload" replace />} />
              </Routes>
            </main>
          </>
        ) : (
          <PendingView />
        )}
      </BrowserRouter>
    </MeContext.Provider>
  );
}
```

- [ ] **Step 5: Commit** (build verified after Task 19)

```bash
git add web/frontend/package.json web/frontend/package-lock.json web/frontend/src/App.tsx web/frontend/src/components/ProfileHeader.tsx web/frontend/src/views/PendingView.tsx
git commit -m "feat(fe): router, profile header, pending gate"
```

### Task 17: Upload + Select views

**Files:** Modify `web/frontend/src/views/UploadView.tsx`; Create `web/frontend/src/views/SelectView.tsx`

- [ ] **Step 1: Replace** `web/frontend/src/views/UploadView.tsx`:
```tsx
import React, { useState } from "react";
import { api } from "../lib/api";
import { uploadAll } from "../lib/upload";

export function UploadView() {
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [pct, setPct] = useState<number[]>([]);
  const [msg, setMsg] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);

  function add(list: FileList | null) {
    if (list) setFiles((f) => [...f, ...Array.from(list).filter((x) => x.type.startsWith("video/") || /\.(mp4|mov|avi|mkv|m4v)$/i.test(x.name))]);
  }

  async function run() {
    setBusy(true); setMsg(null); setPct(files.map(() => 0));
    try {
      const { uploads } = await api.allocate(files.map((f) => ({ name: f.name, size: f.size })));
      const byName = new Map(files.map((f) => [f.name, f]));
      const items = uploads.map((u) => ({ initiateUrl: u.initiate_url, file: byName.get(u.filename)! }));
      await uploadAll(items, 3, (i, sent, total) =>
        setPct((p) => { const n = [...p]; n[i] = Math.round((100 * sent) / total); return n; }));
      for (const u of uploads) await api.finalize(u, byName.get(u.filename)!.size);
      setMsg(`Uploaded ${uploads.length} video(s). Go to Select to analyze.`);
      setFiles([]); setPct([]);
    } catch (e) { setMsg(String(e)); } finally { setBusy(false); }
  }

  return (
    <div>
      <h2>Upload videos</h2>
      <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
           onDragLeave={() => setDrag(false)}
           onDrop={(e) => { e.preventDefault(); setDrag(false); add(e.dataTransfer.files); }}
           style={{ border: `2px dashed ${drag ? "#7c3aed" : "#cbd5e1"}`, borderRadius: 12,
                    padding: 36, textAlign: "center", background: drag ? "#faf5ff" : "#fafafa" }}>
        <p>Drag &amp; drop videos here, or</p>
        <input type="file" multiple accept="video/*" onChange={(e) => add(e.target.files)} />
      </div>
      {files.map((f, i) => (
        <div key={i} style={{ marginTop: 8, fontSize: 13 }}>
          {f.name} — {(f.size / 1e6).toFixed(1)} MB {busy && `(${pct[i] ?? 0}%)`}
        </div>
      ))}
      <button disabled={busy || !files.length} onClick={run} style={{ marginTop: 16 }}>
        {busy ? "Uploading…" : `Upload ${files.length || ""}`}
      </button>
      {msg && <p style={{ marginTop: 10 }}>{msg}</p>}
    </div>
  );
}
```

- [ ] **Step 2: Create** `web/frontend/src/views/SelectView.tsx`:
```tsx
import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, Video } from "../lib/api";

export function SelectView() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const nav = useNavigate();

  useEffect(() => { api.listVideos().then((r) => setVideos(r.videos)).catch((e) => setErr(String(e))); }, []);

  const byDate = useMemo(() => {
    const m = new Map<string, Video[]>();
    for (const v of videos) { if (!m.has(v.date)) m.set(v.date, []); m.get(v.date)!.push(v); }
    return [...m.entries()].sort((a, b) => b[0].localeCompare(a[0]));
  }, [videos]);

  function toggle(id: string) {
    setSel((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  }

  async function runAnalysis() {
    setBusy(true); setErr(null);
    try {
      const run = await api.createRun([...sel]);
      nav("/status");
      return run;
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }

  return (
    <div>
      <h2>Select videos to analyze</h2>
      {err && <p style={{ color: "crimson" }}>{err}</p>}
      {byDate.length === 0 && <p>No uploads yet. Upload some videos first.</p>}
      {byDate.map(([date, vids]) => (
        <div key={date} style={{ marginBottom: 16 }}>
          <div style={{ fontWeight: 700, color: "#374151", margin: "8px 0" }}>{date}</div>
          {vids.map((v) => (
            <label key={v.video_id} style={{ display: "block", padding: "4px 0", fontSize: 14 }}>
              <input type="checkbox" checked={sel.has(v.video_id)} onChange={() => toggle(v.video_id)} />{" "}
              {v.filename} <span style={{ color: "#9ca3af" }}>({(v.size_bytes / 1e6).toFixed(1)} MB)</span>
            </label>
          ))}
        </div>
      ))}
      <button disabled={busy || sel.size === 0} onClick={runAnalysis}>
        {busy ? "Starting…" : `Run analysis (${sel.size})`}
      </button>
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add web/frontend/src/views/UploadView.tsx web/frontend/src/views/SelectView.tsx
git commit -m "feat(fe): drag-drop resumable Upload + date-grouped Select screens"
```

### Task 18: Status + Results + Profile views

**Files:** Create `web/frontend/src/views/StatusView.tsx`, `web/frontend/src/views/ProfileView.tsx`; Create `web/frontend/src/views/ResultsView.tsx` (adapted from the old ResultView); Delete old `ResultView.tsx` if present.

- [ ] **Step 1: Create** `web/frontend/src/views/StatusView.tsx`:
```tsx
import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, RunStatus } from "../lib/api";

export function StatusView() {
  const [runs, setRuns] = useState<RunStatus[]>([]);
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try { const r = await api.listRuns(); if (alive) setRuns(r.runs); } catch {}
      if (alive) setTimeout(tick, 4000);
    };
    tick(); return () => { alive = false; };
  }, []);
  return (
    <div>
      <h2>Runs</h2>
      {runs.length === 0 && <p>No runs yet.</p>}
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
        <thead><tr style={{ textAlign: "left", color: "#6b7280" }}>
          <th>Run</th><th>Videos</th><th>Status</th><th>Done</th><th></th></tr></thead>
        <tbody>
          {runs.map((r) => {
            const done = r.videos.filter((v) => v.status === "done" || v.status === "failed").length;
            return (
              <tr key={r.run_id} style={{ borderTop: "1px solid #eee" }}>
                <td>{r.run_id}</td><td>{r.video_count}</td>
                <td>{r.status === "done" ? "✅ completed" : r.status === "failed" ? "⚠ failed" : "⏳ running"}</td>
                <td>{done}/{r.video_count}</td>
                <td><Link to={`/runs/${r.run_id}`}>view results</Link></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 2: Create** `web/frontend/src/views/ResultsView.tsx` — adapt the existing per-video ΔE rendering (keep `DeltaEChart`, `MetricChart`, `InfoHover`, tooltips, the ≤60s heads-up, hidden metrics panel), driven by `useParams().runId` and `api.run(runId)` + `api.resultUrl(runId, video_id)`:
```tsx
import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api, RunStatus, ResultDoc, VideoStatus } from "../lib/api";
import { DeltaEChart } from "../components/DeltaEChart";
import { MetricChart } from "../components/MetricChart";
import { InfoHover } from "../components/InfoHover";
import { DELTA_E_INFO, METRIC_INFO } from "../lib/tooltips";

const CAP = 60;
const fmt = (v: number | null) => (v == null ? "—" : `${v.toFixed(2)} s`);

function VideoResult({ runId, v }: { runId: string; v: VideoStatus }) {
  const [doc, setDoc] = useState<ResultDoc | null>(null);
  useEffect(() => {
    if (v.status !== "done") return;
    (async () => { const { url } = await api.resultUrl(runId, v.video_id); setDoc(await api.fetchResult(url)); })();
  }, [runId, v.video_id, v.status]);
  if (v.status === "failed") return <div><b>{v.filename}</b> — failed: {v.error}</div>;
  if (v.status !== "done" || !doc) return <div><b>{v.filename}</b> — {v.status}…</div>;
  const long = (v.duration_s ?? 0) > CAP, t = doc.series.timestamp;
  return (
    <div style={{ borderTop: "1px solid #eee", padding: "16px 0" }}>
      <h3>{v.filename} <InfoHover info={DELTA_E_INFO} /></h3>
      <DeltaEChart result={doc} />
      <div style={{ margin: "8px 0" }}>
        <b>Mixing time</b>{" "}
        <span style={{ color: "#f59e0b" }}>90%: {fmt(v.t_mix_90_s)}</span>{"  "}
        <span style={{ color: "#10b981" }}>95%: {fmt(v.t_mix_95_s)}</span>{"  "}
        <span style={{ color: "#3b82f6" }}>99%: {fmt(v.t_mix_99_s)}</span>
        {long && <div style={{ color: "#b45309", fontSize: 12, marginTop: 4 }}>
          ℹ Heads up: {v.duration_s?.toFixed(0)}s clip. Mixing-time numbers are most reliable for short clips
          (≤{CAP}s); for long, highly viscous, or dead-zone-prone reactions, sanity-check against the ΔE curve. The graph is valid.
        </div>}
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

export function ResultsView() {
  const { runId } = useParams();
  const [run, setRun] = useState<RunStatus | null>(null);
  useEffect(() => {
    if (!runId) return;
    let alive = true;
    const tick = async () => {
      try { const s = await api.run(runId); if (!alive) return; setRun(s);
        if (s.status !== "done" && s.status !== "failed") setTimeout(tick, 4000); } catch { if (alive) setTimeout(tick, 4000); }
    };
    tick(); return () => { alive = false; };
  }, [runId]);
  if (!run) return <p>Loading…</p>;
  return (
    <div>
      <h2>Run {run.run_id} — {run.status}</h2>
      {run.videos.map((v) => <VideoResult key={v.video_id} runId={run.run_id} v={v} />)}
    </div>
  );
}
```

- [ ] **Step 3: Create** `web/frontend/src/views/ProfileView.tsx` (history + admin panel):
```tsx
import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, ManagedUser, RunStatus, Video } from "../lib/api";
import { useMe, isAdmin } from "../lib/me";

const ROLES = ["admin", "runner", "viewer"];

function AdminUsers() {
  const [users, setUsers] = useState<ManagedUser[]>([]);
  const [roles, setRoles] = useState<Record<string, string>>({});
  const [err, setErr] = useState<string | null>(null);
  async function refresh() {
    const r = await api.listUsers(); setUsers(r.users);
    setRoles(Object.fromEntries(r.users.map((u) => [u.email, u.role ?? "runner"])));
  }
  useEffect(() => { refresh().catch((e) => setErr(String(e))); }, []);
  const sorted = useMemo(() => [...users].sort((a, b) =>
    a.status === "pending" && b.status !== "pending" ? -1 :
    a.status !== "pending" && b.status === "pending" ? 1 : a.email.localeCompare(b.email)), [users]);
  async function decide(email: string, body: { role?: string; status?: string }) {
    try { await api.setUser(email, body); await refresh(); } catch (e) { setErr(String(e)); }
  }
  return (
    <div style={{ marginTop: 24 }}>
      <h3>User management (admin)</h3>
      {err && <p style={{ color: "crimson" }}>{err}</p>}
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
        <thead><tr style={{ textAlign: "left", color: "#6b7280" }}>
          <th>Email</th><th>Status</th><th>Role</th><th>Actions</th></tr></thead>
        <tbody>{sorted.map((u) => (
          <tr key={u.email} style={{ borderTop: "1px solid #eee" }}>
            <td>{u.email}</td><td>{u.status}</td>
            <td><select value={roles[u.email] ?? "runner"}
                  onChange={(e) => setRoles((r) => ({ ...r, [u.email]: e.target.value }))}>
                  {ROLES.map((r) => <option key={r}>{r}</option>)}</select></td>
            <td>
              <button onClick={() => decide(u.email, { role: roles[u.email], status: "active" })}>Grant</button>{" "}
              <button onClick={() => decide(u.email, { status: "disabled" })}>Disable</button>
            </td>
          </tr>))}
        </tbody>
      </table>
    </div>
  );
}

export function ProfileView() {
  const me = useMe();
  const [runs, setRuns] = useState<RunStatus[]>([]);
  const [videos, setVideos] = useState<Video[]>([]);
  useEffect(() => { api.listRuns().then((r) => setRuns(r.runs)).catch(() => {}); }, []);
  useEffect(() => { api.myVideos().then((r) => setVideos(r.videos)).catch(() => {}); }, []);
  return (
    <div>
      <h2>Profile — {me?.email}</h2>
      <h3>My runs</h3>
      {runs.length === 0 ? <p>No runs.</p> : (
        <ul>{runs.map((r) => <li key={r.run_id}>
          <Link to={`/runs/${r.run_id}`}>{r.run_id}</Link> — {r.status} ({r.video_count} videos)
        </li>)}</ul>)}
      <h3>My uploads</h3>
      {videos.length === 0 ? <p>No uploads.</p> : (
        <ul>{videos.map((v) => <li key={v.video_id}>{v.date} — {v.filename}</li>)}</ul>)}
      {isAdmin(me) && <AdminUsers />}
    </div>
  );
}
```

- [ ] **Step 4: Delete the old single-screen ResultView if it still exists**

```bash
cd "/Users/kartikey/Desktop/work_products/comp_viz_code:files/comp_viz_analysis"
git rm -f web/frontend/src/views/ResultView.tsx 2>/dev/null || true
```

- [ ] **Step 5: Commit**

```bash
git add web/frontend/src/views/StatusView.tsx web/frontend/src/views/ResultsView.tsx web/frontend/src/views/ProfileView.tsx
git commit -m "feat(fe): Status, Results, Profile (history + admin panel) screens"
```

### Task 19: Frontend build verification

**Files:** none

- [ ] **Step 1: Typecheck + build (direct node — colon path blocks `npm run build`)**

Run:
```bash
cd "/Users/kartikey/Desktop/work_products/comp_viz_code:files/comp_viz_analysis/web/frontend"
node node_modules/typescript/bin/tsc --noEmit
VITE_OAUTH_CLIENT_ID=dummy node node_modules/vite/bin/vite.js build
```
Expected: tsc no errors; vite build writes `dist/`. Fix any TS errors before continuing (common: unused imports, missing types).

- [ ] **Step 2: Commit any fixes**

```bash
git add -A web/frontend/src && git commit -m "fix(fe): typecheck/build clean for v2 screens" || echo "nothing to fix"
```

---

## Phase 7 — Deploy plumbing

### Task 20: CORS for resumable + seed-admins env

**Files:** Modify `web/infra/bootstrap-mixinlab.sh`, `web/infra/deploy.sh`

- [ ] **Step 1: Resumable CORS** — in `bootstrap-mixinlab.sh` Phase 3, change the CORS JSON to allow `POST`, the resumable request headers, and **expose the `location` header** (required for the session URI):
```bash
cat > "$TMP_CORS" <<EOF
[ { "origin": ["$URL_HASH", "$URL_PNUM", "http://localhost:5173"],
    "method": ["GET", "PUT", "POST"],
    "responseHeader": ["Content-Type", "Location", "Range", "X-Goog-Resumable"],
    "maxAgeSeconds": 3600 } ]
EOF
```

- [ ] **Step 2: Seed-admins env** — in `deploy.sh`, append `KC_SEED_ADMINS` to the backend service env:
Change the backend `--set-env-vars` to include `,KC_SEED_ADMINS=kartikey.attri@lemnisca.bio,laalchand.kumawat@lemnisca.bio` — but commas separate env vars, so use the `^:^` delimiter form for that one value. Simplest: set it separately with `--update-env-vars` using a custom delimiter:
Replace the backend deploy with two steps:
```bash
gcloud run deploy kineticolor-app --image "${AR}/backend:${SHA}" --region "$REGION" \
  --service-account "$BACKEND_SA" --allow-unauthenticated --cpu 1 --memory 512Mi \
  --set-env-vars "KC_PROJECT=${PROJECT},KC_REGION=${REGION},KC_BUCKET=${BUCKET},KC_OAUTH_CLIENT_ID=${OAUTH_CLIENT_ID},KC_ALLOWED_DOMAIN=lemnisca.bio,KC_WORKER_JOB=kineticolor-worker,KC_BACKEND_SA=${BACKEND_SA}" \
  --project "$PROJECT"
# seed admins contains commas -> set with a non-comma delimiter
gcloud run services update kineticolor-app --region "$REGION" --project "$PROJECT" \
  --update-env-vars "^@@^KC_SEED_ADMINS=kartikey.attri@lemnisca.bio,laalchand.kumawat@lemnisca.bio"
```
Apply the same `^@@^` two-step in the CI workflow's backend deploy step.

- [ ] **Step 3: Lint + commit**

```bash
cd "/Users/kartikey/Desktop/work_products/comp_viz_code:files/comp_viz_analysis"
bash -n web/infra/bootstrap-mixinlab.sh && bash -n web/infra/deploy.sh
git add web/infra/bootstrap-mixinlab.sh web/infra/deploy.sh .github/workflows/deploy.yml
git commit -m "feat(infra): resumable CORS (POST + expose Location) + KC_SEED_ADMINS env"
```

### Task 21: Redeploy + E2E

- [ ] **Step 1: Full backend test suite green**

Run: `/Users/kartikey/kineticolor-web-venv/bin/python -m pytest web/tests/ -q`
Expected: all pass.

- [ ] **Step 2: Redeploy** (Docker running):
```bash
KC_OAUTH_CLIENT_ID="1000658814221-tcltg00v1dcifqn4og75iiq99udh71hh.apps.googleusercontent.com" bash web/infra/deploy.sh
```
Then re-apply resumable CORS by running the Phase-3 CORS block from `bootstrap-mixinlab.sh` (or rerun the bootstrap), so `POST` + `Location` are allowed.

- [ ] **Step 3: E2E checklist** (Incognito, signed in as `kartikey.attri@lemnisca.bio`):
  - Upload screen: drag a 60 MB+ video → resumable upload completes (no `ERR_HTTP2_PING_FAILED`; survives a brief network blip).
  - Select screen: the video appears under today's date; check it → Run analysis.
  - Status screen: run shows running → completed.
  - Results screen: ΔE graph + 0.90/0.95/0.99 + hidden metrics + tooltips.
  - Profile: shows the run + upload; admin panel lists users.
  - Sign in with a **non-admin** lemnisca.bio account → sees "Access pending"; as admin, Grant it `runner` → that user can now use the app.
Expected: all pass.

---

## Carried-over guardrails
- Per-owner scoping on `/api/videos`, `/api/runs`, `/api/me/*`; admins use `/api/admin/*` to see across users (NEW_GCP model).
- `safe_video_name` rejects traversal/non-video (F-009); finalize checks the object exists + belongs to caller (F-001/F-010); run capped at `MAX_TASKS` (F-004); worker `--task-timeout`/`--max-retries 0` (F-003/F-006); shell `set -euo pipefail` (F-014); sign-in token in `sessionStorage` (F-021).
- Seed admins cannot be demoted/disabled; admins cannot demote/disable themselves (NEW_GCP guards).

---

## Self-Review

**Spec coverage:** (1) Upload screen — profile header + drag-drop + upload button: Task 16 (header), Task 17 (UploadView) ✓. (2) Select screen — date → videos, checkboxes, Run: Task 17 (SelectView) ✓; simplified storage = Firestore `kc_videos` grouped by date, no GCS tree (Tasks 4, 10) ✓. (3) Status screen — running/completed: Task 18 (StatusView) ✓. (4) Results screen — ΔE + levels + metrics: Task 18 (ResultsView) ✓. (5) Profile — history runs + uploads, admin/user, two hardcoded admins, grant access: Tasks 1 (seed admins), 2–3 (RBAC), 12 (admin routes), 18 (ProfileView + AdminUsers) ✓. Better upload strategy — resumable chunked: Tasks 5, 14, 20 ✓. "Same logic as NEW_GCP" RBAC — copied verbatim (users.py, rbac.py, routes_admin.py) ✓.

**Placeholder scan:** Task 10 Step 1 deliberately shows a wrong-first-cut then the corrected finalize — the corrected code is complete; ensure only the corrected version is implemented. No TODO/TBD elsewhere.

**Type consistency:** Per-video keys (`idx, video_id, filename, status, duration_s, t_mix_90/95/99_s, error`) match across `new_run_record`, worker `_set_video`, `RunStatus/VideoStatus` schema, `routes_runs._to_status`, and the React `VideoStatus`. `initiate_url` (snake) in API ↔ `initiate_url` in `UploadTarget` (TS) ↔ `signed_resumable_initiate_url`. `RUN_ID` env consistent across runner (Task 8) + worker (Task 7). `kc_users/kc_videos/kc_runs` collection names consistent. Results paths `runs/<run_id>/results/<video_id>.json` match between worker (Task 7) and `routes_runs.result_url` (Task 11).

**One verification dependency:** Task 7 changes the worker's I/O; its analysis core (engine + levels + results_doc) is unchanged and already covered by `test_worker.py` (levels/results_doc tests stay green). The resumable session semantics (308/Range/Location) in Task 14 can only be fully validated against real GCS in the Task 21 E2E.
