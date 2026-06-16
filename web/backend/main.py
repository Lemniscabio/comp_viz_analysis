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
