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
