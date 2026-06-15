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
