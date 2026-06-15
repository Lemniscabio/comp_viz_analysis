from __future__ import annotations
import datetime as dt
from pathlib import PurePosixPath

import google.auth
import google.auth.transport.requests
from google.cloud import storage

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def safe_filename(name: str) -> str:
    p = PurePosixPath(name)
    if name != p.name or name in (".", "..") or name.startswith("/"):
        raise ValueError(f"unsafe filename: {name!r}")
    if p.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"unsupported video type: {name!r}")
    return name


def input_object_path(job_id: str, idx: int, filename: str) -> str:
    return f"jobs/{job_id}/inputs/{idx}__{filename}"


class GcsService:
    def __init__(self, bucket_name: str, signer_email: str):
        self._signer_email = signer_email
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)

    def _token(self) -> str:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(google.auth.transport.requests.Request())
        return creds.token

    def _signed(self, object_path: str, method: str) -> str:
        return self._bucket.blob(object_path).generate_signed_url(
            version="v4", expiration=dt.timedelta(minutes=30), method=method,
            service_account_email=self._signer_email, access_token=self._token())

    def signed_put_url(self, object_path: str) -> str:
        return self._signed(object_path, "PUT")

    def signed_get_url(self, object_path: str) -> str:
        return self._signed(object_path, "GET")

    def upload_json(self, object_path: str, data: bytes) -> None:
        self._bucket.blob(object_path).upload_from_string(data, content_type="application/json")

    def exists(self, object_path: str) -> bool:
        return self._bucket.blob(object_path).exists()
