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
