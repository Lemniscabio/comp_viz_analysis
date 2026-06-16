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
