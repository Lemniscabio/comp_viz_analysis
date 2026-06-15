from __future__ import annotations
from typing import Any, Dict, List

from web.backend.gcs import input_object_path


def new_job_record(job_id: str, owner_email: str, files: List[str], created_at: str) -> Dict[str, Any]:
    videos = [
        {"idx": i, "filename": fn, "object_path": input_object_path(job_id, i, fn),
         "status": "pending", "duration_s": None,
         "t_mix_90_s": None, "t_mix_95_s": None, "t_mix_99_s": None, "error": None}
        for i, fn in enumerate(files)
    ]
    return {"job_id": job_id, "owner_email": owner_email, "created_at": created_at,
            "status": "allocated", "video_count": len(files), "videos": videos}


class FirestoreStore:
    def __init__(self):
        from google.cloud import firestore
        self._col = firestore.Client().collection("jobs")

    def create(self, record: Dict[str, Any]) -> None:
        self._col.document(record["job_id"]).set(record)

    def get(self, job_id: str) -> Dict[str, Any] | None:
        snap = self._col.document(job_id).get()
        return snap.to_dict() if snap.exists else None

    def set_status(self, job_id: str, status: str) -> None:
        self._col.document(job_id).update({"status": status})

    def list_for_owner(self, owner_email: str) -> List[Dict[str, Any]]:
        return [d.to_dict() for d in self._col.where("owner_email", "==", owner_email).stream()]
