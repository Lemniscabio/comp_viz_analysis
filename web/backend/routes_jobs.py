from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from web.backend.auth import User
from web.backend.gcs import safe_filename
from web.backend.firestore_store import new_job_record
from web.backend.schemas import (AllocateReq, AllocateResp, UploadTarget, SubmitReq,
                                 JobStatus, VideoStatus)


def build_router(get_gcs, get_store, get_runner, current_user, settings):
    router = APIRouter(prefix="/api")

    @router.post("/jobs:allocate", response_model=AllocateResp)
    def allocate(req: AllocateReq, user: User = Depends(current_user),
                 gcs=Depends(get_gcs), store=Depends(get_store)):
        if not req.files:
            raise HTTPException(400, "no files")
        try:
            files = [safe_filename(f) for f in req.files]
        except ValueError as e:
            raise HTTPException(400, str(e))
        job_id = uuid.uuid4().hex[:12]
        record = new_job_record(job_id, user.email, files,
                                datetime.now(timezone.utc).isoformat())
        gcs.upload_json(f"jobs/{job_id}/manifest.json",
                        json.dumps({"videos": record["videos"]}).encode())
        store.create(record)
        uploads = [UploadTarget(idx=v["idx"], filename=v["filename"],
                                object_path=v["object_path"],
                                url=gcs.signed_put_url(v["object_path"]))
                   for v in record["videos"]]
        return AllocateResp(job_id=job_id, uploads=uploads)

    @router.post("/jobs:submit", response_model=JobStatus)
    def submit(req: SubmitReq, user: User = Depends(current_user),
               gcs=Depends(get_gcs), store=Depends(get_store), runner=Depends(get_runner)):
        rec = store.get(req.job_id)
        if not rec or rec["owner_email"] != user.email:
            raise HTTPException(404, "job not found")
        missing = [v["filename"] for v in rec["videos"] if not gcs.exists(v["object_path"])]
        if missing:
            raise HTTPException(400, f"inputs not uploaded: {missing}")
        runner.trigger(req.job_id, settings.bucket, rec["video_count"])
        store.set_status(req.job_id, "submitted")
        return _status(store.get(req.job_id))

    @router.get("/jobs/{job_id}", response_model=JobStatus)
    def status(job_id: str, user: User = Depends(current_user), store=Depends(get_store)):
        rec = store.get(job_id)
        if not rec or rec["owner_email"] != user.email:
            raise HTTPException(404, "job not found")
        return _status(rec)

    @router.get("/jobs/{job_id}/result/{idx}")
    def result_url(job_id: str, idx: int, user: User = Depends(current_user),
                   gcs=Depends(get_gcs), store=Depends(get_store)):
        rec = store.get(job_id)
        if not rec or rec["owner_email"] != user.email:
            raise HTTPException(404, "job not found")
        v = next((x for x in rec["videos"] if x["idx"] == idx), None)
        if not v or v["status"] != "done":
            raise HTTPException(404, "result not ready")
        return {"url": gcs.signed_get_url(f"jobs/{job_id}/results/{idx}.json")}

    return router


def _status(rec) -> JobStatus:
    keys = ("idx", "filename", "status", "duration_s",
            "t_mix_90_s", "t_mix_95_s", "t_mix_99_s", "error")
    return JobStatus(job_id=rec["job_id"], status=rec["status"], video_count=rec["video_count"],
                     videos=[VideoStatus(**{k: v.get(k) for k in keys}) for v in rec["videos"]])
