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

    return router
