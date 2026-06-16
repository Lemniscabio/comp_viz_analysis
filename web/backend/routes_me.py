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
