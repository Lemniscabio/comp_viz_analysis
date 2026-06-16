from __future__ import annotations
import dataclasses, datetime
from fastapi import APIRouter, Depends, HTTPException
from web.backend.users import ROLES, STATUSES
from web.backend.schemas import SetUserReq


def build_admin_router(require_admin, get_user_repo, get_run_repo, get_video_repo, settings):
    router = APIRouter(prefix="/api")

    @router.get("/admin/users")
    def list_users(account=Depends(require_admin), repo=Depends(get_user_repo)):
        return {"users": [dataclasses.asdict(u) for u in repo.list_all()]}

    @router.post("/admin/users/{email}")
    def set_user(email: str, req: SetUserReq, account=Depends(require_admin),
                 repo=Depends(get_user_repo)):
        caller = account[0]
        email = email.lower()
        if req.role is not None and req.role not in ROLES:
            raise HTTPException(400, f"bad role {req.role}")
        if req.status is not None and req.status not in STATUSES:
            raise HTTPException(400, f"bad status {req.status}")
        target = repo.get(email)
        if target is None:
            raise HTTPException(404, "unknown user")
        demote = req.role is not None and req.role != "admin"
        disable = req.status == "disabled"
        if email == caller.email.lower() and (disable or demote):
            raise HTTPException(400, "cannot disable or demote yourself")
        if email in settings.seed_admins and (disable or demote):
            raise HTTPException(400, "cannot demote/disable a seed admin")
        now = datetime.datetime.now(datetime.timezone.utc)
        repo.set_decision(email, role=req.role if req.role is not None else target.role,
                          status=req.status if req.status is not None else target.status,
                          decided_by=caller.email, now=now)
        return dataclasses.asdict(repo.get(email))

    @router.get("/admin/runs")
    def admin_runs(user: str | None = None, account=Depends(require_admin),
                   rrepo=Depends(get_run_repo)):
        recs = rrepo.list_by_owner(user) if user else rrepo.list_all()
        return {"runs": recs}

    @router.get("/admin/videos")
    def admin_videos(user: str | None = None, account=Depends(require_admin),
                     vrepo=Depends(get_video_repo)):
        recs = vrepo.list_by_owner(user) if user else vrepo.list_all()
        return {"videos": [dataclasses.asdict(v) for v in recs]}

    return router
