from __future__ import annotations
import datetime, json, uuid
from fastapi import APIRouter, Depends, HTTPException
from web.backend.runs import new_run_record, manifest_for, video_list
from web.backend.runner import MAX_TASKS
from web.backend.schemas import RunReq, RunStatus, VideoStatus


def build_runs_router(get_gcs, get_video_repo, get_run_repo, get_runner,
                      require_runner, require_active, settings):
    router = APIRouter(prefix="/api")

    def _can_view(rec, account) -> bool:
        return bool(rec) and (account[1].role == "admin"
                              or rec["owner_email"] == account[0].email.lower())

    @router.post("/runs", response_model=RunStatus)
    def create_run(req: RunReq, account=Depends(require_runner), gcs=Depends(get_gcs),
                   vrepo=Depends(get_video_repo), rrepo=Depends(get_run_repo),
                   runner=Depends(get_runner)):
        email = account[0].email
        ids = list(dict.fromkeys(req.video_ids))  # dedupe, keep order
        if not ids:
            raise HTTPException(400, "no videos selected")
        if len(ids) > MAX_TASKS:
            raise HTTPException(400, f"max {MAX_TASKS} videos per run")
        recs = []
        for vid in ids:
            v = vrepo.get(vid)
            if v is None or v.owner_email != email.lower():
                raise HTTPException(404, f"video not found: {vid}")
            recs.append(v)
        run_id = uuid.uuid4().hex[:12]
        now = datetime.datetime.now(datetime.timezone.utc)
        run = new_run_record(run_id, email, recs, now)
        gcs.upload_json(f"runs/{run_id}/manifest.json", json.dumps(manifest_for(run)).encode())
        rrepo.create(run)
        runner.trigger(run_id, settings.bucket, run.video_count)
        return _to_status(rrepo.get(run_id))

    @router.get("/runs/{run_id}", response_model=RunStatus)
    def get_run(run_id: str, account=Depends(require_active), rrepo=Depends(get_run_repo)):
        rec = rrepo.get(run_id)
        if not _can_view(rec, account):
            raise HTTPException(404, "run not found")
        return _to_status(_reconcile(rrepo, rec))

    @router.get("/runs")
    def list_runs(account=Depends(require_active), rrepo=Depends(get_run_repo)):
        return {"runs": [_to_status(_reconcile(rrepo, r)).model_dump()
                         for r in rrepo.list_by_owner(account[0].email)]}

    @router.get("/runs/{run_id}/result/{video_id}")
    def result_url(run_id: str, video_id: str, account=Depends(require_active),
                   gcs=Depends(get_gcs), rrepo=Depends(get_run_repo)):
        rec = rrepo.get(run_id)
        if not _can_view(rec, account):
            raise HTTPException(404, "run not found")
        v = next((x for x in rec["videos"] if x["video_id"] == video_id), None)
        if not v or v["status"] != "done":
            raise HTTPException(404, "result not ready")
        return {"url": gcs.signed_get_url(f"runs/{run_id}/results/{video_id}.json")}

    return router


def _reconcile(rrepo, rec):
    """Self-heal a run whose tasks all finished but whose status never flipped
    (e.g. a worker task crashed before it could finalize). Safe no-op otherwise."""
    if rec and rec.get("status") in ("submitted", "running"):
        vids = video_list(rec.get("videos"))
        if vids and not any(v.get("status") in ("pending", "running") for v in vids):
            new = "failed" if any(v.get("status") == "failed" for v in vids) else "done"
            try:
                rrepo.set_status(rec["run_id"], new)
            except Exception:
                pass
            rec["status"] = new
    return rec


def _to_status(rec) -> RunStatus:
    keys = ("idx", "video_id", "filename", "status", "duration_s",
            "t_mix_90_s", "t_mix_95_s", "t_mix_99_s", "error")
    return RunStatus(run_id=rec["run_id"], owner_email=rec["owner_email"],
                     status=rec["status"], video_count=rec["video_count"],
                     videos=[VideoStatus(**{k: v.get(k) for k in keys}) for v in video_list(rec.get("videos"))])
