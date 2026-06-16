from __future__ import annotations
import datetime
from fastapi import Depends, HTTPException
from web.backend.users import UserRecord, resolve_on_login


def _enforce(rec: UserRecord, need: str) -> UserRecord:
    if rec.status != "active":
        raise HTTPException(status_code=403, detail=f"access {rec.status}")
    if need == "runner" and rec.role not in ("runner", "admin"):
        raise HTTPException(status_code=403, detail="requires runner role")
    if need == "admin" and rec.role != "admin":
        raise HTTPException(status_code=403, detail="requires admin role")
    return rec


def make_rbac(current_user, get_user_repo, settings):
    """Returns (current_account, require_active, require_runner, require_admin) deps."""
    def current_account(user=Depends(current_user), repo=Depends(get_user_repo)):
        now = datetime.datetime.now(datetime.timezone.utc)
        if settings.dev_no_auth:
            return user, UserRecord(email=user.email, role="admin", status="active", requested_at=now)
        existing = repo.get(user.email)
        resolved = resolve_on_login(user.email, settings.seed_admins, existing, now)
        if resolved != existing:
            repo.upsert(resolved)
        return user, resolved

    def require_active(account=Depends(current_account)):
        _enforce(account[1], "active"); return account

    def require_runner(account=Depends(current_account)):
        _enforce(account[1], "runner"); return account

    def require_admin(account=Depends(current_account)):
        _enforce(account[1], "admin"); return account

    return current_account, require_active, require_runner, require_admin
