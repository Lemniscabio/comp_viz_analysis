import datetime as dt
import pytest
from fastapi import HTTPException
from web.backend.users import UserRecord
from web.backend.rbac import _enforce

NOW = dt.datetime(2026, 6, 16, tzinfo=dt.timezone.utc)


def rec(role, status):
    return UserRecord(email="u@lemnisca.bio", role=role, status=status, requested_at=NOW)


def test_active_passes_for_any_active():
    assert _enforce(rec("viewer", "active"), "active").role == "viewer"


def test_inactive_blocked():
    with pytest.raises(HTTPException) as e:
        _enforce(rec("admin", "pending"), "active")
    assert e.value.status_code == 403


def test_runner_requires_runner_or_admin():
    assert _enforce(rec("runner", "active"), "runner")
    assert _enforce(rec("admin", "active"), "runner")
    with pytest.raises(HTTPException):
        _enforce(rec("viewer", "active"), "runner")


def test_admin_requires_admin():
    assert _enforce(rec("admin", "active"), "admin")
    with pytest.raises(HTTPException):
        _enforce(rec("runner", "active"), "admin")
