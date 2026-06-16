import datetime as dt
from web.backend.users import UserRecord, ROLES, STATUSES, resolve_on_login

NOW = dt.datetime(2026, 6, 16, tzinfo=dt.timezone.utc)


def test_seed_admin_becomes_admin_active():
    rec = resolve_on_login("Kartikey.Attri@lemnisca.bio", ["kartikey.attri@lemnisca.bio"], None, NOW)
    assert rec.email == "kartikey.attri@lemnisca.bio"
    assert rec.role == "admin" and rec.status == "active" and rec.decided_by == "seed"


def test_new_user_is_pending():
    rec = resolve_on_login("new@lemnisca.bio", ["a@lemnisca.bio"], None, NOW)
    assert rec.role is None and rec.status == "pending"


def test_existing_non_admin_unchanged():
    existing = UserRecord(email="u@lemnisca.bio", role="runner", status="active", requested_at=NOW)
    rec = resolve_on_login("u@lemnisca.bio", ["a@lemnisca.bio"], existing, NOW)
    assert rec is existing


def test_seed_admin_idempotent():
    existing = UserRecord(email="a@lemnisca.bio", role="admin", status="active", requested_at=NOW,
                          decided_by="seed", decided_at=NOW)
    rec = resolve_on_login("a@lemnisca.bio", ["a@lemnisca.bio"], existing, NOW)
    assert rec is existing


def test_role_status_vocab():
    assert ROLES == {"admin", "runner", "viewer"}
    assert STATUSES == {"pending", "active", "disabled"}
