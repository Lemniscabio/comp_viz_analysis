from __future__ import annotations
import dataclasses
import datetime
from dataclasses import dataclass
from typing import Optional, Protocol

ROLES = {"admin", "runner", "viewer"}
STATUSES = {"pending", "active", "disabled"}


@dataclass
class UserRecord:
    email: str
    role: Optional[str]
    status: str
    requested_at: datetime.datetime
    decided_by: Optional[str] = None
    decided_at: Optional[datetime.datetime] = None


def resolve_on_login(email: str, seed_admins: list[str],
                     existing: Optional[UserRecord], now: datetime.datetime) -> UserRecord:
    email = email.lower()
    if email in seed_admins:
        if existing and existing.role == "admin" and existing.status == "active":
            return existing
        return UserRecord(email=email, role="admin", status="active",
                          requested_at=(existing.requested_at if existing else now),
                          decided_by="seed", decided_at=now)
    if existing is None:
        return UserRecord(email=email, role=None, status="pending", requested_at=now)
    return existing


class UserRepository(Protocol):
    def get(self, email: str) -> Optional[UserRecord]: ...
    def upsert(self, record: UserRecord) -> None: ...
    def list_all(self) -> list[UserRecord]: ...
    def set_decision(self, email: str, role: Optional[str], status: str,
                     decided_by: str, now: datetime.datetime) -> None: ...


class FirestoreUserRepository:
    COLLECTION = "kc_users"

    def __init__(self, client, collection: str = COLLECTION) -> None:
        self._c = client
        self._col = collection

    def _doc(self, email: str):
        return self._c.collection(self._col).document(email.lower())

    def _read(self, d: dict) -> UserRecord:
        return UserRecord(email=d["email"], role=d.get("role"),
                          status=d.get("status", "pending"), requested_at=d.get("requested_at"),
                          decided_by=d.get("decided_by"), decided_at=d.get("decided_at"))

    def get(self, email):
        snap = self._doc(email).get()
        return self._read(snap.to_dict()) if snap.exists else None

    def upsert(self, record):
        self._doc(record.email).set({
            "email": record.email.lower(), "role": record.role, "status": record.status,
            "requested_at": record.requested_at, "decided_by": record.decided_by,
            "decided_at": record.decided_at}, merge=True)

    def list_all(self):
        out = [self._read(s.to_dict()) for s in self._c.collection(self._col).stream()]
        return sorted(out, key=lambda u: u.email)

    def set_decision(self, email, role, status, decided_by, now):
        self._doc(email).set({"role": role, "status": status,
                              "decided_by": decided_by, "decided_at": now}, merge=True)
