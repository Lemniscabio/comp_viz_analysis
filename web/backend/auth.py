from __future__ import annotations
from dataclasses import dataclass

from fastapi import Header, HTTPException
from google.auth.transport import requests as g_requests
from google.oauth2 import id_token

from web.backend.config import Settings


@dataclass(frozen=True)
class User:
    email: str
    sub: str


def user_from_idinfo(idinfo: dict, allowed_domain: str) -> User:
    if not idinfo.get("email_verified"):
        raise PermissionError("email not verified")
    if idinfo.get("hd") != allowed_domain:
        raise PermissionError(f"not a {allowed_domain} Workspace account (hd={idinfo.get('hd')!r})")
    return User(email=idinfo.get("email", ""), sub=idinfo.get("sub", ""))


def make_auth_dependency(settings: Settings):
    def current_user(authorization: str = Header(default="")) -> User:
        if settings.dev_no_auth:
            return User(email="dev@lemnisca.bio", sub="dev")
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        token = authorization.split(" ", 1)[1]
        try:
            idinfo = id_token.verify_oauth2_token(token, g_requests.Request(),
                                                  settings.oauth_client_id)
            return user_from_idinfo(idinfo, settings.allowed_domain)
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(401, f"invalid token: {e}")

    return current_user
