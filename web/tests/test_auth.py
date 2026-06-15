import pytest
import web.backend.auth as a


def test_accepts_domain_account():
    u = a.user_from_idinfo({"email": "x@lemnisca.bio", "email_verified": True,
                            "hd": "lemnisca.bio", "sub": "1"}, "lemnisca.bio")
    assert u.email == "x@lemnisca.bio"


def test_rejects_unverified():
    with pytest.raises(PermissionError):
        a.user_from_idinfo({"email": "x@lemnisca.bio", "email_verified": False,
                            "hd": "lemnisca.bio"}, "lemnisca.bio")


def test_rejects_wrong_domain():
    with pytest.raises(PermissionError):
        a.user_from_idinfo({"email": "x@gmail.com", "email_verified": True, "hd": None},
                           "lemnisca.bio")
