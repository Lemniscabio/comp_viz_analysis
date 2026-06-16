import web.backend.config as c


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("KC_PROJECT", "proj-x")
    monkeypatch.setenv("KC_BUCKET", "bucket-x")
    monkeypatch.setenv("KC_OAUTH_CLIENT_ID", "cid")
    monkeypatch.setenv("KC_ALLOWED_DOMAIN", "lemnisca.bio")
    s = c.Settings.from_env()
    assert s.project == "proj-x" and s.bucket == "bucket-x"
    assert s.allowed_domain == "lemnisca.bio"


def test_seed_admins_default_and_override(monkeypatch):
    monkeypatch.delenv("KC_SEED_ADMINS", raising=False)
    import importlib, web.backend.config as c
    importlib.reload(c)
    s = c.Settings.from_env()
    assert "kartikey.attri@lemnisca.bio" in s.seed_admins
    assert "laalchand.kumawat@lemnisca.bio" in s.seed_admins
    monkeypatch.setenv("KC_SEED_ADMINS", "A@x.bio, b@x.bio")
    s2 = c.Settings.from_env()
    assert s2.seed_admins == ["a@x.bio", "b@x.bio"]  # lowercased, trimmed
