import web.backend.config as c


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("KC_PROJECT", "proj-x")
    monkeypatch.setenv("KC_BUCKET", "bucket-x")
    monkeypatch.setenv("KC_OAUTH_CLIENT_ID", "cid")
    monkeypatch.setenv("KC_ALLOWED_DOMAIN", "lemnisca.bio")
    s = c.Settings.from_env()
    assert s.project == "proj-x" and s.bucket == "bucket-x"
    assert s.allowed_domain == "lemnisca.bio"
