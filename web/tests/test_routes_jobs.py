from fastapi.testclient import TestClient
import web.backend.main as m


class FakeGcs:
    def __init__(self): self.objects = set(); self.json = {}
    def signed_put_url(self, p): return f"https://put/{p}"
    def signed_get_url(self, p): return f"https://get/{p}"
    def upload_json(self, p, d): self.json[p] = d
    def exists(self, p): return p in self.objects


class FakeStore:
    def __init__(self): self.db = {}
    def create(self, rec): self.db[rec["job_id"]] = rec
    def get(self, jid): return self.db.get(jid)
    def set_status(self, jid, s): self.db[jid]["status"] = s
    def list_for_owner(self, e): return [r for r in self.db.values() if r["owner_email"] == e]


class FakeRunner:
    def __init__(self): self.triggered = []
    def trigger(self, jid, bucket, n): self.triggered.append((jid, n)); return "exec-1"


def make_client():
    app = m.create_app(dev_no_auth=True)
    gcs, store, runner = FakeGcs(), FakeStore(), FakeRunner()
    app.dependency_overrides[m.get_gcs] = lambda: gcs
    app.dependency_overrides[m.get_store] = lambda: store
    app.dependency_overrides[m.get_runner] = lambda: runner
    return TestClient(app), gcs, store, runner


def test_allocate_returns_urls_and_record():
    client, gcs, store, _ = make_client()
    body = client.post("/api/jobs:allocate", json={"files": ["a.mp4", "b.mp4"]}).json()
    assert len(body["uploads"]) == 2
    assert body["uploads"][0]["url"].startswith("https://put/")
    assert store.get(body["job_id"])["video_count"] == 2


def test_submit_fails_if_inputs_missing():
    client, gcs, store, runner = make_client()
    jid = client.post("/api/jobs:allocate", json={"files": ["a.mp4"]}).json()["job_id"]
    assert client.post("/api/jobs:submit", json={"job_id": jid}).status_code == 400


def test_submit_triggers_when_inputs_present():
    client, gcs, store, runner = make_client()
    body = client.post("/api/jobs:allocate", json={"files": ["a.mp4"]}).json()
    gcs.objects.add(body["uploads"][0]["object_path"])
    assert client.post("/api/jobs:submit", json={"job_id": body["job_id"]}).status_code == 200
    assert runner.triggered == [(body["job_id"], 1)]
    assert store.get(body["job_id"])["status"] == "submitted"


def test_rejects_unsafe_filename():
    client, *_ = make_client()
    assert client.post("/api/jobs:allocate", json={"files": ["../x.mp4"]}).status_code == 400


def test_result_url_only_when_done():
    client, gcs, store, _ = make_client()
    body = client.post("/api/jobs:allocate", json={"files": ["a.mp4"]}).json()
    jid = body["job_id"]
    # not done yet -> 404
    assert client.get(f"/api/jobs/{jid}/result/0").status_code == 404
    store.db[jid]["videos"][0]["status"] = "done"
    r = client.get(f"/api/jobs/{jid}/result/0").json()
    assert r["url"].startswith("https://get/")
