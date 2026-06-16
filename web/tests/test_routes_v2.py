from fastapi.testclient import TestClient
import web.backend.main as m


class FakeGcs:
    def __init__(self): self.objects={}; self.json={}
    def signed_resumable_initiate_url(self, p, content_type="application/octet-stream"): return f"https://init/{p}"
    def signed_get_url(self, p): return f"https://get/{p}"
    def upload_json(self, p, d): self.json[p]=d
    def object_size(self, p): return self.objects.get(p)
    def exists(self, p): return p in self.objects


class FakeVideos:
    def __init__(self): self.db={}
    def create(self, r): self.db[r.video_id]=r
    def get(self, v): return self.db.get(v)
    def list_by_owner(self, e): return [r for r in self.db.values() if r.owner_email==e.lower()]
    def list_all(self): return list(self.db.values())


class FakeRuns:
    def __init__(self): self.db={}
    def create(self, r):
        import dataclasses; self.db[r.run_id]=dataclasses.asdict(r)
    def get(self, r): return self.db.get(r)
    def list_by_owner(self, e): return [x for x in self.db.values() if x["owner_email"]==e.lower()]
    def list_all(self): return list(self.db.values())


class FakeRunner:
    def __init__(self): self.triggered=[]
    def trigger(self, run_id, bucket, n): self.triggered.append((run_id, n))


class FakeUsers:
    # current_account depends on get_user_repo; stub it so tests never build a
    # real firestore.Client() (CI has no GCP credentials).
    def get(self, e): return None
    def upsert(self, r): pass
    def list_all(self): return []
    def set_decision(self, *a, **k): pass


def client():
    app = m.create_app(dev_no_auth=True)
    g,v,r,rn = FakeGcs(),FakeVideos(),FakeRuns(),FakeRunner()
    app.dependency_overrides[m.get_gcs]=lambda: g
    app.dependency_overrides[m.get_video_repo]=lambda: v
    app.dependency_overrides[m.get_run_repo]=lambda: r
    app.dependency_overrides[m.get_runner]=lambda: rn
    app.dependency_overrides[m.get_user_repo]=lambda: FakeUsers()
    return TestClient(app), g, v, r, rn


def test_me_is_admin_in_dev():
    c,*_ = client()
    assert c.get("/api/me").json()["role"] == "admin"


def test_allocate_returns_initiate_urls():
    c,g,v,r,rn = client()
    body = c.post("/api/videos:allocate", json={"files":[{"name":"a.mp4","size":10}]}).json()
    assert body["uploads"][0]["initiate_url"].startswith("https://init/")


def test_finalize_then_run_triggers_worker():
    c,g,v,r,rn = client()
    a = c.post("/api/videos:allocate", json={"files":[{"name":"a.mp4","size":10}]}).json()
    u = a["uploads"][0]
    g.objects[u["object_path"]] = 10  # simulate uploaded bytes
    fin = c.post(f"/api/videos/{u['video_id']}:finalize",
                 json={"video_id":u["video_id"],"filename":"a.mp4",
                       "object_path":u["object_path"],"size_bytes":10})
    assert fin.status_code == 200
    run = c.post("/api/runs", json={"video_ids":[u["video_id"]]}).json()
    assert rn.triggered == [(run["run_id"], 1)]
    assert run["status"] == "submitted"


def test_finalize_rejects_foreign_path():
    c,g,v,r,rn = client()
    assert c.post("/api/videos/x:finalize",
                  json={"video_id":"x","filename":"a.mp4",
                        "object_path":"uploads/other@lemnisca.bio/2026-06-16/x__a.mp4",
                        "size_bytes":1}).status_code == 400
