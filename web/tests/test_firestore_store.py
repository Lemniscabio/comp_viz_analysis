import web.backend.firestore_store as fsx


def test_new_job_record_shape():
    rec = fsx.new_job_record("J1", "x@lemnisca.bio", ["a.mp4", "b.mp4"], "2026-06-15T00:00:00Z")
    assert rec["status"] == "allocated" and rec["video_count"] == 2
    assert rec["videos"][0] == {
        "idx": 0, "filename": "a.mp4", "object_path": "jobs/J1/inputs/0__a.mp4",
        "status": "pending", "duration_s": None,
        "t_mix_90_s": None, "t_mix_95_s": None, "t_mix_99_s": None, "error": None,
    }
