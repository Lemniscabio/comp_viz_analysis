import datetime as dt
from web.backend.videos import VideoRecord
from web.backend.runs import new_run_record, RunRecord

NOW = dt.datetime(2026, 6, 16, tzinfo=dt.timezone.utc)


def vid(i):
    return VideoRecord(video_id=f"v{i}", owner_email="u@lemnisca.bio", filename=f"c{i}.mp4",
                       date="2026-06-16", gcs_path=f"uploads/u@lemnisca.bio/2026-06-16/v{i}__c{i}.mp4",
                       size_bytes=10, uploaded_at=NOW, status="uploaded")


def test_new_run_record_builds_manifest_and_video_status():
    rec = new_run_record("run1", "u@lemnisca.bio", [vid(0), vid(1)], NOW)
    assert rec.run_id == "run1" and rec.status == "submitted" and rec.video_count == 2
    assert rec.videos[0] == {
        "idx": 0, "video_id": "v0", "filename": "c0.mp4",
        "object_path": "uploads/u@lemnisca.bio/2026-06-16/v0__c0.mp4",
        "status": "pending", "duration_s": None,
        "t_mix_90_s": None, "t_mix_95_s": None, "t_mix_99_s": None, "error": None}
