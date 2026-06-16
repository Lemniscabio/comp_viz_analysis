import datetime as dt
from web.backend.videos import (VideoRecord, video_object_path, new_video_record,
                                 VIDEO_EXTS, safe_video_name)
import pytest

NOW = dt.datetime(2026, 6, 16, 9, 30, tzinfo=dt.timezone.utc)


def test_safe_video_name_rejects_bad():
    with pytest.raises(ValueError):
        safe_video_name("../x.mp4")
    with pytest.raises(ValueError):
        safe_video_name("notes.txt")
    assert safe_video_name("clip 1.mp4") == "clip 1.mp4"


def test_object_path_groups_by_email_and_date():
    p = video_object_path("u@lemnisca.bio", "vid123", "clip.mp4", NOW)
    assert p == "uploads/u@lemnisca.bio/2026-06-16/vid123__clip.mp4"


def test_new_video_record_shape():
    rec = new_video_record("vid123", "u@lemnisca.bio", "clip.mp4", 1234, NOW)
    assert rec.video_id == "vid123"
    assert rec.date == "2026-06-16"
    assert rec.gcs_path == "uploads/u@lemnisca.bio/2026-06-16/vid123__clip.mp4"
    assert rec.size_bytes == 1234 and rec.status == "uploaded"
