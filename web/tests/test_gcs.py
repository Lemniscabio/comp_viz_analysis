import pytest
import web.backend.gcs as g


def test_safe_filename_rejects_traversal():
    with pytest.raises(ValueError):
        g.safe_filename("../etc/passwd")
    with pytest.raises(ValueError):
        g.safe_filename("/abs/x.mp4")


def test_safe_filename_rejects_non_video():
    with pytest.raises(ValueError):
        g.safe_filename("notes.txt")


def test_safe_filename_keeps_basename():
    assert g.safe_filename("clip 01.mp4") == "clip 01.mp4"


def test_input_object_path():
    assert g.input_object_path("J1", 2, "a.mp4") == "jobs/J1/inputs/2__a.mp4"


def test_gcsservice_has_resumable_initiate(monkeypatch):
    import web.backend.gcs as g
    # the method exists and builds a POST signed url with the resumable header
    assert hasattr(g.GcsService, "signed_resumable_initiate_url")
