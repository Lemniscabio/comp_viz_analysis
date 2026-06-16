import pytest
import web.backend.runner as r


def test_build_overrides_sets_run_id_env():
    ov = r.build_overrides("run1", "b", 3)
    env = {e["name"]: e["value"] for e in ov["container_overrides"][0]["env"]}
    assert env["RUN_ID"] == "run1" and env["BUCKET"] == "b"


def test_build_overrides_caps_video_count():
    with pytest.raises(ValueError):
        r.build_overrides("J1", "b", r.MAX_TASKS + 1)
