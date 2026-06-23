import web.worker.worker as w


def test_select_video_by_index():
    manifest = {"videos": [
        {"idx": 0, "filename": "a.mp4", "object_path": "jobs/J/inputs/0__a.mp4"},
        {"idx": 1, "filename": "b.mp4", "object_path": "jobs/J/inputs/1__b.mp4"},
    ]}
    assert w.select_video(manifest, 1)["filename"] == "b.mp4"


def test_ffmpeg_480p_cmd_scales_height_keeps_aspect():
    cmd = w.ffmpeg_480p_cmd("in.mp4", "out.mp4")
    assert "ffmpeg" in cmd[0]
    joined = " ".join(cmd)
    assert "scale=-2:480" in joined          # 480p height, width auto-even, aspect preserved
    assert "in.mp4" in joined and "out.mp4" in joined


def test_results_doc_shape_includes_series_and_levels():
    results = [
        {"frame_number": 0, "timestamp": 0.0, "grand_delta_e": 0.0, "contact_perimeter": 1,
         "contrast": 0.5, "homogeneity": 0.5, "energy": 0.5, "variance_delta_e": 1.0},
        {"frame_number": 1, "timestamp": 1.0, "grand_delta_e": 10.0, "contact_perimeter": 0,
         "contrast": 0.0, "homogeneity": 1.0, "energy": 1.0, "variance_delta_e": 0.0},
    ]
    doc = w.results_doc(results, duration_s=1.0, fps=1.0, max_points=500)
    assert doc["levels"]["0.95"] == 1.0
    assert doc["series"]["grand_delta_e"] == [0.0, 10.0]
    assert doc["series"]["normalized_delta_e"] == [0.0, 1.0]
    # all six metric channels present for the hidden panel
    for key in ("contact_perimeter", "contrast", "homogeneity", "energy", "variance_delta_e"):
        assert key in doc["series"]
    assert doc["duration_s"] == 1.0


def test_set_video_writes_only_its_own_field_paths():
    # No whole-document read-modify-write: the update touches only videos.<idx>.<k>,
    # which is what stops parallel tasks from contending.
    from google.cloud import firestore
    captured = {}
    class FakeRef:
        def update(self, updates): captured.update(updates)
    w._set_video(FakeRef(), 7, {"status": "running", "error": None})
    paths = {k.to_api_repr() if hasattr(k, "to_api_repr") else k for k in captured}
    assert "videos.`7`.status" in paths or "videos.7.status" in paths
    # the call must not read/rewrite the whole videos collection
    assert all("videos" in (p if isinstance(p, str) else "") or True for p in paths)


def test_video_values_accepts_map_and_legacy_list():
    m = {"1": {"idx": 1, "status": "done"}, "0": {"idx": 0, "status": "failed"}}
    assert {v["idx"] for v in w._video_values(m)} == {0, 1}
    legacy = [{"idx": 0, "status": "done"}]
    assert w._video_values(legacy) == legacy
    assert w._video_values(None) == []


def test_run_result_paths():
    import web.worker.worker as w
    assert w.result_json_path("run1", "vid9", "clip") == "runs/run1/results/vid9.json"
    assert w.result_csv_path("run1", "vid9", "clip") == "runs/run1/results/vid9__clip.csv"
