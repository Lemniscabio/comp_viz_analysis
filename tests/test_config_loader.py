"""Tests for config loader."""

import pytest
from src.utils.config_loader import load_config, DEFAULT_CONFIG


class TestLoadConfig:
    def test_load_defaults(self):
        """Loading with no file returns all default values."""
        config = load_config()
        assert config["frame_skip"] == 1
        assert config["glcm_frame_skip"] == 1
        assert config["grid_rows"] == 5
        assert config["grid_cols"] == 5
        assert config["glcm_gray_levels"] == 16
        assert config["glcm_offset"] == [1, 1]
        assert config["contact_threshold"] == 128
        assert config["camera_index"] == 0
        assert config["video_fps_override"] is None
        assert config["export_format"] == "csv"
        assert config["brightness_change_threshold"] == 0.2

    def test_load_from_yaml(self, tmp_path):
        """Loading from a YAML file overrides defaults."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("frame_skip: 3\ngrid_rows: 10\n")
        config = load_config(yaml_file)
        assert config["frame_skip"] == 3
        assert config["grid_rows"] == 10
        assert config["grid_cols"] == 5  # unchanged default

    def test_invalid_frame_skip(self, tmp_path):
        """frame_skip < 1 raises ValueError."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("frame_skip: 0\n")
        with pytest.raises(ValueError, match="frame_skip"):
            load_config(yaml_file)

    def test_invalid_contact_threshold(self, tmp_path):
        """contact_threshold outside 0-255 raises ValueError."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("contact_threshold: 300\n")
        with pytest.raises(ValueError, match="contact_threshold"):
            load_config(yaml_file)

    def test_invalid_export_format(self, tmp_path):
        """export_format not csv or xlsx raises ValueError."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("export_format: json\n")
        with pytest.raises(ValueError, match="export_format"):
            load_config(yaml_file)
