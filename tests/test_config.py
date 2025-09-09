from pathlib import Path

import pytest
from pydantic import ValidationError

from sentinel.config import AppConfig, Settings, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_loads():
    config = load_config(ROOT / "configs" / "default.yaml")
    assert config.detector.input_size == 416
    assert config.source.kind == "camera"
    assert config.logging.backup_count == 5


def test_missing_sections_fall_back_to_defaults(tmp_path):
    path = tmp_path / "partial.yaml"
    path.write_text("detector:\n  conf_threshold: 0.5\n", encoding="utf-8")
    config = load_config(path)
    assert config.detector.conf_threshold == 0.5
    assert config.detector.iou_threshold == 0.45
    assert config.source.queue_size == 2


def test_empty_file_gives_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == AppConfig()


def test_threshold_out_of_range_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("detector:\n  conf_threshold: 1.5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("SENTINEL_LLM_MODEL", "claude-test")
    monkeypatch.setenv("SENTINEL_LLM_TEMPERATURE", "0.2")
    settings = Settings()
    assert settings.llm_model == "claude-test"
    assert settings.llm_temperature == 0.2
