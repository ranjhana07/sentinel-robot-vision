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


def test_zone_rules_are_parsed_from_default_yaml():
    config = load_config(ROOT / "configs" / "default.yaml")
    zone = config.speech.zones[0]
    assert zone.name == "restricted"
    assert zone.applies_to == "unknown"
    assert zone.area == (0.55, 0.0, 1.0, 1.0)


def test_zone_area_must_be_ordered(tmp_path):
    path = tmp_path / "zones.yaml"
    path.write_text(
        "speech:\n  zones:\n    - name: bad\n      area: [0.8, 0.0, 0.2, 1.0]\n      phrase: hi\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(path)


def test_quiet_hours_wrap_past_midnight():
    from datetime import time

    from sentinel.config import QuietHours

    quiet = QuietHours(start="22:00", end="06:30")
    assert quiet.contains(time(23, 15))
    assert quiet.contains(time(2, 0))
    assert not quiet.contains(time(6, 30))
    assert not quiet.contains(time(12, 0))


def test_quiet_hours_same_day_window():
    from datetime import time

    from sentinel.config import QuietHours

    quiet = QuietHours(start="13:00", end="14:00")
    assert quiet.contains(time(13, 30))
    assert not quiet.contains(time(14, 0))


def test_quiet_hours_load_from_yaml(tmp_path):
    path = tmp_path / "quiet.yaml"
    path.write_text(
        "speech:\n  quiet_hours:\n    start: '22:00'\n    end: '06:00'\n    suppress: [greeting, unknown]\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.speech.quiet_hours.suppress == ["greeting", "unknown"]


def test_reconnect_defaults_and_validation(tmp_path):
    config = AppConfig()
    assert config.source.reconnect.enabled
    assert config.source.reconnect.max_attempts == 5
    path = tmp_path / "bad.yaml"
    path.write_text("source:\n  reconnect:\n    base_delay: 0\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)
