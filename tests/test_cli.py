from pathlib import Path

import yaml
from typer.testing import CliRunner

from sentinel.cli import app

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_config_show_applies_overrides():
    result = runner.invoke(
        app,
        ["config", "show", "--config", str(ROOT / "configs" / "default.yaml"), "--set", "detector.conf_threshold=0.7"],
    )
    assert result.exit_code == 0
    assert yaml.safe_load(result.stdout)["detector"]["conf_threshold"] == 0.7


def test_config_check_accepts_default_file():
    result = runner.invoke(app, ["config", "check", "--config", str(ROOT / "configs" / "default.yaml")])
    assert result.exit_code == 0
    assert "is valid" in result.stdout


def test_config_check_fails_on_invalid_values(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("tracker:\n  min_hits: 0\n", encoding="utf-8")
    result = runner.invoke(app, ["config", "check", "--config", str(path)])
    assert result.exit_code == 1


def test_quantize_static_requires_calibration_dir(tmp_path):
    result = runner.invoke(app, ["quantize", "model.onnx", "--output", str(tmp_path / "out.onnx")])
    assert result.exit_code != 0


def test_logs_stats_reports_problem_events():
    result = runner.invoke(
        app,
        ["logs", "stats", str(ROOT / "examples" / "sample_robot.jsonl"), "--since", "2025-11-04T10:00:00Z", "--until", "2025-11-04T10:30:00Z"],
    )
    assert result.exit_code == 0
    assert "[error] tts_failure: 3" in result.stdout
