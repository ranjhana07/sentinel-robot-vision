import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("generate_sample_logs", ROOT / "scripts" / "generate_sample_logs.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def test_generation_is_deterministic_and_sorted():
    first = generator.build(11)
    assert first == generator.build(11)
    stamps = [item["ts"] for item in first]
    assert stamps == sorted(stamps)


def test_injected_incidents_are_present():
    events = {item["event"] for item in generator.build(3)}
    assert {"tts_failure", "zone_violation", "camera_reconnect", "disk_low", "thermal_throttle"} <= events


def test_committed_sample_matches_generator():
    import json

    committed = [json.loads(line) for line in (ROOT / "examples" / "sample_robot.jsonl").read_text(encoding="utf-8").splitlines()]
    assert committed == generator.build(11)
