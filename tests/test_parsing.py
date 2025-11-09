from datetime import datetime, timezone

import pytest

from sentinel.rag.parsing import parse_json_line, parse_line, parse_text_line, parse_timestamp, read_records


def test_parse_timestamp_variants():
    expected = datetime(2025, 11, 4, 9, 12, 41, tzinfo=timezone.utc)
    assert parse_timestamp("2025-11-04T09:12:41Z") == expected
    assert parse_timestamp("2025-11-04 09:12:41") == expected
    assert parse_timestamp("2025-11-04T14:42:41+05:30") == expected
    assert parse_timestamp("2025-11-04 09:12:41,250").microsecond == 250000


def test_parse_json_line_builds_message_from_fields():
    line = '{"ts":"2025-11-04T09:12:41.000+00:00","level":"warning","event":"fps_drop","robot":"r1","fps":4.1}'
    record = parse_json_line(line)
    assert record.level == "warning"
    assert record.event == "fps_drop"
    assert record.message == "fps_drop fps=4.1"


@pytest.mark.parametrize("line", ["{broken", '{"level":"info"}', '{"ts":"not a date"}', "[1, 2]"])
def test_parse_json_line_rejects_bad_input(line):
    assert parse_json_line(line) is None


def test_parse_text_line():
    record = parse_text_line("2025-11-04 10:15:03 ERROR tts_failure backend=pyttsx3")
    assert record.level == "error"
    assert record.event == "tts_failure"
    assert record.message == "tts_failure backend=pyttsx3"


def test_level_aliases_are_normalised():
    assert parse_text_line("2025-11-04 10:15:03 WARN something odd").level == "warning"
    assert parse_text_line("2025-11-04 10:15:03 CRITICAL boom").level == "error"


def test_parse_line_dispatches_and_skips_blanks():
    assert parse_line("   ") is None
    assert parse_line("plain text with no timestamp") is None
    assert parse_line('{"ts":"2025-11-04T09:00:00Z","event":"startup"}').event == "startup"
    assert parse_line("2025-11-04 09:00:00 INFO startup").event == "startup"


def test_read_records_skips_garbage(tmp_path):
    path = tmp_path / "mixed.log"
    path.write_text(
        '{"ts":"2025-11-04T09:00:00Z","event":"startup"}\n'
        "garbage line\n"
        "2025-11-04 09:05:00 INFO camera_open source=0\n",
        encoding="utf-8",
    )
    events = [record.event for record in read_records(path)]
    assert events == ["startup", "camera_open"]
