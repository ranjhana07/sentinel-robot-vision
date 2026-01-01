from datetime import datetime, timezone

from sentinel.rag.parsing import LogRecord
from sentinel.rag.stats import compute_stats, render_stats


def rec(hour, minute, level, event):
    return LogRecord(datetime(2025, 11, 4, hour, minute, tzinfo=timezone.utc), level, event, event)


RECORDS = [
    rec(9, 0, "info", "startup"),
    rec(9, 5, "info", "fps"),
    rec(9, 10, "info", "fps"),
    rec(10, 15, "error", "tts_failure"),
    rec(10, 16, "error", "tts_failure"),
    rec(10, 20, "warning", "disk_low"),
    rec(11, 0, "error", "camera_read_failed"),
]


def test_counts_levels_and_events():
    stats = compute_stats(RECORDS)
    assert stats.total == 7
    assert stats.by_level == {"info": 3, "error": 3, "warning": 1}
    assert stats.top_events[0] == ("fps", 2)


def test_problem_events_list_errors_first_by_frequency():
    stats = compute_stats(RECORDS)
    assert stats.problem_events == [
        ("error", "tts_failure", 2),
        ("error", "camera_read_failed", 1),
        ("warning", "disk_low", 1),
    ]


def test_range_is_half_open():
    since = datetime(2025, 11, 4, 10, 0, tzinfo=timezone.utc)
    until = datetime(2025, 11, 4, 11, 0, tzinfo=timezone.utc)
    stats = compute_stats(RECORDS, since, until)
    assert stats.total == 3
    assert stats.first.hour == 10


def test_render_includes_sections():
    text = render_stats(compute_stats(RECORDS))
    assert "records: 7" in text
    assert "[error] tts_failure: 2" in text


def test_render_empty_range():
    assert render_stats(compute_stats([])) == "no records in range\n"
