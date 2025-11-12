from datetime import datetime, timedelta, timezone

from sentinel.rag.chunking import chunk_by_window, format_record, split_pairs, window_start
from sentinel.rag.parsing import LogRecord


def record(minute, second=0, level="info", event="tick"):
    stamp = datetime(2025, 11, 4, 9, minute, second, tzinfo=timezone.utc)
    return LogRecord(stamp, level, event, event)


def test_window_start_aligns_to_bucket():
    stamp = datetime(2025, 11, 4, 9, 47, 12, tzinfo=timezone.utc)
    assert window_start(stamp, timedelta(minutes=10)) == datetime(2025, 11, 4, 9, 40, tzinfo=timezone.utc)


def test_records_group_by_window():
    chunks = chunk_by_window([record(1), record(9), record(11), record(35)], timedelta(minutes=10))
    assert [c.record_count for c in chunks] == [2, 1, 1]
    assert chunks[0].start.minute == 0
    assert chunks[0].end.minute == 10


def test_unsorted_input_is_ordered():
    chunks = chunk_by_window([record(8), record(2)], timedelta(minutes=10))
    assert chunks[0].text.splitlines()[0].endswith("tick")
    assert "09:02:00" in chunks[0].text.splitlines()[0]


def test_severity_counts_per_chunk():
    chunks = chunk_by_window(
        [record(1, level="error"), record(2, level="warning"), record(3, level="error")],
        timedelta(minutes=10),
    )
    assert chunks[0].error_count == 2
    assert chunks[0].warning_count == 1


def test_oversized_window_is_split_into_parts():
    records = [record(1, second=s, event="x" * 40) for s in range(30)]
    chunks = chunk_by_window(records, timedelta(minutes=10), max_chars=300)
    assert len(chunks) > 1
    assert chunks[0].chunk_id.endswith("-0")
    assert chunks[1].chunk_id.endswith("-1")
    assert sum(c.record_count for c in chunks) == 30
    assert all(len(c.text) <= 300 for c in chunks)


def test_split_pairs_keeps_long_single_line():
    assert split_pairs([("a" * 50, "info")], 10) == [[("a" * 50, "info")]]


def test_severity_counts_are_per_part_when_window_splits():
    records = [record(1, second=s, level="error", event="x" * 40) for s in range(5)]
    records += [record(2, second=s, level="info", event="y" * 40) for s in range(20)]
    chunks = chunk_by_window(records, timedelta(minutes=10), max_chars=300)
    assert sum(c.error_count for c in chunks) == 5


def test_metadata_has_epoch_bounds_for_filtering():
    chunk = chunk_by_window([record(1)], timedelta(minutes=10))[0]
    meta = chunk.metadata()
    assert meta["end_epoch"] - meta["start_epoch"] == 600
    assert meta["record_count"] == 1


def test_format_record():
    assert format_record(record(1, level="warning", event="fps_drop")) == "2025-11-04 09:01:00 WARNING fps_drop"


def test_empty_input():
    assert chunk_by_window([]) == []
