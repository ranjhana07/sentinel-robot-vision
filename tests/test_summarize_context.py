from datetime import datetime, timezone

from sentinel.rag.store import RetrievedChunk, to_retrieved, window_filter
from sentinel.rag.summarize import build_context


def chunk(start, text="line", chunk_id="c"):
    return RetrievedChunk(chunk_id, text, start, start, 0, 0)


def test_context_labels_each_chunk_with_its_range():
    context = build_context([chunk("2025-11-04T09:00:00+00:00", "alpha"), chunk("2025-11-04T09:10:00+00:00", "beta")])
    assert context.startswith("[2025-11-04T09:00:00+00:00 to 2025-11-04T09:00:00+00:00]\nalpha")
    assert "beta" in context


def test_context_stops_at_character_budget_but_keeps_first_chunk():
    chunks = [chunk(f"t{i}", "x" * 100) for i in range(10)]
    context = build_context(chunks, max_chars=250)
    assert 1 <= context.count("x" * 100) <= 2
    assert build_context(chunks[:1], max_chars=10).count("x" * 100) == 1


def test_window_filter_uses_epoch_bounds():
    since = datetime(2025, 11, 4, 9, tzinfo=timezone.utc)
    until = datetime(2025, 11, 4, 10, tzinfo=timezone.utc)
    clauses = window_filter(since, until)["$and"]
    assert clauses[0] == {"end_epoch": {"$gt": since.timestamp()}}
    assert clauses[1] == {"start_epoch": {"$lt": until.timestamp()}}


def test_to_retrieved_maps_metadata():
    meta = {"start": "a", "end": "b", "error_count": 2, "warning_count": 1}
    results = to_retrieved(["x"], ["body"], [meta], [0.25])
    assert results[0].error_count == 2
    assert results[0].distance == 0.25
    assert to_retrieved(["x"], ["body"], [meta])[0].distance is None
