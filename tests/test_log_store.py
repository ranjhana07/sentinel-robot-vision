from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("chromadb")

from chromadb.api.types import Documents, EmbeddingFunction


from sentinel.rag.chunking import chunk_by_window
from sentinel.rag.parsing import LogRecord
from sentinel.rag.store import LogStore


class HashEmbedding(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input: Documents):
        vectors = []
        for text in input:
            vector = [0.0] * 16
            for word in text.lower().split():
                vector[hash(word) % 16] += 1.0
            vectors.append(vector)
        return vectors

    @staticmethod
    def name():
        return "hash-embedding"


@pytest.fixture
def store(tmp_path):
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path))
    instance = LogStore.__new__(LogStore)
    instance._collection = client.get_or_create_collection("logs", embedding_function=HashEmbedding())
    return instance


def records():
    base = datetime(2025, 11, 4, 9, 0, tzinfo=timezone.utc)
    rows = [
        (0, "info", "startup"),
        (12, "error", "tts_failure"),
        (25, "warning", "disk_low"),
        (47, "error", "camera_read_failed"),
    ]
    return [LogRecord(base + timedelta(minutes=m), lvl, ev, ev) for m, lvl, ev in rows]


def test_chunks_are_upserted_once(store):
    chunks = chunk_by_window(records(), timedelta(minutes=10))
    assert store.add_chunks(chunks) == 4
    store.add_chunks(chunks)
    assert store.count() == 4


def test_window_returns_only_overlapping_chunks_in_order(store):
    store.add_chunks(chunk_by_window(records(), timedelta(minutes=10)))
    since = datetime(2025, 11, 4, 9, 10, tzinfo=timezone.utc)
    until = datetime(2025, 11, 4, 9, 30, tzinfo=timezone.utc)
    found = store.window(since, until)
    assert [c.chunk_id[:13] for c in found] == ["20251104T0910", "20251104T0920"]


def test_query_respects_time_filter_and_returns_distance(store):
    store.add_chunks(chunk_by_window(records(), timedelta(minutes=10)))
    since = datetime(2025, 11, 4, 9, 0, tzinfo=timezone.utc)
    until = datetime(2025, 11, 4, 9, 30, tzinfo=timezone.utc)
    found = store.query("tts_failure", since, until, top_k=2)
    assert len(found) == 2
    assert all(c.distance is not None for c in found)
    assert all("camera_read_failed" not in c.text for c in found)


def test_add_nothing_is_a_noop(store):
    assert store.add_chunks([]) == 0
