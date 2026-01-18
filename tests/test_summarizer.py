from datetime import datetime, timezone

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from sentinel.rag.store import RetrievedChunk
from sentinel.rag.summarize import LogSummarizer

SINCE = datetime(2025, 11, 4, 9, tzinfo=timezone.utc)
UNTIL = datetime(2025, 11, 4, 10, tzinfo=timezone.utc)


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    def query(self, text, since, until, top_k):
        self.queries.append((text, top_k))
        return self.chunks


def chunk(chunk_id):
    return RetrievedChunk(chunk_id, "09:47:10 WARNING fps_drop fps=4.1", "a", "b", 0, 1)


def test_summary_returns_model_text_and_chunk_ids():
    store = FakeStore([chunk("c1"), chunk("c2")])
    summarizer = LogSummarizer(store, FakeListChatModel(responses=["One fps incident."]), top_k=4)
    summary = summarizer.summarize(SINCE, UNTIL, "Any slowdowns?")
    assert summary.text == "One fps incident."
    assert summary.chunk_ids == ("c1", "c2")
    assert store.queries == [("Any slowdowns?", 4)]


def test_empty_window_skips_the_model():
    summarizer = LogSummarizer(FakeStore([]), FakeListChatModel(responses=["should not be used"]))
    summary = summarizer.summarize(SINCE, UNTIL)
    assert summary.chunk_ids == ()
    assert "No log entries" in summary.text
