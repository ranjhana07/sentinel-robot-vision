from datetime import datetime, timezone

from sentinel.rag.evaluate import (
    EvalExample,
    load_examples,
    mention_coverage,
    retrieval_report,
    summarise_report,
)
from sentinel.rag.store import RetrievedChunk


class FakeStore:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def query(self, question, since, until, top_k):
        self.calls.append((question, since, until, top_k))
        return [RetrievedChunk("c", self.text, "a", "b", 0, 0)]


def test_load_examples_from_repo_dataset():
    examples = load_examples("eval/dataset.jsonl")
    assert len(examples) >= 6
    assert all(example.reference and example.must_mention for example in examples)


def test_mention_coverage_is_case_insensitive():
    assert mention_coverage("TTS_Failure at 10:15", ["tts_failure", "audio device busy"]) == 0.5
    assert mention_coverage("anything", []) == 1.0


def test_retrieval_report_parses_window_and_scores():
    example = EvalExample("q?", "2025-11-04T09:00:00Z", "2025-11-04T10:00:00Z", "ref", ("disk_low",))
    store = FakeStore("15:40 WARNING disk_low free_mb=210")
    rows = retrieval_report(store, [example], top_k=4)
    assert rows[0]["hit"] is True
    assert store.calls[0][1] == datetime(2025, 11, 4, 9, tzinfo=timezone.utc)
    assert store.calls[0][3] == 4


def test_summarise_report():
    rows = [{"hit": True, "coverage": 1.0}, {"hit": False, "coverage": 0.5}]
    assert summarise_report(rows) == {"questions": 2, "hit_rate": 0.5, "mean_coverage": 0.75}
    assert summarise_report([])["questions"] == 0
