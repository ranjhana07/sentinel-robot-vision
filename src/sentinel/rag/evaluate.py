import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sentinel.rag.timeparse import parse_time_argument


@dataclass(frozen=True)
class EvalExample:
    question: str
    since: str
    until: str
    reference: str
    must_mention: tuple[str, ...]


def load_examples(path: Path | str) -> list[EvalExample]:
    examples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            examples.append(
                EvalExample(
                    question=raw["question"],
                    since=raw["since"],
                    until=raw["until"],
                    reference=raw["reference"],
                    must_mention=tuple(raw.get("must_mention", [])),
                )
            )
    return examples


def mention_coverage(text: str, must_mention: Sequence[str]) -> float:
    if not must_mention:
        return 1.0
    lowered = text.lower()
    found = sum(1 for needle in must_mention if needle.lower() in lowered)
    return found / len(must_mention)


def retrieval_report(store, examples: Sequence[EvalExample], top_k: int) -> list[dict]:
    rows = []
    for example in examples:
        since = parse_time_argument(example.since)
        until = parse_time_argument(example.until)
        chunks = store.query(example.question, since, until, top_k)
        context = "\n".join(chunk.text for chunk in chunks)
        rows.append(
            {
                "question": example.question,
                "retrieved_chunks": len(chunks),
                "coverage": mention_coverage(context, example.must_mention),
                "hit": mention_coverage(context, example.must_mention) == 1.0,
            }
        )
    return rows


def summarise_report(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"questions": 0, "hit_rate": 0.0, "mean_coverage": 0.0}
    return {
        "questions": len(rows),
        "hit_rate": sum(1 for row in rows if row["hit"]) / len(rows),
        "mean_coverage": sum(row["coverage"] for row in rows) / len(rows),
    }
