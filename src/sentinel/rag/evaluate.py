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


JUDGE_PROMPT = (
    "You grade answers to questions about robot logs. Compare the candidate answer with the reference. "
    "Score 1 to 5: 5 means every fact in the reference is present and correct, 3 means the main point is "
    "right but details are missing, 1 means wrong or unsupported.\n\n"
    "Question: {question}\n\nReference: {reference}\n\nCandidate: {candidate}"
)


def build_judge(llm):
    from pydantic import BaseModel, Field

    class Verdict(BaseModel):
        score: int = Field(ge=1, le=5)
        reasoning: str

    return llm.with_structured_output(Verdict)


def make_judge_evaluator(judge):
    def judge_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        verdict = judge.invoke(
            JUDGE_PROMPT.format(
                question=inputs["question"],
                reference=reference_outputs["reference"],
                candidate=outputs["answer"],
            )
        )
        return {"key": "judge_correctness", "score": (verdict.score - 1) / 4, "comment": verdict.reasoning}

    return judge_correctness


def retrieval_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    coverage = mention_coverage(outputs.get("context", ""), reference_outputs.get("must_mention", []))
    return {"key": "retrieval_coverage", "score": coverage}


def ensure_dataset(client, name: str, examples: Sequence[EvalExample]):
    if client.has_dataset(dataset_name=name):
        return client.read_dataset(dataset_name=name)
    dataset = client.create_dataset(name, description="robot log question and reference pairs")
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"question": e.question, "since": e.since, "until": e.until} for e in examples],
        outputs=[{"reference": e.reference, "must_mention": list(e.must_mention)} for e in examples],
    )
    return dataset


def run_langsmith_evaluation(summarizer, judge_llm, examples: Sequence[EvalExample], dataset_name: str, prefix: str):
    from langsmith import Client
    from langsmith.evaluation import evaluate

    from sentinel.rag.summarize import build_context

    client = Client()
    ensure_dataset(client, dataset_name, examples)

    def target(inputs: dict) -> dict:
        since = parse_time_argument(inputs["since"])
        until = parse_time_argument(inputs["until"])
        chunks = summarizer.retrieve(since, until, inputs["question"])
        summary = summarizer.summarize(since, until, inputs["question"])
        return {"answer": summary.text, "context": build_context(chunks)}

    return evaluate(
        target,
        data=dataset_name,
        evaluators=[retrieval_evaluator, make_judge_evaluator(build_judge(judge_llm))],
        experiment_prefix=prefix,
        max_concurrency=2,
    )
