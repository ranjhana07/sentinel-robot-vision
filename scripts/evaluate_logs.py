import argparse
import json
from pathlib import Path

from sentinel.config import Settings, load_config
from sentinel.rag.evaluate import (
    load_examples,
    retrieval_report,
    run_langsmith_evaluation,
    summarise_report,
)
from sentinel.rag.ingest import build_chunks
from sentinel.rag.store import LogStore
from sentinel.rag.summarize import LogSummarizer, build_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="evaluate the log summarizer")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset.jsonl"))
    parser.add_argument("--logs", type=Path, nargs="*", default=[Path("examples/sample_robot.jsonl")])
    parser.add_argument("--mode", choices=["retrieval", "langsmith"], default="retrieval")
    parser.add_argument("--dataset-name", default="sentinel-log-qa")
    parser.add_argument("--prefix", default="sentinel-summaries")
    parser.add_argument("--output", type=Path, default=Path("results/retrieval_eval.json"))
    args = parser.parse_args()

    rag = load_config(args.config).rag
    store = LogStore.from_config(rag)
    store.add_chunks(build_chunks(args.logs, rag.window_minutes, rag.max_chunk_chars))
    examples = load_examples(args.dataset)

    if args.mode == "retrieval":
        rows = retrieval_report(store, examples, rag.top_k)
        report = {"summary": summarise_report(rows), "top_k": rag.top_k, "rows": rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["summary"], indent=2))
        return

    settings = Settings()
    llm = build_llm(settings)
    summarizer = LogSummarizer(store, llm, top_k=rag.top_k)
    run_langsmith_evaluation(summarizer, llm, examples, args.dataset_name, args.prefix)


if __name__ == "__main__":
    main()
