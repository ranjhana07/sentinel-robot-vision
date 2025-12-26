import csv
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Sequence

import numpy as np

from sentinel.detection.onnx_engine import build_session
from sentinel.sysinfo import environment_info


@dataclass(frozen=True)
class BenchResult:
    name: str
    size_mb: float
    runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    fps: float
    median_fps: float


def latency_stats(samples_ms: Sequence[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    if values.size == 0:
        raise ValueError("no latency samples")
    mean_ms = float(values.mean())
    return {
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "fps": 1000.0 / mean_ms,
        "median_fps": 1000.0 / float(np.percentile(values, 50)),
    }


def benchmark_model(
    model_path: Path | str,
    input_size: int = 416,
    runs: int = 100,
    warmup: int = 10,
    threads: int = 4,
) -> BenchResult:
    path = Path(model_path)
    session = build_session(path, intra_op_threads=threads)
    input_name = session.get_inputs()[0].name
    sample = np.random.default_rng(0).random((1, 3, input_size, input_size), dtype=np.float32)
    for _ in range(warmup):
        session.run(None, {input_name: sample})
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        session.run(None, {input_name: sample})
        timings.append((time.perf_counter() - started) * 1000.0)
    return BenchResult(
        name=path.name,
        size_mb=path.stat().st_size / 1e6,
        runs=runs,
        **latency_stats(timings),
    )


def benchmark_interleaved(
    model_paths: Sequence[Path | str],
    input_size: int = 416,
    runs: int = 100,
    warmup: int = 10,
    threads: int = 4,
    rounds: int = 5,
) -> list[BenchResult]:
    sessions = [build_session(path, intra_op_threads=threads) for path in model_paths]
    sample = np.random.default_rng(0).random((1, 3, input_size, input_size), dtype=np.float32)
    feeds = [{session.get_inputs()[0].name: sample} for session in sessions]
    for session, feed in zip(sessions, feeds):
        for _ in range(max(warmup, 1)):
            session.run(None, feed)
    timings: list[list[float]] = [[] for _ in sessions]
    per_round = max(runs // rounds, 1)
    for _ in range(rounds):
        for index, (session, feed) in enumerate(zip(sessions, feeds)):
            for _ in range(per_round):
                started = time.perf_counter()
                session.run(None, feed)
                timings[index].append((time.perf_counter() - started) * 1000.0)
    return [
        BenchResult(
            name=Path(path).name,
            size_mb=Path(path).stat().st_size / 1e6,
            runs=len(samples),
            **latency_stats(samples),
        )
        for path, samples in zip(model_paths, timings)
    ]


def benchmark_pytorch(
    weights_path: Path | str,
    input_size: int = 416,
    runs: int = 100,
    warmup: int = 10,
    threads: int = 4,
) -> BenchResult:
    import torch
    from ultralytics import YOLO

    path = Path(weights_path)
    torch.set_num_threads(threads)
    network = YOLO(str(path)).model.float().eval()
    sample = torch.rand(1, 3, input_size, input_size)
    with torch.inference_mode():
        for _ in range(warmup):
            network(sample)
        timings = []
        for _ in range(runs):
            started = time.perf_counter()
            network(sample)
            timings.append((time.perf_counter() - started) * 1000.0)
    return BenchResult(
        name=f"{path.name} (pytorch)",
        size_mb=path.stat().st_size / 1e6,
        runs=runs,
        **latency_stats(timings),
    )


def render_markdown(results: Sequence[BenchResult]) -> str:
    lines = [
        "| model | size (MB) | mean (ms) | p50 (ms) | p95 (ms) | FPS (mean) | FPS (median) | speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline_fps = results[0].median_fps if results else 0.0
    for result in results:
        speedup = result.median_fps / baseline_fps if baseline_fps else 0.0
        lines.append(
            f"| {result.name} | {result.size_mb:.1f} | {result.mean_ms:.1f} "
            f"| {result.p50_ms:.1f} | {result.p95_ms:.1f} | {result.fps:.1f} "
            f"| {result.median_fps:.1f} | {speedup:.2f}x |"
        )
    return "\n".join(lines) + "\n"


def write_csv(results: Sequence[BenchResult], path: Path | str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(BenchResult)])
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_json(
    results: Sequence[BenchResult],
    path: Path | str,
    settings: dict[str, int | str],
    environment: dict[str, str | int],
) -> None:
    payload = {
        "environment": environment,
        "settings": settings,
        "results": [asdict(result) for result in results],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_reports(
    results: Sequence[BenchResult],
    directory: Path | str,
    settings: dict[str, int | str] | None = None,
    stem: str = "benchmark",
) -> tuple[Path, Path, Path]:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    markdown_path = folder / f"{stem}.md"
    csv_path = folder / f"{stem}.csv"
    json_path = folder / f"{stem}.json"
    markdown_path.write_text(render_markdown(results), encoding="utf-8")
    write_csv(results, csv_path)
    write_json(results, json_path, settings or {}, environment_info())
    return markdown_path, csv_path, json_path
