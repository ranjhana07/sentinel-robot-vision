import csv
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Sequence

import numpy as np

from sentinel.detection.onnx_engine import build_session


@dataclass(frozen=True)
class BenchResult:
    name: str
    size_mb: float
    runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    fps: float


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


def render_markdown(results: Sequence[BenchResult]) -> str:
    lines = [
        "| model | size (MB) | mean (ms) | p50 (ms) | p95 (ms) | FPS | speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline_fps = results[0].fps if results else 0.0
    for result in results:
        speedup = result.fps / baseline_fps if baseline_fps else 0.0
        lines.append(
            f"| {result.name} | {result.size_mb:.1f} | {result.mean_ms:.1f} "
            f"| {result.p50_ms:.1f} | {result.p95_ms:.1f} | {result.fps:.1f} | {speedup:.2f}x |"
        )
    return "\n".join(lines) + "\n"


def write_csv(results: Sequence[BenchResult], path: Path | str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(BenchResult)])
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_reports(results: Sequence[BenchResult], directory: Path | str) -> tuple[Path, Path]:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    markdown_path = folder / "benchmark.md"
    csv_path = folder / "benchmark.csv"
    markdown_path.write_text(render_markdown(results), encoding="utf-8")
    write_csv(results, csv_path)
    return markdown_path, csv_path
