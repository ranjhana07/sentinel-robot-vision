import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sentinel.detection.onnx_engine import build_session


@dataclass(frozen=True)
class BenchResult:
    name: str
    size_mb: float
    runs: int
    mean_ms: float
    fps: float


def benchmark_model(
    model_path: Path | str,
    input_size: int = 416,
    runs: int = 100,
    threads: int = 4,
) -> BenchResult:
    path = Path(model_path)
    session = build_session(path, intra_op_threads=threads)
    input_name = session.get_inputs()[0].name
    sample = np.random.default_rng(0).random((1, 3, input_size, input_size), dtype=np.float32)
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        session.run(None, {input_name: sample})
        timings.append((time.perf_counter() - started) * 1000.0)
    mean_ms = sum(timings) / len(timings)
    return BenchResult(
        name=path.name,
        size_mb=path.stat().st_size / 1e6,
        runs=runs,
        mean_ms=mean_ms,
        fps=1000.0 / mean_ms,
    )


def render_markdown(results: list[BenchResult]) -> str:
    lines = [
        "| model | size (MB) | mean (ms) | FPS |",
        "| --- | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.size_mb:.1f} | {result.mean_ms:.1f} | {result.fps:.1f} |"
        )
    return "\n".join(lines) + "\n"
